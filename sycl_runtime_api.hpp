#include <sycl/sycl.hpp>

#include <mutex>
#include <exception>

// map to cuda warpSize
#define SubSize(it)         it.get_sub_group().get_local_range().get(0)

// map to cuda threadIdx.x/y/z 
#define threadIdx_x(it)     it.get_local_id(2)
#define threadIdx_y(it)     it.get_local_id(1)
#define threadIdx_z(it)     it.get_local_id(0)

// map to cuda blockDim.x/y/z 
#define blockDim_x(it)      it.get_local_range().get(2)
#define blockDim_y(it)      it.get_local_range().get(1)
#define blockDim_z(it)      it.get_local_range().get(0)

// map to cuda blockIdx.x/y/z
#define blockIdx_x(it)      it.get_group(2)
#define blockIdx_y(it)      it.get_group(1)
#define blockIdx_z(it)      it.get_group(0)

// map to cuda gridDim.x/y/z
#define gridDim_x(it)       it.get_group_range(2)
#define gridDim_y(it)       it.get_group_range(1)
#define gridDim_z(it)       it.get_group_range(0)

/*
  cube idx(it);
  idx.threadIdx.x;
*/

struct cube {
  struct {
     size_t x, y, z;
  } threadIdx, blockDim, blockIdx, gridDim;
  inline cube (const sycl::nd_item<3>& it) 
      : threadIdx{it.get_local_id(2), it.get_local_id(1), it.get_local_id(0)},
        blockDim{it.get_local_range(2), it.get_local_range(1), it.get_local_range(0)},
        blockIdx{it.get_group(2), it.get_group(1), it.get_group(0)},
        gridDim{it.get_group_range(2), it.get_group_range(1), it.get_group_range(0)} {}
};

/// device manager
namespace detail {

class device_ext : public sycl::device {
public:
  device_ext() : sycl::device(), _ctx(*this) {}
  ~device_ext() {
    std::lock_guard<std::mutex> lock(m_mutex);
    clear_queues();
  }
  device_ext(const sycl::device &base) : sycl::device(base), _ctx(*this) {
    std::lock_guard<std::mutex> lock(m_mutex);
    init_queues();
  }

public:
  void reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    clear_queues();
    init_queues();
  }

  sycl::queue &in_order_queue() {
    return *_q_in_order;
  }

  sycl::queue &out_of_order_queue() {
    return *_q_out_of_order;
  }

  sycl::queue &default_queue() {
#ifdef DPCT_USM_LEVEL_NONE
    return out_of_order_queue();
#else
    return in_order_queue();
#endif // DPCT_USM_LEVEL_NONE
  }

  void queues_wait_and_throw() {
    std::unique_lock<std::mutex> lock(m_mutex);
    std::vector<std::shared_ptr<sycl::queue>> current_queues(_queues);
    lock.unlock();
    for (const auto &q : current_queues) {
      q->wait_and_throw();
    }
    // Guard the destruct of current_queues to make sure the ref count is safe.
    lock.lock();
  }

  sycl::queue *create_queue(bool enable_exception_handler = false) {
#ifdef DPCT_USM_LEVEL_NONE
    return create_out_of_order_queue(enable_exception_handler);
#else
    return create_in_order_queue(enable_exception_handler);
#endif // DPCT_USM_LEVEL_NONE
  }

  sycl::queue *create_in_order_queue(bool enable_exception_handler = false) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return create_queue_impl(enable_exception_handler, sycl::property::queue::in_order());
  }

  sycl::queue *create_out_of_order_queue(bool enable_exception_handler = false) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return create_queue_impl(enable_exception_handler);
  }

  void destroy_queue(sycl::queue *&queue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                  [=](const std::shared_ptr<sycl::queue> &q) -> bool {
                                    return q.get() == queue;
                                  }),
                   _queues.end());
    queue = nullptr;
  }
  sycl::context get_context() const {
    return _ctx;
  }


private:
  void clear_queues() {
    _queues.clear();
    _q_in_order = _q_out_of_order = _saved_queue = nullptr;
  }

  void init_queues() {
    _q_in_order = create_queue_impl(true, sycl::property::queue::in_order());
    _q_out_of_order = create_queue_impl(true);
    _saved_queue = &default_queue();
  }

  /// Caller should acquire resource \p m_mutex before calling this function.
  template <class... Properties>
  sycl::queue *create_queue_impl(bool enable_exception_handler, Properties... properties) {
    sycl::async_handler eh = {};
    if (enable_exception_handler) {
      eh = exception_handler;
    }
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, eh,
        sycl::property_list(
#ifdef SYCL_PROFILING_ENABLED
            sycl::property::queue::enable_profiling(),
#endif
            properties...)));

    return _queues.back().get();
  }

  sycl::queue *_q_in_order, *_q_out_of_order;
  sycl::queue *_saved_queue;
  sycl::context _ctx;
  std::vector<std::shared_ptr<sycl::queue>> _queues;
  mutable std::mutex m_mutex;
};

class dev_mgr {
public:
  /// Returns the instance of device manager singleton.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }

  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;

public:
  sycl::device& get_current_device() {
    return get_device(current_device_id());
  }

  sycl::device& get_cpu_device() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (_cpu_device == -1) {
      throw std::runtime_error("no valid cpu device");
    } else {
      return *_devs[_cpu_device];
    }
  }

  sycl::device& get_device(unsigned int id) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    check_id(id);
    return *_devs[id];
  }

  /// Select device with a device ID.
  /// \param [in] id The id of the device which can
  /// be obtained through get_device_id(const sycl::device).
  void select_device(unsigned int id) {
    /// Replace the top of the stack with the given device id
    if (_dev_stack.empty()) {
      push_device(id);
    } else {
      check_id(id);
      _dev_stack.top() = id;
    }
  }

  unsigned int device_count() {
    return _devs.size();
  }

  unsigned int get_device_id(const sycl::device &dev) {
    unsigned int id = 0;
    for (auto dev_item : _devs) {
      if (*dev_item == dev) {
        return id;
      }
      id++;
    }
    throw std::runtime_error(
        "The device[" + dev.get_info<sycl::info::device::name>() +
        "] is filtered out by dpct::dev_mgr::filter/dpct::filter_device in "
        "current device "
        "list!");
  }

  /// List all the devices with its id in dev_mgr.
  void list_devices() const {
    for (size_t i = 0; i < _devs.size(); ++i) {
      std::cout << "" << i << ": " << _devs[i]->get_info<sycl::info::device::name>() << std::endl;
    }
  }

  /// Update the device stack for the current thread id
  void push_device(unsigned int id) {
    check_id(id);
    _dev_stack.push(id);
  }

  /// Remove the device from top of the stack if it exist
  unsigned int pop_device() {
    if (_dev_stack.empty()) {
      throw std::runtime_error("can't pop an empty dpct device stack");
    }

    auto id = _dev_stack.top();
    _dev_stack.pop();
    return id;
  }

private:
  dev_mgr() {
    sycl::device default_device = sycl::device(sycl::default_selector_v);
    _devs.push_back(std::make_shared<sycl::device>(default_device));

    std::vector<sycl::device> sycl_all_devs = sycl::device::get_devices(sycl::info::device_type::all);
    // Collect other devices except for the default device.
    if (default_device.is_cpu()) {
      _cpu_device = 0;
    }
    for (auto &dev : sycl_all_devs) {
      if (dev == default_device) {
        continue;
      }
      _devs.push_back(std::make_shared<sycl::device>(dev));
      if (_cpu_device == -1 && dev.is_cpu()) {
        _cpu_device = _devs.size() - 1;
      }
    }
  }

  void check_id(unsigned int id) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (id >= _devs.size()) {
      throw std::runtime_error("invalid device id");
    }
  }

  std::vector<std::shared_ptr<sycl::device>> _devs;
  /// stack of devices resulting from CUDA context change;
  static inline thread_local std::stack<unsigned int> _dev_stack;
  /// DEFAULT_DEVICE_ID is used, if current_device_id() finds an empty
  /// _dev_stack, which means the default device should be used for the current
  /// thread.
  const unsigned int DEFAULT_DEVICE_ID = 0;
  int _cpu_device = -1;
  mutable std::recursive_mutex m_mutex;
};
} // detail

// device
static inline sycl::device& get_current_device() {
  return detail::dev_mgr::instance().get_current_device();
}

static inline sycl::device& get_default_device() {
  return detail::dev_mgr::instance().get_default_device();
}

static inline sycl::device& get_device(unsigned int id) {
  return detail::dev_mgr::instance().get_device(id);
}

static inline unsigned int get_device_id(sycl::device& dev) {
  return detail::dev_mgr::instance().get_device_id(dev);
}

// queue
static inline sycl::queue& get_current_queue() {
  return detail::dev_mgr::instance().get_current_device()
}


