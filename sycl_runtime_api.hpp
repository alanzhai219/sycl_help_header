#include <sycl/sycl.hpp>

#include <vector>
#include <unordered_map>
#include <stack>
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

/// cube is a helpful wrapper mapping to cuda spec.
/// cube idx(it);
/// idx.threadIdx.x;
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

/// 1. define kernel
/// void vec_add(sycl::nd_item<3> it, float* c, float* a, float* b, int size) {
///   int i = it.get_global_id(2);
///   for (i < size) {
///     c[i] = a[i] + b[i];
///   }
/// }
/// 2. instance kernel
/// auto func = [=](sycl::nd_item<3> nit) { vec_add(nit, c, a, b); };
/// 3. launch kernel
/// sycl_kernel_launch(q, gws, lws, func);

template <typename KernelObject>
void sycl_kernel_launch(sycl::queue& q, sycl::range<3> gws, sycl::range<3> lws, KernelObject ko) {
    q.parallel_for(sycl::nd_range<3>(gws, lws), ko).wait();
}

/// device manager
namespace detail {
class dev_mgr {
public:
    // 单例实例
    static dev_mgr& instance() {
        static dev_mgr d_m;
        return d_m;
    }

    // 删除拷贝和移动构造/赋值
    dev_mgr(const dev_mgr&) = delete;
    dev_mgr& operator=(const dev_mgr&) = delete;
    dev_mgr(dev_mgr&&) = delete;
    dev_mgr& operator=(dev_mgr&&) = delete;

    // 析构函数
    ~dev_mgr() {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        destroy_queues();
    }

public:
    // dev_mgr 设备管理功能
    void list_devices() const {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        for (size_t i = 0; i < m_devs.size(); ++i) {
            std::cout << i << ": " << m_devs[i]->get_info<sycl::info::device::name>() << std::endl;
        }
    }

    void set_device(unsigned int id) {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        if (m_dev_stack.empty()) {
            push_device(id);
        } else {
            check_id(id);
            m_dev_stack.top() = id;
            update_queues(); // 切换设备时更新队列
        }
    }

    sycl::device& get_device(unsigned int id) const {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        check_id(id);
        return *m_devs[id];
    }

    sycl::device& get_cpu_device() const {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        if (_cpu_device == -1) {
            throw std::runtime_error("no valid cpu device");
        }
        return *m_devs[_cpu_device];
    }

    sycl::device& get_current_device() {
        return get_device(current_device_id());
    }

    unsigned int device_count() const {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        return m_devs.size();
    }

    unsigned int get_device_id(const sycl::device& dev) const {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        unsigned int id = 0;
        for (const auto& dev_item : m_devs) {
            if (*dev_item == dev) {
                return id;
            }
            id++;
        }
        throw std::runtime_error(
            "The device[" + dev.get_info<sycl::info::device::name>() +
            "] is filtered out in current device list!");
    }

    // late initialization
    void push_device(unsigned int id) {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        check_id(id);
        m_dev_stack.push(id);
        update_queues(); // 更新队列以匹配新设备
    }

    // early clear
    unsigned int pop_device() {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        if (m_dev_stack.empty()) {
            throw std::runtime_error("can't pop an empty dpct device stack");
        }
        auto id = m_dev_stack.top();
        m_dev_stack.pop();
        update_queues(); // 更新队列以匹配栈顶设备
        return id;
    }

    // queue manager
		sycl::queue& in_order_queue() {
				std::lock_guard<std::mutex> lock(m_queue_mutex);
        sycl::device& cur_dev = get_current_device();
				if (!m_dq_map[cur_dev]->q_in_order) {
						init_queues();
				}
				return *m_dq_map[cur_dev]->q_in_order;
		}

		sycl::queue& out_of_order_queue() {
				std::lock_guard<std::mutex> lock(m_queue_mutex);
        sycl::device& cur_dev = get_current_device();
				if (!m_dq_map[cur_dev]->q_out_of_order) {
						init_queues();
				}
				return *m_dq_map[cur_dev]->q_out_of_order;
		}

		sycl::queue& default_queue() {
				std::lock_guard<std::mutex> lock(m_queue_mutex);
        sycl::device& cur_dev = get_current_device();
				if (!m_dq_map[cur_dev]->q_default) {
						init_queues();
				}
				return *m_dq_map[cur_dev]->q_default;
		}


private:
    // 单例构造函数
    dev_mgr() {
        // init device
        sycl::device default_device = sycl::device(sycl::default_selector_v);
        m_dq_map[default_device] = std::make_shared<device_queue>();
        m_dq_map[default_device]->ctx = sycl::context(default_device);
        m_devs.push_back(std::make_shared<sycl::device>(default_device));
        m_ctx = sycl::context(default_device);

        std::vector<sycl::device> sycl_all_devs = sycl::device::get_devices(sycl::info::device_type::all);
        if (default_device.is_cpu()) {
            _cpu_device = 0;
        }
        for (const auto& dev : sycl_all_devs) {
            if (dev == default_device) {
                continue;
            }

            m_dq_map[dev] = std::make_shared<device_queue>();
            m_dq_map[dev]->ctx = sycl::context(dev);

            m_devs.push_back(std::make_shared<sycl::device>(dev));
            if (_cpu_device == -1 && dev.is_cpu()) {
                _cpu_device = m_devs.size() - 1;
            }
        }

        // init queue
        init_queues();
    }

    unsigned int current_device_id() const {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        unsigned int id = m_dev_stack.empty() ? DEFAULT_DEVICE_ID : m_dev_stack.top();
        std::cout << "[debug] id = " << id << "\n";
        return id;
    }

    void check_id(unsigned int id) const {
        std::lock_guard<std::recursive_mutex> lock(m_dev_mutex);
        if (id >= m_devs.size()) {
            throw std::runtime_error("invalid device id");
        }
    }

    // 队列管理
    void init_queues() {
        sycl::device& cur_dev = get_current_device();
        m_dq_map[cur_dev]->q_in_order = std::make_unique<sycl::queue>(m_dq_map[cur_dev]->ctx, cur_dev, sycl::property::queue::in_order{});
        m_dq_map[cur_dev]->q_out_of_order = std::make_unique<sycl::queue>(m_dq_map[cur_dev]->ctx, cur_dev);
        m_dq_map[cur_dev]->q_default = std::make_unique<sycl::queue>(m_dq_map[cur_dev]->ctx, cur_dev);
    }

    void clear_queues() {
        sycl::device& cur_dev = get_current_device();
        m_dq_map[cur_dev]->q_in_order.reset();
        m_dq_map[cur_dev]->q_out_of_order.reset();
        m_dq_map[cur_dev]->q_default.reset();
    }

    void update_queues() {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        clear_queues();
        init_queues();
    }

    void destroy_queues() {
      for (auto cur_dev :  m_devs) {
        m_dq_map[*cur_dev]->q_in_order.reset();
        m_dq_map[*cur_dev]->q_out_of_order.reset();
        m_dq_map[*cur_dev]->q_default.reset();
      }
    }

private:
    struct device_queue {
      std::unique_ptr<sycl::queue> q_in_order = nullptr;
      std::unique_ptr<sycl::queue> q_out_of_order = nullptr;
      std::unique_ptr<sycl::queue> q_default = nullptr;
      sycl::context ctx;
    };
    // 设备管理数据
    std::unordered_map<sycl::device, std::shared_ptr<device_queue>> m_dq_map;

    std::vector<std::shared_ptr<sycl::device>> m_devs;
    static inline thread_local std::stack<unsigned int> m_dev_stack;
    const unsigned int DEFAULT_DEVICE_ID = 0;
    int _cpu_device = -1;
    mutable std::recursive_mutex m_dev_mutex;

    // 队列管理数据（来自 device_ext）
    sycl::context m_ctx;
    std::mutex m_queue_mutex;
};
} // detail

// device
static void sycl_list_devices() {
  return detail::dev_mgr::instance().list_devices();
}

static void sycl_set_device(unsigned int id) {
  return detail::dev_mgr::instance().set_device(id);
}

static inline sycl::device& sycl_get_current_device() {
  return detail::dev_mgr::instance().get_current_device();
}

static inline sycl::device& sycl_get_cpu_device() {
  return detail::dev_mgr::instance().get_cpu_device();
}

static inline sycl::device& sycl_get_device(unsigned int id) {
  return detail::dev_mgr::instance().get_device(id);
}

static inline unsigned int sycl_get_device_id(sycl::device& dev) {
  return detail::dev_mgr::instance().get_device_id(dev);
}

// queue
static inline sycl::queue& sycl_get_in_order_queue() {
  return detail::dev_mgr::instance().in_order_queue();
}

static inline sycl::queue& sycl_get_out_of_order_queue() {
  return detail::dev_mgr::instance().out_of_order_queue();
}

static inline sycl::queue& sycl_get_default_quque() {
  return detail::dev_mgr::instance().default_queue();
}