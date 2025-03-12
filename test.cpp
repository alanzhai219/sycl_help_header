#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

// Assuming the sycl_runtime_api.hpp file is available and included
#include "sycl_runtime_api.hpp"

void vec_add_kernel(sycl::nd_item<3> item, float* a, float* b, float* c, size_t N) {
    size_t idx = item.get_global_id(2); // 获取全局线程ID
    if (idx < N) {                      // 边界检查
        c[idx] = a[idx] + b[idx];       // 向量加法
    }
}

int main() {
  // Device selection
  try {
    sycl_list_devices(); // list devices
    sycl_set_device(0); // select device 0 (if you have multiple devices) , you can change the id here.
    sycl::device& dev = sycl_get_current_device();
    std::cout << "Running on device: " << dev.get_info<sycl::info::device::name>() << std::endl;
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  }

  // Data size
  const size_t data_size = 1024;

  // Command queue
  sycl::queue& q = sycl_get_default_quque();
  sycl::device dev = q.get_device();

  // USM allocation
  float* a = sycl::malloc_shared<float>(data_size, q);
  float* b = sycl::malloc_shared<float>(data_size, q);
  float* c = sycl::malloc_shared<float>(data_size, q);

  if (a == nullptr || b == nullptr || c == nullptr) {
    std::cerr << "USM allocation failed!" << std::endl;
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return 1;
  }

  // Initialize data (on host or device)
  for (size_t i = 0; i < data_size; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
    c[i] = 0.0f;
  }
  
    //kernel
  try {
    // q.submit([&](sycl::handler& h) {
    //   h.parallel_for<class vector_add_usm>(sycl::range<1>(data_size), [=](sycl::id<1> i) {
    //     c[i] = a[i] + b[i];
    //   });
    // }).wait(); // Wait for the kernel to complete.
    sycl::range<3> gws{1,1,1024};
    sycl::range<3> lws{1,1,256};
    auto func = [=](sycl::nd_item<3> nit) { vec_add_kernel(nit, a, b, c, data_size);};
    sycl_kernel_launch(q, gws, lws, func);
    
      // Verification
    bool passed = true;
    for (size_t i = 0; i < data_size; ++i) {
      if (c[i] != 3.0f) {
        passed = false;
        std::cerr << "Error at index " << i << ": Expected 3.0, got " << c[i] << std::endl;
      }
    }

    if (passed) {
      std::cout << "Vector addition (USM) test passed!" << std::endl;
    } else {
      std::cerr << "Vector addition (USM) test failed!" << std::endl;
    }
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
  }
  

  // Free USM memory
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);

  return 0;
}
