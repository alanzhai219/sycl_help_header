#include <stdlib.h>
#include "../../sycl_runtime_api.hpp"

void vectorAdd(sycl::nd_item<3>& nit, const float* a, const float* b, float* c, int n) {
  cube index(nit);
  int i = index.blockIdx.x * index.blockDim.x + index.threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
    unsigned int device_id = 0;
    sycl_set_device(0);

    sycl::queue& q = sycl_get_in_order_queue();

    size_t n = 10000000;
    size_t size = n * sizeof(float);

    // host vectors
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate device memory
    d_a = (float*)sycl::malloc_device(size, q);
    d_b = (float*)sycl::malloc_device(size, q);
    d_c = (float*)sycl::malloc_device(size, q);

    // Copy host vectors to device
    q.memcpy(d_a, h_a, size).wait();
    q.memcpy(d_b, h_b, size).wait();

    // Set up execution configuration
    int local_work_item = 256;
    int global_work_item = ((n + local_work_item - 1) / local_work_item) * local_work_item;
    sycl::range<3> lws{1,1,256};
    sycl::range<3> gws{1,1,(size_t)global_work_item};

    // Launch kernel
    auto func = [=](sycl::nd_item<3> nit){ vectorAdd(nit, d_a, d_b, d_c, n); };
    sycl_launch_kernel(q, gws, lws, func);

    // Copy result back to host
    q.memcpy(h_c, d_c, size).wait();

    // Verify results (check first few elements)
    for (int i = 0; i < 5; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free device memory
    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
