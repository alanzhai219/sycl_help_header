# sycl_help_api

It is a library to help understand and develop sycl like cuda. Many concept in sycl will be re-deigned like what cuda does.

## map cuda build-in variable

| cuda | sycl |
| ---- | ---- |
| `warpSize` | `subSize(it)` |
| `threadIdx.x/y/z` | `threadIdx_x/y/z(it)` |
| `blockDim.x/y/z` | `blockDim_x/y/z(it)` |
| `blockIdx.x/y/z` | `blockIdx_x/y/z(it)` |
| `gridDim.x/y/z` | `gridDim_x/y/z(it)` |

> node: `it` means `sycl::nd_item<size_t>`
