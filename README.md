# sycl_help_api

It is a library to help understand and develop sycl like cuda. Many concept in sycl will be re-deigned like what cuda does.

## map cuda build-in variable

| cuda | sycl | sycl |
| ---- | ---- | ---- |
| `warpSize` | `subSize(it)` | |
| `threadIdx.x/y/z` | `threadIdx_x/y/z(it)` | `cube::threadIdx.x/y/z` |
| `blockDim.x/y/z` | `blockDim_x/y/z(it)` | `cube::blockDim.x/y/z` |
| `blockIdx.x/y/z` | `blockIdx_x/y/z(it)` | `cube::blockIdx.x/y/z` |
| `gridDim.x/y/z` | `gridDim_x/y/z(it)` | `cube::gridDim.x/y/z` |

> note: `it` means `sycl::nd_item<size_t>`
> note: `cube index(it); index.threadIdx.x;`
