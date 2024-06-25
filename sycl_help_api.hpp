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
