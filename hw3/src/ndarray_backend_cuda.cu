#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

constexpr size_t TILE = 4;
constexpr size_t BASE_THREAD_NUM = 256;
constexpr size_t SQRT_THREAD = 16;
constexpr size_t L = SQRT_THREAD * TILE;
constexpr size_t S = 16;
constexpr size_t DATA_PER_THREAD = S * L / BASE_THREAD_NUM;

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t M,size_t N){
  CudaDims dim;
  dim.block=dim3(SQRT_THREAD,SQRT_THREAD,1);
  dim.grid=dim3((M+L-1)/L,(N+L-1)/L,1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * 用于 compact 操作的 CUDA 内核。该操作应将未压缩输入数组 a 中的单个条目，
   * 有效地映射到压缩输出数组 out 中的对应位置（位于 gid 处）。
   *
   * 参数：
   *   a: 指向输入数组的 CUDA 指针
   *   out: 指向输出数组的 CUDA 指针
   *   size: 输出数组的大小
   *   shape: 输入和输出数组的形状向量（类型为 CudaVec，用于传递给 CUDA 内核）
   *   strides: 输出数组的步长向量
   *   offset: 输出数组的偏移量
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  size_t idx=0,cur=gid;
  for(int dim=shape.size-1;dim>=0;dim--){
      idx+=cur%shape.data[dim]*strides.data[dim];
      cur=(cur-cur%shape.data[dim])/shape.data[dim];
    }
    out[gid]=a[idx+offset];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
* 压缩内存中的数组。与 C++ 版本不同，在 CUDA 中，此操作将主要调用相关的 CUDA 内核。
   * 在本例中，我们阐明了应如何设置（即，我们提供此函数的代码以及 
   * CompactKernel() 函数的原型）。
   * 但是，对于此后的函数，你需要根据执行底层函数的需要自行定义这些内核。
   * 
   * 参数：
   *   a: 输入的非紧凑表示的数组
   *   out: 待写入的紧凑版本的数组
   *   shape: a 和 out 的每个维度的形状
   *   strides: *a* 数组的步幅（不是 out 的步幅，紧凑版本的 out 具有紧凑的步幅）
   *   offset: *a* 数组的偏移量（不是 out 的偏移量，紧凑时其偏移量为零）   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid=blockIdx.x*blockDim.x+threadIdx.x;

  size_t idx=0,cur=gid;
  for(int dim=shape.size-1;dim>=0;dim--){
    idx+=cur%shape.data[dim]*strides.data[dim];
    cur=(cur-cur%shape.data[dim])/shape.data[dim];
  }
  out[idx+offset]=a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
* 使用 CUDA 在一个（非紧凑）数组中设置项。您很可能希望实现一个
   * EwiseSetitemKernel() 函数（与上述类似），来完成实际工作。
   * 
   * 参数：
   *   a：_紧凑_ 数组，其元素将被写入 out
   *   out：要写入的非紧凑数组
   *   shape：a 和 out 各维度的形状
   *   strides：*out* 数组的步幅（而非 a，a 由于是紧凑数组，拥有紧凑的步幅）
   *   offset：*out* 数组的偏移量（而非 a，a 因其紧凑性偏移量为零）   */
  /// BEGIN SOLUTION
  CudaDims dim= CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid,dim.block>>>(a.ptr,out->ptr,out->size,VecToCuda(shape),
                                        VecToCuda(strides),offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(scalar_t val,scalar_t* out,scalar_t size,
                                    CudaVec shape,CudaVec strides,size_t offset){
  size_t gid=blockIdx.x*blockDim.x+threadIdx.x;

  size_t idx=0,cur=gid;
  for(int dim=shape.size-1;dim>=0;dim--){
    idx+=cur%shape.data[dim]*strides.data[dim];
    cur=(cur-cur%shape.data[dim])/shape.data[dim];
  }
  out[idx+offset]=val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items 操作一个（非紧凑）数组
   * 
   * 参数：
   *   size：要写入 out 数组的元素数量（注意，这不是 out.size，
   * 因为 out 是非紧凑子集数组）；它 _会_ 与 shape 中各维度的元素乘积相同，
   * 但为了方便直接在此传入。
   *   val：要写入的标量值
   *   */
  /// BEGIN SOLUTION
  CudaDims dim= CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid,dim.block>>>(val,out->ptr,out->size,VecToCuda(shape),
                                        VecToCuda(strides),offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


#define EWISE(name,expr)\
  __global__ void name##Kernel(const scalar_t* a, const scalar_t* b,scalar_t* out,size_t size){\
    size_t gid=blockIdx.x*blockDim.x+threadIdx.x;\
    if(gid<size) out[gid]=expr;\
  }\
  \
  void name(const CudaArray& a, const CudaArray& b,CudaArray* out){\
    CudaDims dim=CudaOneDim(out->size);  \
    name##Kernel<<<dim.grid,dim.block>>>(a.ptr,b.ptr,out->ptr,out->size); \
  }

#define SCALAR(name,expr)\
  __global__ void name##Kernel(const scalar_t* a, scalar_t val,scalar_t* out,size_t size){\
    size_t gid=blockIdx.x*blockDim.x+threadIdx.x;\
    if(gid<size) out[gid]=expr;\
  }\
  \
  void name(const CudaArray& a,scalar_t val,CudaArray* out){\
    CudaDims dim=CudaOneDim(out->size);  \
    name##Kernel<<<dim.grid,dim.block>>>(a.ptr,val,out->ptr,out->size); \
  }

#define UNARY(name,expr)\
  __global__ void name##Kernel(const scalar_t* a,scalar_t* out,size_t size){\
    size_t gid=blockIdx.x*blockDim.x+threadIdx.x;\
    if(gid<size) out[gid]=expr;\
  }\
  \
  void name(const CudaArray& a,CudaArray* out){\
    CudaDims dim=CudaOneDim(out->size);  \
    name##Kernel<<<dim.grid,dim.block>>>(a.ptr,out->ptr,out->size); \
  }


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

EWISE(EwiseMul,       a[gid] * b[gid])
EWISE(EwiseDiv,       a[gid] / b[gid])
EWISE(EwiseMaximum,   max(a[gid], b[gid]))
EWISE(EwiseEq,        (scalar_t)(a[gid] == b[gid]))
EWISE(EwiseGe,        (scalar_t)(a[gid] >= b[gid]))

SCALAR(ScalarMul,     a[gid] * val)
SCALAR(ScalarDiv,     a[gid] / val)
SCALAR(ScalarPower,   pow(a[gid], val))
SCALAR(ScalarMaximum, max(a[gid], val))
SCALAR(ScalarEq,      (scalar_t)(a[gid] == val))
SCALAR(ScalarGe,      (scalar_t)(a[gid] >= val))

UNARY(EwiseLog,       log(a[gid]))
UNARY(EwiseExp,       exp(a[gid]))
UNARY(EwiseTanh,      tanh(a[gid]))

//s*l/t=s*v*t**0.5/t=

__global__ void MatmulKernel(const scalar_t* A,const scalar_t* BT,scalar_t* out,
                              size_t M,size_t N,size_t P ){
  __shared__ scalar_t sAT[S][L+1],sB[S][L+1];
  scalar_t c[TILE][TILE]={0};
  scalar_t a[TILE],b[TILE];

  size_t xblock=blockIdx.x;
  size_t yblock=blockIdx.y;
  size_t tid=threadIdx.y*SQRT_THREAD+threadIdx.x;
  size_t xthread=threadIdx.x;
  size_t ythread=threadIdx.y;

  for(size_t ni=0;ni<N;ni+=S){

    __syncthreads();
    for(size_t ti=0;ti<DATA_PER_THREAD;ti++){
      size_t ax=xblock*L+tid/S*DATA_PER_THREAD+ti;
      size_t ay=tid%S+ni;
      size_t bx=ni+tid/L*DATA_PER_THREAD+ti;
      size_t by=yblock*L+tid%L;
      sAT[ay-ni][ax-xblock*L]=ax<M&&ay<N?A[ax*N+ay]:0;
      sB[bx-ni][by-yblock*L]=bx<N&&by<P?BT[bx*P+by]:0;
    }
    __syncthreads();

    for(size_t si=0;si<S;si++){

      for(size_t ti=0;ti<TILE;ti++){
        a[ti]=sAT[si][ythread+ti*SQRT_THREAD];
        b[ti]=sB[si][xthread+ti*SQRT_THREAD];
      }

      for(size_t cx=0;cx<TILE;cx++){
        for(size_t cy=0;cy<TILE;cy++){
          c[cx][cy]+=a[cx]*b[cy];
        }
      }
    }

  }

  for(size_t cx=0;cx<TILE;cx++){
    for(size_t cy=0;cy<TILE;cy++){
      size_t ox=xblock*L+cx*SQRT_THREAD+ythread;
      size_t oy=yblock*L+cy*SQRT_THREAD+xthread;
      if(ox<M&&oy<P) out[ox*P+oy]=c[cx][cy];
    }
  }
                              
}


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * 将两个（紧凑）矩阵相乘，结果存入输出（也是紧凑）矩阵中。
   你需要查阅关于基于 GPU 的线性代数的课程讲座和笔记来了解实现方法。
   * 由于 mugrade 最终只评估正确性，你 _可以_ 
   实现一个简单地将输出数组的 (i,j) 条目并行化的版本。
   * 然而，为了真正充分发挥本题的益处，我们鼓励你采用协作式获取、
   共享内存和寄存器分块以及课程笔记中介绍的其他思想。
   * 注意，与 CPU 后端中的分块矩阵乘法函数不同，
   这里你需要实现一个适用于所有尺寸矩阵的单一函数，
   * 无论矩阵尺寸是否为分块大小的倍数。与之前的 CUDA 实现类似，
   本函数主要只是设置内核调用，
   * 你应把具体逻辑实现在一个单独的 MatmulKernel() 调用中。
   * 
   *
   * 参数：
   *   a: 尺寸为 m x n 的紧凑二维数组
   *   b: 尺寸为 n x p 的紧凑二维数组
   *   out: 尺寸为 m x p 的紧凑二维数组，用于写入输出结果
   *   M: a / out 的行数
   *   N: a 的列数 / b 的行数
   *   P: b / out 的列数
   */

  /// BEGIN SOLUTION
  CudaDims dim=CudaTwoDim(M,P);
  MatmulKernel<<<dim.grid,dim.block>>>(a.ptr,b.ptr,out->ptr,M,N,P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a,scalar_t* out,size_t reduce_size,size_t size ){
  size_t gid=blockIdx.x*blockDim.x+threadIdx.x;
  if(gid<size) {
    out[gid]=-INFINITY;
    for(size_t i=0;i<reduce_size;i++){
      out[gid]=fmaxf(out[gid],a[reduce_size*gid+i]);
    }
  }  
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * 通过对 `reduce_size` 个连续块取最大值来进行归约。尽管这样做效率低下，
   * 但为简化起见，你可将每次归约放在单个 CUDA 线程中执行。
   * 
   * 参数：
   *   a：紧凑数组，大小为 a.size = out.size * reduce_size，待归约的对象
   *   out：紧凑数组，用于写入结果
   *   redice_size：待归约维度的大小
   */
  /// 开始解答
  CudaDims dim=CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid,dim.block>>>(a.ptr,out->ptr,reduce_size,out->size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a,scalar_t* out,size_t reduce_size,size_t size ){
  size_t gid=blockIdx.x*blockDim.x+threadIdx.x;
  if(gid<size) {
    out[gid]=0;
    for(size_t i=0;i<reduce_size;i++){
      out[gid]+=a[reduce_size*gid+i];
    }
  }  
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim=CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid,dim.block>>>(a.ptr,out->ptr,reduce_size,out->size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
