#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
* 这是一个实用结构，用于在内存中维护与 ALIGNMENT 边界对齐的数组。
该对齐应至少为 TILE * ELEM_SIZE，不过我们在此默认情况下将其设得更大。
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * 用 val 填充对齐数组的值
   */
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
*   在内存中压缩数组
   *
   * 参数:
   *   a: 数组的非紧凑表示，作为输入给出
   *   out: 将要写入的数组的紧凑版本
   *   shape: a 和 out 每个维度的形状
   *   strides: *a* 数组的步幅（不是 out，out 具有紧凑步幅）
   *   offset: *a* 数组的偏移量（不是 out，out 由于是紧凑的，偏移量为零）
   *
   * 返回:
   *  void（你需要直接修改 out，而不是返回任何东西；
   * 对于这里将要实现的所有函数都是如此，因此我们不会重复此说明。）
   */
  /// BEGIN SOLUTION
  int64_t len=1;
  for(size_t dim =0;dim<shape.size();dim++){
    len*=shape[dim];
  }
  for(int64_t cnt=0;cnt<len;cnt++){
    size_t idx=0,cur=cnt;
    for(int dim=shape.size()-1;dim>=0;dim--){
      idx+=cur%shape[dim]*strides[dim];
      cur=(cur-cur%shape[dim])/shape[dim];
    }
    out->ptr[cnt]=a.ptr[idx+offset];
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * 在（非紧凑）数组中设置元素
   *
   * 参数：
   *   a：_紧凑_ 数组，其元素将被写入 out
   *   out：非紧凑数组，其元素将被写入
   *   shape：a 和 out 每个维度的形状
   *   strides：*out* 数组的步幅（而不是 a，a 具有紧凑步幅）
   *   offset：*out* 数组的偏移量（而不是 a，由于紧凑，其偏移量为零）
   */
  /// 开始解答
  int64_t len=1;
  for(size_t dim =0;dim<shape.size();dim++){
    len*=shape[dim];
  }
  for(int64_t cnt=0;cnt<len;cnt++){
    size_t idx=0,cur=cnt;
    for(int dim=shape.size()-1;dim>=0;dim--){
      idx+=cur%shape[dim]*strides[dim];
      cur=(cur-cur%shape[dim])/shape[dim];
    }
    out->ptr[idx+offset]=a.ptr[cnt];
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  for(int64_t cnt=0;cnt<size;cnt++){
    size_t idx=0,cur=cnt;
    for(int dim=shape.size()-1;dim>=0;dim--){
      idx+=cur%shape[dim]*strides[dim];
      cur=(cur-cur%shape[dim])/shape[dim];
    }
    out->ptr[idx+offset]=val;
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
""* 在后续的代码中，请使用上述模板为下列函数创建类似的逐元素运算符和标量运算符。
 * 关于这些函数应如何运作，可参考 NumPy 后端中的实现示例。
 *   - EwiseMul（逐元素乘法）、ScalarMul（标量乘法）
 *   - EwiseDiv（逐元素除法）、ScalarDiv（标量除法）
 *   - ScalarPower（标量幂运算）
 *   - EwiseMaximum（逐元素取最大值）、ScalarMaximum（标量取最大值）
 *   - EwiseEq（逐元素等于判断）、ScalarEq（标量等于判断）
 *   - EwiseGe（逐元素大于等于判断）、ScalarGe（标量大于等于判断）
 *   - EwiseLog（逐元素对数运算）
 *   - EwiseExp（逐元素指数运算）
 *   - EwiseTanh（逐元素双曲正切运算）
 *
 * 如果完全按朴素方式实现所有这些函数，会出现大量重复代码，
 * 因此你欢迎（但非强制）使用宏或模板来定义这些函数（具体实现方式不限，
 * 只要函数匹配正确的签名即可）。""
 */

// 定义宏：逐元素操作
#define DEFINE_EWISE_OP(name, expr) \
  void name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
    for (size_t i = 0; i < a.size; i++) { \
      out->ptr[i] = expr; \
    } \
  }

// 定义宏：标量操作
#define DEFINE_SCALAR_OP(name, expr) \
  void name(const AlignedArray& a, scalar_t val, AlignedArray* out) { \
    for (size_t i = 0; i < a.size; i++) { \
      out->ptr[i] = expr; \
    } \
  }

// 定义宏：单参数逐元素操作（如 log, exp, tanh）
#define DEFINE_UNARY_OP(name, expr) \
  void name(const AlignedArray& a, AlignedArray* out) { \
    for (size_t i = 0; i < a.size; i++) { \
      out->ptr[i] = expr; \
    } \
  }

// 使用：一行一个函数

DEFINE_EWISE_OP(EwiseMul,      a.ptr[i] * b.ptr[i])
DEFINE_EWISE_OP(EwiseDiv,      a.ptr[i] / b.ptr[i])
DEFINE_EWISE_OP(EwiseMaximum,  std::max(a.ptr[i], b.ptr[i]))
DEFINE_EWISE_OP(EwiseEq,       (scalar_t)(a.ptr[i] == b.ptr[i]))
DEFINE_EWISE_OP(EwiseGe,       (scalar_t)(a.ptr[i] >= b.ptr[i]))


DEFINE_SCALAR_OP(ScalarMul,    a.ptr[i] * val)
DEFINE_SCALAR_OP(ScalarDiv,    a.ptr[i] / val)
DEFINE_SCALAR_OP(ScalarPower,  std::pow(a.ptr[i], val))
DEFINE_SCALAR_OP(ScalarMaximum, std::max(a.ptr[i], val))
DEFINE_SCALAR_OP(ScalarEq,     (scalar_t)(a.ptr[i] == val))
DEFINE_SCALAR_OP(ScalarGe,     (scalar_t)(a.ptr[i] >= val))

DEFINE_UNARY_OP(EwiseLog,      std::log(a.ptr[i]))
DEFINE_UNARY_OP(EwiseExp,      std::exp(a.ptr[i]))
DEFINE_UNARY_OP(EwiseTanh,     std::tanh(a.ptr[i]))




void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for(int i=0;i<m;i++){
    for(int j=0;j<p;j++){
      out->ptr[i*p+j]=0;
      for(int k=0;k<n;k++){
        out->ptr[i*p+j]+=a.ptr[i*n+k]*b.ptr[k*p+j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * 将两个 TILE x TILE 矩阵相乘，并将结果_加_到 out 中
    （重要的是将结果加到现有的 out 上，而不应事先将 out 清零）。
    我们在此包含了编译器标志，使编译器能够正确使用向量操作符来实现此功能。
    具体来说，__restrict__ 关键字向编译器表明 a、b 和 out 之间没有内存重叠
    （这是确保向量操作与非向量化版本等效的必要条件
    ——试想如果 a、b 和 out 的内存重叠可能会发生什么）。
    同样，__builtin_assume_aligned 关键字告诉
    编译器输入数组将与内存中的适当块对齐，这也有助于编译器对代码进行向量化。
   *
   * 参数：
   *   a：大小为 TILE x TILE 的紧凑二维数组
   *   b：大小为 TILE x TILE 的紧凑二维数组
   *   out：大小为 TILE x TILE 的紧凑二维数组，用于写入结果
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for(int i=0;i<TILE;i++){
    for(int j=0;j<TILE;j++){
      for(int k=0;k<TILE;k++){
        out[i*TILE+j]+=a[i*TILE+k]*b[k*TILE+j];
      }
    }
  }
  /// END SOLUTION
}

#define TILESQ TILE*TILE

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
    * 基于数组分块表示的矩阵乘法。在此情况下，
    a、b 和 out 都是大小合适的 *4维* 紧凑数组，例如 a 的大小为
   *   a[m/TILE][n/TILE][TILE][TILE]
   * 你应该逐块进行乘法以提高数组性能（
   * 即，此函数应调用上面实现的 `AlignedDot()`）。
   *
   * 注意，此函数仅在 m、n、p
   */
  /// BEGIN SOLUTION

  for(int i=0;i<m/TILE;i++){
    for(int j=0;j<p/TILE;j++){
      for(int idx=0;idx<TILESQ;idx++){
        out->ptr[i*p*TILE+j*TILESQ+idx]=0;
      }
      for(int k=0;k<n/TILE;k++){
        AlignedDot(a.ptr+(i*n*TILE+k*TILESQ)
                  ,b.ptr+(k*p*TILE+j*TILESQ)
                  ,out->ptr+(i*p*TILE+j*TILESQ));
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
* 通过对连续的 `reduce_size` 个块取最大值来进行缩减。
   *
   * 参数：
   *   a: 紧凑数组，其大小为 a.size = out.size * reduce_size，将对该数组进行缩减
   *   out: 要写入的紧凑数组
   *   reduce_size: 要缩减的维度的大小
   */

  /// BEGIN SOLUTION
  scalar_t now=-INFINITY;
  size_t out_idx=0;
  for(size_t i=0;i<a.size;i++){
    now=std::max(a.ptr[i],now);
    if((i+1)%reduce_size==0){
      out->ptr[out_idx++]=now;
      now=-INFINITY;
    }
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  scalar_t now=0;
  size_t out_idx=0;
  for(size_t i=0;i<a.size;i++){
    now+=a.ptr[i];
    if((i+1)%reduce_size==0){
      out->ptr[out_idx++]=now;
      now=0;
    }
    
  }
  /// BEGIN SOLUTION
  
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
