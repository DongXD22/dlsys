#!/bin/bash
# Debug 编译脚本 - 保留调试符号，关闭优化
cd "$(dirname "$0")"
rm -rf build_debug
mkdir build_debug
cd build_debug
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-g -O0 -fno-omit-frame-pointer" \
  -DCMAKE_CUDA_FLAGS="-g -O0"
make -j$(nproc)
# 复制 .so 到正确位置
cp -f python/needle/backend_ndarray/ndarray_backend_*.so ../python/needle/backend_ndarray/
echo "Debug build complete. .so files copied."
