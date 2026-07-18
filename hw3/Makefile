.PHONY: lib, debug, pybind, clean, format, all

all: lib

lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

debug:
	@rm -rf build
	@mkdir -p build
	@cd build; cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-g -O0 -fno-omit-frame-pointer"
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
