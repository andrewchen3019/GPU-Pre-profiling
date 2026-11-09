SP ?= ../.venv/lib/python3.12/site-packages/nvidia
CUDA_INC ?= -I$(SP)/cuda_runtime/include -I$(SP)/cuda_nvcc/include
CUDART ?= $(SP)/cuda_runtime/lib/libcudart.so.12
SRCS = device.c runtime.c driver.c nvml.c memory.c launch.c stream.c stubs.c runtime_stubs.c

all: libfakecuda.so lib/libnvidia-ml.so.1

libfakecuda.so: $(SRCS) fakecuda.h integrity.h libcudart.map
	gcc -O2 -Wall -shared -fPIC $(CUDA_INC) -Wl,--version-script=libcudart.map,-soname,libcuda.so.1 -o $@ $(SRCS) -lpthread -ldl

runtime_stubs.c: tools/gen_runtime_stubs.sh
	sh tools/gen_runtime_stubs.sh $(CUDART) > $@

lib/libnvidia-ml.so.1:
	mkdir -p lib
	ln -sf ../libfakecuda.so $@

tests/test_procaddress: tests/test_procaddress.c
	gcc -O2 -Wall -o $@ $< -ldl

test: all tests/test_procaddress
	./run ./tests/test_procaddress
	./run ../.venv/bin/python tests/test_nvml.py
	./run ../.venv/bin/python tests/test_memory.py
	./run ../.venv/bin/python tests/test_kernels.py

clean:
	rm -rf *.so lib tests/test_procaddress runtime_stubs.c

.PHONY: all test clean
