#include "fakecuda.h"
#include <string.h>

cudaError_t cudaGetDeviceCount(int *count) {
  *count = fake_device_count;
  cudaError_t r = fake_device_count > 0 ? cudaSuccess : cudaErrorNoDevice;
  FAKE_LOG("cudaGetDeviceCount() = %d  count=%d", r, *count);
  return r;
}

cudaError_t cudaGetDevice(int *device) {
  *device = fake_current_device;
  FAKE_LOG("cudaGetDevice() = 0  device=%d", *device);
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  cudaError_t r = cudaErrorInvalidDevice;
  if (fake_valid(device)) {
    fake_current_device = device;
    fake_ctx_touch(device);
    r = cudaSuccess;
  }
  FAKE_LOG("cudaSetDevice(%d) = %d", device, r);
  return r;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device) {
  cudaError_t r = cudaErrorInvalidDevice;
  if (fake_valid(device)) {
    *prop = fake_devices[device];
    r = cudaSuccess;
  }
  FAKE_LOG("cudaGetDeviceProperties_v2(dev=%d) = %d", device, r);
  return r;
}

#undef cudaGetDeviceProperties
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  return cudaGetDeviceProperties_v2(prop, device);
}

cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
  cudaError_t r = cudaErrorInvalidDevice;
  if (fake_valid(device)) {
    *value = fake_device_attr(attr, device);
    r = cudaSuccess;
  }
  FAKE_LOG("cudaDeviceGetAttribute(attr=%d, dev=%d) = %d  value=%d", attr, device, r,
           r == cudaSuccess ? *value : -1);
  return r;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
  cudaError_t r = cudaErrorInvalidDevice;
  if (fake_valid(device)) {
    snprintf(pciBusId, len, "0000:%02x:00.0", fake_devices[device].pciBusID);
    r = cudaSuccess;
  }
  FAKE_LOG("cudaDeviceGetPCIBusId(dev=%d) = %d  %s", device, r, r == cudaSuccess ? pciBusId : "");
  return r;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
  if (leastPriority) *leastPriority = 0;
  if (greatestPriority) *greatestPriority = -5;
  FAKE_LOG("cudaDeviceGetStreamPriorityRange() = 0  least=0 greatest=-5");
  return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *v) {
  *v = FAKECUDA_CUDA_VERSION;
  FAKE_LOG("cudaDriverGetVersion() = 0  %d", *v);
  return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *v) {
  *v = FAKECUDA_CUDA_VERSION;
  FAKE_LOG("cudaRuntimeGetVersion() = 0  %d", *v);
  return cudaSuccess;
}

cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr,
                                             unsigned int cudaVersion, unsigned long long flags,
                                             enum cudaDriverEntryPointQueryResult *driverStatus) {
  int r = fake_get_proc_address(symbol, funcPtr, (int)cudaVersion, flags, (int *)driverStatus);
  return r == 0 ? cudaSuccess : cudaErrorSymbolNotFound;
}

cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags,
                                    enum cudaDriverEntryPointQueryResult *driverStatus) {
  return cudaGetDriverEntryPointByVersion(symbol, funcPtr, FAKECUDA_CUDA_VERSION, flags,
                                          driverStatus);
}

const char *cudaGetErrorName(cudaError_t e) {
  switch (e) {
  case cudaSuccess: return "cudaSuccess";
  case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
  case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
  case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
  case cudaErrorNoDevice: return "cudaErrorNoDevice";
  case cudaErrorSymbolNotFound: return "cudaErrorSymbolNotFound";
  case cudaErrorNotSupported: return "cudaErrorNotSupported";
  default: return "cudaErrorUnknown";
  }
}

const char *cudaGetErrorString(cudaError_t e) {
  switch (e) {
  case cudaSuccess: return "no error";
  case cudaErrorInvalidValue: return "invalid argument";
  case cudaErrorMemoryAllocation: return "out of memory";
  case cudaErrorInvalidDevice: return "invalid device ordinal";
  case cudaErrorNoDevice: return "no CUDA-capable device is detected";
  case cudaErrorSymbolNotFound: return "named symbol not found";
  case cudaErrorNotSupported: return "operation not supported (fakecuda: not emulated)";
  default: return "unknown error";
  }
}

cudaError_t cudaGetLastError(void) {
  FAKE_LOG("cudaGetLastError() = 0");
  return cudaSuccess;
}

cudaError_t cudaPeekAtLastError(void) {
  FAKE_LOG("cudaPeekAtLastError() = 0");
  return cudaSuccess;
}

static cudaError_t mem_error(int r) {
  return r == FAKE_MEM_OK ? cudaSuccess
       : r == FAKE_MEM_OOM ? cudaErrorMemoryAllocation : cudaErrorInvalidValue;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  cudaError_t r = mem_error(fake_mem_alloc(fake_current_device, size, devPtr));
  FAKE_LOG("cudaMalloc(%zu) = %d  dev=%d ptr=%p", size, r, fake_current_device, *devPtr);
  return r;
}

cudaError_t cudaFree(void *devPtr) {
  fake_ctx_touch(fake_current_device);
  cudaError_t r = mem_error(fake_mem_free(devPtr));
  FAKE_LOG("cudaFree(%p) = %d", devPtr, r);
  return r;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  *total = fake_devices[fake_current_device].totalGlobalMem;
  *free = *total - fake_mem_used(fake_current_device);
  FAKE_LOG("cudaMemGetInfo() = 0  dev=%d free=%zu total=%zu", fake_current_device, *free, *total);
  return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
  cudaError_t r = mem_error(fake_host_alloc(size, pHost));
  FAKE_LOG("cudaHostAlloc(%zu, flags=%u) = %d  ptr=%p", size, flags, r, *pHost);
  return r;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) { return cudaHostAlloc(ptr, size, 0); }

cudaError_t cudaFreeHost(void *ptr) {
  cudaError_t r = mem_error(fake_host_free(ptr));
  FAKE_LOG("cudaFreeHost(%p) = %d", ptr, r);
  return r;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attr, const void *ptr) {
  int device;
  memset(attr, 0, sizeof *attr);
  if (fake_mem_lookup(ptr, &device)) {
    attr->type = device >= 0 ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
    attr->device = device >= 0 ? device : fake_current_device;
    attr->devicePointer = (void *)ptr;
    attr->hostPointer = device >= 0 ? NULL : (void *)ptr;
  }
  FAKE_LOG("cudaPointerGetAttributes(%p) = 0  type=%d dev=%d", ptr, attr->type, attr->device);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  memmove(dst, src, count);
  FAKE_LOG("cudaMemcpy(%p <- %p, %zu, kind=%d) = 0", dst, src, count, kind);
  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream) {
  memmove(dst, src, count);
  FAKE_LOG("cudaMemcpyAsync(%p <- %p, %zu, kind=%d, stream=%p) = 0", dst, src, count, kind,
           (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  memset(devPtr, value, count);
  FAKE_LOG("cudaMemset(%p, %d, %zu) = 0", devPtr, value, count);
  return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
  memset(devPtr, value, count);
  FAKE_LOG("cudaMemsetAsync(%p, %d, %zu, stream=%p) = 0", devPtr, value, count, (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  FAKE_LOG("cudaStreamSynchronize(%p) = 0", (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
  FAKE_LOG("cudaDeviceSynchronize() = 0");
  return cudaSuccess;
}

