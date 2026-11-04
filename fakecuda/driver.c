#include "fakecuda.h"
#include <string.h>

#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_NOT_FOUND 500

int cuInit(unsigned flags) {
  FAKE_LOG("cuInit(%u) = 0", flags);
  return CUDA_SUCCESS;
}

int cuDriverGetVersion(int *v) {
  *v = FAKECUDA_CUDA_VERSION;
  FAKE_LOG("cuDriverGetVersion() = 0  %d", *v);
  return CUDA_SUCCESS;
}

int cuDeviceGetCount(int *count) {
  *count = fake_device_count;
  FAKE_LOG("cuDeviceGetCount() = 0  count=%d", *count);
  return CUDA_SUCCESS;
}

int cuDeviceGet(int *dev, int ordinal) {
  int r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(ordinal)) {
    *dev = ordinal;
    r = CUDA_SUCCESS;
  }
  FAKE_LOG("cuDeviceGet(ordinal=%d) = %d", ordinal, r);
  return r;
}

int cuDeviceGetAttribute(int *value, int attr, int dev) {
  int r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(dev)) {
    *value = fake_device_attr(attr, dev);
    r = CUDA_SUCCESS;
  }
  FAKE_LOG("cuDeviceGetAttribute(attr=%d, dev=%d) = %d  value=%d", attr, dev, r,
           r == CUDA_SUCCESS ? *value : -1);
  return r;
}

int cuDeviceGetUuid(cudaUUID_t *uuid, int dev) {
  int r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(dev)) {
    *uuid = fake_devices[dev].uuid;
    r = CUDA_SUCCESS;
  }
  FAKE_LOG("cuDeviceGetUuid(dev=%d) = %d", dev, r);
  return r;
}

int cuDevicePrimaryCtxGetState(int dev, unsigned *flags, int *active) {
  int r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(dev)) {
    *flags = 0;
    *active = __atomic_load_n(&fake_ctx_active[dev], __ATOMIC_RELAXED);
    r = CUDA_SUCCESS;
  }
  FAKE_LOG("cuDevicePrimaryCtxGetState(dev=%d) = %d  active=%d", dev, r, r ? -1 : *active);
  return r;
}

int cuGetErrorString(int err, const char **s) {
  *s = err == CUDA_SUCCESS ? "no error" : "fakecuda: operation not emulated";
  FAKE_LOG("cuGetErrorString(%d) = 0", err);
  return CUDA_SUCCESS;
}

int cuGetErrorName(int err, const char **s) {
  *s = err == CUDA_SUCCESS ? "CUDA_SUCCESS" : "CUDA_ERROR_NOT_SUPPORTED";
  FAKE_LOG("cuGetErrorName(%d) = 0", err);
  return CUDA_SUCCESS;
}

static char primary_ctx[FAKECUDA_MAX_DEVICES];
static __thread int ctx_stack[16];
static __thread int ctx_depth;

static void *ctx_of(int dev) { return &primary_ctx[dev]; }
static int dev_of(void *ctx) {
  int dev = (int)((char *)ctx - primary_ctx);
  return ctx && fake_valid(dev) ? dev : -1;
}

int cuDevicePrimaryCtxRetain(void **pctx, int dev) {
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  fake_ctx_touch(dev);
  *pctx = ctx_of(dev);
  FAKE_LOG("cuDevicePrimaryCtxRetain(dev=%d) = 0  %p", dev, *pctx);
  return CUDA_SUCCESS;
}

int cuDevicePrimaryCtxRelease_v2(int dev) {
  FAKE_LOG("cuDevicePrimaryCtxRelease(dev=%d) = 0", dev);
  return fake_valid(dev) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

int cuCtxGetCurrent(void **pctx) {
  int dev = fake_current_device;
  *pctx = __atomic_load_n(&fake_ctx_active[dev], __ATOMIC_RELAXED) ? ctx_of(dev) : NULL;
  FAKE_LOG("cuCtxGetCurrent() = 0  %p", *pctx);
  return CUDA_SUCCESS;
}

int cuCtxSetCurrent(void *ctx) {
  int dev = dev_of(ctx);
  FAKE_LOG("cuCtxSetCurrent(%p) = %d  dev=%d", ctx, ctx && dev < 0 ? CUDA_ERROR_INVALID_VALUE : 0, dev);
  if (!ctx) return CUDA_SUCCESS;
  if (dev < 0) return CUDA_ERROR_INVALID_VALUE;
  fake_current_device = dev;
  return CUDA_SUCCESS;
}

int cuCtxPushCurrent_v2(void *ctx) {
  int dev = dev_of(ctx);
  if (dev < 0 || ctx_depth == 16) return CUDA_ERROR_INVALID_VALUE;
  ctx_stack[ctx_depth++] = fake_current_device;
  fake_current_device = dev;
  FAKE_LOG("cuCtxPushCurrent(dev=%d) = 0", dev);
  return CUDA_SUCCESS;
}

int cuCtxPopCurrent_v2(void **pctx) {
  if (ctx_depth == 0) return CUDA_ERROR_INVALID_VALUE;
  if (pctx) *pctx = ctx_of(fake_current_device);
  fake_current_device = ctx_stack[--ctx_depth];
  FAKE_LOG("cuCtxPopCurrent() = 0");
  return CUDA_SUCCESS;
}

int cuCtxGetDevice(int *dev) {
  *dev = fake_current_device;
  FAKE_LOG("cuCtxGetDevice() = 0  dev=%d", *dev);
  return CUDA_SUCCESS;
}

int cuCtxSynchronize(void) {
  FAKE_LOG("cuCtxSynchronize() = 0");
  return CUDA_SUCCESS;
}

static int mem_error(int r) {
  return r == FAKE_MEM_OK ? CUDA_SUCCESS
       : r == FAKE_MEM_OOM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_VALUE;
}

int cuMemAlloc_v2(unsigned long long *dptr, size_t size) {
  void *p;
  int r = mem_error(fake_mem_alloc(fake_current_device, size, &p));
  *dptr = (unsigned long long)p;
  FAKE_LOG("cuMemAlloc(%zu) = %d  dev=%d ptr=%p", size, r, fake_current_device, p);
  return r;
}

int cuMemFree_v2(unsigned long long dptr) {
  int r = mem_error(fake_mem_free((void *)dptr));
  FAKE_LOG("cuMemFree(%p) = %d", (void *)dptr, r);
  return r;
}

int cuMemGetInfo_v2(size_t *free, size_t *total) {
  *total = fake_devices[fake_current_device].totalGlobalMem;
  *free = *total - fake_mem_used(fake_current_device);
  FAKE_LOG("cuMemGetInfo() = 0  dev=%d free=%zu total=%zu", fake_current_device, *free, *total);
  return CUDA_SUCCESS;
}

#define CUDA_ERROR_NOT_SUPPORTED 801

int cuGetExportTable(const void **table, const unsigned char *uuid) {
  *table = NULL;
  fprintf(stderr,
          "[fakecuda] MISSING (CALLED): cuGetExportTable(%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
          "%02x%02x%02x%02x%02x%02x), private driver interface\n",
          uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7], uuid[8], uuid[9],
          uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
  return CUDA_ERROR_NOT_SUPPORTED;
}

int cuGetProcAddress(const char *symbol, void **pfn, int version, unsigned long long flags);
int cuGetProcAddress_v2(const char *symbol, void **pfn, int version, unsigned long long flags,
                        int *status);

static const struct { const char *name; void *fn; } driver_syms[] = {
    {"cuInit", cuInit},
    {"cuDriverGetVersion", cuDriverGetVersion},
    {"cuDeviceGetCount", cuDeviceGetCount},
    {"cuDeviceGet", cuDeviceGet},
    {"cuDeviceGetAttribute", cuDeviceGetAttribute},
    {"cuDeviceGetUuid", cuDeviceGetUuid},
    {"cuDevicePrimaryCtxGetState", cuDevicePrimaryCtxGetState},
    {"cuGetErrorString", cuGetErrorString},
    {"cuGetErrorName", cuGetErrorName},
    {"cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain},
    {"cuDevicePrimaryCtxRelease", cuDevicePrimaryCtxRelease_v2},
    {"cuCtxGetCurrent", cuCtxGetCurrent},
    {"cuCtxSetCurrent", cuCtxSetCurrent},
    {"cuCtxPushCurrent", cuCtxPushCurrent_v2},
    {"cuCtxPopCurrent", cuCtxPopCurrent_v2},
    {"cuCtxGetDevice", cuCtxGetDevice},
    {"cuCtxSynchronize", cuCtxSynchronize},
    {"cuGetExportTable", cuGetExportTable},
    {"cuMemAlloc", cuMemAlloc_v2},
    {"cuMemFree", cuMemFree_v2},
    {"cuMemGetInfo", cuMemGetInfo_v2},
};

int fake_get_proc_address(const char *symbol, void **pfn, int version,
                          unsigned long long flags, int *status) {
  void *fn = NULL;
  if (strcmp(symbol, "cuGetProcAddress") == 0)
    fn = version >= 12000 ? (void *)cuGetProcAddress_v2 : (void *)cuGetProcAddress;
  for (size_t i = 0; !fn && i < sizeof driver_syms / sizeof *driver_syms; i++)
    if (strcmp(symbol, driver_syms[i].name) == 0) fn = driver_syms[i].fn;

  if (fn) {
    FAKE_LOG("cuGetProcAddress(%s, v%d, flags=%llu) -> emulated", symbol, version, flags);
  } else {
    FAKE_LOG("MISSING (looked up): %s (v%d)", symbol, version);
    fn = fake_stub_for(symbol);
  }

  *pfn = fn;
  if (status) *status = fn ? 0 : 1;
  return fn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

int cuGetProcAddress(const char *symbol, void **pfn, int version, unsigned long long flags) {
  return fake_get_proc_address(symbol, pfn, version, flags, NULL);
}

int cuGetProcAddress_v2(const char *symbol, void **pfn, int version, unsigned long long flags,
                        int *status) {
  return fake_get_proc_address(symbol, pfn, version, flags, status);
}
