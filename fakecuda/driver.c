#include "fakecuda.h"
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

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

int cuDeviceGetName(char *name, int len, int dev) {
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  snprintf(name, (size_t)len, "%s", fake_devices[dev].name);
  FAKE_LOG("cuDeviceGetName(dev=%d) = 0  %s", dev, name);
  return CUDA_SUCCESS;
}

int cuDeviceTotalMem_v2(size_t *bytes, int dev) {
  if (!bytes) return CUDA_ERROR_INVALID_VALUE;
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  *bytes = fake_devices[dev].totalGlobalMem;
  FAKE_LOG("cuDeviceTotalMem(dev=%d) = 0  bytes=%zu", dev, *bytes);
  return CUDA_SUCCESS;
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

static unsigned primary_flags[FAKECUDA_MAX_DEVICES];

int cuDevicePrimaryCtxGetState(int dev, unsigned *flags, int *active) {
  int r = CUDA_ERROR_INVALID_DEVICE;
  if (fake_valid(dev)) {
    *flags = __atomic_load_n(&primary_flags[dev], __ATOMIC_RELAXED);
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

/* Private ABI reference: vosen/ZLUDA dark_api/src/lib.rs, CUDART_INTERFACE.
 */
static int export_module(void **module, const void *image) {
  (void)image;
  if (!module) return CUDA_ERROR_INVALID_VALUE;
  *module = NULL;
  fprintf(stderr, "[fakecuda] MISSING (CALLED): CUDART_INTERFACE[1] module loading\n");
  return CUDA_ERROR_NOT_SUPPORTED;
}

static int export_primary_context(void **ctx, int dev) {
  FAKE_LOG("CUDART_INTERFACE[2](dev=%d)", dev);
  if (!ctx) return CUDA_ERROR_INVALID_VALUE;
  *ctx = NULL;
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  /* Obtaining the handle does not activate the primary context. */
  *ctx = ctx_of(dev);
  return CUDA_SUCCESS;
}

static int export_primary_retain(int dev, unsigned flags) {
  FAKE_LOG("CUDART_INTERFACE[4](dev=%d, flags=%u)", dev, flags);
  if (!fake_valid(dev)) return CUDA_ERROR_INVALID_DEVICE;
  __atomic_store_n(&primary_flags[dev], flags, __ATOMIC_RELAXED);
  fake_ctx_touch(dev);
  return CUDA_SUCCESS;
}

static int export_module_ext1(void **module, const void *image,
                              void *a, void *b, unsigned flags) {
  (void)a; (void)b; (void)flags;
  fprintf(stderr, "[fakecuda] CUDART_INTERFACE[6]\n");
  return export_module(module, image);
}

static void export_release(size_t token) {
  FAKE_LOG("CUDART_INTERFACE[7](%zu)", token);
}

static int export_module_ext2(const void *image, void **module,
                              void *a, void *b, unsigned flags) {
  (void)a; (void)b; (void)flags;
  fprintf(stderr, "[fakecuda] CUDART_INTERFACE[8]\n");
  return export_module(module, image);
}

static int export_load_compilers(void) {
  fprintf(stderr, "[fakecuda] MISSING (CALLED): CUDART_INTERFACE[12] load_compilers\n");
  return CUDA_ERROR_NOT_SUPPORTED;
}

static const void *const cudart_exports[13] = {
  [0] = (const void *)(13 * sizeof(void *)),
  [1] = export_module,
  [2] = export_primary_context,
  [4] = export_primary_retain,
  [6] = export_module_ext1,
  [7] = export_release,
  [8] = export_module_ext2,
  [12] = export_load_compilers,
};

/* ZLUDA models these opaque hook areas as arrays of u32; the returned
 * length is an element count, matching its reference implementation. */
static unsigned hook_buffer1[1024], hook_buffer2[14];
static void export_hook_buffer1(void **ptr, size_t *count) {
  FAKE_LOG("TOOLS_RUNTIME_CALLBACK_HOOKS[2]");
  *ptr = hook_buffer1;
  *count = sizeof hook_buffer1 / sizeof *hook_buffer1;
}
static void export_hook_buffer2(void **ptr, size_t *count) {
  FAKE_LOG("TOOLS_RUNTIME_CALLBACK_HOOKS[6]");
  *ptr = hook_buffer2;
  *count = sizeof hook_buffer2 / sizeof *hook_buffer2;
}
static const void *const hook_exports[7] = {
  [0] = (const void *)(7 * sizeof(void *)),
  [2] = export_hook_buffer1,
  [6] = export_hook_buffer2,
};

static const void *const tls_exports[4] = {
  [0] = (const void *)(4 * sizeof(void *)),
};

typedef void (*storage_destructor)(void *, void *, void *);
struct context_storage {
  void *ctx, *key, *value;
  storage_destructor destructor;
  struct context_storage *next;
};
static struct context_storage *storage_entries;
static pthread_mutex_t storage_lock = PTHREAD_MUTEX_INITIALIZER;

static void *storage_context(void *ctx) {
  if (!ctx) cuCtxGetCurrent(&ctx);
  for (int i = 0; i < fake_device_count; i++)
    if (ctx == ctx_of(i)) return ctx;
  return NULL;
}
static int export_storage_put(void *ctx, void *key, void *value,
                               storage_destructor destructor) {
  FAKE_LOG("CONTEXT_LOCAL_STORAGE[0](ctx=%p, key=%p)", ctx, key);
  ctx = storage_context(ctx);
  if (!ctx) return 201; /* CUDA_ERROR_INVALID_CONTEXT */
  pthread_mutex_lock(&storage_lock);
  struct context_storage *e = storage_entries;
  while (e && (e->ctx != ctx || e->key != key)) e = e->next;
  if (!e) {
    e = calloc(1, sizeof *e);
    if (!e) { pthread_mutex_unlock(&storage_lock); return CUDA_ERROR_OUT_OF_MEMORY; }
    e->ctx = ctx; e->key = key;
    e->next = storage_entries; storage_entries = e;
  }
  e->value = value; e->destructor = destructor;
  pthread_mutex_unlock(&storage_lock);
  return CUDA_SUCCESS;
}
static int export_storage_delete(void *ctx, void *key) {
  FAKE_LOG("CONTEXT_LOCAL_STORAGE[1](ctx=%p, key=%p)", ctx, key);
  ctx = storage_context(ctx);
  if (!ctx) return 201;
  pthread_mutex_lock(&storage_lock);
  struct context_storage **p = &storage_entries;
  while (*p && ((*p)->ctx != ctx || (*p)->key != key)) p = &(*p)->next;
  if (*p) { struct context_storage *e = *p; *p = e->next; free(e); }
  pthread_mutex_unlock(&storage_lock);
  return CUDA_SUCCESS;
}
static int export_storage_get(void **value, void *ctx, void *key) {
  FAKE_LOG("CONTEXT_LOCAL_STORAGE[2](ctx=%p, key=%p)", ctx, key);
  if (!value) return CUDA_ERROR_INVALID_VALUE;
  *value = NULL;
  ctx = storage_context(ctx);
  if (!ctx) return 201;
  pthread_mutex_lock(&storage_lock);
  struct context_storage *e = storage_entries;
  while (e && (e->ctx != ctx || e->key != key)) e = e->next;
  int result = e ? CUDA_SUCCESS : 400; /* CUDA_ERROR_INVALID_HANDLE */
  if (e) *value = e->value;
  pthread_mutex_unlock(&storage_lock);
  return result;
}
static const void *const storage_exports[4] = {
  export_storage_put, export_storage_delete, export_storage_get, NULL
};

#include "integrity.h"

static int export_context_check(void *ctx, unsigned *bits, const void **detail) {
  FAKE_LOG("CONTEXT_CHECKS[2](ctx=%p)", ctx);
  if (!bits) return CUDA_ERROR_INVALID_VALUE;
  *bits = 0;
  if (detail) *detail = NULL;
  return CUDA_SUCCESS;
}
static unsigned export_context_check3(void) {
  FAKE_LOG("CONTEXT_CHECKS[3]");
  return 0;
}
static const void *const context_check_exports[4] = {
  (const void *)(4 * sizeof(void *)), NULL, export_context_check, export_context_check3
};

int cuGetExportTable(const void **table, const unsigned char *uuid) {
  static const unsigned char cudart_uuid[16] = {
    0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,
    0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9
  };
  static const unsigned char hooks_uuid[16] = {
    0xa0,0x94,0x79,0x8c,0x2e,0x74,0x2e,0x74,
    0x93,0xf2,0x08,0x00,0x20,0x0c,0x0a,0x66
  };
  static const unsigned char tls_uuid[16] = {
    0x42,0xd8,0x5a,0x81,0x23,0xf6,0xcb,0x47,
    0x82,0x98,0xf6,0xe7,0x8a,0x3a,0xec,0xdc
  };
  static const unsigned char storage_uuid[16] = {
    0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,
    0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93
  };
  static const unsigned char integrity_uuid[16] = {
    0xd4,0x08,0x20,0x55,0xbd,0xe6,0x70,0x4b,0x8d,0x34,0xba,0x12,0x3c,0x66,0xe1,0xf2
  };
  static const unsigned char checks_uuid[16] = {
    0x26,0x3e,0x88,0x60,0x7c,0xd2,0x61,0x43,0x92,0xf6,0xbb,0xd5,0x00,0x6d,0xfa,0x7e
  };
  if (!table || !uuid) return CUDA_ERROR_INVALID_VALUE;
  if (memcmp(uuid, cudart_uuid, sizeof cudart_uuid) == 0) {
    *table = cudart_exports;
    FAKE_LOG("cuGetExportTable(CUDART_INTERFACE) -> %p", *table);
    return CUDA_SUCCESS;
  }

  if (memcmp(uuid, hooks_uuid, sizeof hooks_uuid) == 0) {
    *table = hook_exports;
    FAKE_LOG("cuGetExportTable(TOOLS_RUNTIME_CALLBACK_HOOKS) -> %p", *table);
    return CUDA_SUCCESS;
  }
  if (memcmp(uuid, tls_uuid, sizeof tls_uuid) == 0) {
    *table = tls_exports;
    FAKE_LOG("cuGetExportTable(TOOLS_TLS) -> %p", *table);
    return CUDA_SUCCESS;
  }
  if (memcmp(uuid, storage_uuid, sizeof storage_uuid) == 0) {
    *table = storage_exports;
    FAKE_LOG("cuGetExportTable(CONTEXT_LOCAL_STORAGE_INTERFACE_V0301) -> %p", *table);
    return CUDA_SUCCESS;
  }
  if (memcmp(uuid, integrity_uuid, 16) == 0) {
    *table = integrity_exports;
    FAKE_LOG("cuGetExportTable(INTEGRITY_CHECK) -> %p", *table);
    return CUDA_SUCCESS;
  }
  if (memcmp(uuid, checks_uuid, 16) == 0) {
    *table = context_check_exports;
    FAKE_LOG("cuGetExportTable(CONTEXT_CHECKS) -> %p", *table);
    return CUDA_SUCCESS;
  }
  *table = NULL;
  fprintf(stderr,
          "[fakecuda] MISSING (CALLED): cuGetExportTable(%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
          "%02x%02x%02x%02x%02x%02x), private driver interface\n",
          uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7], uuid[8], uuid[9],
          uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
  return CUDA_ERROR_NOT_SUPPORTED;
}

int cuModuleGetLoadingMode(int *mode) {
  if (!mode) return CUDA_ERROR_INVALID_VALUE;
  const char *setting = getenv("CUDA_MODULE_LOADING");
  *mode = setting && strcmp(setting, "EAGER") == 0 ? 1 : 2;
  FAKE_LOG("cuModuleGetLoadingMode() = 0  mode=%d", *mode);
  return CUDA_SUCCESS;
}

int cuGetProcAddress(const char *symbol, void **pfn, int version, unsigned long long flags);
int cuGetProcAddress_v2(const char *symbol, void **pfn, int version, unsigned long long flags,
                        int *status);

static const struct { const char *name; void *fn; } driver_syms[] = {
    {"cuInit", cuInit},
    {"cuModuleGetLoadingMode", cuModuleGetLoadingMode},
    {"cuDriverGetVersion", cuDriverGetVersion},
    {"cuDeviceGetCount", cuDeviceGetCount},
    {"cuDeviceGet", cuDeviceGet},
    {"cuDeviceGetName", cuDeviceGetName},
    {"cuDeviceTotalMem", cuDeviceTotalMem_v2},
    {"cuDeviceTotalMem_v2", cuDeviceTotalMem_v2},
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
