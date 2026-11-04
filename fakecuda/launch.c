#include "fakecuda.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct reg {
  const void *host;
  const char *name;
  char *pretty;
  void *var;
  size_t size;
};

static struct reg *regs;
static size_t reg_cap, reg_count;
static pthread_mutex_t reg_lock = PTHREAD_MUTEX_INITIALIZER;

static size_t slot(const void *p, size_t cap) { return ((size_t)p >> 4) * 0x9E3779B97F4A7C15ull & (cap - 1); }

static struct reg *reg_find_locked(const void *host) {
  if (!reg_cap) return NULL;
  for (size_t i = slot(host, reg_cap);; i = (i + 1) & (reg_cap - 1)) {
    if (regs[i].host == host) return &regs[i];
    if (!regs[i].host) return NULL;
  }
}

static void reg_add(struct reg r) {
  pthread_mutex_lock(&reg_lock);
  if (2 * (reg_count + 1) > reg_cap) {
    size_t old_cap = reg_cap;
    struct reg *old = regs;
    reg_cap = reg_cap ? 2 * reg_cap : 1 << 14;
    regs = calloc(reg_cap, sizeof *regs);
    for (size_t i = 0; i < old_cap; i++) {
      if (!old[i].host) continue;
      size_t j = slot(old[i].host, reg_cap);
      while (regs[j].host) j = (j + 1) & (reg_cap - 1);
      regs[j] = old[i];
    }
    free(old);
  }
  struct reg *e = reg_find_locked(r.host);
  if (!e) {
    size_t j = slot(r.host, reg_cap);
    while (regs[j].host) j = (j + 1) & (reg_cap - 1);
    e = &regs[j];
    reg_count++;
  }
  *e = r;
  pthread_mutex_unlock(&reg_lock);
}

static struct reg *reg_find(const void *host) {
  pthread_mutex_lock(&reg_lock);
  struct reg *e = reg_find_locked(host);
  pthread_mutex_unlock(&reg_lock);
  return e;
}

static const char *pretty_name(struct reg *e) {
  if (!e) return "<unregistered>";
  if (!e->pretty) {
    static char *(*demangle)(const char *, char *, size_t *, int *);
    if (!demangle) {
      void *h = dlopen("libstdc++.so.6", RTLD_LAZY | RTLD_NOLOAD);
      if (!h) h = dlopen("libstdc++.so.6", RTLD_LAZY);
      if (h) demangle = (char *(*)(const char *, char *, size_t *, int *))dlsym(h, "__cxa_demangle");
    }
    int status = -1;
    char *d = demangle ? demangle(e->name, NULL, NULL, &status) : NULL;
    e->pretty = status == 0 && d ? d : strdup(e->name);
    if (strlen(e->pretty) > 160) strcpy(e->pretty + 157, "...");
  }
  return e->pretty;
}

void **__cudaRegisterFatBinary(void *fatCubin) {
  void **handle = malloc(sizeof *handle);
  *handle = fatCubin;
  return handle;
}

void __cudaRegisterFatBinaryEnd(void **handle) { (void)handle; }

void __cudaUnregisterFatBinary(void **handle) { free(handle); }

char __cudaInitModule(void **handle) {
  (void)handle;
  return 1;
}

void __cudaRegisterFunction(void **handle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  (void)handle, (void)deviceFun, (void)thread_limit, (void)tid, (void)bid, (void)bDim, (void)gDim,
      (void)wSize;
  reg_add((struct reg){.host = hostFun, .name = deviceName});
}

void __cudaRegisterVar(void **handle, char *hostVar, char *deviceAddress, const char *deviceName,
                       int ext, size_t size, int constant, int global) {
  (void)handle, (void)deviceAddress, (void)ext, (void)constant, (void)global;
  reg_add((struct reg){.host = hostVar, .name = deviceName, .var = calloc(1, size ? size : 1),
                       .size = size});
}

void __cudaRegisterManagedVar(void **handle, void **hostVarPtrAddress, char *deviceAddress,
                              const char *deviceName, int ext, size_t size, int constant,
                              int global) {
  (void)handle, (void)deviceAddress, (void)ext, (void)constant, (void)global;
  void *mem = calloc(1, size ? size : 1);
  *hostVarPtrAddress = mem;
  reg_add((struct reg){.host = hostVarPtrAddress, .name = deviceName, .var = mem, .size = size});
}

void __cudaRegisterHostVar(void) {}
void __cudaRegisterUnifiedTable(void) {}

struct call_config {
  dim3 grid, block;
  size_t shmem;
  cudaStream_t stream;
};
static __thread struct call_config config_stack[8];
static __thread int config_depth;

unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
  if (config_depth == 8) return 1;
  config_stack[config_depth++] = (struct call_config){grid, block, shmem, stream};
  return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shmem, void *stream) {
  if (config_depth == 0) return cudaErrorMissingConfiguration;
  struct call_config c = config_stack[--config_depth];
  *grid = c.grid;
  *block = c.block;
  *shmem = c.shmem;
  *(cudaStream_t *)stream = c.stream;
  return cudaSuccess;
}

static cudaError_t launch(const char *api, const void *func, dim3 g, dim3 b, size_t shmem,
                          cudaStream_t stream) {
  struct reg *e = reg_find(func);
  const struct cudaDeviceProp *p = &fake_devices[fake_current_device];
  cudaError_t r = cudaSuccess;
  if (!e || e->var)
    r = cudaErrorInvalidDeviceFunction;
  else if ((unsigned long long)b.x * b.y * b.z > (unsigned)p->maxThreadsPerBlock || b.x == 0 ||
           b.y == 0 || b.z == 0 || b.x > (unsigned)p->maxThreadsDim[0] ||
           b.y > (unsigned)p->maxThreadsDim[1] || b.z > (unsigned)p->maxThreadsDim[2] ||
           g.x == 0 || g.y == 0 || g.z == 0 || g.x > (unsigned)p->maxGridSize[0] ||
           g.y > (unsigned)p->maxGridSize[1] || g.z > (unsigned)p->maxGridSize[2] ||
           shmem > p->sharedMemPerBlockOptin)
    r = cudaErrorInvalidConfiguration;
  if (fake_log || r != cudaSuccess)
    fprintf(stderr, "[fakecuda] %s %s grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%zu stream=%p = %d\n",
            api, pretty_name(e), g.x, g.y, g.z, b.x, b.y, b.z, shmem, (void *)stream, r);
  if (r == cudaSuccess) fake_ctx_touch(fake_current_device);
  return r;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 grid, dim3 block, void **args, size_t shmem,
                             cudaStream_t stream) {
  (void)args;
  return launch("cudaLaunchKernel", func, grid, block, shmem, stream);
}

cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 grid, dim3 block, void **args,
                                        size_t shmem, cudaStream_t stream) {
  (void)args;
  return launch("cudaLaunchCooperativeKernel", func, grid, block, shmem, stream);
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args) {
  (void)args;
  return launch("cudaLaunchKernelExC", func, config->gridDim, config->blockDim,
                config->dynamicSmemBytes, config->stream);
}

cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *func) {
  *kernel = (cudaKernel_t)func;
  return reg_find(func) ? cudaSuccess : cudaErrorInvalidDeviceFunction;
}

cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid, dim3 block, void **args,
                               size_t shmem, cudaStream_t stream) {
  (void)args;
  return launch("launch", (const void *)kernel, grid, block, shmem, stream);
}

cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid, dim3 block, void **args,
                                    size_t shmem, cudaStream_t stream) {
  return __cudaLaunchKernel(kernel, grid, block, args, shmem, stream);
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
  struct reg *e = reg_find(func);
  if (!e || e->var) return cudaErrorInvalidDeviceFunction;
  memset(attr, 0, sizeof *attr);
  attr->maxThreadsPerBlock = fake_devices[fake_current_device].maxThreadsPerBlock;
  attr->numRegs = 32;
  attr->ptxVersion = 90;
  attr->binaryVersion = 90;
  attr->maxDynamicSharedSizeBytes = 48 << 10;
  attr->preferredShmemCarveout = -1;
  FAKE_LOG("cudaFuncGetAttributes(%s) = 0", pretty_name(e));
  return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
  FAKE_LOG("cudaFuncSetAttribute(%p, attr=%d, %d) = 0", func, attr, value);
  return reg_find(func) ? cudaSuccess : cudaErrorInvalidDeviceFunction;
}

cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache config) {
  FAKE_LOG("cudaFuncSetCacheConfig(%p, %d) = 0", func, config);
  return reg_find(func) ? cudaSuccess : cudaErrorInvalidDeviceFunction;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func,
                                                                  int blockSize, size_t shmem,
                                                                  unsigned int flags) {
  (void)flags;
  if (!reg_find(func)) return cudaErrorInvalidDeviceFunction;
  const struct cudaDeviceProp *p = &fake_devices[fake_current_device];
  if (blockSize <= 0 || blockSize > p->maxThreadsPerBlock) return cudaErrorInvalidValue;
  int warps = (blockSize + p->warpSize - 1) / p->warpSize;
  int n = p->maxThreadsPerMultiProcessor / (warps * p->warpSize);
  if (n > p->maxBlocksPerMultiProcessor) n = p->maxBlocksPerMultiProcessor;
  if (shmem && (int)(p->sharedMemPerMultiprocessor / shmem) < n)
    n = (int)(p->sharedMemPerMultiprocessor / shmem);
  *numBlocks = n;
  FAKE_LOG("cudaOccupancyMaxActiveBlocksPerMultiprocessor(block=%d, shmem=%zu) = 0  %d", blockSize,
           shmem, n);
  return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func,
                                                          int blockSize, size_t shmem) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, shmem, 0);
}

static struct reg *find_var(const void *symbol) {
  struct reg *e = reg_find(symbol);
  return e && e->var ? e : NULL;
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  struct reg *e = find_var(symbol);
  if (!e) return cudaErrorInvalidSymbol;
  *devPtr = e->var;
  return cudaSuccess;
}

cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
  struct reg *e = find_var(symbol);
  if (!e) return cudaErrorInvalidSymbol;
  *size = e->size;
  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset,
                               enum cudaMemcpyKind kind) {
  (void)kind;
  struct reg *e = find_var(symbol);
  if (!e) return cudaErrorInvalidSymbol;
  if (offset + count > e->size) return cudaErrorInvalidValue;
  memmove((char *)e->var + offset, src, count);
  FAKE_LOG("cudaMemcpyToSymbol(%s, %zu, offset=%zu) = 0", e->name, count, offset);
  return cudaSuccess;
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset,
                                 enum cudaMemcpyKind kind) {
  (void)kind;
  struct reg *e = find_var(symbol);
  if (!e) return cudaErrorInvalidSymbol;
  if (offset + count > e->size) return cudaErrorInvalidValue;
  memmove(dst, (char *)e->var + offset, count);
  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset,
                                    enum cudaMemcpyKind kind, cudaStream_t stream) {
  (void)stream;
  return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset,
                                      enum cudaMemcpyKind kind, cudaStream_t stream) {
  (void)stream;
  return cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}
