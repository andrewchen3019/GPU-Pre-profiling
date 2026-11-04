#include "fakecuda.h"
#include <stdlib.h>
#include <string.h>

struct cudaDeviceProp fake_devices[FAKECUDA_MAX_DEVICES];
int fake_device_count = 3;
int fake_log = 0;
__thread int fake_current_device = 0;
int fake_ctx_active[FAKECUDA_MAX_DEVICES];

static void init_h200(struct cudaDeviceProp *p) {
  memset(p, 0, sizeof *p);
  strcpy(p->name, "NVIDIA H200");
  p->totalGlobalMem = 143771ULL << 20;
  p->sharedMemPerBlock = 48 << 10;
  p->sharedMemPerBlockOptin = 227 << 10;
  p->sharedMemPerMultiprocessor = 228 << 10;
  p->regsPerBlock = 65536;
  p->regsPerMultiprocessor = 65536;
  p->warpSize = 32;
  p->memPitch = 2147483647;
  p->maxThreadsPerBlock = 1024;
  p->maxThreadsDim[0] = 1024;
  p->maxThreadsDim[1] = 1024;
  p->maxThreadsDim[2] = 64;
  p->maxGridSize[0] = 2147483647;
  p->maxGridSize[1] = 65535;
  p->maxGridSize[2] = 65535;
  p->clockRate = 1980000;
  p->memoryClockRate = 3201000;
  p->memoryBusWidth = 6144;
  p->totalConstMem = 64 << 10;
  p->major = 9;
  p->minor = 0;
  p->multiProcessorCount = 132;
  p->l2CacheSize = 50 << 20;
  p->persistingL2CacheMaxSize = 78643200;
  p->maxThreadsPerMultiProcessor = 2048;
  p->maxBlocksPerMultiProcessor = 32;
  p->textureAlignment = 512;
  p->texturePitchAlignment = 32;
  p->deviceOverlap = 1;
  p->asyncEngineCount = 3;
  p->unifiedAddressing = 1;
  p->managedMemory = 1;
  p->concurrentManagedAccess = 1;
  p->concurrentKernels = 1;
  p->canMapHostMemory = 1;
  p->ECCEnabled = 1;
  p->isMultiGpuBoard = 0;
  p->computeMode = cudaComputeModeDefault;
  p->cooperativeLaunch = 1;
  p->pageableMemoryAccess = 0;
  p->streamPrioritiesSupported = 1;
  p->globalL1CacheSupported = 1;
  p->localL1CacheSupported = 1;
  p->pciDomainID = 0;
  p->pciDeviceID = 0;
}

__attribute__((constructor)) static void fakecuda_init(void) {
  const char *l = getenv("FAKECUDA_LOG");
  fake_log = l && *l && strcmp(l, "0") != 0;

  const char *s = getenv("FAKECUDA_DEVICE_COUNT");
  if (s) fake_device_count = atoi(s);
  if (fake_device_count < 0) fake_device_count = 0;
  if (fake_device_count > FAKECUDA_MAX_DEVICES) fake_device_count = FAKECUDA_MAX_DEVICES;

  for (int i = 0; i < fake_device_count; i++) {
    struct cudaDeviceProp *p = &fake_devices[i];
    init_h200(p);
    memcpy(p->uuid.bytes, "fakecuda-h200-00", 16);
    p->uuid.bytes[15] = (char)('0' + i);
    p->pciBusID = i + 1;
  }
  FAKE_LOG("loaded: %d x %s", fake_device_count, fake_devices[0].name);
}

int fake_device_attr(int attr, int dev) {
  const struct cudaDeviceProp *p = &fake_devices[dev];
  switch (attr) {
  case cudaDevAttrComputeCapabilityMajor: return p->major;
  case cudaDevAttrComputeCapabilityMinor: return p->minor;
  case cudaDevAttrMultiProcessorCount: return p->multiProcessorCount;
  case cudaDevAttrMaxThreadsPerBlock: return p->maxThreadsPerBlock;
  case cudaDevAttrWarpSize: return p->warpSize;
  case cudaDevAttrMaxSharedMemoryPerBlock: return (int)p->sharedMemPerBlock;
  case cudaDevAttrMaxSharedMemoryPerBlockOptin: return (int)p->sharedMemPerBlockOptin;
  case cudaDevAttrL2CacheSize: return p->l2CacheSize;
  case cudaDevAttrClockRate: return p->clockRate;
  case cudaDevAttrPciBusId: return p->pciBusID;
  default:
    FAKE_LOG("  attr %d not emulated, answering 0", attr);
    return 0;
  }
}
