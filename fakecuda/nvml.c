#include "fakecuda.h"
#include <string.h>

#define NVML_SUCCESS 0
#define NVML_ERROR_INVALID_ARGUMENT 2
#define NVML_ERROR_NOT_SUPPORTED 3
#define NVML_ERROR_NOT_FOUND 6
#define NVML_ERROR_INSUFFICIENT_SIZE 7

typedef struct cudaDeviceProp *nvmlDevice_t;

typedef struct {
  unsigned long long total, free, used;
} nvmlMemory_t;

typedef struct {
  char busIdLegacy[16];
  unsigned int domain, bus, device, pciDeviceId, pciSubSystemId;
  char busId[32];
} nvmlPciInfo_t;

#define H200_PCI_DEVICE_ID 0x233510DEu

static int handle_index(nvmlDevice_t d) {
  int i = (int)(d - fake_devices);
  return d && fake_valid(i) ? i : -1;
}

static void format_uuid(int i, char *out, size_t len) {
  const unsigned char *b = (const unsigned char *)fake_devices[i].uuid.bytes;
  snprintf(out, len,
           "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
           b[13], b[14], b[15]);
}

int nvmlInitWithFlags(unsigned int flags) {
  FAKE_LOG("nvmlInitWithFlags(%u) = 0", flags);
  return NVML_SUCCESS;
}
int nvmlInit_v2(void) { return nvmlInitWithFlags(0); }
int nvmlInit(void) { return nvmlInitWithFlags(0); }

int nvmlShutdown(void) {
  FAKE_LOG("nvmlShutdown() = 0");
  return NVML_SUCCESS;
}

const char *nvmlErrorString(int r) {
  switch (r) {
  case NVML_SUCCESS: return "Success";
  case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
  case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
  case NVML_ERROR_NOT_FOUND: return "Not Found";
  case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
  default: return "fakecuda: NVML error";
  }
}

int nvmlSystemGetDriverVersion(char *version, unsigned int len) {
  int r = (size_t)snprintf(version, len, "%s", FAKECUDA_DRIVER_VERSION) < len
              ? NVML_SUCCESS : NVML_ERROR_INSUFFICIENT_SIZE;
  FAKE_LOG("nvmlSystemGetDriverVersion() = %d  %s", r, FAKECUDA_DRIVER_VERSION);
  return r;
}

int nvmlSystemGetCudaDriverVersion_v2(int *v) {
  *v = FAKECUDA_CUDA_VERSION;
  FAKE_LOG("nvmlSystemGetCudaDriverVersion() = 0  %d", *v);
  return NVML_SUCCESS;
}
int nvmlSystemGetCudaDriverVersion(int *v) { return nvmlSystemGetCudaDriverVersion_v2(v); }

int nvmlDeviceGetCount_v2(unsigned int *count) {
  *count = (unsigned)fake_device_count;
  FAKE_LOG("nvmlDeviceGetCount() = 0  count=%u", *count);
  return NVML_SUCCESS;
}
int nvmlDeviceGetCount(unsigned int *count) { return nvmlDeviceGetCount_v2(count); }

int nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
  int r = NVML_ERROR_INVALID_ARGUMENT;
  if (fake_valid((int)index)) {
    *device = &fake_devices[index];
    r = NVML_SUCCESS;
  }
  FAKE_LOG("nvmlDeviceGetHandleByIndex(%u) = %d", index, r);
  return r;
}
int nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
  return nvmlDeviceGetHandleByIndex_v2(index, device);
}

int nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
  int i = handle_index(device);
  if (i >= 0) *index = (unsigned)i;
  FAKE_LOG("nvmlDeviceGetIndex() = %d  index=%d", i >= 0 ? 0 : NVML_ERROR_INVALID_ARGUMENT, i);
  return i >= 0 ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}

int nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int len) {
  int i = handle_index(device), r = NVML_ERROR_INVALID_ARGUMENT;
  if (i >= 0) {
    char s[48];
    format_uuid(i, s, sizeof s);
    r = strlen(s) < len ? NVML_SUCCESS : NVML_ERROR_INSUFFICIENT_SIZE;
    if (r == NVML_SUCCESS) strcpy(uuid, s);
  }
  FAKE_LOG("nvmlDeviceGetUUID(dev=%d) = %d  %s", i, r, r == NVML_SUCCESS ? uuid : "");
  return r;
}

int nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
  int r = NVML_ERROR_NOT_FOUND;
  for (int i = 0; i < fake_device_count; i++) {
    char s[48];
    format_uuid(i, s, sizeof s);
    if (strcmp(s, uuid) == 0) {
      *device = &fake_devices[i];
      r = NVML_SUCCESS;
      break;
    }
  }
  FAKE_LOG("nvmlDeviceGetHandleByUUID(%s) = %d", uuid, r);
  return r;
}

int nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  int i = handle_index(device);
  if (i >= 0) {
    memset(pci, 0, sizeof *pci);
    pci->domain = (unsigned)fake_devices[i].pciDomainID;
    pci->bus = (unsigned)fake_devices[i].pciBusID;
    pci->device = (unsigned)fake_devices[i].pciDeviceID;
    pci->pciDeviceId = H200_PCI_DEVICE_ID;
    snprintf(pci->busIdLegacy, sizeof pci->busIdLegacy, "%04x:%02x:%02x.0", pci->domain, pci->bus,
             pci->device);
    snprintf(pci->busId, sizeof pci->busId, "%08x:%02x:%02x.0", pci->domain, pci->bus,
             pci->device);
  }
  FAKE_LOG("nvmlDeviceGetPciInfo(dev=%d) = %d  %s", i, i >= 0 ? 0 : NVML_ERROR_INVALID_ARGUMENT,
           i >= 0 ? pci->busId : "");
  return i >= 0 ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}
int nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

int nvmlDeviceGetHandleByPciBusId_v2(const char *busId, nvmlDevice_t *device) {
  unsigned domain, bus, dev, fn;
  int r = NVML_ERROR_NOT_FOUND;
  if (sscanf(busId, "%x:%x:%x.%x", &domain, &bus, &dev, &fn) == 4) {
    for (int i = 0; i < fake_device_count; i++) {
      const struct cudaDeviceProp *p = &fake_devices[i];
      if (domain == (unsigned)p->pciDomainID && bus == (unsigned)p->pciBusID &&
          dev == (unsigned)p->pciDeviceID && fn == 0) {
        *device = &fake_devices[i];
        r = NVML_SUCCESS;
        break;
      }
    }
  }
  FAKE_LOG("nvmlDeviceGetHandleByPciBusId(%s) = %d", busId, r);
  return r;
}
int nvmlDeviceGetHandleByPciBusId(const char *busId, nvmlDevice_t *device) {
  return nvmlDeviceGetHandleByPciBusId_v2(busId, device);
}

int nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int len) {
  int i = handle_index(device), r = NVML_ERROR_INVALID_ARGUMENT;
  if (i >= 0)
    r = (size_t)snprintf(name, len, "%s", fake_devices[i].name) < len
            ? NVML_SUCCESS : NVML_ERROR_INSUFFICIENT_SIZE;
  FAKE_LOG("nvmlDeviceGetName(dev=%d) = %d", i, r);
  return r;
}

int nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *mem) {
  int i = handle_index(device);
  if (i >= 0) {
    mem->total = fake_devices[i].totalGlobalMem;
    mem->used = fake_mem_used(i);
    mem->free = mem->total - mem->used;
  }
  FAKE_LOG("nvmlDeviceGetMemoryInfo(dev=%d) = %d", i, i >= 0 ? 0 : NVML_ERROR_INVALID_ARGUMENT);
  return i >= 0 ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}

int nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
  int i = handle_index(device);
  if (i >= 0) {
    *major = fake_devices[i].major;
    *minor = fake_devices[i].minor;
  }
  FAKE_LOG("nvmlDeviceGetCudaComputeCapability(dev=%d) = %d", i,
           i >= 0 ? 0 : NVML_ERROR_INVALID_ARGUMENT);
  return i >= 0 ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}

int nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *count, void *infos) {
  (void)infos;
  int i = handle_index(device);
  if (i >= 0) *count = 0;
  FAKE_LOG("nvmlDeviceGetComputeRunningProcesses(dev=%d) = %d  count=0", i,
           i >= 0 ? 0 : NVML_ERROR_INVALID_ARGUMENT);
  return i >= 0 ? NVML_SUCCESS : NVML_ERROR_INVALID_ARGUMENT;
}
int nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *count, void *infos) {
  return nvmlDeviceGetComputeRunningProcesses_v3(device, count, infos);
}

int nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, int *type) {
  (void)device, (void)type;
  FAKE_LOG("nvmlDeviceGetNvLinkRemoteDeviceType(link=%u) = %d", link, NVML_ERROR_NOT_SUPPORTED);
  return NVML_ERROR_NOT_SUPPORTED;
}

int nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  (void)device, (void)pci;
  FAKE_LOG("nvmlDeviceGetNvLinkRemotePciInfo(link=%u) = %d", link, NVML_ERROR_NOT_SUPPORTED);
  return NVML_ERROR_NOT_SUPPORTED;
}

int nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, void *info) {
  (void)device, (void)info;
  FAKE_LOG("nvmlDeviceGetGpuFabricInfoV() = %d", NVML_ERROR_NOT_SUPPORTED);
  return NVML_ERROR_NOT_SUPPORTED;
}
