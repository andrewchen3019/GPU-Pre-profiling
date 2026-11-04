#include "fakecuda.h"
#include <pthread.h>
#include <stdlib.h>
#include <sys/mman.h>

struct alloc {
  char *ptr;
  size_t size;
  int device;
};

static struct alloc *allocs;
static size_t n_allocs, cap_allocs;
static size_t used[FAKECUDA_MAX_DEVICES];
static pthread_mutex_t mem_lock = PTHREAD_MUTEX_INITIALIZER;

static void track_add(char *ptr, size_t size, int device) {
  if (n_allocs == cap_allocs) {
    cap_allocs = cap_allocs ? 2 * cap_allocs : 256;
    allocs = realloc(allocs, cap_allocs * sizeof *allocs);
  }
  allocs[n_allocs++] = (struct alloc){ptr, size, device};
  if (device >= 0) used[device] += size;
}

static int track_remove(void *ptr, int want_host, size_t *size) {
  for (size_t i = 0; i < n_allocs; i++) {
    if (allocs[i].ptr != ptr || (allocs[i].device < 0) != want_host) continue;
    *size = allocs[i].size;
    if (allocs[i].device >= 0) used[allocs[i].device] -= allocs[i].size;
    allocs[i] = allocs[--n_allocs];
    return 1;
  }
  return 0;
}

static void *reserve(size_t size) {
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                 -1, 0);
  return p == MAP_FAILED ? NULL : p;
}

int fake_mem_alloc(int device, size_t size, void **out) {
  *out = NULL;
  if (size == 0) return FAKE_MEM_OK;
  pthread_mutex_lock(&mem_lock);
  int r = FAKE_MEM_OOM;
  fake_ctx_touch(device);
  if (size <= fake_devices[device].totalGlobalMem - used[device] && (*out = reserve(size))) {
    track_add(*out, size, device);
    r = FAKE_MEM_OK;
  }
  pthread_mutex_unlock(&mem_lock);
  return r;
}

int fake_mem_free(void *ptr) {
  if (!ptr) return FAKE_MEM_OK;
  size_t size;
  pthread_mutex_lock(&mem_lock);
  int found = track_remove(ptr, 0, &size);
  pthread_mutex_unlock(&mem_lock);
  if (!found) return FAKE_MEM_INVALID;
  munmap(ptr, size);
  return FAKE_MEM_OK;
}

int fake_host_alloc(size_t size, void **out) {
  *out = NULL;
  if (size == 0) return FAKE_MEM_OK;
  if (!(*out = reserve(size))) return FAKE_MEM_OOM;
  pthread_mutex_lock(&mem_lock);
  track_add(*out, size, -1);
  pthread_mutex_unlock(&mem_lock);
  return FAKE_MEM_OK;
}

int fake_host_free(void *ptr) {
  if (!ptr) return FAKE_MEM_OK;
  size_t size;
  pthread_mutex_lock(&mem_lock);
  int found = track_remove(ptr, 1, &size);
  pthread_mutex_unlock(&mem_lock);
  if (!found) return FAKE_MEM_INVALID;
  munmap(ptr, size);
  return FAKE_MEM_OK;
}

size_t fake_mem_used(int device) {
  pthread_mutex_lock(&mem_lock);
  size_t u = used[device];
  pthread_mutex_unlock(&mem_lock);
  return u;
}

int fake_mem_lookup(const void *ptr, int *device) {
  int found = 0;
  pthread_mutex_lock(&mem_lock);
  for (size_t i = 0; i < n_allocs && !found; i++) {
    const struct alloc *a = &allocs[i];
    if ((const char *)ptr >= a->ptr && (const char *)ptr < a->ptr + a->size) {
      *device = a->device;
      found = 1;
    }
  }
  pthread_mutex_unlock(&mem_lock);
  return found;
}
