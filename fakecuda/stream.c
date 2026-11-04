#include "fakecuda.h"
#include <stdlib.h>
#include <time.h>

struct CUstream_st {
  int device, priority;
  unsigned flags;
};

struct CUevent_st {
  unsigned flags;
  int recorded;
  double ms;
};

static int is_default(cudaStream_t s) { return (size_t)s <= 2; }

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority) {
  struct CUstream_st *s = malloc(sizeof *s);
  *s = (struct CUstream_st){fake_current_device, priority, flags};
  fake_ctx_touch(fake_current_device);
  *stream = s;
  FAKE_LOG("cudaStreamCreate(flags=%u, priority=%d) = 0  %p", flags, priority, (void *)s);
  return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
  return cudaStreamCreateWithPriority(stream, flags, 0);
}

cudaError_t cudaStreamCreate(cudaStream_t *stream) {
  return cudaStreamCreateWithPriority(stream, cudaStreamDefault, 0);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (is_default(stream)) return cudaErrorInvalidResourceHandle;
  FAKE_LOG("cudaStreamDestroy(%p) = 0", (void *)stream);
  free(stream);
  return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  FAKE_LOG("cudaStreamQuery(%p) = 0  idle", (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  FAKE_LOG("cudaStreamWaitEvent(%p, %p, %u) = 0", (void *)stream, (void *)event, flags);
  return cudaSuccess;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority) {
  *priority = is_default(stream) ? 0 : stream->priority;
  return cudaSuccess;
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags) {
  *flags = is_default(stream) ? 0 : stream->flags;
  return cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData) {
  FAKE_LOG("cudaLaunchHostFunc(%p) = 0  running now", (void *)stream);
  fn(userData);
  return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags) {
  (void)flags;
  FAKE_LOG("cudaStreamAddCallback(%p) = 0  running now", (void *)stream);
  callback(stream, cudaSuccess, userData);
  return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *status) {
  *status = cudaStreamCaptureStatusNone;
  FAKE_LOG("cudaStreamIsCapturing(%p) = 0  none", (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *status,
                                        unsigned long long *id, cudaGraph_t *graph,
                                        const cudaGraphNode_t **deps, size_t *numDeps) {
  *status = cudaStreamCaptureStatusNone;
  if (id) *id = 0;
  if (graph) *graph = NULL;
  if (deps) *deps = NULL;
  if (numDeps) *numDeps = 0;
  FAKE_LOG("cudaStreamGetCaptureInfo(%p) = 0  none", (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t stream, enum cudaStreamCaptureStatus *status,
                                        unsigned long long *id, cudaGraph_t *graph,
                                        const cudaGraphNode_t **deps,
                                        const cudaGraphEdgeData **edgeData, size_t *numDeps) {
  if (edgeData) *edgeData = NULL;
  return cudaStreamGetCaptureInfo_v2(stream, status, id, graph, deps, numDeps);
}

static __thread enum cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal;

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
  enum cudaStreamCaptureMode old = capture_mode;
  capture_mode = *mode;
  *mode = old;
  return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  struct CUevent_st *e = calloc(1, sizeof *e);
  e->flags = flags;
  *event = e;
  FAKE_LOG("cudaEventCreate(flags=%u) = 0  %p", flags, (void *)e);
  return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  if (!event) return cudaErrorInvalidResourceHandle;
  FAKE_LOG("cudaEventDestroy(%p) = 0", (void *)event);
  free(event);
  return cudaSuccess;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
  (void)flags;
  if (!event) return cudaErrorInvalidResourceHandle;
  event->recorded = 1;
  event->ms = now_ms();
  FAKE_LOG("cudaEventRecord(%p, stream=%p) = 0", (void *)event, (void *)stream);
  return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return cudaEventRecordWithFlags(event, stream, 0);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
  return event ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return event ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  if (!start || !end) return cudaErrorInvalidResourceHandle;
  if ((start->flags | end->flags) & cudaEventDisableTiming) return cudaErrorInvalidResourceHandle;
  if (!start->recorded || !end->recorded) return cudaErrorInvalidResourceHandle;
  *ms = (float)(end->ms - start->ms);
  FAKE_LOG("cudaEventElapsedTime() = 0  %.3f ms", *ms);
  return cudaSuccess;
}
