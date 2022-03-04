
#ifndef CUPTI_UM_H
#define CUPTI_UM_H

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cupti.h>

/* metadata of a memory allocation */
typedef struct MemoryObject{
  uint64_t st;
  uint64_t end;
  char name[16];

  MemoryObject(void* _st, size_t _size, const char* _name)
    :st( (uint64_t)_st ), end( st + _size ) {
        size_t len = strlen(_name);
        len = (len<16) ?len : 15;
        memcpy(name, _name, len);
        name[len]='\0';
    }
}MemObj;

std::vector<MemObj> um_dataobj_map;
std::map<uint64_t, uint64_t> data_movement_h2d;
std::map<uint64_t, uint64_t> data_movement_d2h;

#define CUPTI_CALL(call)                                                    \
do {                                                                        \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      if(_status == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED)              \
            exit(0);                                                        \
      else                                                                  \
            exit(-1);                                                       \
    }                                                                       \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define BUF_SIZE (16 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static const char *
getUvmCounterKindString(CUpti_ActivityUnifiedMemoryCounterKind kind)
{
    switch (kind) 
    {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
        return "BYTES_TRANSFER_H2D";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
        return "BYTES_TRANSFER_D2H";
    default:
        return "<Unknown Uvm Counter>";
    }
}

static void
printUvmActivity(CUpti_Activity *record)
{
    switch (record->kind) 
    {
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
        {
            CUpti_ActivityUnifiedMemoryCounter2 *uvm = (CUpti_ActivityUnifiedMemoryCounter2 *)record;
	        
	        uint64_t uvm_addr_st = uvm->address;
            uint64_t uvm_addr_end = uvm_addr_st + uvm->value;
            uint64_t total_overlaped = 0;
#ifdef VERBOSE
            printf("%s %llu bytes [0x%016lx 0x%016lx] \n",
                    getUvmCounterKindString(uvm->counterKind), 
                    (unsigned long long)uvm->value, uvm_addr_st, uvm_addr_end);
#endif

	        for(auto r : um_dataobj_map){
                //printf("%s:: 0x%016lx 0x%016lx\n", r.name, r.st, r.end);
                //find overlapping regions
                if( uvm_addr_st <= r.st && r.st  < uvm_addr_end ){

                    uint64_t region_end =  (uvm_addr_end<=r.end)  ?uvm_addr_end :r.end;
                    uint64_t overlaped = region_end - r.st;
                    if( uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD)
		                data_movement_h2d[r.st] += overlaped;
                    else if( uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH)
		                data_movement_d2h[r.st] += overlaped;

                    total_overlaped += overlaped;
                    printf("%s::overlaped %lu\n", r.name, overlaped);   
                }else if( r.st <= uvm_addr_st && uvm_addr_st < r.end ){

                    uint64_t region_end = (uvm_addr_end<=r.end) ?uvm_addr_end :r.end;
                    uint64_t overlaped = region_end - uvm_addr_st;
                    if( uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD)
		                data_movement_h2d[r.st] += overlaped;
                    else if( uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH)
		                data_movement_d2h[r.st] += overlaped;

                    total_overlaped += overlaped;
                    printf("%s::overlaped %lu\n", r.name, overlaped); 
                }

	            //if( r.end < uvm_addr_st ){
                //    break; //data objects are sorted in ascending order of addr_st
                //}
	        }
            
            if( total_overlaped < uvm->value ){
	            printf("Warning: UM page migration [0x%016lx 0x%016lx] has [%zu bytes] in unregistered data objects!\n", 
                        uvm_addr_st, uvm_addr_end, (uvm->value-total_overlaped));
            }

            break;
        }
    default:
        printf("  <unknown activity>\n");
        break;
    }
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    uint8_t *rawBuffer;

    *size = BUF_SIZE;
    rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

    *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
    *maxNumRecords = 0;

    if (*buffer == NULL) {
        printf("Error: out of memory\n");
        exit(-1);
    }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            printUvmActivity(record);
        }
        else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        }
        else {
            CUPTI_CALL(status);
        }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
        printf("Dropped %u activity records\n", (unsigned int)dropped);
    }

    free(buffer);
}

template<class T>
__host__ __device__ void checkData(const char *loc, T *data, int size, int expectedVal) {
    int i;

    for (i = 0; i < size / (int)sizeof(T); i++) {
        if (data[i] != expectedVal) {
            printf("Mismatch found on %s\n", loc);
            printf("Address 0x%p, Observed = 0x%x Expected = 0x%x\n", data+i, data[i], expectedVal);
            break;
        }
    }
}

template<class T>
__host__ __device__ void writeData(T *data, int size, int writeVal) {
    int i;

    for (i = 0; i < size / (int)sizeof(T); i++) {
        data[i] = writeVal;
    }
}

static void uvm_profiling_init()
{
    CUptiResult res;
    int deviceCount;
    CUpti_ActivityUnifiedMemoryCounterConfig config[2];

    DRIVER_API_CALL(cuInit(0));

    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit(-1);
    }

    // register cupti activity buffer callbacks
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // configure unified memory counters
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    config[0].deviceId = 0;
    config[0].enable = 1;

    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    config[1].deviceId = 0;
    config[1].enable = 1;

    res = cuptiActivityConfigureUnifiedMemoryCounter(config, 2);
    if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED) {
        printf("Test is waived, unified memory is not supported on the underlying platform.\n");
        //return 0;
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE) {
        printf("Test is waived, unified memory is not supported on the device.\n");
        //return 0;
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES) {
        printf("Test is waived, unified memory is not supported on the non-P2P multi-gpu setup.\n");
        //return 0;
    }
    else {
        CUPTI_CALL(res);
    }

    //printf("init unified memory counter activity\n");
    //return 0;
}

bool sort_um(MemObj const& lhs, MemObj const& rhs) 
{
  bool res = (lhs.st == rhs.st) ?(lhs.end < rhs.end) : (lhs.st < rhs.st);
  return res;
}

static void uvm_profiling_start()
{
    // enable unified memory counter activity
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
    printf("enable unified memory counter activity \n");

    std::sort(um_dataobj_map.begin(), um_dataobj_map.end(), &sort_um);
}

static void uvm_profiling_stop()
{
    CUPTI_CALL(cuptiActivityFlushAll(0));

    // disable unified memory counter activity
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));


    for(auto d : um_dataobj_map){
      printf("%16s [0x%016lx 0x%016lx] %9.3f MB :: H2D %16lu bytes, D2H %16lu bytes.\n", 
        d.name, d.st, d.end, (d.end-d.st)/1024.0/1024.0, data_movement_h2d[d.st], data_movement_d2h[d.st]);
    }
    data_movement_h2d.clear();
    data_movement_d2h.clear();

    printf("disable unified memory counter activity\n");
}

#endif