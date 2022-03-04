#include "cupti_um.hpp"

int main(int argc, char **argv)
{
    int *data = NULL;
    int size = 64*1024;     // 64 KB
    int i = 123;

  /*
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
        return 0;
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE) {
        printf("Test is waived, unified memory is not supported on the device.\n");
        return 0;
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES) {
        printf("Test is waived, unified memory is not supported on the non-P2P multi-gpu setup.\n");
        return 0;
    }
    else {
        CUPTI_CALL(res);
    }
  */
    uvm_profiling_init();

    // enable unified memory counter activity
    //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
    uvm_profiling_start();

    // allocate unified memory
    printf("Allocation size in bytes %d\n", size);
    RUNTIME_API_CALL(cudaMallocManaged(&data, size));
    mem_obj_t a = {(uint64_t) data, (uint64_t) data + size};
    um_dataobj_map.push_back( a );

    // CPU access
    writeData(data, size, i);
    // kernel launch
    testKernel<<<1,1>>>(data, size, i);
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    // CPU access
    checkData("CPU", data, size, -i);

    // free unified memory
    RUNTIME_API_CALL(cudaFree(data));

    
    //CUPTI_CALL(cuptiActivityFlushAll(0));
    // disable unified memory counter activity
    //CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
    uvm_profiling_stop();

    cudaDeviceReset();

    return 0;
}
