/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <unistd.h>
#include <string>
#include <map>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "defs.h"
#include "config.h"

/* ccontains functions for processing traces */
#include "analyzer.h"

/* some global variables for memory objects */
std::vector<mem_obj_t> g_umem_obj_map;
std::vector<mem_obj_t> g_dmem_obj_map;

/* some global variables for data accesses in kernel and global scopes */
std::map<int, std::vector<mem_obj_t>> g_da_kernel_map;
std::map<int, std::vector<mem_obj_t>> g_da_global_map;
std::mutex g_mutex;

/* Channel used to communicate from GPU to CPU receiving thread */
static __managed__ ChannelDev channel_dev[NUM_RECV_BUF];
static ChannelHost channel_host[NUM_RECV_BUF];

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
//volatile bool recv_thread_receiving = false;
volatile int recv_thread_receiving = 0;

/* processing thread */
pthread_t proc_threads[NUM_RECV_BUF];
char *recv_buffer_array[NUM_RECV_BUF];
volatile int buffer_head = 0;


/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint64_t instr_begin_interval = 0;
uint64_t instr_end_interval = UINT64_MAX;
int verbose = 0;
std::string target_kernel;


/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* function map */
std::unordered_set<CUfunction> already_instrumented;


/* This function is called as soon as the program starts, 
 * no GPU calls should be made a this moment */
void nvbit_at_init() 
{

    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(instr_begin_interval, "INSTR_BEGIN", 0,
		            "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(instr_end_interval, "INSTR_END", 0, //UINT64_MAX,
		            "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");

    GET_VAR_STR(target_kernel, "TARGET_KERNEL", "Selected kernel to instrument");

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());


    for(int i=0; i<NUM_RECV_BUF; i++){
      recv_buffer_array[i] = (char *)calloc(sizeof(int) + CHANNEL_SIZE, 1);
      assert( recv_buffer_array[i] && "Failed to allocate recv_buffer_array");
    }

}

/* This function is called just before the program terminates, no GPU calls
 * should be made a this moment */
void nvbit_at_term()
{
  for(int i=0; i<NUM_RECV_BUF; i++){
    free(recv_buffer_array[i]);
  }

  std::string pad(100, '-');
  printf("\n%s\nNVBit exit\n", pad.c_str());
}

/* This function starts a pthread on the host at nvbit_at_ctx_init
 * to process traces from the channel. */
void *recv_thread_fun(void *) {
    
    const size_t size_ma = sizeof(mem_access_t);
    const size_t size_int = sizeof(int);
    
    while (recv_thread_started) {

        if ( recv_thread_receiving > 0 ){

          for(int i=0; i<NUM_RECV_BUF; i++){
            int* num_ma = (int*) (recv_buffer_array[i]);
            char *recv_buffer = recv_buffer_array[i] + size_int;

            //while( *num_ma != 0){
            //  pthread_yield();
            //}
            if( *num_ma != 0 )
              continue;

            uint32_t num_recv_bytes = channel_host[i].recv(recv_buffer, CHANNEL_SIZE);
            
            if(num_recv_bytes > 0){
              assert( num_recv_bytes%size_ma == 0);
              size_t num_recv_ma = num_recv_bytes/size_ma;

              mem_access_t *ma = (mem_access_t *) recv_buffer;
              /*printf("recv_thread_fun:: recv_buffer[%d] num_ma%zu size_ma%zu :: Instr%d - %d bytes\n", 
                      i, num_recv_ma, size_ma,
                      ma[0].instr_id, ma[0].memop_size);*/

              /* check the last event 
               * cta_id_x = -1 means that the kernel has completed */              
              if (ma[num_recv_ma-1].memop_size == -1) {
                //recv_thread_receiving = false;
                recv_thread_receiving --;
                //printf("num_ma%d buffer_head%d \n", num_ma, buffer_head);
              }

              /* signal to proc_thread to process */
              *num_ma = num_recv_ma;
            }
          }
        }
    }
 
    return NULL;
}


/* This function starts a pthread on the host at nvbit_at_ctx_init
 * to process traces from the channel. */
void *proc_thread_fun( void* arg ) {
    
    char *recv_buffer = *((char **) arg);
    //printf("proc_thread %d \n", proc_id);
    int* num_ma = (int*)recv_buffer;
    mem_access_t* ma = (mem_access_t*)(recv_buffer + sizeof(int));

    while ( recv_thread_started ) {
        //for(int i=0; i<NUM_RECV_BUF; i++){
            //recv_buffer = recv_buffer_array[i];
            //ma = (mem_access_t*)(recv_buffer + sizeof(int));
            //num_ma = (int*)recv_buffer;

          if( num_ma[0] != 0 ){
              /* printf("num_ma%d buffer_head%d buffer_tail%d Instr%d - CTA %d,%d,%d - block id %d - warp %d - opcode %s - %s - %d bytes\n",*num_ma, buffer_head,buffer_tail,
                        ma->instr_id, 
		                    ma->cta_id_x, ma->cta_id_y, ma->cta_id_z, 
		                    ma->block_id, ma->warp_id,
                        id_to_opcode_map[ma->opcode_id].c_str(),
			                  Instr::memOpTypeStr[ma->memop_type],
			                  ma->memop_size);*/

            /*printf("proc_thread_fun:: recv_buffer %p num_ma%d Instr%d - %d bytes\n",
                    recv_buffer, *num_ma, ma->instr_id, ma->memop_size);*/

          process_ma_um(ma, *num_ma);
          num_ma[0] = 0;
          
          //break;
        }
    }

    return NULL;
}


/* This function is called as soon as a GPU context is started and it should
 * contain any code that we would like to execute at that moment. */
void nvbit_at_ctx_init(CUcontext ctx) {

    for(int i=0; i<NUM_RECV_BUF; i++){
      channel_host[i].init(i, CHANNEL_SIZE, &(channel_dev[i]), NULL);
    }

    recv_thread_started = true;
    pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);

    for(int i=0; i<NUM_RECV_BUF; i++){
      pthread_create(&(proc_threads[i]), NULL, proc_thread_fun, (void*)&recv_buffer_array[i] );
    }
    
    /* Notify nvbit of a pthread used by the tool, this pthread will not
     * trigger any call backs even if executing CUDA events of kernel launches.
     * Multiple pthreads can be registered one after the other. */
    //void nvbit_set_tool_pthread(pthread_t tool_pthread);
    printf("nvbit_at_ctx_init: channel_dev %p channel_dev[0] %p\n", channel_dev, &(channel_dev[0]));
}

/* This function is called as soon as the GPU context is terminated and it
 * should contain any code that we would like to execute at that moment. */
void nvbit_at_ctx_term(CUcontext ctx) {
    
    if (recv_thread_started) {
        recv_thread_started = false;
        pthread_join(recv_thread, NULL);
        for(int i=0; i<NUM_RECV_BUF; i++){
          pthread_join(proc_threads[i], NULL);
        }
        //void nvbit_unset_tool_pthread(pthread_t tool_pthread);
    }

    process_global_data_access() ;

    printf("nvbit_at_ctx_term\n");
}

/* The function is called at the exit of a functino 
 * to flush traces from GPU to host */
__global__ void flush_channel() {
    /* push memory access with negative cta id to communicate the kernel is
     * completed */
    mem_access_t ma;
    //ma.cta_id_x = -1;
    ma.memop_size = -1;

    for(int i=0; i<NUM_RECV_BUF; i++){
      channel_dev[i].push(&ma, sizeof(mem_access_t));

      /* flush channel */
      channel_dev[i].flush();
    }
}


/* Set used to avoid re-instrumenting the same functions multiple times */
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
        //if (verbose) {
            printf("Inspecting function %s at address 0x%lx\n",
                   nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
        //}
	
        uint64_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
          if (//cnt < instr_begin_interval || cnt >= instr_end_interval ||
                instr->getMemOpType() == Instr::memOpType::NONE) {
                cnt++;
                continue;
          }
          if (verbose) {
            instr->printDecoded();
          }

          /* capture opcode */
          if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
            int opcode_id = opcode_to_id_map.size();
            opcode_to_id_map[instr->getOpcode()] = opcode_id;
            id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
          }
          int opcode_id = opcode_to_id_map[instr->getOpcode()];

          /* capture instruction id */
	        uint32_t instr_id = instr->getIdx();

          /* iterate on the operands */
          for (int i = 0; i < instr->getNumOperands(); i++) {
            
            const Instr::operand_t *op = instr->getOperand(i);

            if (op->type == Instr::operandType::MREF &&  
                    instr->getMemOpType() == Instr::memOpType::GLOBAL) {
                    /* insert call to the instrumentation function with its
                     * arguments */
                    nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                    /* predicate value */
                    nvbit_add_call_arg_pred_val(instr);
		                /* instruction id*/
		                nvbit_add_call_arg_const_val32(instr, instr_id);
                    /* opcode id */
                    //nvbit_add_call_arg_const_val32(instr, opcode_id);
                    /* memory reference 64 bit address */
                    nvbit_add_call_arg_mref_addr64(instr);
		                /* memOpType: "NONE", "LOCAL", "GENERIC", "GLOBAL", "SHARED", "CONSTANT" */
		                //nvbit_add_call_arg_const_val32(instr, (int) instr->getMemOpType());
		                /* get mem op size */
		                nvbit_add_call_arg_const_val32(instr, instr->getSize());

                    /* add pointer to channel_dev */
                    nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
            }
          }
          cnt++;
        }/* end of iterate on all the static instructions in the function */
    }/* end of iterate on function */

}


/* This is the function called every beginning (is_exit = 0) and
 * end (is_exit = 1) of a CUDA driver call. 
 * It calls instrument function to inject code */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (skip_flag) return;

    /* Capture memory allocation on the device */
    if( is_exit ){
      switch(cbid){
        case API_CUDA_cuMemAlloc :
        {
          cuMemAlloc_params *p = (cuMemAlloc_params *) params;
          uint64_t addr_st = (uint64_t) (*(p->dptr));
          uint64_t addr_end = addr_st + p->bytesize;
          printf("\tD_OBJ%5zu :: cudaMalloc %u bytes [0x%lx 0x%lx]\n", 
                  g_dmem_obj_map.size(), p->bytesize, addr_st, addr_end);
          g_dmem_obj_map.push_back(mem_obj_t({addr_st, addr_end}));      
          return;
        }
  
        case API_CUDA_cuMemAlloc_v2 :
        {
          cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *) params;
          uint64_t addr_st = (uint64_t) (*(p->dptr));
          uint64_t addr_end = addr_st + p->bytesize;
          printf("\tD_OBJ%5zu :: cuMemAlloc_v2 %zu bytes [0x%lx 0x%lx]\n",
                  g_dmem_obj_map.size(), p->bytesize, addr_st, addr_end);
          g_dmem_obj_map.push_back(mem_obj_t({addr_st, addr_end}));        
          return;
        }

        case API_CUDA_cuMemAllocPitch_v2 :
        {
          cuMemAllocPitch_v2_params *p = (cuMemAllocPitch_v2_params *) params;
          uint64_t addr_st = (uint64_t) (*(p->dptr));
          uint64_t addr_end = addr_st + p->WidthInBytes * p->Height;
          printf("\tD_OBJ%5zu :: cuMemAllocPitch_v2 %zu bytes [0x%lx 0x%lx]\n",
                  g_dmem_obj_map.size(), (p->WidthInBytes * p->Height), addr_st, addr_end);
          g_dmem_obj_map.push_back(mem_obj_t({addr_st, addr_end}));        
          return;
        }

        /* Capture memory allocation in Unified Memmory */ 
        case API_CUDA_cuMemAllocManaged :
        {      
          cuMemAllocManaged_params *p = (cuMemAllocManaged_params *) params;
          uint64_t addr_st = (uint64_t) (*(p->dptr));
          uint64_t addr_end = addr_st + p->bytesize;
          g_umem_obj_map.push_back(mem_obj_t({addr_st, addr_end}));            
          printf("\tcudaMallocManaged %zu bytes [0x%lx 0x%lx]\n", 
                  p->bytesize, addr_st, addr_end);
          return;
        }

        /* Capture data movement */
        case API_CUDA_cuMemcpyHtoD_v2 :
        {
          cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *) params;
          uint64_t addr_h = (uint64_t) p->srcHost;
          uint64_t addr_d = (uint64_t) p->dstDevice;
          size_t   bytes_moved = p->ByteCount;
          int obj_id = find_data_obj( addr_d );
          printf("\tData Movement Host[0x%lx] -> D_OBJ %5d (%zu bytes) \n", 
                addr_h, obj_id, bytes_moved);
          return;
        }

        case API_CUDA_cuMemcpyHtoDAsync_v2 :
        { 
          cuMemcpyHtoDAsync_v2_params *p = (cuMemcpyHtoDAsync_v2_params *) params;
          uint64_t addr_h = (uint64_t) p->srcHost;
          uint64_t addr_d = (uint64_t) p->dstDevice;
          size_t   bytes_moved = p->ByteCount;
          int obj_id = find_data_obj( addr_d );
          printf("\tData Movement Host[0x%lx] -> D_OBJ %5d (%zu bytes) \n", 
                addr_h, obj_id, bytes_moved);
          return;
        }

        case API_CUDA_cuMemcpyDtoHAsync_v2 :
        { 
          cuMemcpyDtoHAsync_v2_params *p = (cuMemcpyDtoHAsync_v2_params *) params;
          uint64_t addr_h = (uint64_t) p->dstHost;
          uint64_t addr_d = (uint64_t) p->srcDevice;
          size_t   bytes_moved = p->ByteCount;
          int obj_id = find_data_obj( addr_d );
          if(obj_id>-1)
            printf("\tData Movement D_OBJ %5d -> Host[0x%lx] (%zu bytes) \n", 
                  obj_id, addr_h, bytes_moved);
          return;
        } 

        case API_CUDA_cuMemcpyDtoH_v2 :
        { 
          cuMemcpyDtoH_v2_params *p = (cuMemcpyDtoH_v2_params *) params;
          uint64_t addr_h = (uint64_t) p->dstHost;
          uint64_t addr_d = (uint64_t) p->srcDevice;
          size_t   bytes_moved = p->ByteCount;
          int obj_id = find_data_obj( addr_d );
          if(obj_id>-1)
            printf("\tData Movement D_OBJ %5d -> Host[0x%lx] (%zu bytes) \n", 
                  obj_id, addr_h, bytes_moved);
          return;
        } 

        default:
          break;
      }
    }

    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel) {

        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        const char* kernel_name = nvbit_get_func_name(ctx, p->f);

        /* If user specifies a target kernel 
         * only instrument the target kernel */
	      if( target_kernel.size() > 0 ){
          const char* target_kernel_c = target_kernel.c_str();
	        if( !strstr(kernel_name, target_kernel_c) ){
	          //printf("Kernel %s is skipped \n", kernel_name);
	          return;
	        }
	      }

        if (!is_exit) {
            int nregs;
            CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

            int shmem_static_nbytes;
            CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

            instrument_function_if_needed(ctx, p->f);

            nvbit_enable_instrumented(ctx, p->f, true);

            //recv_thread_receiving = true;
            recv_thread_receiving = NUM_RECV_BUF;

            printf(
                "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
                "%d - shmem %d - cuda stream id %ld\n",
                nvbit_get_func_name(ctx, p->f), p->gridDimX, p->gridDimY,
                p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ, nregs,
                shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);

        } else {
            /* make sure current kernel is completed */
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
	          if( error != cudaSuccess){
              printf("CUDA_ERROR :: %s\n", cudaGetErrorString(error));
              assert(false);
            }

            /* make sure we prevent re-entry on the nvbit_callback when issuing
             * the flush_channel kernel */
            skip_flag = true;

            /* issue flush of channel so we are sure all the memory accesses
             * have been pushed */
            flush_channel<<<1, 1>>>();
            cudaDeviceSynchronize();
            error = cudaGetLastError();
	          if( error != cudaSuccess){
              printf("CUDA_ERROR :: %s\n", cudaGetErrorString(error));
              assert(false);
            }

            /* unset the skip flag */
            skip_flag = false;

            /* wait here until the receiving thread has not finished with the
             * current kernel */
            while (recv_thread_receiving>0) {
                pthread_yield();
            }

            for(int i=0; i<NUM_RECV_BUF; i++){
              int* num_ma = (int*)(recv_buffer_array[i]);
              while(num_ma[0] != 0){
                pthread_yield();
              }
            }
            process_kernel_data_access();
            printf("\n\n"); 
            //printf("Kernel %s exit \n\n", nvbit_get_func_name(ctx, p->f));

	      }
    }
}



