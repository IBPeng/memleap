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

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "defs.h"

/* contains channel configurations */
#include "config.h"

extern "C" __device__ __noinline__ void instrument_mem(int pred,
                                                        uint32_t instr_id,
       	   	      		   		                        //int opcode_id,
                                                        uint64_t addr,
						                                //int memop_type,
						                                uint32_t memop_size,
                                                        //uint64_t pchannel_dev
                                                        uint64_t pchannel_dev_array
                                                        ) 
{
    if (!pred) {
        return;
    }

    int active_mask = ballot(1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
        
    mem_access_t ma;

    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = shfl(addr, i);
    }

    //int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

/*  int4 cta = get_ctaid();
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.warp_id = get_global_warp_id();//get_warpid();
    ma.block_id = smid;//block_id;
    ma.opcode_id = opcode_id;
    ma.memop_type = memop_type;*/

    ma.instr_id  = instr_id;
    ma.memop_size = memop_size;

    /* first active lane pushes information on the channel */
    if ( first_laneid == laneid ) { // && get_global_warp_id()<32  && ma.block_id == 0
    
        ChannelDev *chanel_dev_arr = (ChannelDev *)pchannel_dev_array;
        ChannelDev *channel_dev = chanel_dev_arr + (smid%NUM_RECV_BUF);//+ smid%8;
        //printf("chanel_dev_arr %p, channel_dev %p\n", chanel_dev_arr, channel_dev);
        //ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;

        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}
