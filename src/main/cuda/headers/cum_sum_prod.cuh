/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef __CUM_SUM_PROD_H
#define __CUM_SUM_PROD_H

#pragma once

#include <cuda_runtime.h>

/**
 * Cumulative Sum-Product
 * Applies column-wise x_n = x_n + y_n * sum with sum = x_n-1 on a two column matrix in a down-up-sweep cascade of
 * thread blocks.
 * ToDo: Put phases of sum-prod into separate kernels
 */
template<typename T, typename Vec2Accessor>
__device__ void cumulative_sum_prod(T *g_idata, ///< input data stored in device memory (of size n)
		T *g_odata, ///< output/temporary array stored in device memory (of size n)
		T *g_tiData,  // data input to final stage from intermediate stage
		T *g_toData,  // temporary data produced by cascading thread blocks
		uint rows,  ///< rows in input and temporary/output arrays
		uint block_height, // number of rows in a block
		uint phase, // up and down sweep are implemented in this one kernel for now. <phase> indicates what to do.
		// phase 0: initial down-sweep; 1: subsequent down-sweeps of the cascade;
		//       2: subsequent up-sweeps; 3: final up-sweep
		bool verbose = false) // toggle printf output (for debugging purposes)
{
	uint idx = blockIdx.x * block_height;

	// return if thread of (last) block is out of bounds
	// ToDo: should not happen - remove after checking
	if (idx > rows - 1)
		return;

	uint n = min(idx + block_height, rows);
	auto value = Vec2Accessor::init();

	// if up-sweep, fetch initial value from intermediate block results
	if(phase > 1)
	{
		int read_idx = (blockIdx.x - 1) * 2;
		T a = 0.0;
		if(read_idx >= 0)
			a = g_tiData[read_idx];

		value = Vec2Accessor::get(g_idata, idx * 2);

		if(verbose)
			printf("In: blockIdx.x: %d, 2*idx: %d, val_x: %f, val_y: %f, a: %f, phase=%d\n",
				blockIdx.x, 2*idx, value.x, value.y, a, phase);

		value.x = value.x + value.y * a;
		n = min(idx + block_height, rows);
	}
	else // fetch initial value from input
	{
		value = Vec2Accessor::get(g_idata, 2 * idx);

		if(verbose)
			printf("Read: blockIdx.x: %d, 2 * idx: %d, val_x: %f, val_y: %f, n: %d, phase=%d\n",
				blockIdx.x, 2 * idx, value.x, value.y, n, phase);
	}

	T sumprod = value.x;
	T prod = value.y;

	// write to one or two column output in up-sweep
	if(phase == 2)
		Vec2Accessor::put(g_odata, 2 * idx, sumprod, prod);
	else if (phase == 3)
		g_odata[idx] = value.x;

	if(phase > 1 && verbose)
		printf("Out[i=0]: blockIdx.x: %d, idx: %d, sumprod: %f, prod: %f, n: %d, phase=%d\n",
			blockIdx.x, idx, sumprod, prod, n, phase);

	// loop over 2nd to n rows of thread block
	for (int i = idx + 1; i < n; ++i)
	{
		value = Vec2Accessor::get(g_idata, 2 * i);

		prod = prod * value.y;
		sumprod = value.x + value.y * sumprod;

		// write subsequent row results of that block to one or two column output in up-sweep
		if(phase == 2)
			Vec2Accessor::put(g_odata, 2 * i, sumprod, prod);
		else if (phase == 3)
			g_odata[i] = sumprod;

		if(verbose)
			printf("Loop[i=%d]: blockIdx.x: %d, read_i: %d, sumprod: %f, prod: %f, n: %d, phase=%d\n",
				i, blockIdx.x, i * 2, sumprod, prod, n, phase);
	}

	// down sweep (phase 0/1)
	// write out accumulated block offset to intermediate buffer
	if (g_toData != nullptr)
	{
		uint write_idx = blockIdx.x * 2;
		if(blockIdx.x < gridDim.x - 1)
		{
			Vec2Accessor::put(g_toData, write_idx, sumprod, prod);

			if(verbose)
				printf("Carry: blockIdx.x: %d, write_idx: %d, sumprod: %f, prod: %f, n: %d, phase=%d\n",
					blockIdx.x, write_idx, sumprod, prod, n, phase);
		}
	}
}

/**
 * Cumulative sum-product instantiation for double
 */
extern "C"
__global__ void cumulative_sum_prod_d(double *g_idata, double *g_odata, double *g_tiData, double *g_toData, uint rows,
    uint block_height, uint offset)
{
	cumulative_sum_prod<double, double2Accessor>(g_idata, g_odata, g_tiData, g_toData, rows, block_height, offset);
}

/**
 * Cumulative sum-product instantiation for float
 */
extern "C" __global__ void cumulative_sum_prod_f(float *g_idata, float *g_odata, float *g_tiData, float *g_toData,
    uint rows, uint block_height, uint offset)
{
	cumulative_sum_prod<float, float2Accessor>(g_idata, g_odata, g_tiData, g_toData, rows, block_height, offset);
}

#endif // __CUM_SUM_PROD_H
