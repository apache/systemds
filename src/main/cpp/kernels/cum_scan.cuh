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

#ifndef __CUM_SCAN_H
#define __CUM_SCAN_H

#pragma once

/**
 * Cumulative Scan - Applies <scanOp> to accumulate values over columns of an input matrix
 * @param scanOp - Type of the functor object that implements the scan operation
 */
// --------------------------------------------------------
template<typename scanOp, typename T>
__device__ void cumulative_scan_up_sweep(T *g_idata, T *g_tdata, uint rows, uint cols, uint block_height, 
		scanOp scan_op)  
{
	// check if the current thread is within row-length
	if (blockIdx.x * blockDim.x + threadIdx.x > cols - 1)
		return;

	uint offset = blockIdx.y * cols * block_height + blockIdx.x * blockDim.x;
	uint idx = offset + threadIdx.x;

	// initial accumulator value
	T acc = g_idata[idx];

	// loop through <block_height> number of items colwise
	uint last_idx = min(idx + block_height * cols, rows * cols);

	// loop from 2nd  value
	for (int i = idx + cols; i < last_idx; i += cols)
		acc = scan_op(acc, g_idata[i]);

	// write out accumulated block offset
	if (block_height < rows)
	{
		g_tdata[blockIdx.y * cols + blockIdx.x * blockDim.x + threadIdx.x] = acc;
		// if(threadIdx.x == 0)
		// 	printf("blockIdx.y=%d, acc=%f\n", blockIdx.y, acc);
	}
}

// --------------------------------------------------------
template<typename scanOp, typename NeutralElement, typename T>
__device__ void cumulative_scan_down_sweep(T *g_idata, T *g_odata, T *g_tdata, uint rows, uint cols, uint block_height, 
	scanOp scan_op)
{
	// check if the current thread is within row-length
	if (blockIdx.x * blockDim.x + threadIdx.x > cols - 1)
		return;

	uint idx = blockIdx.y * cols * block_height + blockIdx.x * blockDim.x + threadIdx.x;
	int offset_idx = blockIdx.y * cols + blockIdx.x * blockDim.x + threadIdx.x;
	
	// initial accumulator value
	T acc = (gridDim.y > 1) ? ((blockIdx.y > 0) ? g_tdata[offset_idx-1] : NeutralElement::get()) : NeutralElement::get();

	// if(threadIdx.x == 0)
	// {
	// 	printf("gridDim.y=%d, blockIdx.y=%d, down sweep acc=%f\n", gridDim.y, blockIdx.y, acc);
	// 	printf("gridDim.y=%d, blockIdx.y=%d, g_tdata[%d]=%f\n", gridDim.y, blockIdx.y, idx, g_tdata[offset_idx]);
	// }

	g_odata[idx] = acc = scan_op(acc, g_idata[idx]);
	
	// loop through <block_height> number of items colwise
	uint last_idx = min(idx + block_height * cols, rows * cols);

	// loop from 2nd  value
	 for (int i = idx + cols; i < last_idx; i += cols)
		g_odata[i] = acc = scan_op(acc, g_idata[i]);
}

#endif // __CUM_SCAN_H
