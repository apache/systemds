
#ifndef __CUM_MAX_H
#define __CUM_MAX_H

#pragma once
#include <cuda_runtime.h>

/**
 * Do a cumulative maximum over all columns of a matrix
 * @param g_idata   input data stored in device memory (of size rows x cols)
 * @param g_odata   output/temporary array stored in device memory (of size rows x cols)
 * @param g_tdata temporary accumulated block offsets
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 * @param block_height number of rows processed per block
 */

/**
 * Cumulative maximum instantiation for double
 */
extern "C"
__global__ void cumulative_max_up_sweep_d(double *g_idata, double* g_tdata, uint rows, uint cols,
    uint block_height)
{
	MaxOp<double> op;
	cumulative_scan_up_sweep<MaxOp<double>, double>(g_idata, g_tdata, rows, cols, block_height, op);
}

/**
 * Cumulative maximum instantiation for float
 */
extern "C"
__global__ void cumulative_max_up_sweep_f(float *g_idata, float* g_tdata, uint rows, uint cols,
    uint block_height)
{
	MaxOp<float> op;
	cumulative_scan_up_sweep<MaxOp<float>, float>(g_idata, g_tdata, rows, cols, block_height, op);
}

/**
 * Cumulative maximum instantiation for double
 */
extern "C" __global__ void cumulative_max_down_sweep_d(double *g_idata, double *g_odata, double* g_tdata, uint rows,
    uint cols, uint block_height)
{
	MaxOp<double> op;
	cumulative_scan_down_sweep<MaxOp<double>, MaxNeutralElement<double>, double>(g_idata, g_odata, g_tdata, rows, cols, block_height, op);
}

/**
 * Cumulative maximum instantiation for float
 */
extern "C" __global__ void cumulative_max_down_sweep_f(float *g_idata, float *g_odata, float* g_tdata, uint rows,
    uint cols, uint block_height)
{
	MaxOp<float> op;
	cumulative_scan_down_sweep<MaxOp<float>, MaxNeutralElement<float>, float>(g_idata, g_odata, g_tdata, rows, cols, block_height, op);
}

#endif // __CUM_MAX_H
