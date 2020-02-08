
#ifndef __CUM_MIN_H
#define __CUM_MIN_H

#pragma once
#include <cuda_runtime.h>

/**
 * Do a cumulative minimum over all columns of a matrix
 * @param g_idata   input data stored in device memory (of size rows x cols)
 * @param g_odata   output/temporary array stored in device memory (of size rows x cols)
 * @param g_tdata temporary accumulated block offsets
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 * @param block_height number of rows processed per block
 */

/**
 * Cumulative minimum instantiation for double
 */
extern "C"
__global__ void cumulative_min_up_sweep_d(double *g_idata, double* g_tdata, uint rows, uint cols,
    uint block_height)
{
	MinOp<double> op;
	cumulative_scan_up_sweep<MinOp<double>, double>(g_idata, g_tdata, rows, cols, block_height, op);
}

/**
 * Cumulative minimum instantiation for float
 */
extern "C"
__global__ void cumulative_min_up_sweep_f(float *g_idata, float* g_tdata, uint rows, uint cols,
    uint block_height)
{
	MinOp<float> op;
	cumulative_scan_up_sweep<MinOp<float>, float>(g_idata, g_tdata, rows, cols, block_height, op);
}

/**
 * Cumulative minimum instantiation for double
 */
extern "C" __global__ void cumulative_min_down_sweep_d(double *g_idata, double *g_odata, double* g_tdata, uint rows,
    uint cols, uint block_height)
{
	MinOp<double> op;
	cumulative_scan_down_sweep<MinOp<double>, MinNeutralElement<double>, double>(g_idata, g_odata, g_tdata, rows, cols, block_height, op);
}

/**
 * Cumulative minimum instantiation for float
 */
extern "C" __global__ void cumulative_min_down_sweep_f(float *g_idata, float *g_odata, float* g_tdata, uint rows,
    uint cols, uint block_height)
{
	MinOp<float> op;
	cumulative_scan_down_sweep<MinOp<float>, MinNeutralElement<float>, float>(g_idata, g_odata, g_tdata, rows, cols, block_height, op);
}

#endif // __CUM_MIN_H
