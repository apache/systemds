package org.apache.sysml.runtime.matrix.data;

import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardData;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardFilter;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolutionNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilterNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import jcuda.Pointer;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.JCudaObject;
import org.apache.sysml.utils.Statistics;

public class LibMatrixCUDA {
	
	public static cudnnHandle cudnnHandle;
	public static cublasHandle cublasHandle;
	
	public static void conv2d(MatrixObject image, MatrixObject filter, MatrixObject outputBlock, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q)
			throws DMLRuntimeException {
		cudnnTensorDescriptor srcTensorDesc = null;
		cudnnTensorDescriptor dstTensorDesc = null;
		cudnnFilterDescriptor filterDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		Pointer workSpace = null;
		long sizeInBytes = 0;
		Pointer alpha = null;
		Pointer beta = null;
		try {
			// Allocate descriptors
			srcTensorDesc = allocateTensorDescriptor(N, C, H, W);
			dstTensorDesc = allocateTensorDescriptor(N, K, P, Q);
			filterDesc = allocateFilterDescriptor(K, C, R, S);
			
			// Allocate data
			// (Pointer) gpuCtx.prepare(image, true, true);
			// (Pointer) gpuCtx.prepare(filter, true, true);
			
			Pointer imagePointer = ((JCudaObject)image._gpuHandle).jcudaPointer; 
			Pointer filterPointer = ((JCudaObject)filter._gpuHandle).jcudaPointer; 
			Pointer dstPointer = ((JCudaObject)outputBlock._gpuHandle).jcudaPointer; 
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
			
			long sizeInBytesArray[] = { 0 };
            workSpace = new Pointer();
            cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, 
                    srcTensorDesc, filterDesc, convDesc, dstTensorDesc, 
                    algo, sizeInBytesArray);
            
			alpha = pointerTo(1.0); // TODO
			beta = pointerTo(0.0f);
			long start = System.nanoTime();
			int status = cudnnConvolutionForward(cudnnHandle, alpha, 
					srcTensorDesc, imagePointer, 
					filterDesc, filterPointer,
					convDesc, algo, workSpace, sizeInBytes, beta,
					dstTensorDesc, dstPointer);
			Statistics.cudaConvFwdTime.addAndGet(System.nanoTime()-start);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			
			if(srcTensorDesc != null)
				cudnnDestroyTensorDescriptor(srcTensorDesc);
			if(dstTensorDesc != null)
				cudnnDestroyTensorDescriptor(dstTensorDesc);
			if(filterDesc != null)
				cudnnDestroyFilterDescriptor(filterDesc);
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
	}
	
	private static cudnnConvolutionDescriptor allocateConvolutionDescriptor(int padding [], int strides []) {
		cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convDesc);
		int upscale[] = { 1, 1 };
		cudnnSetConvolutionNdDescriptor(convDesc, 2, padding, strides, upscale, 
				CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);
		return convDesc;
	}
	
	private static  Pointer pointerTo(double value) {
        return Pointer.to(new double[] { value });
    }
	
	private static  cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H, int W) {
		cudnnTensorDescriptor ret = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(ret);
		cudnnSetTensor4dDescriptor(ret, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W);
		return ret;
	}
	
	private static cudnnFilterDescriptor allocateFilterDescriptor(int K, int C, int R, int S) {
		cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(filterDesc);
		int filterDim[] = { K, C, R, S };
		cudnnSetFilterNdDescriptor(filterDesc, CUDNN_DATA_DOUBLE, 4, filterDim);
		return filterDesc;
	}


	public static void conv2d_backward_filter(MatrixObject image, MatrixObject dout,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor xTensorDesc = null;
		cudnnTensorDescriptor doutTensorDesc = null;
		cudnnFilterDescriptor dwDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		
		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {
			// Allocate descriptors
			xTensorDesc = allocateTensorDescriptor(N, C, H, W);
			doutTensorDesc = allocateTensorDescriptor(N, K, P, Q);
			dwDesc = allocateFilterDescriptor(K, C, R, S);
			
			// Allocate data
			Pointer imagePointer = ((JCudaObject)image._gpuHandle).jcudaPointer; 
			Pointer doutPointer = ((JCudaObject)dout._gpuHandle).jcudaPointer; 
			Pointer dwPointer = ((JCudaObject)outputBlock._gpuHandle).jcudaPointer; 
			
			alpha = pointerTo(1.0); // TODO
			beta = pointerTo(0.0f);
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			workSpace = new Pointer();
			cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
					xTensorDesc, doutTensorDesc, convDesc, dwDesc, algo, sizeInBytesArray);
			
			int status = cudnnConvolutionBackwardFilter(cudnnHandle, alpha, xTensorDesc, imagePointer, 
					doutTensorDesc, doutPointer, convDesc, algo, workSpace, sizeInBytes, beta, dwDesc, dwPointer);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardFilter: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(xTensorDesc != null)
				cudnnDestroyTensorDescriptor(xTensorDesc);
			if(doutTensorDesc != null)
				cudnnDestroyTensorDescriptor(doutTensorDesc);
			if(dwDesc != null)
				cudnnDestroyFilterDescriptor(dwDesc);
			
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
		
	}

	public static void matmult(MatrixObject left1, MatrixObject right1, MatrixObject output, 
			boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {
		if(isInSparseFormat(left1) || isInSparseFormat(right1)) {
			throw new DMLRuntimeException("Sparse GPU matrix multiplication is not implemented");
		}
		
		// Since CuBLAS expects inputs in column-major format,
		// reverse the order of matrix-multiplication and take care of dimension mismatch.
		MatrixObject left = right1; 
		MatrixObject right = left1;
		boolean isLeftTransposed = isRightTransposed1; 
		boolean isRightTransposed = isLeftTransposed1; 
		
		char transa = isLeftTransposed ? 'T' : 'N';
		char transb = isRightTransposed ? 'T' : 'N';
		// Note: the dimensions are swapped
		int m = (int) (isLeftTransposed ? left.getNumRows() : left.getNumColumns()) ;
		int n = (int) (isRightTransposed ? right.getNumColumns() : right.getNumRows());
		int k = (int) (isLeftTransposed ?  left.getNumColumns() : left.getNumRows());
		int k1 = (int) (isRightTransposed ?  right.getNumRows() : right.getNumColumns());
		if(k != k1) 
			throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1);
		
		if(m == -1 || n == -1 || k == -1)
			throw new DMLRuntimeException("Incorrect dimensions");
		
		double alpha = 1;
		double beta = 0;
		
		int lda = isLeftTransposed ?  k : m;
		int ldb = isRightTransposed ? n : k;
		int ldc = m;
		
		if(!left.getGPUObject().isAllocated || !right.getGPUObject().isAllocated)
			throw new DMLRuntimeException("One of input is not allocated:" + left.getGPUObject().isAllocated + " " + right.getGPUObject().isAllocated);
		if(!output.getGPUObject().isAllocated)
			throw new DMLRuntimeException("Output is not allocated:" + output.getGPUObject().isAllocated);
		
		Pointer A = ((JCudaObject)left.getGPUObject()).jcudaPointer;
		Pointer B = ((JCudaObject)right.getGPUObject()).jcudaPointer;
		Pointer C = ((JCudaObject)output.getGPUObject()).jcudaPointer;
		
		long start = System.nanoTime();
		JCublas.cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		Statistics.cudaMultTime.addAndGet(System.nanoTime()-start);
	}
	
//	private void transpose(Pointer A, Pointer ret, int numRows, int numCols) {
//		Pointer alpha = null; 
//		Pointer beta = null;
//		try {
//			alpha = pointerTo(1.0);
//			beta = pointerTo(0.0);
//			JCublas2.cublasDgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, numCols, numRows, 
//					alpha, A, numRows, beta, A, numCols, ret, numCols);
//		}
//		finally {
//			if(alpha != null)
//				cudaFree(alpha);
//			if(beta != null)
//				cudaFree(beta);
//		}
//	}

	public static void conv2d_backward_data(MatrixObject filter, MatrixObject dout,
			MatrixObject output, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor dyDesc = null;
		cudnnTensorDescriptor dxDesc = null;
		cudnnFilterDescriptor wDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		
		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {
			// Allocate descriptors
			wDesc = allocateFilterDescriptor(K, C, R, S);
			dyDesc = allocateTensorDescriptor(N, K, P, Q);
			dxDesc = allocateTensorDescriptor(N, C, H, W);
			
			// Allocate data
			Pointer w = ((JCudaObject)filter._gpuHandle).jcudaPointer; 
			Pointer dy = ((JCudaObject)dout._gpuHandle).jcudaPointer; 
			Pointer dx = ((JCudaObject)output._gpuHandle).jcudaPointer; 
			
			alpha = pointerTo(1.0); // TODO
			beta = pointerTo(0.0f);
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			workSpace = new Pointer();
			cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
					wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytesArray);
			
			int status = cudnnConvolutionBackwardData(cudnnHandle, alpha, wDesc, w, 
					dyDesc, dy, convDesc, algo, workSpace, sizeInBytes, beta, dxDesc, dx);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardData: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(dyDesc != null)
				cudnnDestroyTensorDescriptor(dyDesc);
			if(dxDesc != null)
				cudnnDestroyTensorDescriptor(dxDesc);
			if(wDesc != null)
				cudnnDestroyFilterDescriptor(wDesc);
			
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
	}
	
	public static boolean isInSparseFormat(MatrixObject mo) {
		if(mo.getGPUObject() != null && mo.getGPUObject().isAllocated)
			return mo.getGPUObject().isInSparseFormat;
		return MatrixBlock.evalSparseFormatInMemory(mo.getNumRows(), mo.getNumColumns(), mo.getNnz());
	}
}
