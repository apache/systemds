package org.apache.sysml.api;

import static jcuda.jcusparse.JCusparse.cusparseDcsr2dense;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject.CSRPointer;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class TryCusparse {

	public static void main(String[] args){
		
		try {
			DMLScript.USE_ACCELERATOR = true;
			DMLScript.FORCE_ACCELERATOR = true;
			GPUContext.createGPUContext();
			JCudaObject.CSRPointer csr = JCudaObject.CSRPointer.allocateEmpty(7, 4);
			double[] vals = {1.0, 2.0, 3.0, 8.0, 9.0, 14.0, 18.0};
			int[] rowsPtr = {0, 3, 5, 6, 7};
			int[] colIndx = {0, 1, 2, 2, 3, 3, 2};
			
			JCuda.cudaMemcpy(csr.val, Pointer.to(vals), Sizeof.DOUBLE * vals.length, cudaMemcpyKind.cudaMemcpyHostToDevice);
			JCuda.cudaMemcpy(csr.rowPtr, Pointer.to(rowsPtr), Sizeof.INT * rowsPtr.length, cudaMemcpyKind.cudaMemcpyHostToDevice);
			JCuda.cudaMemcpy(csr.colInd, Pointer.to(colIndx), Sizeof.INT * colIndx.length, cudaMemcpyKind.cudaMemcpyHostToDevice);

			
			Pointer C = new Pointer();
			JCuda.cudaMalloc(C, Sizeof.DOUBLE * 4 * 5);
			
			cusparseDcsr2dense(LibMatrixCUDA.cusparseHandle, 4, 5, CSRPointer.getDefaultCuSparseMatrixDescriptor(), csr.val, csr.rowPtr, csr.colInd, C, 4);

			double[] O = new double[20];
			JCuda.cudaMemcpy(Pointer.to(O), C, 20 * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
			
			int k=0;
			for (int i=0; i<4; i++){
				for (int j=0; j<5; j++){
					System.out.print(O[k] + " ");
					k++;
				}
				System.out.println();
			}
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		
	}
	
}
