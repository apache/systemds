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

package org.apache.sysml.runtime.util;

import java.util.Arrays;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;


public class ConvolutionUtils {
	
	public static String getConv2dOutputMap(String H, String R, String verticalStride, String heightPadding) {
		long padX2 = -1;
		try {
			padX2 = Long.parseLong(heightPadding)*2;
			return "" + getP(Long.parseLong(H), Long.parseLong(R), Long.parseLong(verticalStride), Long.parseLong(heightPadding));
		} catch(Exception e) {
			  if(padX2 == -1) 			return "((" + H + " + 2*" + heightPadding + " - " + R + ") / " + verticalStride + "+ 1)";
			  else if(padX2 == 0) 	return "((" + H + " - " + R + ") / " + verticalStride + "+ 1)";
			  else 									return "((" + H + " + " + padX2 + " - " + R + ") / " + verticalStride + "+ 1)";
		}
	}
	
	public static long getP(long H, long R, long verticalStride, long heightPadding) {
		long ret = (H + 2 * heightPadding - R) / verticalStride + 1;
		if(ret <= 0) {
			throw new RuntimeException("Incorrect output patch size: "
					+ "(image_height + 2 * pad_h - filter_height) / verticalStride + 1) needs to be positive, but is " + ret
					+ " (" + H + " + 2 * " + heightPadding + " - " + R + ") / " + verticalStride + " + 1))");
		}
		return ret;
	}
	public static long getQ(long W, long S, long horizontalStride, long widthPadding) {
		long ret = (W + 2 * widthPadding - S) / horizontalStride + 1;
		if(ret <= 0) {
			throw new RuntimeException("Incorrect output patch size: (image_width + 2 * pad_w - filter_width) / horizontalStride + 1) needs to be positive, but is " + ret
					+ " (" + W + " + 2 * " + widthPadding + " - " + S + ") / " + horizontalStride + " + 1))");
		}
		return ret;
	}

	
	// Performs dest[destPos...] op= thatValue[src_rl:src_ru,]
	public static void binaryOperationInPlace(MatrixBlock src, double [] dest, 
			int destPos, int destNumCols, int src_rl, int src_ru, BinaryOperator op) throws DMLRuntimeException {
		if(src.isInSparseFormat()) {
			if(src.isEmptyBlock() && op.fn == Plus.getPlusFnObject()) {
				// Do nothing: Inplace addition by zero
			}
			else if(src.isEmptyBlock() && op.fn == Multiply.getMultiplyFnObject()) {
				// Inplace multiplication by zero
				Arrays.fill(dest, destPos, destPos + (src_ru-src_rl)*destNumCols, 0);
			}
			else if(op.fn == Plus.getPlusFnObject()) {
				for(int i = src_rl, cix = destPos; i < src_ru; i++, cix += destNumCols) {
					if( !src.getSparseBlock().isEmpty(i) ) {
						int apos = src.getSparseBlock().pos(i);
						int alen = src.getSparseBlock().size(i);
						int[] aix = src.getSparseBlock().indexes(i);
						double[] avals = src.getSparseBlock().values(i);
						for(int j = apos; j < apos+alen; j++) {
							dest[ cix+aix[j] ] += avals[j];
						}
					}
				}
			}
			else if(op.fn == Multiply.getMultiplyFnObject()) {
				// Unsafe operation
				for(int i = src_rl, cix = destPos; i < src_ru; i++, cix += destNumCols) {
					if( !src.getSparseBlock().isEmpty(i) ) {
						int apos = src.getSparseBlock().pos(i);
						int alen = src.getSparseBlock().size(i);
						int[] aix = src.getSparseBlock().indexes(i);
						double[] avals = src.getSparseBlock().values(i);
						int prevDestIndex = 0;
						for(int j = apos; j < apos+alen; j++) {
							// Multiplication by zero. Assumption: aix is sorted.
							Arrays.fill(dest, cix+prevDestIndex, aix[j], 0);
							prevDestIndex = aix[j]+1;
							dest[ cix+aix[j] ] *= avals[j];
						}
						Arrays.fill(dest, cix+prevDestIndex, cix+destNumCols, 0);
					}
					else {
						Arrays.fill(dest, cix, cix + destNumCols, 0);
					}
				}
			}
			else {
				// As operation could be safe or unsafe. This will be caught at development time.
				throw new DMLRuntimeException("Unimplemented sparse operation");
			}
		}
		else {
			double [] inputArr = src.getDenseBlock();
			if(op.fn == Plus.getPlusFnObject()) {
				for(int i = destPos; i < src_ru*destNumCols; i++) {
					dest[i] += inputArr[i];
				}
			}
			else if(op.fn == Multiply.getMultiplyFnObject()) {
				for(int i = destPos; i < src_ru*destNumCols; i++) {
					dest[i] *= inputArr[i];
				}
			}
			else {
				for(int i = destPos; i < src_ru*destNumCols; i++) {
					dest[i] = op.fn.execute(dest[i], inputArr[i]);
				}
			}
		}
	}
	
	// Performs dest[destPos...] = src[src_rl:src_ru,] op scalar
	public static void scalarOperations(MatrixBlock src, double [] dest, 
			int destPos, int destNumCols, int src_rl, int src_ru, ScalarOperator scalarOp) throws DMLRuntimeException {
		if(src.isInSparseFormat()) {
			for(int i = src_rl, cix = destPos; i < src_ru; i++, cix += destNumCols) {
				if( !src.getSparseBlock().isEmpty(i) ) {
					int apos = src.getSparseBlock().pos(i);
					int alen = src.getSparseBlock().size(i);
					int[] aix = src.getSparseBlock().indexes(i);
					double[] avals = src.getSparseBlock().values(i);
					for(int j = apos; j < apos+alen; j++) {
						dest[ cix+aix[j] ] = scalarOp.executeScalar(avals[j]);
					}
				}
			}
		}
		else {
			double [] inputArr = src.getDenseBlock();
			for(int i = destPos; i < src_ru*destNumCols; i++) {
				dest[i] = scalarOp.executeScalar(inputArr[i]);
			}
		}
	}
	
	public static void fillBias(MatrixBlock bias, double [] outputArray, int src_rl, int src_ru, int N, int K, int PQ) throws DMLRuntimeException {
		// bias.getNumColumns() == 1 checked outside
		if(bias.isInSparseFormat()) {
			for(int k = 0; k < K; k++) {
				if( !bias.getSparseBlock().isEmpty(k) ) {
					int apos = bias.getSparseBlock().pos(k);
					double[] avals = bias.getSparseBlock().values(k);
					double val = avals[apos];
					for(int n = src_rl; n < src_ru; n++) {
						int fromIndex = n*K*PQ + k*PQ;
						Arrays.fill(outputArray, fromIndex, fromIndex + PQ, val);
					}
				}
			}
		}
		else {
			double [] biasArr = bias.getDenseBlock();
			for(int n = src_rl; n < src_ru; n++) {
				for(int k = 0; k < K; k++) {
					int fromIndex = n*K*PQ + k*PQ;
					double val = biasArr[k];
					Arrays.fill(outputArray, fromIndex, fromIndex + PQ, val);
				}
			}
		}
	}
	
}
