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

package org.apache.sysds.runtime.util;

import java.util.Arrays;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;


public class DnnUtils {
	
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
		if(H <= 0 || R <= 0 || heightPadding < 0 || verticalStride < 0) {
			throw new RuntimeException("Incorrect parameters: height=" + H + " filter_height=" + R + " stride=" + verticalStride + " pad=" + heightPadding);
		}
		long padded_image_height = H + 2 * heightPadding;
		long ret = (padded_image_height - R) / verticalStride + 1;
		if(ret <= 0 || ret > Integer.MAX_VALUE) {
			// Check for valid output activation height
			if(padded_image_height < R)
				throw new RuntimeException("Incorrect parameters: padded image height:" + padded_image_height + " cannot be less than filter_height:" + R);
			else
				throw new RuntimeException("Incorrect parameters: height=" + H + " filter_height=" + R + " stride=" + verticalStride + " pad=" + heightPadding + " as P=" + ret);
		}
		return ret;
	}
	public static long getQ(long W, long S, long horizontalStride, long widthPadding) {
		if(W <= 0 || S <= 0 || widthPadding < 0 || horizontalStride < 0) {
			throw new RuntimeException("Incorrect parameters: width=" + W + " filter_width=" + S + " stride=" + horizontalStride + " pad=" + widthPadding);
		}
		long padded_image_width = W + 2 * widthPadding;
		long ret = (padded_image_width - S) / horizontalStride + 1;
		if(ret <= 0 || ret > Integer.MAX_VALUE) {
			// Check for valid output activation width
			if(padded_image_width < S)
				throw new RuntimeException("Incorrect parameters: padded image width:" + padded_image_width + " cannot be less than filter width:" + S);
			else
				throw new RuntimeException("Incorrect parameters: width=" + W + " filter_width=" + S + " stride=" + horizontalStride + " pad=" + widthPadding + " as Q=" + ret);
		}
		return ret;
	}
	
	public static void fillBias(MatrixBlock bias, double [] outputArray, int src_rl, int src_ru, int N, int K, int PQ) {
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
			double [] biasArr = bias.getDenseBlockValues();
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
