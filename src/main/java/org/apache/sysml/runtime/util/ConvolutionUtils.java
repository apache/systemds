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


public class ConvolutionUtils {
	
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
	
}
