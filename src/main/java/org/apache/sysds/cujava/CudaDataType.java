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

package org.apache.sysds.cujava;

/**
 * CUDA data-type constants (mirror of cudaDataType_t). Grouped: all real (R) types first, then complex (C) companions.
 */

public class CudaDataType {

	/* ─────── Real scalars ─────── */

	/** 16-bit IEEE half-precision float (fp16) */
	public static final int CUDA_R_16F = 2;
	/** 16-bit bfloat16 */
	public static final int CUDA_R_16BF = 14;
	/** 32-bit IEEE single-precision float */
	public static final int CUDA_R_32F = 0;
	/** 64-bit IEEE double-precision float */
	public static final int CUDA_R_64F = 1;

	/** 4-bit  signed integer */
	public static final int CUDA_R_4I = 16;
	/** 4-bit  unsigned integer */
	public static final int CUDA_R_4U = 18;
	/** 8-bit  signed integer */
	public static final int CUDA_R_8I = 3;
	/** 8-bit  unsigned integer */
	public static final int CUDA_R_8U = 8;
	/** 16-bit signed integer */
	public static final int CUDA_R_16I = 20;
	/** 16-bit unsigned integer */
	public static final int CUDA_R_16U = 22;
	/** 32-bit signed integer */
	public static final int CUDA_R_32I = 10;
	/** 32-bit unsigned integer */
	public static final int CUDA_R_32U = 12;
	/** 64-bit signed integer */
	public static final int CUDA_R_64I = 24;
	/** 64-bit unsigned integer */
	public static final int CUDA_R_64U = 26;

	/** 8-bit float, FP-8 format E4M3 */
	public static final int CUDA_R_8F_E4M3 = 28;
	/** 8-bit float, FP-8 format E5M2 */
	public static final int CUDA_R_8F_E5M2 = 29;


	/* ─────── Complex pairs (real + imaginary) ─────── */

	/** two fp16 numbers: (real, imag) */
	public static final int CUDA_C_16F = 6;
	/** two bfloat16 numbers */
	public static final int CUDA_C_16BF = 15;
	/** two 32-bit floats */
	public static final int CUDA_C_32F = 4;
	/** two 64-bit doubles */
	public static final int CUDA_C_64F = 5;

	/** two 4-bit  signed integers */
	public static final int CUDA_C_4I = 17;
	/** two 4-bit  unsigned integers */
	public static final int CUDA_C_4U = 19;
	/** two 8-bit  signed integers */
	public static final int CUDA_C_8I = 7;
	/** two 8-bit  unsigned integers */
	public static final int CUDA_C_8U = 9;
	/** two 16-bit signed integers */
	public static final int CUDA_C_16I = 21;
	/** two 16-bit unsigned integers */
	public static final int CUDA_C_16U = 23;
	/** two 32-bit signed integers */
	public static final int CUDA_C_32I = 11;
	/** two 32-bit unsigned integers */
	public static final int CUDA_C_32U = 13;
	/** two 64-bit signed integers */
	public static final int CUDA_C_64I = 25;
	/** two 64-bit unsigned integers */
	public static final int CUDA_C_64U = 27;

	private CudaDataType() { /* utility class – no instantiation */ }

}
