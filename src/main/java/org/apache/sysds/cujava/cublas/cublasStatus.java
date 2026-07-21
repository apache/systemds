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

package org.apache.sysds.cujava.cublas;

public class cublasStatus {

	public static final int CUBLAS_STATUS_SUCCESS = 0;

	public static final int CUBLAS_STATUS_NOT_INITIALIZED = 1;

	public static final int CUBLAS_STATUS_ALLOC_FAILED = 3;

	public static final int CUBLAS_STATUS_INVALID_VALUE = 7;

	public static final int CUBLAS_STATUS_ARCH_MISMATCH = 8;

	public static final int CUBLAS_STATUS_MAPPING_ERROR = 11;

	public static final int CUBLAS_STATUS_EXECUTION_FAILED = 13;

	public static final int CUBLAS_STATUS_INTERNAL_ERROR   = 14;

	public static final int CUBLAS_STATUS_NOT_SUPPORTED    = 15;

	private cublasStatus() {
	}

	public static String statusString(int err) {
		return switch(err) {
			case CUBLAS_STATUS_SUCCESS -> "CUBLAS_STATUS_SUCCESS";
			case CUBLAS_STATUS_NOT_INITIALIZED -> "CUBLAS_STATUS_NOT_INITIALIZED";
			case CUBLAS_STATUS_ALLOC_FAILED -> "CUBLAS_STATUS_ALLOC_FAILED";
			case CUBLAS_STATUS_INVALID_VALUE -> "CUBLAS_STATUS_INVALID_VALUE";
			case CUBLAS_STATUS_ARCH_MISMATCH -> "CUBLAS_STATUS_ARCH_MISMATCH";
			case CUBLAS_STATUS_MAPPING_ERROR -> "CUBLAS_STATUS_MAPPING_ERROR";
			case CUBLAS_STATUS_EXECUTION_FAILED -> "CUBLAS_STATUS_EXECUTION_FAILED";
			case CUBLAS_STATUS_INTERNAL_ERROR -> "CUBLAS_STATUS_INTERNAL_ERROR";
			case CUBLAS_STATUS_NOT_SUPPORTED -> "CUBLAS_STATUS_NOT_SUPPORTED";
			default -> "Invalid error";
		};
	}

}
