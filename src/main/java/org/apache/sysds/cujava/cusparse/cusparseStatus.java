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

/**
 * Source for the numerical value:
 * https://gitlab.com/nvidia/headers/cuda-individual/cusparse/-/blob/d4fd9303b8a5a770d11c2c60211e3f9e76410e51/cusparse.h
 */

package org.apache.sysds.cujava.cusparse;

public class cusparseStatus {

	public static final int CUSPARSE_STATUS_SUCCESS = 0;

	public static final int CUSPARSE_STATUS_NOT_INITIALIZED = 1;

	public static final int CUSPARSE_STATUS_ALLOC_FAILED = 2;

	public static final int CUSPARSE_STATUS_INVALID_VALUE = 3;

	public static final int CUSPARSE_STATUS_ARCH_MISMATCH = 4;

	public static final int CUSPARSE_STATUS_MAPPING_ERROR = 5;

	public static final int CUSPARSE_STATUS_EXECUTION_FAILED = 6;

	public static final int CUSPARSE_STATUS_INTERNAL_ERROR = 7;

	public static final int CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8;

	public static final int CUSPARSE_STATUS_ZERO_PIVOT = 9;

	public static final int CUSPARSE_STATUS_NOT_SUPPORTED = 10;

	public static final int CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11;

	public static String statusString(int err) {
		return switch(err) {
			case CUSPARSE_STATUS_SUCCESS -> "CUSPARSE_STATUS_SUCCESS";
			case CUSPARSE_STATUS_NOT_INITIALIZED -> "CUSPARSE_STATUS_NOT_INITIALIZED";
			case CUSPARSE_STATUS_ALLOC_FAILED -> "CUSPARSE_STATUS_ALLOC_FAILED";
			case CUSPARSE_STATUS_INVALID_VALUE -> "CUSPARSE_STATUS_INVALID_VALUE";
			case CUSPARSE_STATUS_ARCH_MISMATCH -> "CUSPARSE_STATUS_ARCH_MISMATCH";
			case CUSPARSE_STATUS_MAPPING_ERROR -> "CUSPARSE_STATUS_MAPPING_ERROR";
			case CUSPARSE_STATUS_EXECUTION_FAILED -> "CUSPARSE_STATUS_EXECUTION_FAILED";
			case CUSPARSE_STATUS_INTERNAL_ERROR -> "CUSPARSE_STATUS_INTERNAL_ERROR";
			case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED -> "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
			case CUSPARSE_STATUS_ZERO_PIVOT -> "CUSPARSE_STATUS_ZERO_PIVOT";
			case CUSPARSE_STATUS_NOT_SUPPORTED -> "CUSPARSE_STATUS_NOT_SUPPORTED";
			case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES -> "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
			default -> "Invalid error";
		};
	}

}
