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

package org.apache.sysml.api.mlcontext;

/**
 * MatrixFormat represents the different matrix formats supported by the
 * MLContext API.
 *
 */
public enum MatrixFormat {
	/**
	 * Comma-separated value format (dense).
	 */
	CSV,

	/**
	 * (I J V) format (sparse). I and J represent matrix coordinates and V
	 * represents the value. The I J and V values are space-separated.
	 */
	IJV,

	/**
	 * DataFrame of doubles with a row index.
	 */
	DF_DOUBLES_WITH_INDEX,

	/**
	 * DataFrame of doubles with no row index.
	 */
	DF_DOUBLES,

	/**
	 * Vector DataFrame with a row index.
	 */
	DF_VECTOR_WITH_INDEX,

	/**
	 * Vector DataFrame with no row index.
	 */
	DF_VECTOR;

	/**
	 * Is the matrix format vector-based?
	 * 
	 * @return {@code true} if matrix is a vector-based DataFrame, {@code false}
	 *         otherwise.
	 */
	public boolean isVectorBased() {
		if (this == DF_VECTOR_WITH_INDEX) {
			return true;
		} else if (this == DF_VECTOR) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * Does the DataFrame have a row index?
	 * 
	 * @return {@code true} if the DataFrame has a row index, {@code false}
	 *         otherwise.
	 */
	public boolean hasIDColumn() {
		if (this == DF_DOUBLES_WITH_INDEX) {
			return true;
		} else if (this == DF_VECTOR_WITH_INDEX) {
			return true;
		} else {
			return false;
		}
	}

}
