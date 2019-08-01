/*
 * Copyright 2018 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.common;

import org.tugraz.sysds.runtime.DMLRuntimeException;

public class Types
{
	/**
	 * Execution mode for entire script. 
	 */
	public enum ExecMode { 
		SINGLE_NODE, // execute all matrix operations in CP
		HYBRID,      // execute matrix operations in CP or MR
		SPARK        // execute matrix operations in Spark
	}
	
	/**
	 * Execution type of individual operations.
	 */
	public enum ExecType { CP, CP_FILE, SPARK, GPU, INVALID }
	
	/**
	 * Data types (tensor, matrix, scalar, frame, object, unknown).
	 */
	public enum DataType {
		TENSOR, MATRIX, SCALAR, FRAME, LIST, UNKNOWN;
		
		public boolean isMatrix() {
			return this == MATRIX;
		}
		public boolean isTensor() {
			return this == TENSOR;
		}
		public boolean isFrame() {
			return this == FRAME;
		}
		public boolean isScalar() {
			return this == SCALAR;
		}
		public boolean isList() {
			return this == LIST;
		}
		public boolean isUnknown() {
			return this == UNKNOWN;
		}
	}

	/**
	 * Value types (int, double, string, boolean, unknown).
	 */
	public enum ValueType {
		FP32, FP64, INT32, INT64, BOOLEAN, STRING, UNKNOWN;
		public boolean isNumeric() {
			return this == INT32 || this == INT64 || this == FP32 || this == FP64;
		}
		public boolean isUnknown() {
			return this == UNKNOWN;
		}
		public boolean isPseudoNumeric() {
			return isNumeric() || this == BOOLEAN;
		}
		public String toExternalString() {
			switch(this) {
				case FP32:
				case FP64:    return "DOUBLE";
				case INT32:
				case INT64:   return "INT";
				case BOOLEAN: return "BOOLEAN";
				default:      return toString();
			}
		}
		public static ValueType fromExternalString(String value) {
			String lvalue = (value != null) ? value.toUpperCase() : null;
			switch(lvalue) {
				case "DOUBLE":   return FP64;
				case "INT":      return INT64;
				case "BOOLEAN" : return BOOLEAN;
				default:
					throw new DMLRuntimeException("Unknown value type: "+value);
			}
		}
	}
	
	/**
	 * Serialization block types (empty, dense, sparse, ultra-sparse)
	 */
	public enum BlockType{
		EMPTY_BLOCK,
		ULTRA_SPARSE_BLOCK,
		SPARSE_BLOCK,
		DENSE_BLOCK,
	}
	
	/**
	 * Type of builtin or user-defined function with regard to its
	 * number of return variables.
	 */
	public enum ReturnType {
		NO_RETURN,
		SINGLE_RETURN,
		MULTI_RETURN
	}
}
