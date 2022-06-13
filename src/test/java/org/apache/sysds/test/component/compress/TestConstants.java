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

package org.apache.sysds.test.component.compress;

/**
 * Class Containing Testing Constants, for easy enumeration of typical Parameters classes
 */
public class TestConstants {

	public enum SparsityType {
		DENSE, SPARSE, ULTRA_SPARSE, EMPTY, FULL,THIRTY
	}

	public enum ValueType {
		RAND, // UC
		CONST, // RLE
		RAND_ROUND, // Values rounded to nearest whole numbers.
		OLE_COMPRESSIBLE, // Ideal inputs for OLE Compression.
		RLE_COMPRESSIBLE, // Ideal inputs for RLE Compression.
		ONE_HOT, UNBALANCED_SPARSE, // An input where some columns are super dense and some very sparse
	}

	public enum MatrixTypology {
		SMALL, // A Small Matrix
		LARGE, // A "Large" Matrix
		FEW_ROW, // A Matrix with a large number of rows and less columns
		FEW_COL, // A Matrix with a large number of columns
		SINGLE_ROW, // Single Row with some columns
		SINGLE_COL, // Single Column with some rows
		L_ROWS, // Many Rows
		XL_ROWS, // A LOT of rows.
		SINGLE_COL_L, // Single Column large.
		COL_16,
	}

	public enum ValueRange {
		SMALL, LARGE, BYTE, BOOLEAN, NEGATIVE, POSITIVE, CONST
	}

	public enum OverLapping {
		COL, MATRIX, NONE, MATRIX_PLUS, MATRIX_MULT_NEGATIVE, SQUASH, PLUS, APPEND_EMPTY, APPEND_CONST, PLUS_LARGE,
		C_BIND_SELF, PLUS_ROW_VECTOR;

		public static boolean effectOnOutput(OverLapping opcode) {
			switch(opcode) {
				case MATRIX_MULT_NEGATIVE:
				case MATRIX:
				case SQUASH:
				case COL:
				case MATRIX_PLUS:
				case PLUS_LARGE:
					return true;
				default:
					return false;
			}
		}
	}

	public static double getSparsityValue(SparsityType sparsityType) {
		switch(sparsityType) {
			case DENSE:
				return 0.8;
			case SPARSE:
				return 0.1;
			case ULTRA_SPARSE:
				return 0.0001;
			case EMPTY:
				return 0.0;
			case FULL:
				return 1.0;
			case THIRTY:
				return 0.3;
			default:
				throw new RuntimeException("Invalid Sparsity type");
		}
	}

	public static int getMinRangeValue(ValueRange valueRange) {
		switch(valueRange) {
			case SMALL:
				return -1;
			case LARGE:
				return -127 * 2;
			case BYTE:
				return -127;
			case BOOLEAN:
				return 0;
			case NEGATIVE:
				return -132;
			case CONST:
				return 14;
			default:
				throw new RuntimeException("Invalid range value enum type");
		}
	}

	public static int getMaxRangeValue(ValueRange valueRange) {
		switch(valueRange) {
			case SMALL:
				return 5;
			case LARGE:
				return -127;
			case BYTE:
				return 127;
			case BOOLEAN:
				return 1;
			case NEGATIVE:
				return -23;
			case CONST:
				return 14;
			default:
				throw new RuntimeException("Invalid range value enum type");
		}
	}

	public static int getNumberOfRows(MatrixTypology matrixTypology) {
		switch(matrixTypology) {
			case SMALL:
				return 500;
			case LARGE:
				return 1000;
			case SINGLE_COL:
				return 5000;
			case XL_ROWS:
				return 3000;
			case COL_16:
				return 3000;
			default:
				throw new RuntimeException("Invalid matrix enum type");
		}
	}

	public static int getNumberOfColumns(MatrixTypology matrixTypology) {
		switch(matrixTypology) {
			case SMALL:
				return 3;
			case LARGE:
				return 10;
			case SINGLE_COL:
				return 1;
			case COL_16:
				return 16;
			case XL_ROWS:
				return 100;
			default:
				throw new RuntimeException("Invalid matrix enum type");
		}
	}
}
