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
		DENSE, SPARSE, ULTRA_SPARSE, EMPTY, FULL
	}

	public enum ValueType {
		RAND, // UC
		CONST, // RLE
		RAND_ROUND, // Values rounded to nearest whole numbers.
		OLE_COMPRESSIBLE, // Ideal inputs for OLE Compression.
		RLE_COMPRESSIBLE, // Ideal inputs for RLE Compression.
		ONE_HOT_ENCODED,
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
	}

	public enum ValueRange {
		SMALL, LARGE, BYTE, BOOLEAN, NEGATIVE, POSITIVE
	}

	public enum OverLapping {
		COL, MATRIX, NONE, MATRIX_PLUS, MATRIX_MULT_NEGATIVE, SQUASH, PLUS, APPEND_EMPTY, APPEND_CONST, PLUS_LARGE;

		public static boolean effectOnOutput(OverLapping opcode) {
			switch(opcode) {
				case MATRIX_MULT_NEGATIVE:
				case MATRIX:
				case SQUASH:
				case COL:
				case MATRIX_PLUS:
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
				return 0.01;
			case EMPTY:
				return 0.0;
			case FULL:
				return 1.0;
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
			default:
				throw new RuntimeException("Invalid range value enum type");
		}
	}

	public static int getNumberOfRows(MatrixTypology matrixTypology) {
		switch(matrixTypology) {
			case SMALL:
				return 4;
			case LARGE:
				return 200;
			case FEW_ROW:
				return 1283;
			case FEW_COL:
				return 500;
			case SINGLE_ROW:
				return 1;
			case SINGLE_COL:
				return 100;
			case L_ROWS:
				return 5000;
			case XL_ROWS:
				return 66000;
			case SINGLE_COL_L:
				return 64000 * 2;
			default:
				throw new RuntimeException("Invalid matrix enum type");
		}
	}

	public static int getNumberOfColumns(MatrixTypology matrixTypology) {
		switch(matrixTypology) {
			case SMALL:
				return 20;
			case LARGE:
				return 8;
			case FEW_ROW:
				return 13;
			case FEW_COL:
				return 3;
			case SINGLE_ROW:
				return 321;
			case SINGLE_COL:
				return 1;
			case L_ROWS:
				return 5;
			case XL_ROWS:
				return 10;
			case SINGLE_COL_L:
				return 1;
			default:
				throw new RuntimeException("Invalid matrix enum type");
		}
	}
}
