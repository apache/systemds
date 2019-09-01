/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.api.mlcontext;

import org.tugraz.sysds.runtime.meta.DataCharacteristics;

/**
 * Matrix metadata, such as the number of rows, the number of columns, the
 * number of non-zero values, the number of rows per block, and the number of
 * columns per block in the matrix.
 *
 */
public class MatrixMetadata extends Metadata {

	private MatrixFormat matrixFormat;

	public MatrixMetadata() {
	}

	/**
	 * Constructor to create a MatrixMetadata object based on matrix format.
	 *
	 * @param matrixFormat
	 *            The matrix format.
	 */
	public MatrixMetadata(MatrixFormat matrixFormat) {
		this.matrixFormat = matrixFormat;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on matrix format, the
	 * number of rows, and the number of columns in a matrix.
	 *
	 * @param matrixFormat
	 *            The matrix format.
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 */
	public MatrixMetadata(MatrixFormat matrixFormat, Long numRows, Long numColumns) {
		this.matrixFormat = matrixFormat;
		this.numRows = numRows;
		this.numColumns = numColumns;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on matrix format, the
	 * number of rows, and the number of columns in a matrix.
	 *
	 * @param matrixFormat
	 *            The matrix format.
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 */
	public MatrixMetadata(MatrixFormat matrixFormat, int numRows, int numColumns) {
		this.matrixFormat = matrixFormat;
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on matrix format, the
	 * number of rows, the number of columns, and the number of non-zero values
	 * in a matrix.
	 *
	 * @param matrixFormat
	 *            The matrix format.
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numNonZeros
	 *            The number of non-zero values in the matrix.
	 */
	public MatrixMetadata(MatrixFormat matrixFormat, Long numRows, Long numColumns, Long numNonZeros) {
		this.matrixFormat = matrixFormat;
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on matrix format, the
	 * number of rows, the number of columns, and the number of non-zero values
	 * in a matrix.
	 *
	 * @param matrixFormat
	 *            The matrix format.
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numNonZeros
	 *            The number of non-zero values in the matrix.
	 */
	public MatrixMetadata(MatrixFormat matrixFormat, int numRows, int numColumns, int numNonZeros) {
		this.matrixFormat = matrixFormat;
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.numNonZeros = (long) numNonZeros;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on matrix format, the
	 * number of rows, the number of columns, the number of non-zero values, the
	 * number of rows per block, and the number of columns per block in a
	 * matrix.
	 *
	 * @param matrixFormat
	 *            The matrix format.
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numNonZeros
	 *            The number of non-zero values in the matrix.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the matrix.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the matrix.
	 */
	public MatrixMetadata(MatrixFormat matrixFormat, Long numRows, Long numColumns, Long numNonZeros, Integer blen) {
		this.matrixFormat = matrixFormat;
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on matrix format, the
	 * number of rows, the number of columns, the number of non-zero values, the
	 * number of rows per block, and the number of columns per block in a
	 * matrix.
	 *
	 * @param matrixFormat
	 *            The matrix format.
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numNonZeros
	 *            The number of non-zero values in the matrix.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the matrix.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the matrix.
	 */
	public MatrixMetadata(MatrixFormat matrixFormat, int numRows, int numColumns, int numNonZeros, int blen) {
		this.matrixFormat = matrixFormat;
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.numNonZeros = (long) numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on the number of rows
	 * and the number of columns in a matrix.
	 *
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 */
	public MatrixMetadata(Long numRows, Long numColumns) {
		this.numRows = numRows;
		this.numColumns = numColumns;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on the number of rows
	 * and the number of columns in a matrix.
	 *
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 */
	public MatrixMetadata(int numRows, int numColumns) {
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on the number of
	 * rows, the number of columns, and the number of non-zero values in a
	 * matrix.
	 *
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numNonZeros
	 *            The number of non-zero values in the matrix.
	 */
	public MatrixMetadata(Long numRows, Long numColumns, Long numNonZeros) {
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on the number of
	 * rows, the number of columns, the number of rows per block, and the number
	 * of columns per block in a matrix.
	 *
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the matrix.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the matrix.
	 */
	public MatrixMetadata(Long numRows, Long numColumns, Integer blen) {
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on the number of
	 * rows, the number of columns, the number of rows per block, and the number
	 * of columns per block in a matrix.
	 *
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the matrix.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the matrix.
	 */
	public MatrixMetadata(int numRows, int numColumns, int blen) {
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on the number of
	 * rows, the number of columns, the number of non-zero values, the number of
	 * rows per block, and the number of columns per block in a matrix.
	 *
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numNonZeros
	 *            The number of non-zero values in the matrix.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the matrix.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the matrix.
	 */
	public MatrixMetadata(Long numRows, Long numColumns, Long numNonZeros, Integer blen) {
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on the number of
	 * rows, the number of columns, the number of non-zero values, the number of
	 * rows per block, and the number of columns per block in a matrix.
	 *
	 * @param numRows
	 *            The number of rows in the matrix.
	 * @param numColumns
	 *            The number of columns in the matrix.
	 * @param numNonZeros
	 *            The number of non-zero values in the matrix.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the matrix.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the matrix.
	 */
	public MatrixMetadata(int numRows, int numColumns, int numNonZeros, int blen) {
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.numNonZeros = (long) numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a MatrixMetadata object based on a
	 * DataCharacteristics object.
	 *
	 * @param dataCharacteristics
	 *            the matrix metadata as a DataCharacteristics object
	 */
	public MatrixMetadata(DataCharacteristics dataCharacteristics) {
		this.numRows = dataCharacteristics.getRows();
		this.numColumns = dataCharacteristics.getCols();
		this.numNonZeros = dataCharacteristics.getNonZeros();
		this.blockSize = dataCharacteristics.getBlocksize();
	}

	/**
	 * Obtain the matrix format
	 *
	 * @return the matrix format
	 */
	public MatrixFormat getMatrixFormat() {
		return matrixFormat;
	}

	/**
	 * Set the matrix format
	 *
	 * @param matrixFormat
	 *            the matrix format
	 */
	public void setMatrixFormat(MatrixFormat matrixFormat) {
		this.matrixFormat = matrixFormat;
	}

}
