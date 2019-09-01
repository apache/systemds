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
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;

/**
 * Frame metadata, such as the number of rows, the number of columns, the number
 * of non-zero values, the number of rows per block, and the number of columns
 * per block in the frame.
 *
 */
public class FrameMetadata extends Metadata {

	private FrameFormat frameFormat;
	private FrameSchema frameSchema;

	public FrameMetadata() {
	}

	/**
	 * Constructor to create a FrameMetadata object based on a string
	 * representation of a frame schema.
	 *
	 * @param schema
	 *            String representation of the frame schema.
	 */
	public FrameMetadata(String schema) {
		this.frameSchema = new FrameSchema(schema);
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format.
	 *
	 * @param frameFormat
	 *            The frame format.
	 */
	public FrameMetadata(FrameFormat frameFormat) {
		this.frameFormat = frameFormat;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame schema.
	 *
	 * @param frameSchema
	 *            The frame schema.
	 */
	public FrameMetadata(FrameSchema frameSchema) {
		this.frameSchema = frameSchema;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format and
	 * frame schema.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param frameSchema
	 *            The frame schema.
	 */
	public FrameMetadata(FrameFormat frameFormat, FrameSchema frameSchema) {
		this.frameFormat = frameFormat;
		this.frameSchema = frameSchema;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, frame
	 * schema, the number of rows, and the number of columns in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param frameSchema
	 *            The frame schema.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, FrameSchema frameSchema, Long numRows, Long numColumns) {
		this.frameFormat = frameFormat;
		this.frameSchema = frameSchema;
		this.numRows = numRows;
		this.numColumns = numColumns;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, frame
	 * schema, the number of rows, and the number of columns in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param frameSchema
	 *            The frame schema.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, FrameSchema frameSchema, int numRows, int numColumns) {
		this.frameFormat = frameFormat;
		this.frameSchema = frameSchema;
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, frame
	 * schema, the number of rows, the number of columns, the number of non-zero
	 * values, the number of rows per block, and the number of columns per block
	 * in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param frameSchema
	 *            The frame schema.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the frame.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, FrameSchema frameSchema, Long numRows, Long numColumns,
			Long numNonZeros, Integer blen) {
		this.frameFormat = frameFormat;
		this.frameSchema = frameSchema;
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, frame
	 * schema, the number of rows, the number of columns, the number of non-zero
	 * values, the number of rows per block, and the number of columns per block
	 * in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param frameSchema
	 *            The frame schema.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the frame.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, FrameSchema frameSchema, int numRows, int numColumns, int numNonZeros, int blen) {
		this.frameFormat = frameFormat;
		this.frameSchema = frameSchema;
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.numNonZeros = (long) numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, the
	 * number of rows, and the number of columns in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, Long numRows, Long numColumns) {
		this.frameFormat = frameFormat;
		this.numRows = numRows;
		this.numColumns = numColumns;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, the
	 * number of rows, and the number of columns in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, int numRows, int numColumns) {
		this.frameFormat = frameFormat;
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, the
	 * number of rows, the number of columns, and the number of non-zero values
	 * in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, Long numRows, Long numColumns, Long numNonZeros) {
		this.frameFormat = frameFormat;
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, the
	 * number of rows, the number of columns, and the number of non-zero values
	 * in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, int numRows, int numColumns, int numNonZeros) {
		this.frameFormat = frameFormat;
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.numNonZeros = (long) numNonZeros;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, the
	 * number of rows, the number of columns, the number of non-zero values, the
	 * number of rows per block, and the number of columns per block in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the frame.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, Long numRows, Long numColumns, Long numNonZeros, Integer blen) {
		this.frameFormat = frameFormat;
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a FrameMetadata object based on frame format, the
	 * number of rows, the number of columns, the number of non-zero values, the
	 * number of rows per block, and the number of columns per block in a frame.
	 *
	 * @param frameFormat
	 *            The frame format.
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the frame.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the frame.
	 */
	public FrameMetadata(FrameFormat frameFormat, int numRows, int numColumns, int numNonZeros, int blen) {
		this.frameFormat = frameFormat;
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.numNonZeros = (long) numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a FrameMetadata object based on the number of rows
	 * and the number of columns in a frame.
	 *
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 */
	public FrameMetadata(Long numRows, Long numColumns) {
		this.numRows = numRows;
		this.numColumns = numColumns;
	}

	/**
	 * Constructor to create a FrameMetadata object based on the number of rows
	 * and the number of columns in a frame.
	 *
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 */
	public FrameMetadata(int numRows, int numColumns) {
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
	}

	/**
	 * Constructor to create a FrameMetadata object based on the number of rows,
	 * the number of columns, and the number of non-zero values in a frame.
	 *
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 */
	public FrameMetadata(Long numRows, Long numColumns, Long numNonZeros) {
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
	}

	/**
	 * Constructor to create a FrameMetadata object based on the number of rows,
	 * the number of columns, and the number of non-zero values in a frame.
	 *
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 */
	public FrameMetadata(int numRows, int numColumns, int numNonZeros) {
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.numNonZeros = (long) numNonZeros;
	}

	/**
	 * Constructor to create a FrameMetadata object based on the number of rows,
	 * the number of columns, the number of rows per block, and the number of
	 * columns per block in a frame.
	 *
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the frame.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the frame.
	 */
	public FrameMetadata(Long numRows, Long numColumns, Integer blen) {
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a FrameMetadata object based on the number of rows,
	 * the number of columns, the number of non-zero values, the number of rows
	 * per block, and the number of columns per block in a frame.
	 *
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the frame.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the frame.
	 */
	public FrameMetadata(Long numRows, Long numColumns, Long numNonZeros, Integer blen) {
		this.numRows = numRows;
		this.numColumns = numColumns;
		this.numNonZeros = numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a FrameMetadata object based on the number of rows,
	 * the number of columns, the number of non-zero values, the number of rows
	 * per block, and the number of columns per block in a frame.
	 *
	 * @param numRows
	 *            The number of rows in the frame.
	 * @param numColumns
	 *            The number of columns in the frame.
	 * @param numNonZeros
	 *            The number of non-zero values in the frame.
	 * @param numRowsPerBlock
	 *            The number of rows per block in the frame.
	 * @param numColumnsPerBlock
	 *            The number of columns per block in the frame.
	 */
	public FrameMetadata(int numRows, int numColumns, int numNonZeros, int blen) {
		this.numRows = (long) numRows;
		this.numColumns = (long) numColumns;
		this.numNonZeros = (long) numNonZeros;
		this.blockSize = blen;
	}

	/**
	 * Constructor to create a FrameMetadata object based on a
	 * DataCharacteristics object.
	 *
	 * @param dataCharacteristics
	 *            the frame metadata as a DataCharacteristics object
	 */
	public FrameMetadata(DataCharacteristics dataCharacteristics) {
		this.numRows = dataCharacteristics.getRows();
		this.numColumns = dataCharacteristics.getCols();
		this.numNonZeros = dataCharacteristics.getNonZeros();
		this.blockSize = dataCharacteristics.getBlocksize();
	}

	/**
	 * Constructor to create a FrameMetadata object based on the frame schema
	 * and a DataCharacteristics object.
	 *
	 * @param frameSchema
	 *            The frame schema.
	 * @param matrixCharacteristics
	 *            the frame metadata as a DataCharacteristics object
	 */
	public FrameMetadata(FrameSchema frameSchema, MatrixCharacteristics matrixCharacteristics) {
		this.frameSchema = frameSchema;
		this.numRows = matrixCharacteristics.getRows();
		this.numColumns = matrixCharacteristics.getCols();
		this.numNonZeros = matrixCharacteristics.getNonZeros();
		this.blockSize = matrixCharacteristics.getBlocksize();
	}

	/**
	 * Set the FrameMetadata fields based on a DataCharacteristics object.
	 *
	 * @param matrixCharacteristics
	 *            the frame metadata as a DataCharacteristics object
	 */
	@Override
	public void setMatrixCharacteristics(MatrixCharacteristics matrixCharacteristics) {
		super.setMatrixCharacteristics(matrixCharacteristics);
	}

	/**
	 * Obtain the frame format
	 *
	 * @return the frame format
	 */
	public FrameFormat getFrameFormat() {
		return frameFormat;
	}

	/**
	 * Set the frame format
	 *
	 * @param frameFormat
	 *            the frame format
	 */
	public void setFrameFormat(FrameFormat frameFormat) {
		this.frameFormat = frameFormat;
	}

	/**
	 * Obtain the frame schema
	 *
	 * @return the frame schema
	 */
	public FrameSchema getFrameSchema() {
		return frameSchema;
	}

	/**
	 * Set the frame schema
	 *
	 * @param frameSchema
	 *            the frame schema
	 */
	public void setFrameSchema(FrameSchema frameSchema) {
		this.frameSchema = frameSchema;
	}

}
