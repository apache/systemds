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

package org.apache.sysds.runtime.util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import org.apache.log4j.Logger;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.BitSetArray;
import org.apache.sysds.runtime.frame.data.columns.BooleanArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Utils for converting python data to java.
 */
public class Py4jConverterUtils {
	private static final Logger LOG = Logger.getLogger(Py4jConverterUtils.class);
	public static MatrixBlock convertPy4JArrayToMB(byte[] data, int rlen, int clen) {
		return convertPy4JArrayToMB(data, rlen, clen, false, Types.ValueType.FP64);
	}

	public static MatrixBlock convertPy4JArrayToMB(byte[] data, int rlen, int clen, Types.ValueType valueType) {
		return convertPy4JArrayToMB(data, rlen, clen, false, valueType);
	}

	public static MatrixBlock convertSciPyCOOToMB(byte[] data, byte[] row, byte[] col, int rlen, int clen, int nnz) {
		MatrixBlock mb = new MatrixBlock(rlen, clen, true);
		mb.allocateSparseRowsBlock(false);
		ByteBuffer buf1 = ByteBuffer.wrap(data);
		buf1.order(ByteOrder.nativeOrder());
		ByteBuffer buf2 = ByteBuffer.wrap(row);
		buf2.order(ByteOrder.nativeOrder());
		ByteBuffer buf3 = ByteBuffer.wrap(col);
		buf3.order(ByteOrder.nativeOrder());
		for(int i = 0; i < nnz; i++) {
			double val = buf1.getDouble();
			int rowIndex = buf2.getInt();
			int colIndex = buf3.getInt();
			mb.set(rowIndex, colIndex, val);
		}
		mb.recomputeNonZeros();
		mb.examSparsity();
		return mb;
	}

	public static MatrixBlock convertSciPyCSRToMB(byte[] data, byte[] indices, byte[] indptr, int rlen, int clen, int nnz) {
		LOG.debug("Converting compressed sparse row matrix to MatrixBlock");
		MatrixBlock mb = new MatrixBlock(rlen, clen, true);
		mb.allocateSparseRowsBlock(false);
		ByteBuffer dataBuf = ByteBuffer.wrap(data);
		dataBuf.order(ByteOrder.nativeOrder());
		ByteBuffer indicesBuf = ByteBuffer.wrap(indices);
		indicesBuf.order(ByteOrder.nativeOrder());
		ByteBuffer indptrBuf = ByteBuffer.wrap(indptr);
		indptrBuf.order(ByteOrder.nativeOrder());
		
		// Read indptr array to get row boundaries
		int[] rowPtrs = new int[rlen + 1];
		for(int i = 0; i <= rlen; i++) {
			rowPtrs[i] = indptrBuf.getInt();
		}
		
		// Iterate through each row
		for(int row = 0; row < rlen; row++) {
			int startIdx = rowPtrs[row];
			int endIdx = rowPtrs[row + 1];
			
			// Set buffer positions to the start of this row
			dataBuf.position(startIdx * Double.BYTES);
			indicesBuf.position(startIdx * Integer.BYTES);
			
			// Process all non-zeros in this row sequentially
			for(int idx = startIdx; idx < endIdx; idx++) {
				double val = dataBuf.getDouble();
				int colIndex = indicesBuf.getInt();
				mb.set(row, colIndex, val);
			}
		}
		
		mb.recomputeNonZeros();
		mb.examSparsity();
		return mb;
	}

	public static MatrixBlock allocateDenseOrSparse(int rlen, int clen, boolean isSparse) {
		MatrixBlock ret = new MatrixBlock(rlen, clen, isSparse);
		ret.allocateBlock();
		return ret;
	}

	public static MatrixBlock allocateDenseOrSparse(long rlen, long clen, boolean isSparse) {
		if(rlen > Integer.MAX_VALUE || clen > Integer.MAX_VALUE) {
			throw new DMLRuntimeException(
				"Dimensions of matrix are too large to be passed via NumPy/SciPy:" + rlen + " X " + clen);
		}
		return allocateDenseOrSparse((int) rlen, (int) clen, isSparse);
	}

	public static MatrixBlock convertPy4JArrayToMB(byte[] data, int rlen, int clen, boolean isSparse,
		Types.ValueType valueType) {
		MatrixBlock mb = new MatrixBlock(rlen, clen, isSparse, -1);
		if(isSparse) {
			throw new DMLRuntimeException("Convertion to sparse format not supported");
		}
		else {
			long limit = (long) rlen * clen;
			if(limit > Integer.MAX_VALUE)
				throw new DMLRuntimeException(
					"Dense NumPy array of size " + limit + " cannot be converted to MatrixBlock");
			double[] denseBlock = new double[(int) limit];
			ByteBuffer buf = ByteBuffer.wrap(data);
			buf.order(ByteOrder.nativeOrder());

			switch(valueType) {
				case UINT8:
					for(int i = 0; i< limit; i++)
						denseBlock[i] = buf.get() & 0xFF;
					break;
				case INT32:
					for(int i = 0; i < limit; i++)
						denseBlock[i] = buf.getInt();
					break;
				case FP32:
					for(int i = 0; i < limit; i++)
						denseBlock[i] = buf.getFloat();
					break;
				case FP64:
					for(int i = 0; i < limit; i++)
						denseBlock[i] = buf.getDouble();
					break;
				default:
					throw new DMLRuntimeException("Unsupported value type: " + valueType.name());
			}
			mb.init(denseBlock, rlen, clen);
		}
		mb.recomputeNonZeros();
		mb.examSparsity();
		return mb;
	}

	public static Array<?> convert(byte[] data, int numElements, Types.ValueType valueType) {
		if(data == null || valueType == null) {
			throw new DMLRuntimeException("Invalid input data or value type.");
		}

		ByteBuffer buffer = ByteBuffer.wrap(data);
		buffer.order(ByteOrder.LITTLE_ENDIAN);

		Array<?> array = ArrayFactory.allocate(valueType, numElements);
		readBufferIntoArray(buffer, array, valueType, numElements);

		return array;
	}

	// Right now row conversion is only supported for if all columns have the same datatype, so this is a placeholder for now that essentially just casts to Object[]
	public static Object[] convertRow(byte[] data, int numElements, Types.ValueType valueType) {
		Array<?> converted = convert(data, numElements, valueType);

		Object[] row = new Object[numElements];
		for(int i = 0; i < numElements; i++) {
			row[i] = converted.get(i);
		}

		return row;
	}

	public static Array<?>[] convertFused(byte[] data, int numElements, Types.ValueType[] valueTypes) {
		int numOperations = valueTypes.length;

		ByteBuffer buffer = ByteBuffer.wrap(data);
		buffer.order(ByteOrder.LITTLE_ENDIAN);

		Array<?>[] arrays = new Array<?>[numOperations];

		for (int i = 0; i < numOperations; i++) {
			arrays[i] = ArrayFactory.allocate(valueTypes[i], numElements);
			readBufferIntoArray(buffer, arrays[i], valueTypes[i], numElements);
		}

        return arrays;
    }

	private static void readBufferIntoArray(ByteBuffer buffer, Array<?> array, Types.ValueType valueType, int numElements) {
		for (int i = 0; i < numElements; i++) {
			switch (valueType) {
				case UINT8:
					array.set(i, (int) (buffer.get() & 0xFF));
					break;
				case INT32:
                case HASH32:
                    array.set(i, buffer.getInt());
					break;
				case INT64:
                case HASH64:
                    array.set(i, buffer.getLong());
					break;
				case FP32:
					array.set(i, buffer.getFloat());
					break;
				case FP64:
					array.set(i, buffer.getDouble());
					break;
				case BOOLEAN:
					if (array instanceof BooleanArray) {
						((BooleanArray) array).set(i, buffer.get() != 0);
					} else if (array instanceof BitSetArray) {
						((BitSetArray) array).set(i, buffer.get() != 0);
					} else {
						throw new DMLRuntimeException("Array factory returned invalid array type for boolean values.");
					}
					break;
				case STRING:
					int strLength = buffer.getInt();
					byte[] strBytes = new byte[strLength];
					buffer.get(strBytes);
					array.set(i, new String(strBytes, StandardCharsets.UTF_8));
					break;
				case CHARACTER:
					array.set(i, buffer.getChar());
					break;
                default:
					throw new DMLRuntimeException("Unsupported value type: " + valueType);
			}
		}
	}

	public static byte[] convertMBtoPy4JDenseArr(MatrixBlock mb) {
		byte[] ret = null;
		if(mb.isInSparseFormat()) {
			LOG.debug("Converting sparse matrix to dense");
			mb.sparseToDense();
		}

		long limit = mb.getNumRows() * mb.getNumColumns();
		int times = Double.SIZE / Byte.SIZE;
		if(limit > Integer.MAX_VALUE / times)
			throw new DMLRuntimeException("MatrixBlock of size " + limit + " cannot be converted to dense numpy array");
		ret = new byte[(int) (limit * times)];

		double[] denseBlock = mb.getDenseBlockValues();
		if(mb.isEmptyBlock()) {
			for(int i = 0; i < limit; i++) {
				ByteBuffer.wrap(ret, i * times, times).order(ByteOrder.nativeOrder()).putDouble(0);
			}
		}
		else if(denseBlock == null) {
			throw new DMLRuntimeException("Error while dealing with empty blocks.");
		}
		else {
			for(int i = 0; i < denseBlock.length; i++) {
				ByteBuffer.wrap(ret, i * times, times).order(ByteOrder.nativeOrder()).putDouble(denseBlock[i]);
			}
		}

		return ret;
	}
}
