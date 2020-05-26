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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Utils for converting python data to java.
 */
public class Py4jConverterUtils {
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
			mb.setValue(rowIndex, colIndex, val);
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
				case INT32:
					for(int i = 0; i < rlen * clen; i++)
						denseBlock[i] = buf.getInt();
					break;
				case FP32:
					for(int i = 0; i < rlen * clen; i++)
						denseBlock[i] = buf.getFloat();
					break;
				case FP64:
					for(int i = 0; i < rlen * clen; i++)
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

	public static byte[] convertMBtoPy4JDenseArr(MatrixBlock mb) {
		byte[] ret = null;
		if(mb.isInSparseFormat()) {
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
