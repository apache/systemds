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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.lib.CLALibTable;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.util.UtilFunctions;

public class LibMatrixTable {

	public static boolean ALLOW_COMPRESSED_TABLE_SEQ = false;

	protected static final Log LOG = LogFactory.getLog(LibMatrixTable.class.getName());

	private LibMatrixTable() {
		// empty private constructor
	}

	/**
	 * 
	 * The DML code to activate this function:
	 * 
	 * ret = table(seq(1, nrow(A)), A, w)
	 * 
	 * @param seqHeight A sequence vector height.
	 * @param A         The MatrixBlock vector to encode.
	 * @param w         The weight matrix to multiply on output cells.
	 * @return A new MatrixBlock with the table result.
	 */
	public static MatrixBlock tableSeqOperations(int seqHeight, MatrixBlock A, double w) {
		return tableSeqOperations(seqHeight, A, w, null, true);
	}

	/**
	 * The DML code to activate this function:
	 * 
	 * ret = table(seq(1, nrow(A)), A, w)
	 * 
	 * @param seqHeight  A sequence vector height.
	 * @param A          The MatrixBlock vector to encode.
	 * @param w          The weight matrix to multiply on output cells.
	 * @param ret        The output MatrixBlock, does not have to be used, but depending on updateClen determine the
	 *                   output size.
	 * @param updateClen Update clen, if set to true, ignore dimensions of ret, otherwise use the column dimension of
	 *                   ret.
	 * @return A new MatrixBlock or ret.
	 */
	public static MatrixBlock tableSeqOperations(int seqHeight, MatrixBlock A, double w, MatrixBlock ret,
		boolean updateClen) {

		if(A.getNumRows() != seqHeight)
			throw new DMLRuntimeException(
				"Invalid input sizes for table \"table(seq(1, nrow(A)), A, w)\" : sequence height is: " + seqHeight
					+ " while A is: " + A.getNumRows());

		if(A.getNumColumns() > 1)
			throw new DMLRuntimeException(
				"Invalid input A in table(seq(1, nrow(A)), A, w): A should only have one column but has: "
					+ A.getNumColumns());

		if(!Double.isNaN(w)) {
			if(compressedTableSeq() && w == 1)
				return CLALibTable.tableSeqOperations(seqHeight, A, updateClen ? -1 : ret.getNumColumns());
			else
				return tableSeqSparseBlock(seqHeight, A, w, ret, updateClen);
		}
		else {
			if(ret == null) {
				ret = new MatrixBlock();
				updateClen = true;
			}

			ret.rlen = seqHeight;
			// empty output.
			ret.denseBlock = null;
			ret.sparseBlock = null;
			ret.sparse = true;
			ret.nonZeros = 0;
			updateClen(ret, 0, updateClen);
			return ret;
		}

	}

	private static MatrixBlock tableSeqSparseBlock(final int rlen, final MatrixBlock A, final double w, MatrixBlock ret,
		boolean updateClen) {

		int maxCol = 0;
		// prepare allocation of CSR sparse block
		final int[] rowPointers = new int[rlen + 1];
		final int[] indexes = new int[rlen];
		final double[] values = new double[rlen];

		// sparse-unsafe table execution
		// (because input values of 0 are invalid and have to result in errors)
		// resultBlock guaranteed to be allocated for table expand
		// each row in resultBlock will be allocated and will contain exactly one value
		for(int i = 0; i < rlen; i++) {
			maxCol = execute(i, A.get(i, 0), w, maxCol, indexes, values);
			rowPointers[i] = i;
		}

		rowPointers[rlen] = rlen;

		if(ret == null) {
			ret = new MatrixBlock();
			updateClen = true;
		}

		ret.rlen = rlen;
		// assign the output
		ret.sparse = true;
		ret.denseBlock = null;
		// construct sparse CSR block from filled arrays
		ret.sparseBlock = new SparseBlockCSR(rowPointers, indexes, values, rlen);
		// compact all the null entries.
		((SparseBlockCSR) ret.sparseBlock).compact();
		ret.setNonZeros(ret.sparseBlock.size());

		updateClen(ret, maxCol, updateClen);
		return ret;
	}

	private static void updateClen(MatrixBlock ret, int maxCol, boolean updateClen) {
		// update meta data (initially unknown number of columns)
		// Only allowed if we enable the update flag.
		if(updateClen)
			ret.clen = maxCol;
	}

	public static int execute(int row, double v2, double w, int maxCol, int[] retIx, double[] retVals) {
		// If any of the values are NaN (i.e., missing) then
		// we skip this tuple, proceed to the next tuple
		if(Double.isNaN(v2))
			return maxCol;

		// safe casts to long for consistent behavior with indexing
		int col = UtilFunctions.toInt(v2);
		if(col <= 0)
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (value <= zero): " + v2);

		// set weight as value (expand is guaranteed to address different cells)
		retIx[row] = col - 1;
		retVals[row] = w;

		// maintain max seen col
		return Math.max(maxCol, col);
	}

	private static boolean compressedTableSeq() {
		return ALLOW_COMPRESSED_TABLE_SEQ || ConfigurationManager.isCompressionEnabled();
	}
}
