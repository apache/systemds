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

package org.apache.sysml.runtime.instructions.flink.functions;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.UtilFunctions;

import java.util.ArrayList;

public class ExtractBlockForBinaryReblock
		implements FlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>> {

	private final long rlen;
	private final long clen;
	private final int in_brlen;
	private final int in_bclen;
	private final int out_brlen;
	private final int out_bclen;

	public ExtractBlockForBinaryReblock(MatrixCharacteristics mcIn, MatrixCharacteristics mcOut)
			throws DMLRuntimeException {
		rlen = mcIn.getRows();
		clen = mcIn.getCols();
		in_brlen = mcIn.getRowsPerBlock();
		in_bclen = mcIn.getColsPerBlock();
		out_brlen = mcOut.getRowsPerBlock();
		out_bclen = mcOut.getColsPerBlock();

		//sanity check block sizes
		if (in_brlen <= 0 || in_bclen <= 0 || out_brlen <= 0 || out_bclen <= 0) {
			throw new DMLRuntimeException("Block sizes not unknown:" +
					in_brlen + "," + in_bclen + "," + out_brlen + "," + out_bclen);
		}
	}

	@Override
	public void flatMap(Tuple2<MatrixIndexes, MatrixBlock> arg0, Collector<Tuple2<MatrixIndexes, MatrixBlock>> out)
			throws Exception {
		MatrixIndexes ixIn = arg0.f0;
		MatrixBlock in = arg0.f1;

		// The global cell indexes don't change in reblock operations
		long startRowGlobalCellIndex = UtilFunctions.computeCellIndex(ixIn.getRowIndex(), in_brlen, 0);
		long endRowGlobalCellIndex = getEndGlobalIndex(ixIn.getRowIndex(), true, true);
		long startColGlobalCellIndex = UtilFunctions.computeCellIndex(ixIn.getColumnIndex(), in_bclen, 0);
		long endColGlobalCellIndex = getEndGlobalIndex(ixIn.getColumnIndex(), true, false);
		assert (startRowGlobalCellIndex <= endRowGlobalCellIndex && startColGlobalCellIndex <= endColGlobalCellIndex);

		long out_startRowBlockIndex = UtilFunctions.computeBlockIndex(startRowGlobalCellIndex, out_brlen);
		long out_endRowBlockIndex = UtilFunctions.computeBlockIndex(endRowGlobalCellIndex, out_brlen);
		long out_startColBlockIndex = UtilFunctions.computeBlockIndex(startColGlobalCellIndex, out_bclen);
		long out_endColBlockIndex = UtilFunctions.computeBlockIndex(endColGlobalCellIndex, out_bclen);
		assert (out_startRowBlockIndex <= out_endRowBlockIndex && out_startColBlockIndex <= out_endColBlockIndex);

		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();

		for (long i = out_startRowBlockIndex; i <= out_endRowBlockIndex; i++) {
			for (long j = out_startColBlockIndex; j <= out_endColBlockIndex; j++) {
				MatrixIndexes indx = new MatrixIndexes(i, j);
				long rowLower = Math.max(UtilFunctions.computeCellIndex(i, out_brlen, 0), startRowGlobalCellIndex);
				long rowUpper = Math.min(getEndGlobalIndex(i, false, true), endRowGlobalCellIndex);
				long colLower = Math.max(UtilFunctions.computeCellIndex(j, out_bclen, 0), startColGlobalCellIndex);
				long colUpper = Math.min(getEndGlobalIndex(j, false, false), endColGlobalCellIndex);

				int new_lrlen = UtilFunctions.computeBlockSize(rlen, i, out_brlen);
				int new_lclen = UtilFunctions.computeBlockSize(clen, j, out_bclen);
				MatrixBlock blk = new MatrixBlock(new_lrlen, new_lclen, true);

				int in_i1 = UtilFunctions.computeCellInBlock(rowLower, in_brlen);
				int out_i1 = UtilFunctions.computeCellInBlock(rowLower, out_brlen);

				for (long i1 = rowLower; i1 <= rowUpper; i1++, in_i1++, out_i1++) {
					int in_j1 = UtilFunctions.computeCellInBlock(colLower, in_bclen);
					int out_j1 = UtilFunctions.computeCellInBlock(colLower, out_bclen);
					for (long j1 = colLower; j1 <= colUpper; j1++, in_j1++, out_j1++) {
						double val = in.getValue(in_i1, in_j1);
						blk.appendValue(out_i1, out_j1, val);
					}
				}
				out.collect(new Tuple2<MatrixIndexes, MatrixBlock>(indx, blk));
			}
		}
	}

	/**
	 * @param blockIndex
	 * @param isIn
	 * @param isRow
	 * @return
	 */
	private long getEndGlobalIndex(long blockIndex, boolean isIn, boolean isRow) {
		//determine dimension and block sizes
		long len = isRow ? rlen : clen;
		int blen = isIn ? (isRow ? in_brlen : in_bclen)
				: (isRow ? out_brlen : out_bclen);

		//compute 1-based global cell index in block
		int new_len = UtilFunctions.computeBlockSize(len, blockIndex, blen);
		return UtilFunctions.computeCellIndex(blockIndex, blen, new_len - 1);
	}
}
