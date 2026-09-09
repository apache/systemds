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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;

import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

public class ReshapeOOCInstruction extends ComputationOOCInstruction {
	private final CPOperand _opRows;
	private final CPOperand _opCols;
	// private final CPOperand _opDims;
	private final CPOperand _opByRow;

	private ReshapeOOCInstruction(Operator op, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, CPOperand byRow, String opcode, String istr) {
		super(OOCType.Reshape, op, in, out, opcode, istr);
		_opRows = rows;
		_opCols = cols;
		// _opDims = dims;
		_opByRow = byRow;
	}

	public static ReshapeOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 6);
		String opcode = parts[0];

		if(!opcode.equalsIgnoreCase(Opcodes.RESHAPE.toString()))
			throw new DMLRuntimeException("Unknown opcode while parsing ReshapeInstruction: " + str);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand rows = new CPOperand(parts[2]);
		CPOperand cols = new CPOperand(parts[3]);
		CPOperand dims = new CPOperand(parts[4]);
		CPOperand byRow = new CPOperand(parts[5]);
		CPOperand out = new CPOperand(parts[6]);

		return new ReshapeOOCInstruction(new Operator(true), in, out, rows, cols, dims, byRow, opcode, str);
	}

	public void processInstruction(ExecutionContext ec) {
		long rows = ec.getScalarInput(_opRows).getLongValue();
		long cols = ec.getScalarInput(_opCols).getLongValue();
		boolean byRow = ec.getScalarInput(_opByRow).getBooleanValue();

		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(qOut);

		MatrixObject in = ec.getMatrixObject(input1);
		OOCStream<IndexedMatrixValue> qIn = in.getStreamHandle();
		int blen = in.getBlocksize();
		long rlen = in.getNumRows();
		long clen = in.getNumColumns();

		if(rlen * clen != rows * cols)
			throw new DMLRuntimeException("Reshape matrix requires consistent numbers of input/output cells (" + rlen
				+ ":" + clen + ", " + rows + ":" + cols + ").");

		if(rlen == rows) {
			mapOOC(qIn, qOut, tmp -> tmp);
			return;
		}

		if(clen <= blen && rlen <= blen && cols <= blen && rows <= blen) {
			mapOOC(qIn, qOut, tmp -> {
				MatrixBlock res = ((MatrixBlock) tmp.getValue()).reshape((int) rows, (int) cols, byRow);
				return new IndexedMatrixValue(tmp.getIndexes(), res);
			});
			return;
		}

		int numBlocksPerRowIn = (int) Math.ceil((double) clen / blen);
		int numBlocksPerColIn = (int) Math.ceil((double) rlen / blen);
		int numBlocksPerRowOut = (int) Math.ceil((double) cols / blen);
		int numBlocksPerColOut = (int) Math.ceil((double) rows / blen);

		if(byRow) {
			OOCStream<IndexedMatrixValue> singleRowBlocks = new SubscribableTaskQueue<>();
			// split blocks into single rows and adapt index
			CompletableFuture<Void> f = expandOOC(qIn, singleRowBlocks, tmp -> {
				ArrayList<IndexedMatrixValue> out = new ArrayList<>();
				MatrixBlock blk = (MatrixBlock) tmp.getValue();
				for(int i = 0; i < blk.getNumRows(); i++) {
					MatrixBlock slice = blk.slice(i, i);
					long r = tmp.getIndexes().getRowIndex();
					long c = tmp.getIndexes().getColumnIndex();
					r = (r - 1) * blen + i + 1;
					MatrixIndexes idx = new MatrixIndexes(r, c);
					out.add(new IndexedMatrixValue(idx, slice));
				}
				return out;
			});

			if(clen % blen == 0 && cols % blen == 0) {
				// singleRowBlocks do not need to be split
				if(rows == 1) {
					// result is one single row
					mapOOC(singleRowBlocks.getReadStream(), qOut, tmp -> {
						long r = tmp.getIndexes().getRowIndex();
						long c = tmp.getIndexes().getColumnIndex();
						// adapt index to new position in row
						return new IndexedMatrixValue(new MatrixIndexes(1, (r - 1) * numBlocksPerRowIn + c), tmp.getValue());
					});
				}
				else {
					f.join();
					reshapeFullColBlocks(rows, cols, blen, numBlocksPerRowIn, numBlocksPerRowOut, numBlocksPerColOut, singleRowBlocks, qOut);
				}
			}
			else {
				f.join();
				reshapePartialColBlocks(rlen, clen, rows, cols, blen, numBlocksPerRowIn, numBlocksPerRowOut, numBlocksPerColOut, singleRowBlocks, qOut);
			}
		}
		else {
			OOCStream<IndexedMatrixValue> singleColBlocks = new SubscribableTaskQueue<>();
			// split blocks into single cols and adapt index
			CompletableFuture<Void> f = expandOOC(qIn, singleColBlocks, tmp -> {
				ArrayList<IndexedMatrixValue> out = new ArrayList<>();
				MatrixBlock blk = (MatrixBlock) tmp.getValue();
				for(int i = 0; i < blk.getNumColumns(); i++) {
					MatrixBlock slice = blk.slice(0, blk.getNumRows() - 1, i, i);
					long r = tmp.getIndexes().getRowIndex();
					long c = tmp.getIndexes().getColumnIndex();
					c = (c - 1) * blen + i + 1;
					MatrixIndexes idx = new MatrixIndexes(r, c);
					out.add(new IndexedMatrixValue(idx, slice));
				}
				return out;
			});

			if(rlen % blen == 0 && rows % blen == 0) {
				// cols do not need to be split
				if(cols == 1) {
					// result is one single col
					mapOOC(singleColBlocks.getReadStream(), qOut, tmp -> {
						long r = tmp.getIndexes().getRowIndex();
						long c = tmp.getIndexes().getColumnIndex();
						// adapt index to new position in col
						return new IndexedMatrixValue(new MatrixIndexes((c - 1) * numBlocksPerColIn + r, 1), tmp.getValue());
					});
				}
				else {
					f.join();
					reshapeFullRowBlocks(rows, cols, blen, numBlocksPerRowOut, numBlocksPerColIn, numBlocksPerColOut, singleColBlocks, qOut);
				}
			}
			else {
				f.join();
				reshapePartialRowBlocks(rlen, clen, rows, cols, blen, numBlocksPerRowOut, numBlocksPerColIn, numBlocksPerColOut, singleColBlocks, qOut);
			}
		}
	}

	private void reshapeFullColBlocks(long rows, long cols, int blen, int numBlocksPerRowIn, int numBlocksPerRowOut,
		int numBlocksPerColOut, OOCStream<IndexedMatrixValue> singleRowBlocks, OOCStream<IndexedMatrixValue> qOut) {
		// use cache for accessing input rows by index
		CachingStream singleRowBlockCache = new CachingStream(singleRowBlocks);
		singleRowBlockCache.incrSubscriberCount(1);
		singleRowBlockCache.scheduleDeletion();

		// totalRowIdx corresponds to index of row block when all aligned in one row
		// br * numBlocksPerRowOut * blen + b + r * numBlocksPerRowOut;
		// with numBlocksPerRowOut * blen = cols
		long totalIdx = -cols - 1 - numBlocksPerRowOut;

		// iterate through rows of output blocks
		for(int br = 0; br < numBlocksPerColOut; br++) {
			totalIdx += cols;
			long tmp = totalIdx;
			// for each block in row
			for(int b = 0; b < numBlocksPerRowOut; b++) {
				totalIdx += 1;
				int localRows = (br == numBlocksPerColOut - 1 && rows % blen != 0) ? (int) rows % blen : blen;
				MatrixBlock res = new MatrixBlock(localRows, blen, false);
				long tmp2 = totalIdx;
				// for each row in block
				for(int r = 0; r < blen && r < localRows; r++) {
					totalIdx += numBlocksPerRowOut;
					// calc col idx for input
					long colBlockIn = totalIdx % numBlocksPerRowIn + 1;
					// calc row idx for input
					long rowBlockIn = totalIdx / numBlocksPerRowIn + 1;

					try(OOCStream.QueueCallback<IndexedMatrixValue> cb = singleRowBlockCache
						.findCached(new MatrixIndexes(rowBlockIn, colBlockIn))) {
						MatrixBlock blk = (MatrixBlock) cb.get().getValue();
						res.setRow(r, blk.getDenseBlockValues());
					}
				}
				totalIdx = tmp2;
				qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(br + 1, b + 1), res));
			}
			totalIdx = tmp;
		}
		qOut.closeInput();
	}

	private void reshapePartialColBlocks(long rlen, long clen, long rows, long cols, int blen, int numBlocksPerRowIn,
		int numBlocksPerRowOut, int numBlocksPerColOut, OOCStream<IndexedMatrixValue> singleRowBlocks, OOCStream<IndexedMatrixValue> qOut) {
		// use cache for accessing input rows by index
		CachingStream singleRowBlockCache = new CachingStream(singleRowBlocks);
		singleRowBlockCache.incrSubscriberCount(1);
		singleRowBlockCache.scheduleDeletion();

		int br = 0;
		int bc = 0;
		int r = 0;

		// allocate row of output blocks
		MatrixBlock[] outputBlockRow = allocateSliceBlocks(br, rows, cols, blen, numBlocksPerRowOut, numBlocksPerColOut,true);

		int offsetOut = 0;
		int localColsOut = (cols > blen) ? blen : (int) cols;
		// iterate through input rows and add to row of output blocks
		for(int i = 1; i <= rlen; i++) {
			for(int j = 1; j <= numBlocksPerRowIn; j++) {
				try(OOCStream.QueueCallback<IndexedMatrixValue> qcb = singleRowBlockCache.findCached(new MatrixIndexes(i, j))) {
					MatrixBlock blk = (MatrixBlock) qcb.get().getValue();

					int offsetIn = 0;
					int localColsIn = (j == numBlocksPerRowIn && clen % blen != 0) ? (int) clen % blen : blen;
					while(offsetIn < localColsIn) {
						// until input row fully processed
						int remIn = localColsIn - offsetIn;
						int remOut = localColsOut - offsetOut;
						if(remIn < remOut) {
							// next input
							setOutputEntries(blk, outputBlockRow[bc], r, offsetIn, offsetOut, remIn, true);
							offsetIn += remIn;
							offsetOut += remIn;
							continue;
						}
						else if(remIn == remOut) {
							// next input and next row
							setOutputEntries(blk, outputBlockRow[bc], r, offsetIn, offsetOut, remIn, true);
							offsetIn += remIn;
						}
						else {
							// next row
							setOutputEntries(blk, outputBlockRow[bc], r, offsetIn, offsetOut, remOut, true);
							offsetIn += remOut;
						}
						bc++;
						offsetOut = 0;
						if(bc == numBlocksPerRowOut) {
							// next row
							r++;
							if(r == outputBlockRow[0].getNumRows()) {
								// enqueue filled output blocks and allocate new ones
								for(int b = 0; b < outputBlockRow.length; b++)
									qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(br + 1, b + 1), outputBlockRow[b]));
								br++;
								// allocate new block row
								outputBlockRow = allocateSliceBlocks(br, rows, cols, blen, numBlocksPerRowOut, numBlocksPerColOut, true);
								r = 0;
							}
							bc = 0;
						}
						localColsOut = (bc == numBlocksPerRowOut - 1 && cols % blen != 0) ? (int) cols % blen : blen;
					}
				}
			}
		}
		qOut.closeInput();
	}

	private void reshapeFullRowBlocks(long rows, long cols, int blen, int numBlocksPerRowOut, int numBlocksPerColIn,
		int numBlocksPerColOut, OOCStream<IndexedMatrixValue> singleColBlocks, OOCStream<IndexedMatrixValue> qOut) {
		// use cache for accessing input cols by index
		CachingStream singleColBlockCache = new CachingStream(singleColBlocks);
		singleColBlockCache.incrSubscriberCount(1);
		singleColBlockCache.scheduleDeletion();

		// totalColIdx corresponds to index of col block when all aligned in one col
		// bc * numBlocksPerColOut * blen + b + c * numBlocksPerColOut;
		// with numBlocksPerColOut * blen = rows
		long totalIdx = -rows - 1 - numBlocksPerColOut;

		// iterate through cols of output blocks
		for(int bc = 0; bc < numBlocksPerRowOut; bc++) {
			totalIdx += rows;
			long tmp = totalIdx;
			// for each block in col
			for(int b = 0; b < numBlocksPerColOut; b++) {
				totalIdx += 1;
				int localCols = (bc == numBlocksPerRowOut - 1 && cols % blen != 0) ? (int) cols % blen : blen;
				MatrixBlock res = new MatrixBlock(blen, localCols, false);
				res.allocateDenseBlock();
				long tmp2 = totalIdx;
				// for each col in block
				for(int c = 0; c < blen && c < localCols; c++) {
					totalIdx += numBlocksPerColOut;
					// calc col idx for input
					long colBlockIn = totalIdx / numBlocksPerColIn + 1;
					// calc row idx for input
					long rowBlockIn = totalIdx % numBlocksPerColIn + 1;

					try(OOCStream.QueueCallback<IndexedMatrixValue> cb = singleColBlockCache
						.findCached(new MatrixIndexes(rowBlockIn, colBlockIn))) {
						MatrixBlock blk = (MatrixBlock) cb.get().getValue();
						res.getDenseBlock().set(0, blen, c, c + 1, blk.getDenseBlock());
					}
				}
				totalIdx = tmp2;
				res.recomputeNonZeros();
				qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(b + 1, bc + 1), res));
			}
			totalIdx = tmp;
		}
		qOut.closeInput();
	}

	private void reshapePartialRowBlocks(long rlen, long clen, long rows, long cols, int blen, int numBlocksPerRowOut,
		int numBlocksPerColIn, int numBlocksPerColOut, OOCStream<IndexedMatrixValue> singleColBlocks, OOCStream<IndexedMatrixValue> qOut) {
		// use cache for accessing input cols by index
		CachingStream singleRowBlockCache = new CachingStream(singleColBlocks);
		singleRowBlockCache.incrSubscriberCount(1);
		singleRowBlockCache.scheduleDeletion();

		int br = 0;
		int bc = 0;
		int c = 0;

		// allocate col of output blocks
		MatrixBlock[] outputBlockCol = allocateSliceBlocks(bc, rows, cols, blen, numBlocksPerRowOut, numBlocksPerColOut, false);

		int offsetOut = 0;
		int localRowsOut = (rows > blen) ? blen : (int) rows;
		// iterate through input cols and add to col of output blocks
		for(int j = 1; j <= clen; j++) {
			for(int i = 1; i <= numBlocksPerColIn; i++) {
				try(OOCStream.QueueCallback<IndexedMatrixValue> qcb = singleRowBlockCache.findCached(new MatrixIndexes(i, j))) {
					MatrixBlock blk = (MatrixBlock) qcb.get().getValue();

					int offsetIn = 0;
					int localRowsIn = (i == numBlocksPerColIn && rlen % blen != 0) ? (int) rlen % blen : blen;
					while(offsetIn < localRowsIn) {
						// until input col fully processed
						int remIn = localRowsIn - offsetIn;
						int remOut = localRowsOut - offsetOut;
						if(remIn < remOut) {
							// next input
							setOutputEntries(blk, outputBlockCol[br], c, offsetIn, offsetOut, remIn, false);
							offsetIn += remIn;
							offsetOut += remIn;
							continue;
						}
						else if(remIn == remOut) {
							// next input and next col
							setOutputEntries(blk, outputBlockCol[br], c, offsetIn, offsetOut, remIn, false);
							offsetIn += remIn;
						}
						else {
							// next col
							setOutputEntries(blk, outputBlockCol[br], c, offsetIn, offsetOut, remOut, false);
							offsetIn += remOut;
						}
						br++;
						offsetOut = 0;
						if(br == numBlocksPerColOut) {
							// next col
							c++;
							if(c == outputBlockCol[0].getNumColumns()) {
								// enqueue filled output blocks and allocate new ones
								for(int b = 0; b < outputBlockCol.length; b++) {
									outputBlockCol[b].recomputeNonZeros();
									qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(b + 1, bc + 1), outputBlockCol[b]));
								}
								bc++;
								// allocate new block col
								outputBlockCol = allocateSliceBlocks(bc, rows, cols, blen, numBlocksPerRowOut, numBlocksPerColOut, false);
								c = 0;
							}
							br = 0;
						}
						localRowsOut = (br == numBlocksPerColOut - 1 && rows % blen != 0) ? (int) rows % blen : blen;
					}
				}
			}
		}
		qOut.closeInput();
	}

	private MatrixBlock[] allocateSliceBlocks(int idx, long rows, long cols, int blen, int numBlocksPerRow, int numBlocksPerCol, boolean isBlockRowSlice) {
		int num = isBlockRowSlice ? numBlocksPerRow : numBlocksPerCol;
		MatrixBlock[] res = new MatrixBlock[num];

		// full inner blocks, adjust for outer blocks
		int localRows = ((!isBlockRowSlice || idx == numBlocksPerCol - 1) && rows % blen != 0) ? (int) rows % blen : blen;
		int localCols = ((isBlockRowSlice || idx == numBlocksPerRow - 1) && cols % blen != 0) ? (int) cols % blen : blen;

		for(int k = 0; k < num - 1; k++) {
			res[k] = isBlockRowSlice ? new MatrixBlock(localRows, blen, false) : new MatrixBlock(blen, localCols, false);
			res[k].allocateDenseBlock();
		}
		res[num - 1] = new MatrixBlock(localRows, localCols, false);
		res[num - 1].allocateDenseBlock();
		return res;
	}

	private void setOutputEntries(MatrixBlock src, MatrixBlock dest, int idx, int srcOffset, int destOffset, int length, boolean rowWise) {
		if(rowWise)
			((DenseBlockFP64) dest.getDenseBlock()).setPartialRow(src.getDenseBlock(), idx, srcOffset, destOffset, length);
		else
			((DenseBlockFP64) dest.getDenseBlock()).setPartialCol(src.getDenseBlock(), idx, srcOffset, destOffset, length);
	}
}
