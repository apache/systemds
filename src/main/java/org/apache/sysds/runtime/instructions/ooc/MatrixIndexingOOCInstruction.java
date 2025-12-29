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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class MatrixIndexingOOCInstruction extends IndexingOOCInstruction {

	public MatrixIndexingOOCInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
		CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		IndexRange ix = getIndexRange(ec);

		MatrixObject mo = ec.getMatrixObject(input1.getName());
		int blocksize = mo.getBlocksize();
		long firstBlockRow = ix.rowStart / blocksize;
		long lastBlockRow = ix.rowEnd / blocksize;
		long firstBlockCol = ix.colStart / blocksize;
		long lastBlockCol = ix.colEnd / blocksize;

		boolean inRange = ix.rowStart < mo.getNumRows() && ix.colStart < mo.getNumColumns();

		OOCStream<IndexedMatrixValue> qIn = mo.getStreamHandle();
		OOCStream<IndexedMatrixValue> qOut = createWritableStream();

		addInStream(qIn);
		addOutStream(qOut);

		MatrixObject mOut = ec.getMatrixObject(output);
		mOut.setStreamHandle(qOut);

		//right indexing
		if(opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString())) {
			if(output.isScalar() && inRange) {
				IndexedMatrixValue tmp;

				while((tmp = qIn.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
					if(tmp.getIndexes().getRowIndex() == firstBlockRow &&
						tmp.getIndexes().getColumnIndex() == firstBlockCol) {
						ec.setScalarOutput(output.getName(), new DoubleObject(
							tmp.getValue().get((int) ix.rowStart % blocksize, (int) ix.rowEnd % blocksize)));
						return;
					}
				}

				throw new DMLRuntimeException("Desired block not found");
			}

			if(ix.rowStart % blocksize == 0 && ix.colStart % blocksize == 0) {
				// Aligned case: interior blocks can be forwarded directly, borders may require slicing
				final int outBlockRows = (int) Math.ceil((double) (ix.rowSpan() + 1) / blocksize);
				final int outBlockCols = (int) Math.ceil((double) (ix.colSpan() + 1) / blocksize);
				final int totalBlocks = outBlockRows * outBlockCols;
				final AtomicInteger producedBlocks = new AtomicInteger(0);
				CompletableFuture<Void> future = new  CompletableFuture<>();

				filterOOC(qIn, tmp -> {
					MatrixIndexes inIdx = tmp.getIndexes();
					long blockRow = inIdx.getRowIndex() - 1;
					long blockCol = inIdx.getColumnIndex() - 1;

					MatrixBlock block = (MatrixBlock) tmp.getValue();

					int rowStartLocal = (blockRow == firstBlockRow) ? (int) (ix.rowStart % blocksize) : 0;
					int rowEndLocal = (blockRow == lastBlockRow) ? Math.min(block.getNumRows() - 1,
						(int) (ix.rowEnd % blocksize)) : block.getNumRows() - 1;
					int colStartLocal = (blockCol == firstBlockCol) ? (int) (ix.colStart % blocksize) : 0;
					int colEndLocal = (blockCol == lastBlockCol) ? Math.min(block.getNumColumns() - 1,
						(int) (ix.colEnd % blocksize)) : block.getNumColumns() - 1;

					MatrixBlock outBlock;
					if(rowStartLocal == 0 && rowEndLocal == block.getNumRows() - 1 && colStartLocal == 0 &&
						colEndLocal == block.getNumColumns() - 1) {
						outBlock = block;
					}
					else {
						outBlock = block.slice(rowStartLocal, rowEndLocal, colStartLocal, colEndLocal);
					}

					long outBlockRow = blockRow - firstBlockRow + 1;
					long outBlockCol = blockCol - firstBlockCol + 1;
					qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(outBlockRow, outBlockCol), outBlock));

					if(producedBlocks.incrementAndGet() >= totalBlocks)
						future.complete(null);
				}, tmp -> {
					if (future.isDone()) // Then we may skip blocks and avoid submitting tasks
						return false;

					long blockRow = tmp.getIndexes().getRowIndex() - 1;
					long blockCol = tmp.getIndexes().getColumnIndex() - 1;
					return blockRow >= firstBlockRow && blockRow <= lastBlockRow && blockCol >= firstBlockCol &&
						blockCol <= lastBlockCol;
				}, qOut::closeInput);
				return;
			}

			final BlockAligner<MatrixIndexes> aligner = new BlockAligner<>(ix, blocksize);
			final ConcurrentHashMap<MatrixIndexes, Integer> consumptionCounts = new ConcurrentHashMap<>();

			// We may need to construct our own intermediate stream to properly manage the cached items
			boolean hasIntermediateStream = !qIn.hasStreamCache();
			final CachingStream cachedStream = hasIntermediateStream ? new CachingStream(new SubscribableTaskQueue<>()) : qOut.getStreamCache();
			cachedStream.activateIndexing();
			cachedStream.incrSubscriberCount(1); // We may require re-consumption of blocks (up to 4 times)
			final CompletableFuture<Void> future = new CompletableFuture<>();

			filterOOC(qIn.getReadStream(), tmp -> {
				if (hasIntermediateStream) {
					// We write to an intermediate stream to ensure that these matrix blocks are properly cached
					cachedStream.getWriteStream().enqueue(tmp);
				}

				boolean completed = aligner.putNext(tmp.getIndexes(), tmp.getIndexes(), (idx, sector) -> {
					int targetBlockRow = (int) (idx.getRowIndex() - 1);
					int targetBlockCol = (int) (idx.getColumnIndex() - 1);

					long targetRowStartGlobal = ix.rowStart + (long) targetBlockRow * blocksize;
					long targetRowEndGlobal = Math.min(ix.rowEnd, targetRowStartGlobal + blocksize - 1);
					long targetColStartGlobal = ix.colStart + (long) targetBlockCol * blocksize;
					long targetColEndGlobal = Math.min(ix.colEnd, targetColStartGlobal + blocksize - 1);

					int nRows = (int) (targetRowEndGlobal - targetRowStartGlobal + 1);
					int nCols = (int) (targetColEndGlobal - targetColStartGlobal + 1);

					long firstSrcBlockRow = targetRowStartGlobal / blocksize;
					long lastSrcBlockRow = targetRowEndGlobal / blocksize;
					int rowSegments = (int) (lastSrcBlockRow - firstSrcBlockRow + 1);

					long firstSrcBlockCol = targetColStartGlobal / blocksize;
					long lastSrcBlockCol = targetColEndGlobal / blocksize;
					int colSegments = (int) (lastSrcBlockCol - firstSrcBlockCol + 1);

					MatrixBlock target = null;

					for(int r = 0; r < rowSegments; r++) {
						for(int c = 0; c < colSegments; c++) {
							MatrixIndexes mIdx = sector.get(r, c);
							if(mIdx == null)
								continue;

							try (OOCStream.QueueCallback<IndexedMatrixValue> cb = cachedStream.peekCached(mIdx)) {
								IndexedMatrixValue mv = cb.get();
								MatrixBlock srcBlock = (MatrixBlock) mv.getValue();

								if(target == null)
									target = new MatrixBlock(nRows, nCols, srcBlock.isInSparseFormat());

								long srcBlockRowStart = (mIdx.getRowIndex() - 1) * blocksize;
								long srcBlockColStart = (mIdx.getColumnIndex() - 1) * blocksize;
								long sliceRowStartGlobal = Math.max(targetRowStartGlobal, srcBlockRowStart);
								long sliceRowEndGlobal = Math.min(targetRowEndGlobal,
									srcBlockRowStart + srcBlock.getNumRows() - 1);
								long sliceColStartGlobal = Math.max(targetColStartGlobal, srcBlockColStart);
								long sliceColEndGlobal = Math.min(targetColEndGlobal,
									srcBlockColStart + srcBlock.getNumColumns() - 1);

								int sliceRowStart = (int) (sliceRowStartGlobal - srcBlockRowStart);
								int sliceRowEnd = (int) (sliceRowEndGlobal - srcBlockRowStart);
								int sliceColStart = (int) (sliceColStartGlobal - srcBlockColStart);
								int sliceColEnd = (int) (sliceColEndGlobal - srcBlockColStart);

								int targetRowOffset = (int) (sliceRowStartGlobal - targetRowStartGlobal);
								int targetColOffset = (int) (sliceColStartGlobal - targetColStartGlobal);

								MatrixBlock sliced = srcBlock.slice(sliceRowStart, sliceRowEnd, sliceColStart,
									sliceColEnd);
								sliced.putInto(target, targetRowOffset, targetColOffset, true);
							}

							final int maxConsumptions = aligner.getNumConsumptions(mIdx);

							Integer con = consumptionCounts.compute(mIdx, (k, v) -> {
								if (v == null)
									v = 0;
								v = v+1;
								if (v == maxConsumptions)
									return null;
								return v;
							});

							if (con == null)
								cachedStream.incrProcessingCount(cachedStream.findCachedIndex(mIdx), 1);
						}
					}

					qOut.enqueue(new IndexedMatrixValue(idx, target));
				});

				if(completed)
					future.complete(null);
			}, tmp -> {
				if (future.isDone()) // Then we may skip blocks and avoid submitting tasks
					return false;

				// Pre-filter incoming blocks to avoid unnecessary task submission
				long blockRow = tmp.getIndexes().getRowIndex() - 1;
				long blockCol = tmp.getIndexes().getColumnIndex() - 1;
				return blockRow >= firstBlockRow && blockRow <= lastBlockRow && blockCol >= firstBlockCol &&
					blockCol <= lastBlockCol;
			}, () -> {
				aligner.close();
				qOut.closeInput();
			}, tmp -> {
				// If elements are not processed in an existing caching stream, we increment the process counter to allow block deletion
				if (!hasIntermediateStream)
					cachedStream.incrProcessingCount(cachedStream.findCachedIndex(tmp.getIndexes()), 1);
			});

			if (hasIntermediateStream)
				cachedStream.scheduleDeletion(); // We can immediately delete blocks after consumption
		}
		//left indexing
		else if(opcode.equalsIgnoreCase(Opcodes.LEFT_INDEX.toString())) {
			throw new NotImplementedException();
		}
		else
			throw new DMLRuntimeException(
				"Invalid opcode (" + opcode + ") encountered in MatrixIndexingOOCInstruction.");
	}
}
