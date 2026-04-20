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
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.ooc.stream.FilteredOOCStream;
import org.apache.sysds.runtime.ooc.stream.SubOOCStream;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class MatrixIndexingOOCInstruction extends IndexingOOCInstruction {

	public MatrixIndexingOOCInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
		CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	public MatrixIndexingOOCInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru,
		CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
		super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
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

		//right indexing
		if(opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString())) {
			OOCStream<IndexedMatrixValue> qIn = mo.getStreamHandle();
			addInStream(qIn);

			if(output.isScalar()) {
				if(!inRange)
					throw new DMLRuntimeException(
						"Invalid values for matrix indexing: [" + (ix.rowStart + 1) + ":" + (ix.rowEnd + 1) + "," +
							(ix.colStart + 1) + ":" + (ix.colEnd + 1) + "] must be within matrix dimensions [" +
							mo.getNumRows() + "x" + mo.getNumColumns() + "].");

				Double scalarOut = null;
				IndexedMatrixValue tmp;

				while((tmp = qIn.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
					if(tmp.getIndexes().getRowIndex() == firstBlockRow + 1 &&
						tmp.getIndexes().getColumnIndex() == firstBlockCol + 1) {
						scalarOut = ((MatrixBlock) tmp.getValue()).get((int) (ix.rowStart % blocksize),
							(int) (ix.colStart % blocksize));
					}
				}
				if(scalarOut == null)
					throw new DMLRuntimeException("Desired block not found");
				ec.setScalarOutput(output.getName(), new DoubleObject(scalarOut));
				return;
			}

			if(ix.rowStart < 0 || ix.rowStart >= mo.getNumRows() || ix.rowEnd < ix.rowStart ||
				ix.rowEnd >= mo.getNumRows() || ix.colStart < 0 || ix.colStart >= mo.getNumColumns() ||
				ix.colEnd < ix.colStart || ix.colEnd >= mo.getNumColumns()) {
				String dbg = "inst=\"" + instString + "\", input=" + input1.getName() + ", output=" + output.getName() +
					", rowLower=" + debugScalarOperand(rowLower, ec) + ", rowUpper=" +
					debugScalarOperand(rowUpper, ec) + ", colLower=" + debugScalarOperand(colLower, ec) +
					", colUpper=" + debugScalarOperand(colUpper, ec) + ", resolvedRange=[" + (ix.rowStart + 1) + ":" +
					(ix.rowEnd + 1) + "," + (ix.colStart + 1) + ":" + (ix.colEnd + 1) + "]" + ", matrixDims=[" +
					mo.getNumRows() + "x" + mo.getNumColumns() + "]" + ", blocksize=" + blocksize;
				System.out.println("[WARN] OOC rightIndex bounds violation: " + dbg);
				throw new DMLRuntimeException(
					"Invalid values for matrix indexing: [" + (ix.rowStart + 1) + ":" + (ix.rowEnd + 1) + "," +
						(ix.colStart + 1) + ":" + (ix.colEnd + 1) + "] must be within matrix dimensions [" +
						mo.getNumRows() + "x" + mo.getNumColumns() + "]. " + dbg);
			}

			MatrixObject mOut = ec.getMatrixObject(output);
			ec.getDataCharacteristics(output.getName()).set(ix.rowSpan() + 1, ix.colSpan() + 1, blocksize, -1);
			OOCStream<IndexedMatrixValue> qOut = createWritableStream();
			addOutStream(qOut);
			mOut.setStreamHandle(qOut);

			qIn.setDownstreamMessageRelay(qOut::messageDownstream);
			qOut.setUpstreamMessageRelay(qIn::messageUpstream);
			qOut.setIXTransform((downstream, range) -> {
				if(downstream) {
					long rs = range.rowStart - ix.rowStart + 1;
					long re = range.rowEnd - ix.rowStart + 1;
					long cs = range.colStart - ix.colStart + 1;
					long ce = range.colEnd - ix.colStart + 1;
					// TODO What happens if range is out of bounds?
					rs = Math.max(1, rs);
					cs = Math.max(1, cs);
					re = Math.min(ix.rowSpan(), re);
					ce = Math.min(ix.colSpan(), ce);
					return new IndexRange(rs, re, cs, ce);
				}
				else {
					long rs = range.rowStart + ix.rowStart;
					long re = range.rowEnd + ix.rowStart;
					long cs = range.colStart + ix.colStart;
					long ce = range.colEnd + ix.colStart;
					return new IndexRange(rs, re, cs, ce);
				}
			});

			if(firstBlockRow == lastBlockRow && firstBlockCol == lastBlockCol) {
				MatrixIndexes srcBlock = new MatrixIndexes(firstBlockRow + 1, firstBlockCol + 1);
				OOCStream<IndexedMatrixValue> filteredStream = new FilteredOOCStream<>(qIn,
					tmp -> tmp.getIndexes().equals(srcBlock));
				mapOOC(filteredStream, qOut, tmp -> {
					MatrixBlock block = (MatrixBlock) tmp.getValue();
					int rowStartLocal = (int) (ix.rowStart % blocksize);
					int rowEndLocal = Math.min(block.getNumRows() - 1, (int) (ix.rowEnd % blocksize));
					int colStartLocal = (int) (ix.colStart % blocksize);
					int colEndLocal = Math.min(block.getNumColumns() - 1, (int) (ix.colEnd % blocksize));
					MatrixBlock outBlock = block.slice(rowStartLocal, rowEndLocal, colStartLocal, colEndLocal);
					return new IndexedMatrixValue(new MatrixIndexes(1, 1), outBlock);
				});
				return;
			}

			if(ix.rowStart % blocksize == 0 && ix.colStart % blocksize == 0) {
				// Aligned case: interior blocks can be forwarded directly, borders may require slicing
				final int outBlockRows = (int) Math.ceil((double) (ix.rowSpan() + 1) / blocksize);
				final int outBlockCols = (int) Math.ceil((double) (ix.colSpan() + 1) / blocksize);
				final int totalBlocks = outBlockRows * outBlockCols;
				final boolean isCached = qIn.hasStreamCache();
				final AtomicInteger producedBlocks = new AtomicInteger(0);
				CompletableFuture<Void> future = new CompletableFuture<>();

				mapOptionalOOC(qIn, qOut, tmp -> {
					if(future.isDone())
						return Optional.empty();

					long blockRow = tmp.getIndexes().getRowIndex() - 1;
					long blockCol = tmp.getIndexes().getColumnIndex() - 1;
					boolean within =
						blockRow >= firstBlockRow && blockRow <= lastBlockRow && blockCol >= firstBlockCol &&
							blockCol <= lastBlockCol;
					if(!within)
						return Optional.empty();

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
						// If the block is cached, we need to copy because otherwise it could lead to nullpointers
						outBlock = isCached ? new MatrixBlock(block) : block;
					}
					else {
						outBlock = block.slice(rowStartLocal, rowEndLocal, colStartLocal, colEndLocal);
					}

					long outBlockRow = blockRow - firstBlockRow + 1;
					long outBlockCol = blockCol - firstBlockCol + 1;

					if(producedBlocks.incrementAndGet() >= totalBlocks)
						future.complete(null);
					return Optional.of(new IndexedMatrixValue(new MatrixIndexes(outBlockRow, outBlockCol), outBlock));
				});
				return;
			}

			final BlockAligner<MatrixIndexes> aligner = new BlockAligner<>(ix, blocksize);
			final ConcurrentHashMap<MatrixIndexes, Integer> consumptionCounts = new ConcurrentHashMap<>();

			// We may need to construct our own intermediate stream to properly manage the cached items
			boolean hasIntermediateStream = !qIn.hasStreamCache();
			final CompletableFuture<Void> future = new CompletableFuture<>();

			OOCStream<IndexedMatrixValue> filteredStream = filteredOOCStream(qIn, tmp -> {
				boolean pass = !future.isDone();
				// Pre-filter incoming blocks to avoid unnecessary task submission
				long blockRow = tmp.getIndexes().getRowIndex() - 1;
				long blockCol = tmp.getIndexes().getColumnIndex() - 1;
				pass &= blockRow >= firstBlockRow && blockRow <= lastBlockRow && blockCol >= firstBlockCol &&
					blockCol <= lastBlockCol;

				if(!pass && !hasIntermediateStream)
					qIn.getStreamCache().incrProcessingCount(tmp.getIndexes(), 1);
				return pass;
			});

			final CachingStream cachedStream = hasIntermediateStream ? new CachingStream(
				filteredStream) : qIn.getStreamCache();
			cachedStream.activateIndexing();
			cachedStream.incrSubscriberCount(1); // We may require re-consumption of blocks (up to 4 times)
			OOCStream<IndexedMatrixValue> readStream = cachedStream.getReadStream();

			submitOOCTasks(readStream, tmp -> {
				boolean completed = aligner.putNext(tmp.get().getIndexes(), tmp.get().getIndexes(), (idx, sector) -> {
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

							try(OOCStream.QueueCallback<IndexedMatrixValue> cb = cachedStream.peekCached(mIdx)) {
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
								if(v == null)
									v = 0;
								v = v + 1;
								if(v == maxConsumptions)
									return null;
								return v;
							});

							if(con == null)
								cachedStream.incrProcessingCount(mIdx, 1);
						}
					}

					qOut.enqueue(new IndexedMatrixValue(idx, target));
				});

				if(completed)
					future.complete(null);
			}).thenRun(() -> {
				aligner.close();
				qOut.closeInput();
			}).exceptionally(err -> {
				qOut.propagateFailure(DMLRuntimeException.of(err));
				return null;
			});

			if(hasIntermediateStream)
				cachedStream.scheduleDeletion(); // We can immediately delete blocks after consumption
		}
		else if(opcode.equalsIgnoreCase(Opcodes.LEFT_INDEX.toString())) {
			MatrixObject mOut = ec.getMatrixObject(output);
			ec.getDataCharacteristics(output.getName()).set(mo.getNumRows(), mo.getNumColumns(), blocksize, -1);
			if(input2.getDataType().isScalar()) {
				if(!ix.isScalar())
					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: " + ix + ".");
				if(ix.rowStart < 0 || ix.rowStart >= mo.getNumRows() || ix.colStart < 0 ||
					ix.colStart >= mo.getNumColumns()) {
					throw new DMLRuntimeException(
						"Invalid values for matrix indexing: [" + (ix.rowStart + 1) + ":" + (ix.rowEnd + 1) + "," +
							(ix.colStart + 1) + ":" + (ix.colEnd + 1) + "] must be within matrix dimensions [" +
							mo.getNumRows() + "x" + mo.getNumColumns() + "].");
				}

				final ScalarObject scalar = ec.getScalarInput(input2.getName(), ValueType.FP64, input2.isLiteral());
				final double scalarValue = scalar.getDoubleValue();
				final long targetBlockRow = ix.rowStart / blocksize + 1;
				final long targetBlockCol = ix.colStart / blocksize + 1;
				final int targetLocalRow = (int) (ix.rowStart % blocksize);
				final int targetLocalCol = (int) (ix.colStart % blocksize);

				OOCStream<IndexedMatrixValue> qLhs = mo.getStreamHandle();
				OOCStream<IndexedMatrixValue> qOutRaw = createWritableStream();
				SubOOCStream<IndexedMatrixValue> qOut = new SubOOCStream<>(qOutRaw);
				addInStream(qLhs);
				addOutStream(qOut);
				mOut.setStreamHandle(qOut);

				submitOOCTasks(qLhs, cb -> {
					IndexedMatrixValue lhs = cb.get();
					MatrixIndexes idx = lhs.getIndexes();
					if(idx.getRowIndex() != targetBlockRow || idx.getColumnIndex() != targetBlockCol) {
						qOut.enqueue(cb.keepOpen());
						return;
					}

					MatrixBlock src = (MatrixBlock) lhs.getValue();
					MatrixBlock updated = new MatrixBlock(src);
					updated.set(targetLocalRow, targetLocalCol, scalarValue);
					updated.examSparsity();
					qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(idx), updated));
				}).thenRun(() -> {
					qOut.closeInput();
					qOutRaw.closeInput();
				}).exceptionally(err -> {
					DMLRuntimeException dmlErr = DMLRuntimeException.of(err);
					qOut.propagateFailure(dmlErr);
					qOutRaw.propagateFailure(dmlErr);
					qOutRaw.closeInput();
					return null;
				});
				return;
			}

			final MatrixObject rhsMo = ec.getMatrixObject(input2.getName());
			final long lhsRows = mo.getNumRows();
			final long lhsCols = mo.getNumColumns();
			final long rhsRows = rhsMo.getNumRows();
			final long rhsCols = rhsMo.getNumColumns();

			if(ix.rowSpan() + 1 != rhsRows || ix.colSpan() + 1 != rhsCols) {
				throw new DMLRuntimeException(
					"Invalid index range of leftindexing: [" + (ix.rowStart + 1) + ":" + (ix.rowEnd + 1) + "," +
						(ix.colStart + 1) + ":" + (ix.colEnd + 1) + "] vs [" + rhsRows + "x" + rhsCols + "].");
			}
			if(ix.rowStart < 0 || ix.rowStart >= lhsRows || ix.rowEnd < ix.rowStart || ix.rowEnd >= lhsRows ||
				ix.colStart < 0 || ix.colStart >= lhsCols || ix.colEnd < ix.colStart || ix.colEnd >= lhsCols) {
				throw new DMLRuntimeException(
					"Invalid values for matrix indexing: [" + (ix.rowStart + 1) + ":" + (ix.rowEnd + 1) + "," +
						(ix.colStart + 1) + ":" + (ix.colEnd + 1) + "] must be within matrix dimensions [" + lhsRows +
						"x" + lhsCols + "].");
			}

			final IndexRange shiftRange = new IndexRange(ix.rowStart + 1, ix.rowEnd + 1, ix.colStart + 1,
				ix.colEnd + 1);
			final BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());

			OOCStream<IndexedMatrixValue> qLhs = mo.getStreamHandle();
			OOCStream<IndexedMatrixValue> qRhs = rhsMo.getStreamHandle();
			OOCStream<IndexedMatrixValue> qOutRaw = createWritableStream();
			SubOOCStream<IndexedMatrixValue> qOut = new SubOOCStream<>(qOutRaw);

			addInStream(qLhs, qRhs);
			addOutStream(qOut);
			mOut.setStreamHandle(qOut);

			final Map<MatrixIndexes, LeftIndexAccumulator> aggregators = new ConcurrentHashMap<>();
			submitOOCTasks(List.of(qLhs, qRhs), (streamIdx, cb) -> {
				if(streamIdx == 0) {
					IndexedMatrixValue lhs = cb.get();
					MatrixIndexes lhsIx = lhs.getIndexes();
					if(!UtilFunctions.isInBlockRange(lhsIx, blocksize, shiftRange)) {
						qOut.enqueue(cb.keepOpen());
						return;
					}

					MatrixIndexes key = new MatrixIndexes(lhsIx);
					int expectedRhsContribs = getExpectedRhsContribs(key, shiftRange, blocksize, lhsRows, lhsCols);
					LeftIndexAccumulator acc = aggregators.computeIfAbsent(key,
						k -> new LeftIndexAccumulator(expectedRhsContribs));

					IndexRange zeroRange = UtilFunctions.getSelectedRangeForZeroOut(lhs, blocksize, shiftRange);
					MatrixBlock lhsZeroed = ((MatrixBlock) lhs.getValue()).zeroOutOperations(new MatrixBlock(),
						zeroRange);

					MatrixBlock out = acc.addLhs(lhsZeroed, plus);
					if(out != null) {
						if(!aggregators.remove(key, acc))
							throw new DMLRuntimeException(
								"Failed to remove completed LEFT_INDEX accumulator for " + key);
						out.examSparsity();
						qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(key), out));
					}
				}
				else {
					IndexedMatrixValue rhs = cb.get();
					ArrayList<IndexedMatrixValue> shifted = new ArrayList<>();
					OperationsOnMatrixValues.performShift(rhs, shiftRange, blocksize, lhsRows, lhsCols, shifted);

					for(IndexedMatrixValue part : shifted) {
						MatrixIndexes key = new MatrixIndexes(part.getIndexes());
						LeftIndexAccumulator acc = aggregators.computeIfAbsent(key, k -> new LeftIndexAccumulator(
							getExpectedRhsContribs(k, shiftRange, blocksize, lhsRows, lhsCols)));

						MatrixBlock out = acc.addRhs((MatrixBlock) part.getValue(), plus);
						if(out != null) {
							if(!aggregators.remove(key, acc))
								throw new DMLRuntimeException(
									"Failed to remove completed LEFT_INDEX accumulator for " + key);
							out.examSparsity();
							qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(key), out));
						}
					}
				}
			}).thenRun(() -> {
				if(!aggregators.isEmpty())
					throw new DMLRuntimeException(
						"LEFT_INDEX finished with unfinished aggregators: " + aggregators.size());
				qOut.closeInput();
				qOutRaw.closeInput();
			}).exceptionally(err -> {
				DMLRuntimeException dmlErr = DMLRuntimeException.of(err);
				qOut.propagateFailure(dmlErr);
				qOutRaw.propagateFailure(dmlErr);
				qOutRaw.closeInput();
				return null;
			});
		}
		else
			throw new DMLRuntimeException(
				"Invalid opcode (" + opcode + ") encountered in MatrixIndexingOOCInstruction.");
	}

	private static String debugScalarOperand(CPOperand op, ExecutionContext ec) {
		try {
			return op.getName() + "=" + ec.getScalarInput(op).getStringValue() + (op.isLiteral() ? " [lit]" : " [var]");
		}
		catch(Exception ex) {
			return op.getName() + "=<unavailable:" + ex.getClass().getSimpleName() + ">";
		}
	}

	private static int getExpectedRhsContribs(MatrixIndexes lhsIx, IndexRange shift, int bs, long lhsRows,
		long lhsCols) {

		long lrs = UtilFunctions.computeCellIndex(lhsIx.getRowIndex(), bs, 0);
		long lcs = UtilFunctions.computeCellIndex(lhsIx.getColumnIndex(), bs, 0);
		long lre = lrs + UtilFunctions.computeBlockSize(lhsRows, lhsIx.getRowIndex(), bs) - 1;
		long lce = lcs + UtilFunctions.computeBlockSize(lhsCols, lhsIx.getColumnIndex(), bs) - 1;

		long ors = Math.max(lrs, shift.rowStart), ore = Math.min(lre, shift.rowEnd);
		long ocs = Math.max(lcs, shift.colStart), oce = Math.min(lce, shift.colEnd);
		if(ors > ore || ocs > oce)
			return 0;

		long rhsRowStart = ors - shift.rowStart + 1;
		long rhsColStart = ocs - shift.colStart + 1;
		long rowLen = ore - ors + 1;
		long colLen = oce - ocs + 1;

		long rBlocks = UtilFunctions.computeBlockIndex(rhsRowStart + rowLen - 1, bs) -
			UtilFunctions.computeBlockIndex(rhsRowStart, bs) + 1;
		long cBlocks = UtilFunctions.computeBlockIndex(rhsColStart + colLen - 1, bs) -
			UtilFunctions.computeBlockIndex(rhsColStart, bs) + 1;

		return Math.toIntExact(rBlocks * cBlocks);
	}

	private static class LeftIndexAccumulator {
		private final int _expectedRhsContribs;
		private MatrixBlock _lhs;
		private MatrixBlock _rhsAgg;
		private int _rhsCtr;
		private boolean _lhsSeen;
		private boolean _emitted;

		private LeftIndexAccumulator(int expectedRhsContribs) {
			_expectedRhsContribs = expectedRhsContribs;
			_rhsCtr = 0;
			_lhsSeen = false;
			_emitted = false;
		}

		public synchronized MatrixBlock addLhs(MatrixBlock lhs, BinaryOperator plus) {
			if(_lhsSeen)
				throw new DMLRuntimeException("Duplicate LEFT_INDEX lhs contribution encountered");
			_lhs = lhs;
			_lhsSeen = true;
			return emitIfReady(plus);
		}

		public synchronized MatrixBlock addRhs(MatrixBlock rhs, BinaryOperator plus) {
			if(_emitted)
				throw new DMLRuntimeException("LEFT_INDEX accumulator received rhs after completion");
			_rhsCtr++;
			if(_rhsCtr > _expectedRhsContribs)
				throw new DMLRuntimeException(
					"LEFT_INDEX accumulator rhs overflow: " + _rhsCtr + " > " + _expectedRhsContribs);
			if(_rhsAgg == null)
				_rhsAgg = rhs;
			else
				_rhsAgg = _rhsAgg.binaryOperationsInPlace(plus, rhs);
			return emitIfReady(plus);
		}

		private MatrixBlock emitIfReady(BinaryOperator plus) {
			if(_emitted || !_lhsSeen || _rhsCtr < _expectedRhsContribs)
				return null;
			if(_rhsCtr > _expectedRhsContribs)
				throw new DMLRuntimeException("LEFT_INDEX accumulator encountered invalid rhs contribution count");
			_emitted = true;
			if(_rhsAgg != null)
				_lhs = _lhs.binaryOperationsInPlace(plus, _rhsAgg);
			return _lhs;
		}
	}
}
