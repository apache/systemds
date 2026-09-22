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

package org.apache.sysds.runtime.ooc.primitives;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.ManagedPayload;
import org.apache.sysds.runtime.ooc.memory.ReservationBudget;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.store.StateTable;
import org.apache.sysds.runtime.ooc.store.StoreLease;
import org.apache.sysds.runtime.ooc.stream.AllocatedOOCStream;
import org.apache.sysds.runtime.ooc.stream.StreamContext;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;
import org.apache.sysds.runtime.ooc.util.OOCUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class ReshapeOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _input;
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final boolean _byRow;
	private final long _rows;
	private final long _cols;

	private long _rlen;
	private long _clen;
	private int _blen;

	OOCStream<IndexedMatrixValue> _in;
	OOCStream<IndexedMatrixValue> _out;
	StateTable<IndexedMatrixValue> _table;

	private int _numColBlocksIn;
	private int _numRowBlocksIn;
	private int _numColBlocksOut;
	private int _numRowBlocksOut;
	private long _numRowsBlockOut;
	private long _numColsBlockOut;
	private long _blockBytesOut;
	private long _sliceBytes;

	public ReshapeOOCPrimitive(OOCStreamable<IndexedMatrixValue> input, OOCStreamable<IndexedMatrixValue> output,
		long rows, long cols, boolean byRow, StreamContext context) {
		super(context, input);
		_input = input;
		_output = output;
		_byRow = byRow;
		_rows = rows;
		_cols = cols;
		_pattern = byRow ? OOCAccessPattern.ROW_MAJOR : OOCAccessPattern.COL_MAJOR;
	}

	@Override
	protected void inferPatternsInternal() {
		for(OOCPrimitive child : getChildren())
			child.requestPattern(_pattern);
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		for(OOCPrimitive child : getChildren())
			child.requestPattern(_pattern);
	}

	@Override
	protected void startExecution() {
		initExecution();

		if(_rlen * _clen != _rows * _cols) {
			// non matching dims
			onComplete();
			throw new DMLRuntimeException("Reshape matrix requires consistent numbers of input/output cells (" + _rlen
				+ ":" + _clen + ", " + _rows + ":" + _cols + ").");
		}

		if(_rlen == _rows) {
			// same block dims
			OOCInstructionUtils
				.submitAdmittedOOCTasks(_in, _out,
					value -> new IndexedMatrixValue(value.getIndexes(), value.getValue()), _allowance, getContext())
				.thenRun(this::onComplete);
			return;
		}

		initBlocking();

		if(_clen <= _blen && _rlen <= _blen && _cols <= _blen && _rows <= _blen) {
			// single block
			OOCInstructionUtils.submitAdmittedOOCTasks(_in, _out,
				value -> new IndexedMatrixValue(value.getIndexes(),
					((MatrixBlock) value.getValue()).reshape((int) _rows, (int) _cols, _byRow)),
				_allowance, getContext()).thenRun(this::onComplete);
			return;
		}

		if(_byRow) {
			if(_clen % _blen == 0 && _cols % _blen == 0) {
				// no need to split singleRowBlocks
				if(_rows == 1) {
					// result is one single row
					submitSingleRowColTask();
				}
				else {
					CompletableFuture<Void> f = splitIntoTable();
					f.thenRun(() -> OOCInstructionUtils.submitOOCTask(this::reshapeFullColBlocks, getContext()));
				}
			}
			else {
				CompletableFuture<Void> f = splitIntoTable();
				f.thenRun(() -> OOCInstructionUtils.submitOOCTask(this::reshapePartialColBlocks, getContext()));
			}
		}
		else {
			if(_rlen % _blen == 0 && _rows % _blen == 0) {
				// no need to split singleColBlocks
				if(_cols == 1) {
					// result is one single col
					submitSingleRowColTask();
				}
				else {
					CompletableFuture<Void> f = splitIntoTable();
					f.thenRun(() -> OOCInstructionUtils.submitOOCTask(this::reshapeFullRowBlocks, getContext()));
				}
			}
			else {
				CompletableFuture<Void> f = splitIntoTable();
				f.thenRun(() -> OOCInstructionUtils.submitOOCTask(this::reshapePartialRowBlocks, getContext()));
			}
		}
	}

	private CompletableFuture<Void> splitIntoTable() {

		AllocatedOOCStream<IndexedMatrixValue> allocated = new AllocatedOOCStream<>(_in, _allowance,
			ignored -> _blen * _sliceBytes);

		return OOCInstructionUtils.submitOOCTasks(allocated, callback -> {
			try(ReservationBudget budget = AllocatedOOCStream.detachBudget(callback)) {
				if(budget == null)
					throw new DMLRuntimeException("Missing admitted output budget");

				IndexedMatrixValue imv = callback.get();
				MatrixBlock blk = (MatrixBlock) imv.getValue();
				long r = imv.getIndexes().getRowIndex();
				long c = imv.getIndexes().getColumnIndex();
				long rIdx;
				long cIdx;

				int n = _byRow ? blk.getNumRows() : blk.getNumColumns();
				for(int i = 0; i < n; i++) {
					MatrixBlock slice;
					if(_byRow) {
						slice = blk.slice(i, i);
						rIdx = (r - 1) * _blen + i + 1;
						cIdx = c;
					}
					else {
						slice = blk.slice(0, blk.getNumRows() - 1, i, i);
						cIdx = (c - 1) * _blen + i + 1;
						rIdx = r;
					}

					long targetIdx = _byRow ? (rIdx - 1) * _numColBlocksIn + c - 1 : (cIdx - 1) * _numRowBlocksIn + r - 1;
					IndexedMatrixValue sliceImv = new IndexedMatrixValue(new MatrixIndexes(rIdx, cIdx), slice);
					budget.reserveBlocking(_sliceBytes);
					_table.put((int) targetIdx, new ManagedPayload<>(sliceImv, _sliceBytes, budget));
				}
			}
			catch(IllegalStateException e) {
				throw new DMLRuntimeException(e);
			}
		}, getContext());
	}

	private void submitSingleRowColTask() {

		// one input block is split into blen output blocks
		AllocatedOOCStream<IndexedMatrixValue> allocated = new AllocatedOOCStream<>(_in, _allowance,
			ignored -> _blen * _blockBytesOut);

		OOCInstructionUtils.submitOOCTasks(allocated, callback -> {
			try(ReservationBudget budget = AllocatedOOCStream.detachBudget(callback)) {
				if(budget == null)
					throw new DMLRuntimeException("Missing admitted output budget");

				IndexedMatrixValue imv = callback.get();
				MatrixBlock blk = (MatrixBlock) imv.getValue();
				long r = imv.getIndexes().getRowIndex();
				long c = imv.getIndexes().getColumnIndex();
				long rIdx;
				long cIdx;

				int n = _byRow ? blk.getNumRows() : blk.getNumColumns();
				for(int i = 0; i < n; i++) {
					MatrixBlock slice;
					if(_byRow) {
						// split and adjust idx
						slice = blk.slice(i, i);
						// total row, 1 based
						rIdx = (r - 1) * _blen + i + 1;
						// all in single row
						cIdx = (rIdx - 1) * _numColBlocksIn + c;
						rIdx = 1;
					}
					else {
						// split and adjust idx
						slice = blk.slice(0, blk.getNumRows() - 1, i, i);
						cIdx = (c - 1) * _blen + i + 1;
						// all in single col
						rIdx = (cIdx - 1) * _numRowBlocksIn + r;
						cIdx = 1;
					}

					IndexedMatrixValue sliceImv = new IndexedMatrixValue(new MatrixIndexes(rIdx, cIdx), slice);
					OOCUtils.enqueueExact(_out, sliceImv, budget, false);
				}
			}
			catch(IllegalStateException e) {
				throw new DMLRuntimeException(e);
			}
		}, getContext()).thenRun(this::onComplete).thenRun(_out::closeInput).exceptionally(error -> {
			_out.propagateFailure(DMLRuntimeException.of(error));
			return null;
		});
	}

	private void reshapeFullColBlocks() {

		List<OOCFuture<StoreLease<IndexedMatrixValue>>> futures = new ArrayList<>();
		long outputBytes = _numRowsBlockOut * _sliceBytes + _blockBytesOut;
		ReservationBudget budget = null;

		// totalRowIdx corresponds to index of row block when all aligned in one row
		// br * numColBlocksOut * blen + b + r * numColBlocksOut;
		// with numColBlocksOut * blen = cols
		long totalIdx = -_cols - 1 - _numColBlocksOut;

		try {
			// iterate through rows of output blocks
			for(int br = 0; br < _numRowBlocksOut; br++) {
				totalIdx += _cols;
				long tmp = totalIdx;
				int localRows = (br == _numRowBlocksOut - 1 && _rows % _blen != 0) ? (int) _rows % _blen : _blen;
				// for each block in row
				for(int b = 0; b < _numColBlocksOut; b++) {
					totalIdx += 1;
					long tmp2 = totalIdx;
					budget = OOCUtils.reserveBudget(_allowance, outputBytes);
					// for each row in block
					for(int r = 0; r < _blen && r < localRows; r++) {
						totalIdx += _numColBlocksOut;
						OOCFuture<StoreLease<IndexedMatrixValue>> rowFuture = _table.take((int) totalIdx, budget);
						futures.add(rowFuture);
					}
					totalIdx = tmp2;
					OOCFuture<List<StoreLease<IndexedMatrixValue>>> future = OOCFuture.allOf(futures, StoreLease::close);
					MatrixIndexes idx = new MatrixIndexes(br + 1, b + 1);

					ReservationBudget finalBudget = budget;
					future.whenComplete((leases, error) -> {
						MatrixBlock block = new MatrixBlock(localRows, _blen, false);
						for(int r = 0; r < leases.size(); r++) {
							StoreLease<IndexedMatrixValue> lease = leases.get(r);
							MatrixBlock row = (MatrixBlock) lease.value().getValue();
							block.setRow(r, row.getDenseBlockValues());
							lease.close();
						}
						block.recomputeNonZeros();
						OOCUtils.enqueueExact(_out, new IndexedMatrixValue(idx, block), finalBudget, true);
						futures.clear();
					});
					budget = null;
				}
				totalIdx = tmp;
			}
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			closeResourcesAndComplete(budget);
		}
	}

	private void reshapeFullRowBlocks() {

		List<OOCFuture<StoreLease<IndexedMatrixValue>>> futures = new ArrayList<>();
		long outputBytes = _numColsBlockOut * _sliceBytes + _blockBytesOut;
		ReservationBudget budget = null;

		// totalColIdx corresponds to index of col block when all aligned in one col
		// bc * numRowBlocksOut * blen + b + c * numRowBlocksOut;
		// with numRowBlocksOut * blen = rows
		long totalIdx = -_rows - 1 - _numRowBlocksOut;

		try {
			// iterate through cols of output blocks
			for(int bc = 0; bc < _numColBlocksOut; bc++) {
				totalIdx += _rows;
				long tmp = totalIdx;
				int localCols = (bc == _numColBlocksOut - 1 && _cols % _blen != 0) ? (int) _cols % _blen : _blen;
				// for each block in col
				for(int b = 0; b < _numRowBlocksOut; b++) {
					totalIdx += 1;
					long tmp2 = totalIdx;
					budget = OOCUtils.reserveBudget(_allowance, outputBytes);
					// for each col in block
					for(int c = 0; c < _blen && c < localCols; c++) {
						totalIdx += _numRowBlocksOut;
						OOCFuture<StoreLease<IndexedMatrixValue>> colFuture = _table.take((int) totalIdx, budget);
						futures.add(colFuture);
					}
					totalIdx = tmp2;
					OOCFuture<List<StoreLease<IndexedMatrixValue>>> future = OOCFuture.allOf(futures, StoreLease::close);
					MatrixIndexes idx = new MatrixIndexes(b + 1, bc + 1);

					ReservationBudget finalBudget = budget;
					future.whenComplete((leases, error) -> {
						MatrixBlock block = new MatrixBlock(_blen, localCols, false);
						block.allocateDenseBlock();
						for(int c = 0; c < leases.size(); c++) {
							StoreLease<IndexedMatrixValue> lease = leases.get(c);
							MatrixBlock col = (MatrixBlock) lease.value().getValue();
							block.getDenseBlock().set(0, _blen, c, c + 1, col.getDenseBlock());
							lease.close();
						}
						block.recomputeNonZeros();
						OOCUtils.enqueueExact(_out, new IndexedMatrixValue(idx, block), finalBudget, true);
						futures.clear();
					});
					budget = null;
				}
				totalIdx = tmp;
			}
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			closeResourcesAndComplete(budget);
		}
	}

	private void reshapePartialColBlocks() {

		long numNeededRowsIn = 2 + (long) Math
			.ceil((((double) _numRowsBlockOut * _numColsBlockOut) / _clen) * _numColBlocksIn * _numColBlocksOut);
		long outputBytes = numNeededRowsIn * _sliceBytes + _numColBlocksOut * _blockBytesOut;

		ReservationBudget budget = null;
		int br = 0;

		try {
			List<OOCFuture<StoreLease<IndexedMatrixValue>>> futures = new ArrayList<>();
			// new row of output blocks
			budget = OOCUtils.reserveBudget(_allowance, outputBytes);
			int missing = getNumMissingRowSlices(1, br, 0);
			int startJ = 1;
			final int[] offset = {0};

			// iterate through input rows and add to row of output blocks
			for(int i = 1; i <= _rlen; i++) {
				for(int j = 1; j <= _numColBlocksIn; j++) {
					int totalIdx = (i - 1) * _numColBlocksIn + j - 1;
					OOCFuture<StoreLease<IndexedMatrixValue>> blkFuture = _table.take(totalIdx, budget);
					futures.add(blkFuture);

					if(futures.size() < missing)
						continue;

					final ReservationBudget finalBudget = budget;
					final int finalJ = startJ;
					final int finalBr = br;

					OOCFuture<List<StoreLease<IndexedMatrixValue>>> future = OOCFuture.allOf(futures, StoreLease::close);
					future.whenComplete((leases, error) -> {
						int localJ = finalJ;
						int offsetIn = offset[0];

						int bc = 0;
						int r = 0;
						int offsetOut = 0;
						MatrixBlock[] outputBlockRow = allocateSliceBlocks(finalBr);
						int localColsOut = (_cols > _blen) ? _blen : (int) _cols;

						for(int k = 0; k < leases.size(); k++) {
							StoreLease<IndexedMatrixValue> lease = leases.get(k);
							IndexedMatrixValue slice = lease.value();
							MatrixBlock sliceVal = (MatrixBlock) slice.getValue();

							int localColsIn = (localJ == _numColBlocksIn && _clen % _blen != 0) ? (int) _clen % _blen : _blen;
							while(offsetIn < localColsIn) {
								// until input row fully processed
								int remIn = localColsIn - offsetIn;
								int remOut = localColsOut - offsetOut;
								if(remIn < remOut) {
									// next input
									setOutputEntries(sliceVal, outputBlockRow[bc], r, offsetIn, offsetOut, remIn);
									offsetIn += remIn;
									offsetOut += remIn;
									continue;
								}
								else if(remIn == remOut) {
									// next input and next row
									setOutputEntries(sliceVal, outputBlockRow[bc], r, offsetIn, offsetOut, remIn);
									offsetIn += remIn;
								}
								else {
									// next row
									setOutputEntries(sliceVal, outputBlockRow[bc], r, offsetIn, offsetOut, remOut);
									offsetIn += remOut;
								}
								bc++;
								offsetOut = 0;
								if(bc == _numColBlocksOut) {
									// next row
									r++;
									if(r == outputBlockRow[0].getNumRows()) {
										offset[0] = offsetIn == localColsIn ? 0 : offsetIn;
										break;
									}
									bc = 0;
								}
								localColsOut = (bc == _numColBlocksOut - 1 && _cols % _blen != 0) ? (int) _cols % _blen : _blen;
							}
							localJ++;
							if(localJ == _numColBlocksIn + 1)
								localJ = 1;

							lease.close();
							offsetIn = 0;

							if(k == leases.size() - 1 && offset[0] != 0) {
								// put current slice back into table, to be able to reserve new budget
								finalBudget.reserveBlocking(_sliceBytes);
								_table.put(totalIdx, new ManagedPayload<>(slice, _sliceBytes, finalBudget));
							}
						}

						// enqueue filled output blocks and allocate new ones
						for(int b = 0; b < outputBlockRow.length; b++) {
							outputBlockRow[b].recomputeNonZeros();
							OOCUtils.enqueueExact(_out,
								new IndexedMatrixValue(new MatrixIndexes(finalBr + 1, b + 1), outputBlockRow[b]),
								finalBudget, false);
						}
					});

					budget.close();
					futures.clear();

					if(br == _numRowBlocksOut - 1)
						break;

					// new block row
					br++;
					budget = OOCUtils.reserveBudget(_allowance, outputBytes);

					if(offset[0] != 0) {
						// get slice back from table
						blkFuture = _table.take(totalIdx, budget);
						futures.add(blkFuture);
						startJ = j;
					}
					else {
						startJ = (j == _numColBlocksIn) ? 1 : j + 1;
					}

					missing = getNumMissingRowSlices(startJ, br, offset[0]);
				}
			}
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			closeResourcesAndComplete(budget);
		}
	}

	private int getNumMissingRowSlices(int j, int br, int offsetIn) {
		long localColsIn = (j == _numColBlocksIn && _clen % _blen != 0) ? _clen % _blen : _blen;
		long localRowsOut = (br == _numRowBlocksOut - 1 && _rows % _blen != 0) ? _rows % _blen : _blen;

		long totalIdx = _cols * localRowsOut;
		int cnt = 0;

		if(offsetIn != 0) {
			// reuse table entry
			totalIdx -= (localColsIn - offsetIn);
			cnt++;
		}

		int numRows = (int) Math.floor((double) totalIdx / _clen);
		cnt += numRows * _numColBlocksIn;
		totalIdx -= numRows * _clen;

		long numBlenSlices = Math.max(0, _numColBlocksIn - (j - 1));
		long restRow = numBlenSlices * _blen + localColsIn;
		if(totalIdx - restRow > 0) {
			totalIdx -= restRow;
			cnt += _numColBlocksIn - j;
		}
		cnt += (int) Math.ceil((double) totalIdx / _blen);

		return cnt;
	}

	private void reshapePartialRowBlocks() {

		long numNeededColsIn = 2 + (long) Math
			.ceil((((double) _numRowsBlockOut * _numColsBlockOut) / _rlen) * _numRowBlocksIn * _numRowBlocksOut);
		long outputBytes = numNeededColsIn * _sliceBytes + _numRowBlocksOut * _blockBytesOut;

		ReservationBudget budget = null;
		int bc = 0;

		try {
			List<OOCFuture<StoreLease<IndexedMatrixValue>>> futures = new ArrayList<>();
			// new col of output blocks
			budget = OOCUtils.reserveBudget(_allowance, outputBytes);
			int missing = getNumMissingColSlices(1, bc, 0);
			int startI = 1;
			final int[] offset = {0};

			// iterate through input cols and add to col of output blocks
			for(int j = 1; j <= _clen; j++) {
				for(int i = 1; i <= _numRowBlocksIn; i++) {
					int totalIdx = (j - 1) * _numRowBlocksIn + i - 1;
					OOCFuture<StoreLease<IndexedMatrixValue>> blkFuture = _table.take(totalIdx, budget);
					futures.add(blkFuture);

					if(futures.size() < missing)
						continue;

					final ReservationBudget finalBudget = budget;
					final int finalI = startI;
					final int finalBc = bc;

					OOCFuture<List<StoreLease<IndexedMatrixValue>>> future = OOCFuture.allOf(futures, StoreLease::close);
					future.whenComplete((leases, error) -> {
						int localI = finalI;
						int offsetIn = offset[0];

						int br = 0;
						int c = 0;
						int offsetOut = 0;

						MatrixBlock[] outputBlockCol = allocateSliceBlocks(finalBc);
						int localRowsOut = (_rows > _blen) ? _blen : (int) _rows;

						for(int k = 0; k < leases.size(); k++) {
							StoreLease<IndexedMatrixValue> lease = leases.get(k);
							IndexedMatrixValue slice = lease.value();
							MatrixBlock sliceVal = (MatrixBlock) slice.getValue();

							int localRowsIn = (localI == _numRowBlocksIn && _rlen % _blen != 0) ? (int) _rlen % _blen : _blen;
							while(offsetIn < localRowsIn) {
								// until input col fully processed
								int remIn = localRowsIn - offsetIn;
								int remOut = localRowsOut - offsetOut;
								if(remIn < remOut) {
									// next input
									setOutputEntries(sliceVal, outputBlockCol[br], c, offsetIn, offsetOut, remIn);
									offsetIn += remIn;
									offsetOut += remIn;
									continue;
								}
								else if(remIn == remOut) {
									// next input and next col
									setOutputEntries(sliceVal, outputBlockCol[br], c, offsetIn, offsetOut, remIn);
									offsetIn += remIn;
								}
								else {
									// next col
									setOutputEntries(sliceVal, outputBlockCol[br], c, offsetIn, offsetOut, remOut);
									offsetIn += remOut;
								}
								br++;
								offsetOut = 0;
								if(br == _numRowBlocksOut) {
									// next col
									c++;
									if(c == outputBlockCol[0].getNumColumns()) {
										offset[0] = offsetIn == localRowsIn ? 0 : offsetIn;
										break;
									}
									br = 0;
								}
								localRowsOut = (br == _numRowBlocksOut - 1 && _rows % _blen != 0) ? (int) _rows % _blen : _blen;
							}
							localI++;
							if(localI == _numRowBlocksIn + 1)
								localI = 1;

							lease.close();
							offsetIn = 0;

							if(k == leases.size() - 1 && offset[0] != 0) {
								// put current slice back into table, to be able to reserve new budget
								finalBudget.reserveBlocking(_sliceBytes);
								_table.put(totalIdx, new ManagedPayload<>(slice, _sliceBytes, finalBudget));
							}
						}

						// enqueue filled output blocks and allocate new ones
						for(int b = 0; b < outputBlockCol.length; b++) {
							outputBlockCol[b].recomputeNonZeros();
							OOCUtils.enqueueExact(_out,
								new IndexedMatrixValue(new MatrixIndexes(b + 1, finalBc + 1), outputBlockCol[b]),
								finalBudget, false);
						}
					});

					budget.close();
					futures.clear();

					if(bc == _numColBlocksOut - 1)
						break;

					// new block row
					bc++;
					budget = OOCUtils.reserveBudget(_allowance, outputBytes);

					if(offset[0] != 0) {
						// get slice back from table
						blkFuture = _table.take(totalIdx, budget);
						futures.add(blkFuture);
						startI = i;
					}
					else {
						startI = (i == _numRowBlocksIn) ? 1 : i + 1;
					}
					missing = getNumMissingColSlices(startI, bc, offset[0]);
				}
			}
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			closeResourcesAndComplete(budget);
		}
	}

	private int getNumMissingColSlices(int i, int bc, int offsetIn) {
		long localRowsIn = (i == _numRowBlocksIn && _rlen % _blen != 0) ? _rlen % _blen : _blen;
		long localColsOut = (bc == _numColBlocksOut - 1 && _cols % _blen != 0) ? _cols % _blen : _blen;

		long totalIdx = _rows * localColsOut;
		int cnt = 0;

		if(offsetIn != 0) {
			// reuse table entry
			totalIdx -= (localRowsIn - offsetIn);
			cnt++;
		}

		int numCols = (int) Math.floor((double) totalIdx / _rlen);
		cnt += numCols * _numRowBlocksIn;
		totalIdx -= numCols * _rlen;

		long numBlenSlices = Math.max(0, _numRowBlocksIn - (i - 1));
		long restCol = numBlenSlices * _blen + localRowsIn;
		if(totalIdx - restCol > 0) {
			totalIdx -= restCol;
			cnt += _numRowBlocksIn - i;
		}
		cnt += (int) Math.ceil((double) totalIdx / _blen);

		return cnt;
	}

	private MatrixBlock[] allocateSliceBlocks(int idx) {
		int n = _byRow ? _numColBlocksOut : _numRowBlocksOut;
		MatrixBlock[] res = new MatrixBlock[n];

		// full inner blocks, adjust for outer blocks
		int localRows = ((!_byRow || idx == _numRowBlocksOut - 1) && _rows % _blen != 0) ? (int) _rows % _blen : _blen;
		int localCols = ((_byRow || idx == _numColBlocksOut - 1) && _cols % _blen != 0) ? (int) _cols % _blen : _blen;

		for(int k = 0; k < n - 1; k++) {
			res[k] = _byRow ? new MatrixBlock(localRows, _blen, false) : new MatrixBlock(_blen, localCols, false);
			res[k].allocateDenseBlock();
		}
		res[n - 1] = new MatrixBlock(localRows, localCols, false);
		res[n - 1].allocateDenseBlock();

		return res;
	}

	private void setOutputEntries(MatrixBlock src, MatrixBlock dest, int idx, int srcOffset, int destOffset, int length) {
		if(_byRow)
			((DenseBlockFP64) dest.getDenseBlock()).setPartialRow(src.getDenseBlock(), idx, srcOffset, destOffset, length);
		else
			((DenseBlockFP64) dest.getDenseBlock()).setPartialCol(src.getDenseBlock(), idx, srcOffset, destOffset, length);
	}

	private void closeResourcesAndComplete(ReservationBudget budget) {
		if(budget != null) {
			budget.close();
		}
		try {
			_table.close();
			onComplete();
		}
		finally {
			_out.closeInput();
		}
	}

	private void initExecution() {
		DataCharacteristics inputDc = _input.getDataCharacteristics();
		if(inputDc == null || !inputDc.dimsKnown() || inputDc.getBlocksize() <= 0)
			throw new DMLRuntimeException(
				"Reshape OOC reduction requires known input dimensions and block size.");

		_in = getInputReadStream(0);
		_out = _output.getWriteStream();
		getContext().addOutStream(_out);

		_rlen = inputDc.getRows();
		_clen = inputDc.getCols();
		_blen = Math.toIntExact(inputDc.getBlocksize());
	}

	private void initBlocking() {
		DataCharacteristics inputDc = _input.getDataCharacteristics();

		_numColBlocksIn = Math.toIntExact(inputDc.getNumColBlocks());
		_numRowBlocksIn = Math.toIntExact(inputDc.getNumRowBlocks());

		_numColBlocksOut = (int) Math.ceil((double) _cols / _blen);
		_numRowBlocksOut = (int) Math.ceil((double) _rows / _blen);

		_numRowsBlockOut = Math.min(_rows, _blen);
		_numColsBlockOut = Math.min(_cols, _blen);

		_blockBytesOut = OOCUtils.estimateOutputTileBytes(_out.getDataCharacteristics());
		_sliceBytes = (OOCUtils.estimateFullTileBytes(_in.getDataCharacteristics()) +
			(_blen - 1) * MatrixBlock.getHeaderSize()) / _blen;

		_table = new StateTable<>(OOCCacheManager.getGlobalCache(), CachingStream._streamSeq.getNextID());
	}
}
