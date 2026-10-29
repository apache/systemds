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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class ReshapeOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _input;
	private final OOCStreamable<IndexedMatrixValue> _output;
	private final boolean _byRow;
	private final long _rows;
	private final long _cols;
	private final AtomicInteger _pending;

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
		_pending = new AtomicInteger(1);
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

		if(_clen <= _blen && _rlen <= _blen && _cols <= _blen && _rows <= _blen) {
			// single block
			OOCInstructionUtils.submitAdmittedOOCTasks(_in, _out,
				value -> new IndexedMatrixValue(value.getIndexes(),
					((MatrixBlock) value.getValue()).reshape((int) _rows, (int) _cols, _byRow)),
				_allowance, getContext()).thenRun(this::onComplete);
			return;
		}

		initBlocking();

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
		return OOCInstructionUtils.submitOOCTasks(allocated, this::splitBlockIntoTable, getContext());
	}

	private void splitBlockIntoTable(OOCStream.QueueCallback<IndexedMatrixValue> callback) {
		try(ReservationBudget budget = AllocatedOOCStream.detachBudget(callback)) {
			if(budget == null)
				throw new DMLRuntimeException("Missing admitted output budget");

			IndexedMatrixValue imv = callback.get();
			MatrixBlock blk = (MatrixBlock) imv.getValue();
			long r = imv.getIndexes().getRowIndex();
			long c = imv.getIndexes().getColumnIndex();

			int n = _byRow ? blk.getNumRows() : blk.getNumColumns();
			for(int i = 0; i < n; i++)
				putSliceIntoTable(blk, r, c, i, budget);
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
	}

	private void putSliceIntoTable(MatrixBlock blk, long r, long c, int i, ReservationBudget budget) {
		// split block into individual row or column slices and adapt indices
		MatrixBlock slice = createSlice(blk, i);
		long rIdx = _byRow ? (r - 1) * _blen + i + 1 : r;
		long cIdx = _byRow ? c : (c - 1) * _blen + i + 1;
		// compute position in row-wise or column-wise linearized order
		long tableIdx = _byRow ? (rIdx - 1) * _numColBlocksIn + c - 1 : (cIdx - 1) * _numRowBlocksIn + r - 1;

		IndexedMatrixValue sliceImv = new IndexedMatrixValue(new MatrixIndexes(rIdx, cIdx), slice);
		budget.reserveBlocking(_sliceBytes);
		_table.put((int) tableIdx, new ManagedPayload<>(sliceImv, _sliceBytes, budget));
	}

	private void submitSingleRowColTask() {
		// one input block is split into blen output blocks
		AllocatedOOCStream<IndexedMatrixValue> allocated = new AllocatedOOCStream<>(_in, _allowance,
			ignored -> _blen * _blockBytesOut);

		OOCInstructionUtils.submitOOCTasks(allocated, this::processSingleRowColBlock, getContext())
			.thenRun(this::onComplete).thenRun(_out::closeInput).exceptionally(error -> {
				_out.propagateFailure(DMLRuntimeException.of(error));
				return null;
			});
	}

	private void processSingleRowColBlock(OOCStream.QueueCallback<IndexedMatrixValue> callback) {
		try(ReservationBudget budget = AllocatedOOCStream.detachBudget(callback)) {
			if(budget == null)
				throw new DMLRuntimeException("Missing admitted output budget");

			IndexedMatrixValue imv = callback.get();
			MatrixBlock blk = (MatrixBlock) imv.getValue();
			long r = imv.getIndexes().getRowIndex();
			long c = imv.getIndexes().getColumnIndex();

			int n = _byRow ? blk.getNumRows() : blk.getNumColumns();
			for(int i = 0; i < n; i++)
				enqueueSlice(blk, r, c, i, budget);
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
	}

	private void enqueueSlice(MatrixBlock blk, long r, long c, int i, ReservationBudget budget) {
		MatrixBlock slice = createSlice(blk, i);
		// compute output indices in row-wise or column-wise linearized order
		long rIdx = _byRow ? 1 : ((c - 1) * _blen + i) * _numRowBlocksIn + r;
		long cIdx = _byRow ? ((r - 1) * _blen + i) * _numColBlocksIn + c : 1;
		IndexedMatrixValue sliceImv = new IndexedMatrixValue(new MatrixIndexes(rIdx, cIdx), slice);
		OOCUtils.enqueueExact(_out, sliceImv, budget, false);
	}

	private MatrixBlock createSlice(MatrixBlock block, int i) {
		return _byRow ? block.slice(i, i) : block.slice(0, block.getNumRows() - 1, i, i);
	}

	private void reshapeFullColBlocks() {

		List<OOCFuture<StoreLease<IndexedMatrixValue>>> futures = new ArrayList<>();
		long outputBytes = _numRowsBlockOut * _sliceBytes + _blockBytesOut;
		ReservationBudget budget = null;

		// iterate the output blocks in linearized row-major order
		// tableIdx is the state table index of the corresponding row slice
		// tableIdx(br,b,r) = br * cols + b + r * numColBlocksOut;
		// where cols = numColBlocksOut * blen

		long tableIdx = 0;
		try {
			// iterate through rows of output blocks
			for(int br = 0; br < _numRowBlocksOut; br++) {
				long rowStart = tableIdx;
				int localRows = (br == _numRowBlocksOut - 1 && _rows % _blen != 0) ? (int) _rows % _blen : _blen;

				// for each block in row
				for(int b = 0; b < _numColBlocksOut; b++) {
					long blockStart = tableIdx;
					budget = OOCUtils.reserveBudget(_allowance, outputBytes);

					// for each row in block
					for(int r = 0; r < _blen && r < localRows; r++) {
						OOCFuture<StoreLease<IndexedMatrixValue>> rowFuture = _table.take((int) tableIdx, budget);
						futures.add(rowFuture);
						tableIdx += _numColBlocksOut;
					}
					tableIdx = blockStart;
					tableIdx += 1;

					OOCFuture<List<StoreLease<IndexedMatrixValue>>> future = OOCFuture.allOf(futures, StoreLease::close);
					MatrixIndexes idx = new MatrixIndexes(br + 1, b + 1);
					ReservationBudget finalBudget = budget;

					process(future, finalBudget, leases -> processFullColBlock(idx, localRows, leases, finalBudget));
					futures.clear();
					budget = null;
				}
				tableIdx = rowStart;
				tableIdx += _cols;
			}
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			cleanup(budget);
		}
	}

	private void processFullColBlock(MatrixIndexes idx, int localRows, List<StoreLease<IndexedMatrixValue>> leases,
		ReservationBudget budget) {

		MatrixBlock block = new MatrixBlock(localRows, _blen, false);

		for(int r = 0; r < leases.size(); r++) {
			StoreLease<IndexedMatrixValue> lease = leases.get(r);
			MatrixBlock row = (MatrixBlock) lease.value().getValue();
			if(row.isInSparseFormat()) row.sparseToDense();
			if(row.getDenseBlock() == null)
				row.allocateDenseBlock(false);
			block.setRow(r, row.getDenseBlockValues());
			lease.close();
		}

		block.recomputeNonZeros();
		block.examSparsity();
		OOCUtils.enqueueExact(_out, new IndexedMatrixValue(idx, block), budget, true);
	}

	private void reshapeFullRowBlocks() {

		List<OOCFuture<StoreLease<IndexedMatrixValue>>> futures = new ArrayList<>();
		long outputBytes = _numColsBlockOut * _sliceBytes + _blockBytesOut;
		ReservationBudget budget = null;

		// iterate the output blocks in linearized column-major order
		// tableIdx is the state table index of the corresponding column slice
		// tableIdx(bc,b,c) = bc * rows + b + c * numRowBlocksOut,
		// where rows = numRowBlocksOut * blen

		long tableIdx = 0;
		try {
			// iterate through cols of output blocks
			for(int bc = 0; bc < _numColBlocksOut; bc++) {
				long colStart = tableIdx;
				int localCols = (bc == _numColBlocksOut - 1 && _cols % _blen != 0) ? (int) _cols % _blen : _blen;

				// for each block in col
				for(int b = 0; b < _numRowBlocksOut; b++) {
					long blockStart = tableIdx;
					budget = OOCUtils.reserveBudget(_allowance, outputBytes);

					// for each col in block
					for(int c = 0; c < _blen && c < localCols; c++) {
						OOCFuture<StoreLease<IndexedMatrixValue>> colFuture = _table.take((int) tableIdx, budget);
						futures.add(colFuture);
						tableIdx += _numRowBlocksOut;
					}
					tableIdx = blockStart;
					tableIdx += 1;

					OOCFuture<List<StoreLease<IndexedMatrixValue>>> future = OOCFuture.allOf(futures, StoreLease::close);
					MatrixIndexes idx = new MatrixIndexes(b + 1, bc + 1);
					ReservationBudget finalBudget = budget;

					process(future, finalBudget, leases -> processFullRowBlock(idx, localCols, leases, finalBudget));
					futures.clear();
					budget = null;
				}
				tableIdx = colStart;
				tableIdx += _rows;
			}
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			cleanup(budget);
		}
	}

	private void processFullRowBlock(MatrixIndexes idx, int localCols, List<StoreLease<IndexedMatrixValue>> leases,
		ReservationBudget budget) {

		MatrixBlock block = new MatrixBlock(_blen, localCols, false);
		block.allocateDenseBlock();

		for(int c = 0; c < leases.size(); c++) {
			StoreLease<IndexedMatrixValue> lease = leases.get(c);
			MatrixBlock col = (MatrixBlock) lease.value().getValue();
			if(col.isInSparseFormat()) col.sparseToDense();
			if(col.getDenseBlock() == null)
				col.allocateDenseBlock(false);
			block.getDenseBlock().set(0, _blen, c, c + 1, col.getDenseBlock());
			lease.close();
		}

		block.recomputeNonZeros();
		block.examSparsity();
		OOCUtils.enqueueExact(_out, new IndexedMatrixValue(idx, block), budget, true);
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
			int offset = 0;

			// iterate through input rows and add to row of output blocks
			for(int i = 1; i <= _rlen; i++) {
				for(int j = 1; j <= _numColBlocksIn; j++) {
					int tableIdx = (i - 1) * _numColBlocksIn + j - 1;
					OOCFuture<StoreLease<IndexedMatrixValue>> blkFuture = _table.take(tableIdx, budget);
					futures.add(blkFuture);

					while(futures.size() >= missing) {
						final ReservationBudget finalBudget = budget;
						final int finalJ = startJ;
						final int finalBr = br;

						OOCFuture<List<StoreLease<IndexedMatrixValue>>> future = OOCFuture.allOf(futures, StoreLease::close);
						int blockOffset = offset;
						OOCFuture<Void> processed = process(future, finalBudget,
							leases -> processPartialColBlocks(finalBr, finalJ, blockOffset, tableIdx, leases, finalBudget));

						futures.clear();
						budget = null;

						if(br == _numRowBlocksOut - 1)
							break;

						// new block row
						br++;
						offset = (int) (Math.min((long) br * _blen, _rows) * _cols % _clen % _blen);
						budget = OOCUtils.reserveBudget(_allowance, outputBytes);

						if(offset != 0) {
							// get slice back from table
							ReservationBudget nextBudget = budget;
							blkFuture = processed.thenCompose(ignored -> _table.take(tableIdx, nextBudget));
							futures.add(blkFuture);
							startJ = j;
						}
						else {
							startJ = (j == _numColBlocksIn) ? 1 : j + 1;
						}

						missing = getNumMissingRowSlices(startJ, br, offset);
					}
				}
			}
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			cleanup(budget);
		}
	}

	private void processPartialColBlocks(int br, int j, int startOffset, int tableIdx,
		List<StoreLease<IndexedMatrixValue>> leases, ReservationBudget budget) {

		int offsetIn = startOffset;
		int offsetOut = 0;
		int remainingOffset = 0;
		int bc = 0;
		int r = 0;

		MatrixBlock[] outputBlockRow = allocateSliceBlocks(br);
		int localColsOut = (_cols > _blen) ? _blen : (int) _cols;

		for(int k = 0; k < leases.size(); k++) {
			StoreLease<IndexedMatrixValue> lease = leases.get(k);
			IndexedMatrixValue slice = lease.value();
			MatrixBlock sliceVal = (MatrixBlock) slice.getValue();

			int localColsIn = (j == _numColBlocksIn && _clen % _blen != 0) ? (int) _clen % _blen : _blen;
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
						remainingOffset = offsetIn == localColsIn ? 0 : offsetIn;
						break;
					}
					bc = 0;
				}
				localColsOut = (bc == _numColBlocksOut - 1 && _cols % _blen != 0) ? (int) _cols % _blen : _blen;
			}
			j++;
			if(j == _numColBlocksIn + 1)
				j = 1;

			lease.close();
			offsetIn = 0;

			if(k == leases.size() - 1 && remainingOffset != 0) {
				// put current slice back into table, to be able to reserve new budget
				budget.reserveBlocking(_sliceBytes);
				_table.put(tableIdx, new ManagedPayload<>(slice, _sliceBytes, budget));
			}
		}

		// enqueue filled output blocks and allocate new ones
		for(int b = 0; b < outputBlockRow.length; b++) {
			outputBlockRow[b].recomputeNonZeros();
			outputBlockRow[b].examSparsity();
			OOCUtils.enqueueExact(_out, new IndexedMatrixValue(new MatrixIndexes(br + 1, b + 1), outputBlockRow[b]),
				budget, false);
		}

		budget.close();
	}

	private int getNumMissingRowSlices(int j, int br, int offsetIn) {
		long lastBlockColsIn = (_clen % _blen != 0) ? _clen % _blen : _blen;
		long localColsIn = (j == _numColBlocksIn) ? lastBlockColsIn : _blen;
		long localRowsOut = (br == _numRowBlocksOut - 1 && _rows % _blen != 0) ? _rows % _blen : _blen;

		long remaining = _cols * localRowsOut;
		int missing = 0;

		if(offsetIn != 0) {
			// reuse table entry
			remaining -= (localColsIn - offsetIn);
			missing++;
			j = (j == _numColBlocksIn) ? 1 : j + 1;
		}

		int numRows = (int) Math.floor((double) remaining / _clen);
		missing += numRows * _numColBlocksIn;
		remaining -= numRows * _clen;

		int numBlenSlices = Math.max(0, _numColBlocksIn - j);
		long restRow = (long) numBlenSlices * _blen + lastBlockColsIn;
		if(remaining >= restRow) {
			remaining -= restRow;
			missing += numBlenSlices + 1;
		}
		missing += (int) Math.ceil((double) remaining / _blen);

		return missing;
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
			int offset = 0;

			// iterate through input cols and add to col of output blocks
			for(int j = 1; j <= _clen; j++) {
				for(int i = 1; i <= _numRowBlocksIn; i++) {
					int tableIdx = (j - 1) * _numRowBlocksIn + i - 1;
					OOCFuture<StoreLease<IndexedMatrixValue>> blkFuture = _table.take(tableIdx, budget);
					futures.add(blkFuture);

					while(futures.size() >= missing) {
						final ReservationBudget finalBudget = budget;
						final int finalI = startI;
						final int finalBc = bc;

						OOCFuture<List<StoreLease<IndexedMatrixValue>>> future = OOCFuture.allOf(futures, StoreLease::close);
						int blockOffset = offset;
						OOCFuture<Void> processed = process(future, finalBudget,
							leases -> processPartialRowBlocks(finalBc, finalI, blockOffset, tableIdx, leases, finalBudget));

						futures.clear();
						budget = null;

						if(bc == _numColBlocksOut - 1)
							break;

						// new block col
						bc++;
						offset = (int) (Math.min((long) bc * _blen, _cols) * _rows % _rlen % _blen);
						budget = OOCUtils.reserveBudget(_allowance, outputBytes);

						if(offset != 0) {
							// get slice back from table
							ReservationBudget nextBudget = budget;
							blkFuture = processed.thenCompose(ignored -> _table.take(tableIdx, nextBudget));
							futures.add(blkFuture);
							startI = i;
						}
						else {
							startI = (i == _numRowBlocksIn) ? 1 : i + 1;
						}
						missing = getNumMissingColSlices(startI, bc, offset);
					}
				}
			}
		}
		catch(IllegalStateException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			cleanup(budget);
		}
	}

	private void processPartialRowBlocks(int bc, int i, int startOffset, int tableIdx,
		List<StoreLease<IndexedMatrixValue>> leases, ReservationBudget budget) {

		int offsetIn = startOffset;
		int offsetOut = 0;
		int remainingOffset = 0;
		int br = 0;
		int c = 0;

		MatrixBlock[] outputBlockCol = allocateSliceBlocks(bc);
		int localRowsOut = (_rows > _blen) ? _blen : (int) _rows;

		for(int k = 0; k < leases.size(); k++) {
			StoreLease<IndexedMatrixValue> lease = leases.get(k);
			IndexedMatrixValue slice = lease.value();
			MatrixBlock sliceVal = (MatrixBlock) slice.getValue();

			int localRowsIn = (i == _numRowBlocksIn && _rlen % _blen != 0) ? (int) _rlen % _blen : _blen;
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
						remainingOffset = offsetIn == localRowsIn ? 0 : offsetIn;
						break;
					}
					br = 0;
				}
				localRowsOut = (br == _numRowBlocksOut - 1 && _rows % _blen != 0) ? (int) _rows % _blen : _blen;
			}
			i++;
			if(i == _numRowBlocksIn + 1)
				i = 1;

			lease.close();
			offsetIn = 0;

			if(k == leases.size() - 1 && remainingOffset != 0) {
				// put current slice back into table, to be able to reserve new budget
				budget.reserveBlocking(_sliceBytes);
				_table.put(tableIdx, new ManagedPayload<>(slice, _sliceBytes, budget));
			}
		}

		// enqueue filled output blocks and allocate new ones
		for(int b = 0; b < outputBlockCol.length; b++) {
			outputBlockCol[b].recomputeNonZeros();
			outputBlockCol[b].examSparsity();
			OOCUtils.enqueueExact(_out, new IndexedMatrixValue(new MatrixIndexes(b + 1, bc + 1), outputBlockCol[b]),
				budget, false);
		}

		budget.close();
	}

	private int getNumMissingColSlices(int i, int bc, int offsetIn) {
		long lastBlockRowsIn = (_rlen % _blen != 0) ? _rlen % _blen : _blen;
		long localRowsIn = (i == _numRowBlocksIn) ? lastBlockRowsIn : _blen;
		long localColsOut = (bc == _numColBlocksOut - 1 && _cols % _blen != 0) ? _cols % _blen : _blen;

		long remaining = _rows * localColsOut;
		int missing = 0;

		if(offsetIn != 0) {
			// reuse table entry
			remaining -= (localRowsIn - offsetIn);
			missing++;
			i = (i == _numRowBlocksIn) ? 1 : i + 1;
		}

		int numCols = (int) Math.floor((double) remaining / _rlen);
		missing += numCols * _numRowBlocksIn;
		remaining -= numCols * _rlen;

		int numBlenSlices = Math.max(0, _numRowBlocksIn - i);
		long restCol = (long) numBlenSlices * _blen + lastBlockRowsIn;
		if(remaining >= restCol) {
			remaining -= restCol;
			missing += numBlenSlices + 1;
		}
		missing += (int) Math.ceil((double) remaining / _blen);

		return missing;
	}

	private MatrixBlock[] allocateSliceBlocks(int blockGroupIdx) {
		int n = _byRow ? _numColBlocksOut : _numRowBlocksOut;
		MatrixBlock[] res = new MatrixBlock[n];

		// allocate output blocks for the current block row/column
		// adjust dimensions for the last block
		int localRows = ((!_byRow || blockGroupIdx == _numRowBlocksOut - 1) && _rows % _blen != 0) ?
			(int) _rows % _blen : _blen;
		int localCols = ((_byRow || blockGroupIdx == _numColBlocksOut - 1) && _cols % _blen != 0) ?
			(int) _cols % _blen : _blen;

		for(int k = 0; k < n - 1; k++) {
			res[k] = _byRow ? new MatrixBlock(localRows, _blen, false) : new MatrixBlock(_blen, localCols, false);
			res[k].allocateDenseBlock();
		}
		res[n - 1] = new MatrixBlock(localRows, localCols, false);
		res[n - 1].allocateDenseBlock();

		return res;
	}

	private void setOutputEntries(MatrixBlock src, MatrixBlock dest, int idx, int srcOffset, int destOffset, int length) {
		if(src.isInSparseFormat()) src.sparseToDense();
		if(src.getDenseBlock() == null)
			src.allocateDenseBlock(false);
		if(_byRow)
			((DenseBlockFP64) dest.getDenseBlock()).setPartialRow(src.getDenseBlock(), idx, srcOffset, destOffset, length);
		else
			((DenseBlockFP64) dest.getDenseBlock()).setPartialCol(src.getDenseBlock(), idx, srcOffset, destOffset, length);
	}

	private OOCFuture<Void> process(OOCFuture<List<StoreLease<IndexedMatrixValue>>> future, ReservationBudget budget,
		Consumer<List<StoreLease<IndexedMatrixValue>>> action) {
		OOCFuture<Void> completion = new OOCFuture<>();
		_pending.incrementAndGet();
		future.whenComplete((leases, error) -> {
			try {
				if(error != null)
					throw DMLRuntimeException.of(error);
				action.accept(leases);
				completion.complete(null);
			}
			catch(Throwable failure) {
				try {
					if(leases != null)
						leases.forEach(StoreLease::close);
					budget.close();
					_out.propagateFailure(DMLRuntimeException.of(failure));
				}
				finally {
					completion.completeExceptionally(failure);
				}
			}
			finally {
				cleanup(null);
			}
		});
		return completion;
	}

	private void cleanup(ReservationBudget budget) {
		if(budget != null)
			budget.close();
		if(_pending.decrementAndGet() == 0) {
			try {
				_table.close();
				onComplete();
			}
			finally {
				_out.closeInput();
			}
		}
	}

	private void initExecution() {
		DataCharacteristics inputDc = _input.getDataCharacteristics();
		if(inputDc == null || !inputDc.dimsKnown() || inputDc.getBlocksize() <= 0)
			throw new DMLRuntimeException("Reshape OOC reduction requires known input dimensions and block size.");

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
