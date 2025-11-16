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
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.IndexingCPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

public abstract class IndexingOOCInstruction extends UnaryOOCInstruction {
	protected final CPOperand rowLower, rowUpper, colLower, colUpper;

	public static IndexingOOCInstruction parseInstruction(String str) {
		IndexingCPInstruction cpInst = IndexingCPInstruction.parseInstruction(str);
		return parseInstruction(cpInst);
	}

	public static IndexingOOCInstruction parseInstruction(IndexingCPInstruction cpInst) {
		String opcode = cpInst.getOpcode();

		if(opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString())) {
			if(cpInst.input1.getDataType().isMatrix()) {
				return new MatrixIndexingOOCInstruction(cpInst.input1, cpInst.getRowLower(), cpInst.getRowUpper(),
					cpInst.getColLower(), cpInst.getColUpper(), cpInst.output, cpInst.getOpcode(),
					cpInst.getInstructionString());
			}
			else {
				throw new NotImplementedException();
			}
		}

		throw new NotImplementedException();
	}

	protected IndexingOOCInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
		CPOperand out, String opcode, String istr) {
		super(OOCInstruction.OOCType.MatrixIndexing, null, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}

	protected IndexingOOCInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl,
		CPOperand cu, CPOperand out, String opcode, String istr) {
		super(OOCInstruction.OOCType.MatrixIndexing, null, lhsInput, rhsInput, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}

	protected IndexRange getIndexRange(ExecutionContext ec) {
		return new IndexRange( //rl, ru, cl, ru
			(int) (ec.getScalarInput(rowLower).getLongValue() - 1),
			(int) (ec.getScalarInput(rowUpper).getLongValue() - 1),
			(int) (ec.getScalarInput(colLower).getLongValue() - 1),
			(int) (ec.getScalarInput(colUpper).getLongValue() - 1));
	}

	public static class BlockAligner<T> {
		private final int _blocksize;
		private final IndexRange _indexRange;
		private final IndexRange _blockRange;
		private final int _outRows;
		private final int _outCols;
		private final Sector<T>[] _blocks;
		private final AtomicInteger _emitCtr;

		@SuppressWarnings("unchecked")
		public BlockAligner(IndexRange range, int blocksize) {
			_indexRange = range;
			_blocksize = blocksize;

			long firstBlockRow = range.rowStart / blocksize;
			long lastBlockRow = range.rowEnd / blocksize;
			long firstBlockCol = range.colStart / blocksize;
			long lastBlockCol = range.colEnd / blocksize;
			_blockRange = new IndexRange(firstBlockRow, lastBlockRow + 1, firstBlockCol, lastBlockCol + 1);

			long totalRows = range.rowSpan() + 1;
			long totalCols = range.colSpan() + 1;
			_outRows = (int) ((totalRows + blocksize - 1) / blocksize);
			_outCols = (int) ((totalCols + blocksize - 1) / blocksize);

			_blocks = (Sector<T>[]) new Sector[_outRows * _outCols];
			_emitCtr = new AtomicInteger(0);
		}

		public boolean isAligned() {
			return (_indexRange.rowStart % _blocksize) == 0 && (_indexRange.colStart % _blocksize) == 0;
		}

		public boolean putNext(MatrixIndexes index, T data, BiConsumer<MatrixIndexes, Sector<T>> emitter) {
			long blockRow = index.getRowIndex() - 1;
			long blockCol = index.getColumnIndex() - 1;

			if(!_blockRange.isWithin(blockRow, blockCol))
				return false;

			long blockRowStart = blockRow * _blocksize;
			long blockRowEnd = blockRowStart + _blocksize - 1;
			long blockColStart = blockCol * _blocksize;
			long blockColEnd = blockColStart + _blocksize - 1;

			long overlapRowStart = Math.max(_indexRange.rowStart, blockRowStart);
			long overlapRowEnd = Math.min(_indexRange.rowEnd, blockRowEnd);
			long overlapColStart = Math.max(_indexRange.colStart, blockColStart);
			long overlapColEnd = Math.min(_indexRange.colEnd, blockColEnd);

			int outRowStart = (int) ((overlapRowStart - _indexRange.rowStart) / _blocksize);
			int outRowEnd = (int) ((overlapRowEnd - _indexRange.rowStart) / _blocksize);
			int outColStart = (int) ((overlapColStart - _indexRange.colStart) / _blocksize);
			int outColEnd = (int) ((overlapColEnd - _indexRange.colStart) / _blocksize);

			int emitCtr = -1;

			for(int outRow = outRowStart; outRow <= outRowEnd; outRow++) {
				long targetRowStartGlobal = _indexRange.rowStart + (long) outRow * _blocksize;
				long targetRowEndGlobal = Math.min(_indexRange.rowEnd, targetRowStartGlobal + _blocksize - 1);
				long targetStartBlockRow = targetRowStartGlobal / _blocksize;
				long targetEndBlockRow = targetRowEndGlobal / _blocksize;
				int rowSegments = (int) (targetEndBlockRow - targetStartBlockRow + 1);

				for(int outCol = outColStart; outCol <= outColEnd; outCol++) {
					long targetColStartGlobal = _indexRange.colStart + (long) outCol * _blocksize;
					long targetColEndGlobal = Math.min(_indexRange.colEnd, targetColStartGlobal + _blocksize - 1);
					long targetStartBlockCol = targetColStartGlobal / _blocksize;
					long targetEndBlockCol = targetColEndGlobal / _blocksize;
					int colSegments = (int) (targetEndBlockCol - targetStartBlockCol + 1);

					int rowOffset = (rowSegments == 1) ? 0 : (blockRow == targetStartBlockRow ? 0 : 1);
					int colOffset = (colSegments == 1) ? 0 : (blockCol == targetStartBlockCol ? 0 : 1);

					Sector<T> sector = getOrCreate(outRow, outCol, rowSegments, colSegments);
					if(sector == null)
						continue;

					boolean emit = sector.place(rowOffset, colOffset, data);
					if(emit) {
						int idxPos = resolveIndex(outRow, outCol);
						_blocks[idxPos] = null;
						emitCtr = _emitCtr.incrementAndGet();
						emitter.accept(new MatrixIndexes(outRow + 1, outCol + 1), sector);
					}
				}
			}

			return emitCtr >= _blocks.length;
		}

		private int resolveIndex(int row, int col) {
			if(row < 0 || row >= _outRows || col < 0 || col >= _outCols)
				return -1;
			return row * _outCols + col;
		}

		private synchronized Sector<T> getOrCreate(int outRow, int outCol, int rowSegments, int colSegments) {
			int idx = resolveIndex(outRow, outCol);
			if(idx == -1)
				return null;

			Sector<T> s = _blocks[idx];
			if(s == null) {
				if(rowSegments == 1 && colSegments == 1)
					s = new Sector1<>();
				else if(rowSegments == 1 && colSegments == 2)
					s = new Sector2Col<>();
				else if(rowSegments == 2 && colSegments == 1)
					s = new Sector2Row<>();
				else
					s = new Sector4<>();
				_blocks[idx] = s;
			}

			return s;
		}

		public synchronized void close() {
			if(_emitCtr.get() != _blocks.length)
				throw new DMLRuntimeException("BlockAligner still has some unfinished sectors");
		}
	}

	public static abstract class Sector<T> {
		public abstract boolean place(int rowOffset, int colOffset, T data);

		public abstract T get(int rowOffset, int colOffset);

		public abstract int count();
	}

	public static class Sector1<T> extends Sector<T> {
		private T _data;

		@Override
		public synchronized boolean place(int rowOffset, int colOffset, T data) {
			if(rowOffset != 0 || colOffset != 0)
				return false;

			_data = data;
			return true;
		}

		@Override
		public synchronized T get(int rowOffset, int colOffset) {
			return (rowOffset == 0 && colOffset == 0) ? _data : null;
		}

		@Override
		public synchronized int count() {
			return _data == null ? 0 : 1;
		}
	}

	public static class Sector4<T> extends Sector<T> {
		private int _count;
		private final T[] _data;

		@SuppressWarnings("unchecked")
		public Sector4() {
			_count = 0;
			_data = (T[]) new Object[4];
		}

		@Override
		public synchronized boolean place(int rowOffset, int colOffset, T data) {
			int pos = rowOffset * 2 + colOffset;
			if(_data[pos] == null) {
				_data[pos] = data;
				_count++;
			}
			return _count == 4;
		}

		@Override
		public synchronized T get(int rowOffset, int colOffset) {
			return _data[rowOffset * 2 + colOffset];
		}

		@Override
		public synchronized int count() {
			return _count;
		}
	}

	public static class Sector2Col<T> extends Sector<T> {
		private int _count;
		private final T[] _data;

		@SuppressWarnings("unchecked")
		public Sector2Col() {
			_count = 0;
			_data = (T[]) new Object[2];
		}

		@Override
		public synchronized boolean place(int rowOffset, int colOffset, T data) {
			if(rowOffset != 0 || colOffset < 0 || colOffset > 1)
				return false;

			if(_data[colOffset] == null) {
				_data[colOffset] = data;
				_count++;
			}

			return _count == 2;
		}

		@Override
		public synchronized T get(int rowOffset, int colOffset) {
			return (rowOffset == 0 && colOffset >= 0 && colOffset < 2) ? _data[colOffset] : null;
		}

		@Override
		public synchronized int count() {
			return _count;
		}
	}

	public static class Sector2Row<T> extends Sector<T> {
		private int _count;
		private final T[] _data;

		@SuppressWarnings("unchecked")
		public Sector2Row() {
			_count = 0;
			_data = (T[]) new Object[2];
		}

		@Override
		public synchronized boolean place(int rowOffset, int colOffset, T data) {
			if(colOffset != 0 || rowOffset < 0 || rowOffset > 1)
				return false;

			if(_data[rowOffset] == null) {
				_data[rowOffset] = data;
				_count++;
			}

			return _count == 2;
		}

		@Override
		public synchronized T get(int rowOffset, int colOffset) {
			return (colOffset == 0 && rowOffset >= 0 && rowOffset < 2) ? _data[rowOffset] : null;
		}

		@Override
		public synchronized int count() {
			return _count;
		}
	}
}
