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

package org.apache.sysds.runtime.transform.encode;

import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.tasks.ColumnApply;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

public class ColumnEncoderDummycode extends ColumnEncoder {
	private static final long serialVersionUID = 5832130477659116489L;

	public int _domainSize = -1; // length = #of dummycoded columns

	public ColumnEncoderDummycode() {
		super(-1);
	}

	public ColumnEncoderDummycode(int colID) {
		super(colID);
	}

	public ColumnEncoderDummycode(int colID, int domainSize) {
		super(colID);
		_domainSize = domainSize;
	}

	@Override
	public void build(FrameBlock in) {
		// do nothing
	}

	@Override
	public List<DependencyTask<?>> getBuildTasks(FrameBlock in, int blockSize) {
		return null;
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol) {
		return apply(in, out, outputCol, 0, -1);
	}

	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol) {
		return apply(in, out, outputCol, 0, -1);
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		throw new DMLRuntimeException("Called DummyCoder with FrameBlock");
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		// Out Matrix should already be correct size!
		// append dummy coded or unchanged values to output
		for(int i = rowStart; i < getEndIndex(in.getNumRows(), rowStart, blk); i++) {
			// Using outputCol here as index since we have a MatrixBlock as input where dummycoding could have been
			// applied in a previous encoder
			double val = in.quickGetValueThreadSafe(i, outputCol);
			int nCol = outputCol + (int) val - 1;
			// Setting value to 0 first in case of sparse so the row vector does not need to be resized
			if(nCol != outputCol)
				out.quickSetValueThreadSafe(i, outputCol, 0);
			out.quickSetValueThreadSafe(i, nCol, 1);
		}
		return out;
	}

	@Override
	public List<DependencyTask<?>> getApplyTasks(MatrixBlock in, MatrixBlock out, int outputCol) {
		List<Callable<Object>> tasks = new ArrayList<>();
		if(out.isInSparseFormat())
			tasks.add(new DummycodeSparseApplyTask(this, in, out, outputCol));
		else
			return super.getApplyTasks(in, out, outputCol);
		return DependencyThreadPool.createDependencyTasks(tasks, null);
	}


	@Override
	public void mergeAt(ColumnEncoder other) {
		if(other instanceof ColumnEncoderDummycode) {
			assert other._colID == _colID;
			// temporary, will be updated later
			_domainSize = 0;
			return;
		}
		super.mergeAt(other);
	}

	@Override
	public void updateIndexRanges(long[] beginDims, long[] endDims, int colOffset) {

		// new columns inserted in this (federated) block
		beginDims[1] += colOffset;
		endDims[1] += _domainSize - 1 + colOffset;
	}

	public void updateDomainSizes(List<ColumnEncoder> columnEncoders) {
		if(_colID == -1)
			return;
		for(ColumnEncoder columnEncoder : columnEncoders) {
			int distinct = -1;
			if(columnEncoder instanceof ColumnEncoderRecode) {
				ColumnEncoderRecode columnEncoderRecode = (ColumnEncoderRecode) columnEncoder;
				distinct = columnEncoderRecode.getNumDistinctValues();
			}
			else if(columnEncoder instanceof ColumnEncoderBin) {
				distinct = ((ColumnEncoderBin) columnEncoder)._numBin;
			}

			if(distinct != -1) {
				_domainSize = distinct;
			}
		}
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		// initialize domain sizes and output num columns
		_domainSize = -1;
		_domainSize = (int) meta.getColumnMetadata()[_colID - 1].getNumDistinct();
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeInt(_domainSize);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		_domainSize = in.readInt();
	}

	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o == null || getClass() != o.getClass())
			return false;
		ColumnEncoderDummycode that = (ColumnEncoderDummycode) o;
		return _colID == that._colID && (_domainSize == that._domainSize);
	}

	@Override
	public int hashCode() {
		int result = Objects.hash(_colID);
		result = 31 * result + Objects.hashCode(_domainSize);
		return result;
	}

	public int getDomainSize() {
		return _domainSize;
	}



	private static class DummycodeSparseApplyTask implements ColumnApply {
		private final ColumnEncoderDummycode _encoder;
		private final MatrixBlock _input;
		private final MatrixBlock _out;
		private int _outputCol;


		private DummycodeSparseApplyTask(ColumnEncoderDummycode encoder, MatrixBlock input, MatrixBlock out, int outputCol) {
			_encoder = encoder;
			_input = input;
			_out = out;
			_outputCol = outputCol;
		}

		@Override
		public Object call() throws Exception {
			for(int r = 0; r < _input.getNumRows(); r++) {
				synchronized (_out.getSparseBlock().get(r)){
					// Since the recoded values are already offset in the output matrix (same as input at this point)
					// the dummycoding only needs to offset them within their column domain. Which means that the
					// indexes in the SparseRowVector do not need to be sorted anymore and can be updated directly.
					//
					// Input:                                Output:
					//
					//   1  |  0  |  2  |  0               1  |  0  |  0  |  1
					//   2  |  0  |  1  |  0     ===>      0  |  1  |  1  |  0
					//   1  |  0  |  2  |  0               1  |  0  |  0  |  1
					//   1  |  0  |  1  |  0               1  |  0  |  1  |  0
					//
					//  Example SparseRowVector Internals (1. row):
					//
					//  indexes = [0,2]         ===>      indexes = [0,3]
					//  values = [1,2]                    values = [1,1]
					int index = ((SparseRowVector)_out.getSparseBlock().get(r)).getIndex(_outputCol);
					double val = _out.getSparseBlock().get(r).values()[index];
					int nCol = _outputCol + (int) val - 1;

					_out.getSparseBlock().get(r).indexes()[index] = nCol;
					_out.getSparseBlock().get(r).values()[index] = 1;
				}
			}
			return null;
		}

		@Override
		public void setOutputCol(int outputCol) {
			_outputCol = outputCol;
		}

		@Override
		public int getOutputCol() {
			return _outputCol;
		}
	}


}
