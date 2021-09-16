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
import java.util.List;
import java.util.Objects;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.utils.Statistics;

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
	public List<DependencyTask<?>> getBuildTasks(FrameBlock in) {
		return null;
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		throw new DMLRuntimeException("Called DummyCoder with FrameBlock");
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Out Matrix should already be correct size!
		// append dummy coded or unchanged values to output
		for(int i = rowStart; i < getEndIndex(in.getNumRows(), rowStart, blk); i++) {
			// Using outputCol here as index since we have a MatrixBlock as input where dummycoding could have been
			// applied in a previous encoder
			// FIXME: we need a clear way of separating input/output (org input, pre-allocated output)
			// need input index to avoid inconsistencies; also need to set by position not binarysearch
			double val = in.quickGetValueThreadSafe(i, outputCol);
			int nCol = outputCol + (int) val - 1;
			// Set value, w/ robustness for val=NaN (unknown categories)
			if( nCol >= 0 && !Double.isNaN(val) ) { // filter unknowns
				out.quickSetValue(i, outputCol, 0); //FIXME remove this workaround (see above)
				out.quickSetValue(i, nCol, 1);
			}
			else
				out.quickSetValue(i, outputCol, 0);
		}
		if (DMLScript.STATISTICS)
			Statistics.incTransformDummyCodeApplyTime(System.nanoTime()-t0);
		return out;
	}


	@Override
	protected ColumnApplyTask<? extends ColumnEncoder> 
		getSparseTask(MatrixBlock in, MatrixBlock out, int outputCol, int startRow, int blk) {
		return new DummycodeSparseApplyTask(this, in, out, outputCol, startRow, blk);
	}

	@Override
	protected ColumnApplyTask<? extends ColumnEncoder> 
		getSparseTask(FrameBlock in, MatrixBlock out, int outputCol, int startRow, int blk) {
		throw new DMLRuntimeException("Called DummyCoder with FrameBlock");
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
				LOG.debug("DummyCoder for column: " + _colID + " has domain size: " + _domainSize);
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

	private static class DummycodeSparseApplyTask extends ColumnApplyTask<ColumnEncoderDummycode> {

		protected DummycodeSparseApplyTask(ColumnEncoderDummycode encoder, MatrixBlock input, 
				MatrixBlock out, int outputCol) {
			super(encoder, input, out, outputCol);
		}

		protected DummycodeSparseApplyTask(ColumnEncoderDummycode encoder, MatrixBlock input, 
				MatrixBlock out, int outputCol, int startRow, int blk) {
			super(encoder, input, out, outputCol, startRow, blk);
		}

		public Object call() throws Exception {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			assert _inputM != null;
			if(_out.getSparseBlock() == null)
				return null;
			for(int r = _startRow; r < getEndIndex(_inputM.getNumRows(), _startRow, _blk); r++) {
				// Since the recoded values are already offset in the output matrix (same as input at this point)
				// the dummycoding only needs to offset them within their column domain. Which means that the
				// indexes in the SparseRowVector do not need to be sorted anymore and can be updated directly.
				//
				// Input: Output:
				//
				// 1 | 0 | 2 | 0 		1 | 0 | 0 | 1
				// 2 | 0 | 1 | 0 ===> 	0 | 1 | 1 | 0
				// 1 | 0 | 2 | 0 		1 | 0 | 0 | 1
				// 1 | 0 | 1 | 0 		1 | 0 | 1 | 0
				//
				// Example SparseRowVector Internals (1. row):
				//
				// indexes = [0,2] ===> indexes = [0,3]
				// values = [1,2] values = [1,1]
				int index = _encoder._colID - 1;
				double val = _out.getSparseBlock().get(r).values()[index];
				int nCol = _outputCol + (int) val - 1;
				_out.getSparseBlock().get(r).indexes()[index] = nCol;
				_out.getSparseBlock().get(r).values()[index] = 1;
			}
			if (DMLScript.STATISTICS)
				Statistics.incTransformDummyCodeApplyTime(System.nanoTime()-t0);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}

	}

}
