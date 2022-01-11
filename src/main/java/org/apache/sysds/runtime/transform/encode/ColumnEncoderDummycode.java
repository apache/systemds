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
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
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
	protected TransformType getTransformType() {
		return TransformType.DUMMYCODE;
	}

	@Override
	public void build(CacheBlock in) {
		// do nothing
	}

	@Override
	public List<DependencyTask<?>> getBuildTasks(CacheBlock in) {
		return null;
	}

	@Override
	protected double getCode(CacheBlock in, int row) {
		throw new DMLRuntimeException("DummyCoder does not have a code");
	}

	@Override
	protected double[] getCodeCol(CacheBlock in, int startInd, int blkSize) {
		throw new DMLRuntimeException("DummyCoder does not have a code");
	}

	protected void applySparse(CacheBlock in, MatrixBlock out, int outputCol, int rowStart, int blk){
		if (!(in instanceof MatrixBlock)){
			throw new DMLRuntimeException("ColumnEncoderDummycode called with: " + in.getClass().getSimpleName() +
					" and not MatrixBlock");
		}
		boolean mcsr = MatrixBlock.DEFAULT_SPARSEBLOCK == SparseBlock.Type.MCSR;
		mcsr = false; //force CSR for transformencode
		Set<Integer> sparseRowsWZeros = null;
		int index = _colID - 1;
		for(int r = rowStart; r < getEndIndex(in.getNumRows(), rowStart, blk); r++) {
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
			if (mcsr) {
				double val = out.getSparseBlock().get(r).values()[index];
				if(Double.isNaN(val)){
					if(sparseRowsWZeros == null)
						sparseRowsWZeros = new HashSet<>();
					sparseRowsWZeros.add(r);
					out.getSparseBlock().get(r).values()[index] = 0;
					continue;
				}
				int nCol = outputCol + (int) val - 1;
				out.getSparseBlock().get(r).indexes()[index] = nCol;
				out.getSparseBlock().get(r).values()[index] = 1;
			}
			else { //csr
				SparseBlockCSR csrblock = (SparseBlockCSR)out.getSparseBlock();
				int rptr[] = csrblock.rowPointers();
				double val = csrblock.values()[rptr[r]+index];
				if(Double.isNaN(val)){
					if(sparseRowsWZeros == null)
						sparseRowsWZeros = new HashSet<>();
					sparseRowsWZeros.add(r);
					csrblock.values()[rptr[r]+index] = 0; //test
					continue;
				}
				// Manually fill the column-indexes and values array
				int nCol = outputCol + (int) val - 1;
				csrblock.indexes()[rptr[r]+index] = nCol;
				csrblock.values()[rptr[r]+index] = 1;
			}
		}
		if(sparseRowsWZeros != null){
			addSparseRowsWZeros(sparseRowsWZeros);
		}
	}

	protected void applyDense(CacheBlock in, MatrixBlock out, int outputCol, int rowStart, int blk){
		if (!(in instanceof MatrixBlock)){
			throw new DMLRuntimeException("ColumnEncoderDummycode called with: " + in.getClass().getSimpleName() +
					" and not MatrixBlock");
		}
		int rowEnd = getEndIndex(in.getNumRows(), rowStart, blk);
		double vals[] = new double[rowEnd -rowStart];
		for (int i=rowStart; i<rowEnd; i++)
			vals[i-rowStart] = in.getDouble(i, outputCol);

		// Using outputCol here as index since we have a MatrixBlock as input where 
		// dummycoding might have been applied in a previous encoder
		int B = 32;
		for(int i=rowStart; i<rowEnd; i+=B) {
			// Apply loop tiling to exploit CPU caches
			int lim = Math.min(i+B, rowEnd);
			for (int ii=i; ii<lim; ii++) {
				double val = vals[ii-rowStart];
				if(Double.isNaN(val)) {
					out.quickSetValue(ii, outputCol, 0); //0 if NaN
					continue;
				}
				int nCol = outputCol + (int) val - 1;
				if(nCol != outputCol)
					out.quickSetValue(ii, outputCol, 0);
				out.quickSetValue(ii, nCol, 1);
			}
		}
	}

	@Override
	protected ColumnApplyTask<? extends ColumnEncoder> 
		getSparseTask(CacheBlock in, MatrixBlock out, int outputCol, int startRow, int blk) {
		if (!(in instanceof MatrixBlock)){
			throw new DMLRuntimeException("ColumnEncoderDummycode called with: " + in.getClass().getSimpleName() +
					" and not MatrixBlock");
		}
		return new DummycodeSparseApplyTask(this, (MatrixBlock) in, out, outputCol, startRow, blk);
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
			else if(columnEncoder instanceof ColumnEncoderFeatureHash){
				distinct = (int) ((ColumnEncoderFeatureHash) columnEncoder).getK();
			}

			if(distinct != -1) {
				_domainSize = distinct;
				LOG.debug("DummyCoder for column: " + _colID + " has domain size: " + _domainSize);
			}
		}
	}

	@Override
	public void allocateMetaData(FrameBlock meta) {
		return;
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
			if(_out.getSparseBlock() == null)
				return null;
			_encoder.applySparse(_input, _out, _outputCol, _startRow, _blk);
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
