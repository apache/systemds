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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.utils.Statistics;

public class ColumnEncoderPassThrough extends ColumnEncoder {
	private static final long serialVersionUID = -8473768154646831882L;

	protected ColumnEncoderPassThrough(int ptCols) {
		super(ptCols); // 1-based
	}

	public ColumnEncoderPassThrough() {
		this(-1);
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
	protected ColumnApplyTask<? extends ColumnEncoder>
		getSparseTask(CacheBlock in, MatrixBlock out, int outputCol, int startRow, int blk) {
		return new PassThroughSparseApplyTask(this, in, out, outputCol, startRow, blk);
	}

	@Override
	protected double getCode(CacheBlock in, int row) {
		return in.getDoubleNaN(row, _colID - 1);
	}

	@Override
	protected double[] getCodeCol(CacheBlock in, int startInd, int blkSize) {
		int endInd = getEndIndex(in.getNumRows(), startInd, blkSize);
		double codes[] = new double[endInd-startInd];
		for (int i=startInd; i<endInd; i++) {
			codes[i-startInd] = in.getDoubleNaN(i, _colID-1);
		}
		return codes;
	}

	protected void applySparse(CacheBlock in, MatrixBlock out, int outputCol, int rowStart, int blk){
		Set<Integer> sparseRowsWZeros = null;
		boolean mcsr = MatrixBlock.DEFAULT_SPARSEBLOCK == SparseBlock.Type.MCSR;
		mcsr = false; //force CSR for transformencode
		int index = _colID - 1;
		// Apply loop tiling to exploit CPU caches
		double[] codes = getCodeCol(in, rowStart, blk);
		int rowEnd = getEndIndex(in.getNumRows(), rowStart, blk);
		int B = 32; //tile size
		for(int i = rowStart; i < rowEnd; i+=B) {
			int lim = Math.min(i+B, rowEnd);
			for (int ii=i; ii<lim; ii++) {
				double v = codes[ii-rowStart];
				if(v == 0) {
					if(sparseRowsWZeros == null)
						sparseRowsWZeros = new HashSet<>();
					sparseRowsWZeros.add(ii);
				}
				if (mcsr) {
					SparseRowVector row = (SparseRowVector) out.getSparseBlock().get(ii);
					row.values()[index] = v;
					row.indexes()[index] = outputCol;
				}
				else { //csr
					if(v == 0) {
						if(sparseRowsWZeros == null)
							sparseRowsWZeros = new HashSet<>();
						sparseRowsWZeros.add(ii);
					}
					// Manually fill the column-indexes and values array
					SparseBlockCSR csrblock = (SparseBlockCSR)out.getSparseBlock();
					int rptr[] = csrblock.rowPointers();
					csrblock.indexes()[rptr[ii]+index] = outputCol;
					csrblock.values()[rptr[ii]+index] = codes[ii-rowStart];
				}
			}
		}
		if(sparseRowsWZeros != null){
			addSparseRowsWZeros(sparseRowsWZeros);
		}
	}

	@Override
	protected TransformType getTransformType() {
		return TransformType.PASS_THROUGH;
	}


	@Override
	public void mergeAt(ColumnEncoder other) {
		if(other instanceof ColumnEncoderPassThrough) {
			return;
		}
		super.mergeAt(other);
	}

	@Override
	public void allocateMetaData(FrameBlock meta) {
		// do nothing
		return;
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		// do nothing
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		// do nothing
	}

	public static class PassThroughSparseApplyTask extends ColumnApplyTask<ColumnEncoderPassThrough>{


		protected PassThroughSparseApplyTask(ColumnEncoderPassThrough encoder, CacheBlock input,
				MatrixBlock out, int outputCol) {
			super(encoder, input, out, outputCol);
		}

		protected PassThroughSparseApplyTask(ColumnEncoderPassThrough encoder, 
				CacheBlock input, MatrixBlock out, int outputCol, int startRow, int blk) {
			super(encoder, input, out, outputCol, startRow, blk);
		}

		@Override
		public Object call() throws Exception {
			if(_out.getSparseBlock() == null)
				return null;
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			_encoder.applySparse(_input, _out, _outputCol, _startRow, _blk);
			if(DMLScript.STATISTICS)
				Statistics.incTransformPassThroughApplyTime(System.nanoTime()-t0);
			return null;
		}

		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}

	}

}
