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

import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.utils.stats.TransformStatistics;

public class ColumnEncoderPassThrough extends ColumnEncoder {
	private static final long serialVersionUID = -8473768154646831882L;

	protected ColumnEncoderPassThrough(int ptCols) {
		super(ptCols); // 1-based
	}

	public ColumnEncoderPassThrough() {
		this(-1);
	}

	@Override
	public void build(CacheBlock<?> in) {
		// do nothing
	}

	@Override
	public List<DependencyTask<?>> getBuildTasks(CacheBlock<?> in) {
		return null;
	}

	@Override
	protected ColumnApplyTask<? extends ColumnEncoder>
		getSparseTask(CacheBlock<?> in, MatrixBlock out, int outputCol, int startRow, int blk) {
		return new PassThroughSparseApplyTask(this, in, out, outputCol, startRow, blk);
	}

	@Override
	protected double getCode(CacheBlock<?> in, int row) {
		return in.getDoubleNaN(row, _colID - 1);
	}

	@Override
	protected double[] getCodeCol(CacheBlock<?> in, int startInd, int endInd, double[] tmp) {
		try{
			final int endLength = endInd - startInd;
			final double[] codes = tmp != null && tmp.length == endLength ? tmp : new double[endLength];
			for (int i = startInd; i < endInd; i++) {
				codes[i - startInd] = in.getDoubleNaN(i, _colID - 1);
			}
			return codes;
		}
		catch(Exception e){
			throw new RuntimeException("Failed to get code for col: " + _colID + " on range: " + startInd + "->" + endInd, e);
		}
	}

	protected void applySparse(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk){
		final SparseBlock sb = out.getSparseBlock();
		final boolean mcsr = sb instanceof SparseBlockMCSR;
		final int index = _colID - 1;
		final int rowEnd = getEndIndex(in.getNumRows(), rowStart, blk);
		final int bs = 32;
		double[] tmp = null;
		for(int i = rowStart; i < rowEnd; i+= bs) {
			int end = Math.min(i + bs , rowEnd);
			tmp = getCodeCol(in, i, end,tmp);
			if(mcsr)
				applySparseBlockMCSR(in, (SparseBlockMCSR) sb, index, outputCol, i, end, tmp);
			else
				applySparseBlockCSR(in, (SparseBlockCSR) sb, index, outputCol, i, end, tmp);
			
		}
	}

	private void applySparseBlockMCSR(CacheBlock<?> in, final SparseBlockMCSR sb, final int index,
		final int outputCol, int rl, int ru, double[] tmpCodes) {
		for(int i = rl; i < ru; i ++) {
			final double v = tmpCodes[i - rl];
			SparseRowVector row = (SparseRowVector) sb.get(i);
			row.indexes()[index] = outputCol;
			if(v == 0) 
				containsZeroOut = true;
			else 
				row.values()[index] = v;
		}
	}

	private void applySparseBlockCSR(CacheBlock<?> in, final SparseBlockCSR sb, final int index, final int outputCol,
		int rl, int ru, double[] tmpCodes) {
		final int[] rptr = sb.rowPointers();
		final int[] idx = sb.indexes();
		final double[] val = sb.values();
		for(int i = rl; i < ru; i++) {
			final double v = tmpCodes[i - rl];
			idx[rptr[i] + index] = outputCol;
			if(v == 0)
			containsZeroOut = true;
			else
			val[rptr[i] + index] = v;
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

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName());
		sb.append(": ");
		sb.append(_colID);
		return sb.toString();
	}

	public static class PassThroughSparseApplyTask extends ColumnApplyTask<ColumnEncoderPassThrough>{
		protected PassThroughSparseApplyTask(ColumnEncoderPassThrough encoder, 
				CacheBlock<?> input, MatrixBlock out, int outputCol, int startRow, int blk) {
			super(encoder, input, out, outputCol, startRow, blk);
		}

		@Override
		public Object call() throws Exception {
			if(_out.getSparseBlock() == null)
				return null;
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			_encoder.applySparse(_input, _out, _outputCol, _startRow, _blk);
			if(DMLScript.STATISTICS)
				TransformStatistics.incPassThroughApplyTime(System.nanoTime()-t0);
			return null;
		}

		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}

	}

}
