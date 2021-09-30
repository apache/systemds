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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.Transformable;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.utils.Statistics;

public class ColumnEncoderPassThrough extends ColumnEncoder {
	private static final long serialVersionUID = -8473768154646831882L;
	private List<Integer> _sparseRowsWZeros = null;

	protected ColumnEncoderPassThrough(int ptCols) {
		super(ptCols); // 1-based
	}

	public ColumnEncoderPassThrough() {
		this(-1);
	}

	public List<Integer> getSparseRowsWZeros(){
		return _sparseRowsWZeros;
	}

	@Override
	public void build(Transformable in) {
		// do nothing
	}

	@Override
	public List<DependencyTask<?>> getBuildTasks(Transformable in) {
		return null;
	}

	@Override
	protected ColumnApplyTask<? extends ColumnEncoder>
		getSparseTask(Transformable in, MatrixBlock out, int outputCol, int startRow, int blk) {
		return new PassThroughSparseApplyTask(this, in, out, outputCol, startRow, blk);
	}

	@Override
	protected double getCode(Transformable in, int row) {
		return in.getDoubleValue(row, _colID - 1);
	}


	protected void applySparse(Transformable in, MatrixBlock out, int outputCol, int rowStart, int blk){
		List<Integer> sparseRowsWZeros = null;
		int index = _colID - 1;
		for(int r = rowStart; r < getEndIndex(in.getNumRows(), rowStart, blk); r++) {
			double v = getCode(in, r);
			SparseRowVector row = (SparseRowVector) out.getSparseBlock().get(r);
			if(v == 0) {
				if(sparseRowsWZeros == null)
					sparseRowsWZeros = new ArrayList<>();
				sparseRowsWZeros.add(r);
			}
			row.values()[index] = v;
			row.indexes()[index] = outputCol;
		}
		if(sparseRowsWZeros != null){
			synchronized (this){
				if(_sparseRowsWZeros == null)
					_sparseRowsWZeros = new ArrayList<>();
				_sparseRowsWZeros.addAll(sparseRowsWZeros);
			}
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
	public FrameBlock getMetaData(FrameBlock meta) {
		// do nothing
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		// do nothing
	}

	public static class PassThroughSparseApplyTask extends ColumnApplyTask<ColumnEncoderPassThrough>{


		protected PassThroughSparseApplyTask(ColumnEncoderPassThrough encoder, Transformable input,
				MatrixBlock out, int outputCol) {
			super(encoder, input, out, outputCol);
		}

		protected PassThroughSparseApplyTask(ColumnEncoderPassThrough encoder, Transformable input, MatrixBlock out,
				int outputCol, int startRow, int blk) {
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
