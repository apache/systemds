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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.Statistics;

public class ColumnEncoderPassThrough extends ColumnEncoder {
	private static final long serialVersionUID = -8473768154646831882L;
	private List<Integer> sparseRowsWZeros = null;

	protected ColumnEncoderPassThrough(int ptCols) {
		super(ptCols); // 1-based
	}

	public ColumnEncoderPassThrough() {
		this(-1);
	}

	public List<Integer> getSparseRowsWZeros(){
		return sparseRowsWZeros;
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
	protected ColumnApplyTask<? extends ColumnEncoder> getSparseTask(FrameBlock in, MatrixBlock out,
																	 int outputCol, int startRow, int blk) {
		return new PassThroughSparseApplyTask(this, in, out, outputCol, startRow, blk);
	}

	@Override
	protected ColumnApplyTask<? extends ColumnEncoder> getSparseTask(MatrixBlock in, MatrixBlock out,
																	 int outputCol, int startRow, int blk) {
		throw new NotImplementedException("Sparse PassThrough for MatrixBlocks not jet implemented");
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		int col = _colID - 1; // 1-based
		ValueType vt = in.getSchema()[col];
		for(int i = rowStart; i < getEndIndex(in.getNumRows(), rowStart, blk); i++) {
			Object val = in.get(i, col);
			double v = (val == null ||
				(vt == ValueType.STRING && val.toString().isEmpty())) ? Double.NaN : UtilFunctions.objectToDouble(vt,
					val);
			out.quickSetValue(i, outputCol, v);
		}
		if(DMLScript.STATISTICS)
			Statistics.incTransformPassThroughApplyTime(System.nanoTime()-t0);
		return out;
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		// only transfer from in to out
		if(in == out)
			return out;
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		int col = _colID - 1; // 1-based
		int end = getEndIndex(in.getNumRows(), rowStart, blk);
		for(int i = rowStart; i < end; i++) {
			double val = in.quickGetValueThreadSafe(i, col);
			out.quickSetValue(i, outputCol, val);
		}
		if(DMLScript.STATISTICS)
			Statistics.incTransformPassThroughApplyTime(System.nanoTime()-t0);
		return out;
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


		protected PassThroughSparseApplyTask(ColumnEncoderPassThrough encoder, FrameBlock input, MatrixBlock out,
											 int outputCol) {
			super(encoder, input, out, outputCol);
		}

		protected PassThroughSparseApplyTask(ColumnEncoderPassThrough encoder, FrameBlock input, MatrixBlock out,
											 int outputCol, int startRow, int blk) {
			super(encoder, input, out, outputCol, startRow, blk);
		}

		@Override
		public Object call() throws Exception {
			if(_out.getSparseBlock() == null)
				return null;
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			int index = _encoder._colID - 1;
			assert _inputF != null;
			ValueType vt = _inputF.getSchema()[index];
			for(int r = _startRow; r < getEndIndex(_inputF.getNumRows(), _startRow, _blk); r++) {
				Object val = _inputF.get(r, index);
				double v = (val == null || (vt == ValueType.STRING && val.toString().isEmpty())) ?
						Double.NaN : UtilFunctions.objectToDouble(vt, val);
				SparseRowVector row = (SparseRowVector) _out.getSparseBlock().get(r);
				if(v == 0) {
					if(_encoder.sparseRowsWZeros == null)
						_encoder.sparseRowsWZeros = new ArrayList<>();
					_encoder.sparseRowsWZeros.add(r);
				}
				row.values()[index] = v;
				row.indexes()[index] = _outputCol;
			}
			if(DMLScript.STATISTICS)
				Statistics.incTransformPassThroughApplyTime(System.nanoTime()-t0);
			return null;
		}

		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}

	}

}
