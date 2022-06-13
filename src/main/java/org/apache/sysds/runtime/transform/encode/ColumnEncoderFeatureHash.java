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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.TransformStatistics;

/**
 * Class used for feature hashing transformation of frames.
 */
public class ColumnEncoderFeatureHash extends ColumnEncoder {
	private static final long serialVersionUID = 7435806042138687342L;
	private long _K;

	/*
	 * public EncoderFeatureHash(JSONObject parsedSpec, String[] colnames, int clen, int minCol, int maxCol) throws
	 * JSONException { super(null, clen); _colList = TfMetaUtils.parseJsonIDList(parsedSpec, colnames,
	 * TfMethod.HASH.toString(), minCol, maxCol); _K = getK(parsedSpec); }
	 * 
	 */
	public ColumnEncoderFeatureHash(int colID, long K) {
		super(colID);
		_K = K;
	}

	public ColumnEncoderFeatureHash() {
		super(-1);
		_K = 0;
	}
	public long getK(){
		return _K;
	}

	@Override
	protected TransformType getTransformType() {
		return TransformType.FEATURE_HASH;
	}

	@Override
	protected double getCode(CacheBlock in, int row) {
		// hash a single row
		String key = in.getString(row, _colID - 1);
		if(key == null)
			return Double.NaN;
		return (key.hashCode() % _K) + 1;
	}

	protected double[] getCodeCol(CacheBlock in, int startInd, int blkSize) {
		// hash a block of rows
		int endInd = getEndIndex(in.getNumRows(), startInd, blkSize);
		double codes[] = new double[endInd-startInd];
		for (int i=startInd; i<endInd; i++) {
			String key = in.getString(i, _colID - 1);
			if(key == null || key.isEmpty())
				codes[i-startInd] = Double.NaN;
			else {
				// Calculate non-negative modulo
				//double mod = key.hashCode() % _K > 0 ? key.hashCode() % _K : _K + key.hashCode() % _K;
				double mod = (key.hashCode() % _K) + 1;
				if (mod < 0)
					mod += _K;
				codes[i - startInd] = mod;
			}
		}
		return codes;
	}

	@Override
	public void build(CacheBlock in) {
		// do nothing (no meta data other than K)
	}

	@Override
	public List<DependencyTask<?>> getBuildTasks(CacheBlock in) {
		return null;
	}

	@Override
	protected ColumnApplyTask<? extends ColumnEncoder> 
		getSparseTask(CacheBlock in, MatrixBlock out, int outputCol, int startRow, int blk) {
		return new FeatureHashSparseApplyTask(this, in, out, outputCol, startRow, blk);
	}

	@Override
	public void mergeAt(ColumnEncoder other) {
		if(other instanceof ColumnEncoderFeatureHash) {
			assert other._colID == _colID;
			if(((ColumnEncoderFeatureHash) other)._K != 0 && _K == 0)
				_K = ((ColumnEncoderFeatureHash) other)._K;
			return;
		}
		super.mergeAt(other);
	}

	@Override
	public void allocateMetaData(FrameBlock meta) {
		if (isApplicable())
			meta.ensureAllocatedColumns(1);
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		if(!isApplicable())
			return meta;

		meta.ensureAllocatedColumns(1);
		meta.set(0, _colID - 1, String.valueOf(_K));
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		if(meta == null || meta.getNumRows() <= 0)
			return;
		_K = UtilFunctions.parseToLong(meta.get(0, _colID - 1).toString());
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeLong(_K);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		_K = in.readLong();
	}

	public static class FeatureHashSparseApplyTask extends ColumnApplyTask<ColumnEncoderFeatureHash>{

		public FeatureHashSparseApplyTask(ColumnEncoderFeatureHash encoder, CacheBlock input,
				MatrixBlock out, int outputCol, int startRow, int blk) {
			super(encoder, input, out, outputCol, startRow, blk);
		}

		public FeatureHashSparseApplyTask(ColumnEncoderFeatureHash encoder, CacheBlock input,
				MatrixBlock out, int outputCol) {
			super(encoder, input, out, outputCol);
		}

		@Override
		public Object call() throws Exception {
			if(_out.getSparseBlock() == null)
				return null;
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			_encoder.applySparse(_input, _out, _outputCol, _startRow, _blk);
			if(DMLScript.STATISTICS)
				TransformStatistics.incFeatureHashingApplyTime(System.nanoTime()-t0);
			return null;
		}
	}

}
