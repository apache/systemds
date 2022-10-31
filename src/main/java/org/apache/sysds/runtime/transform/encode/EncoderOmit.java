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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.TransformStatistics;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class EncoderOmit extends LegacyEncoder {
	/*
	 * THIS CLASS IS ONLY FOR LEGACY SUPPORT!!! and will be fazed out slowly.
	 */

	protected static final Log LOG = LogFactory.getLog(Encoder.class.getName());
	private static final long serialVersionUID = 1978852120416654195L;
	private boolean _federated = false;
	private boolean[] _rmRows = new boolean[0];

	public EncoderOmit(JSONObject parsedSpec, String[] colnames, int clen, int minCol, int maxCol)
		throws JSONException {
		this(null, clen);
		if(!parsedSpec.containsKey(TfMethod.OMIT.toString()))
			return;
		int[] collist = TfMetaUtils.parseJsonIDList(parsedSpec, colnames, TfMethod.OMIT.toString(), minCol, maxCol);
		initColList(collist);
		_federated = minCol != -1 || maxCol != -1;
	}

	public EncoderOmit() {
		super(new int[0], 0);
	}

	public EncoderOmit(int[] colList, int clen) {
		super(colList, clen);
	}

	public EncoderOmit(boolean federated) {
		this();
		_federated = federated;
	}

	private EncoderOmit(int[] colList, int clen, boolean[] rmRows) {
		this(colList, clen);
		_rmRows = rmRows;
		_federated = true;
	}

	public int initColList(int[] colList) {
		_colList = colList;
		return _colList.length;
	}

	/**
	 * Indicates if this encoder is applicable, i.e, if there is at least one column to encode.
	 *
	 * @return true if at least one column to encode
	 */
	public boolean isApplicable() {
		return(_colList != null && _colList.length > 0);
	}

	public int getNumRemovedRows(boolean[] rmRows) {
		int cnt = 0;
		for(boolean v : rmRows)
			cnt += v ? 1 : 0;
		return cnt;
	}

	public int getNumRemovedRows() {
		return getNumRemovedRows(_rmRows);
	}

	public boolean omit(String[] words, TfUtils agents) {
		if(!isApplicable())
			return false;

		for (int colID : _colList) {
			if (TfUtils.isNA(agents.getNAStrings(), UtilFunctions.unquote(words[colID - 1].trim())))
				return true;
		}
		return false;
	}

	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		return apply(in, out);
	}

	public void build(FrameBlock in) {
		if(_federated)
			_rmRows = computeRmRows(in);
	}

	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		// local rmRows for broadcasting encoder in spark
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		boolean[] rmRows;
		if(_federated)
			rmRows = _rmRows;
		else
			rmRows = computeRmRows(in);

		// determine output size
		int numRows = out.getNumRows() - getNumRemovedRows(rmRows);

		// copy over valid rows into the output
		MatrixBlock ret = new MatrixBlock(numRows, out.getNumColumns(), false);
		int pos = 0;
		for(int i = 0; i < in.getNumRows(); i++) {
			// copy row if necessary
			if(!rmRows[i]) {
				for(int j = 0; j < out.getNumColumns(); j++)
					ret.quickSetValue(pos, j, out.quickGetValue(i, j));
				pos++;
			}
		}

		_rmRows = rmRows;
		if(DMLScript.STATISTICS)
			TransformStatistics.incOmitApplyTime(System.nanoTime()-t0);
		return ret;
	}

	private boolean[] computeRmRows(FrameBlock in) {
		boolean[] rmRows = new boolean[in.getNumRows()];
		ValueType[] schema = in.getSchema();
		// TODO perf evaluate if column-wise scan more efficient
		// (sequential but less impact of early abort)
		for(int i = 0; i < in.getNumRows(); i++) {
			for(int colID : _colList) {
				Object val = in.get(i, colID - 1);
				if(val == null || (schema[colID - 1] == ValueType.STRING && val.toString().isEmpty())) {
					rmRows[i] = true;
					break; // early abort
				}
			}
		}
		return rmRows;
	}

	public EncoderOmit subRangeEncoder(IndexRange ixRange) {
		int[] colList = subRangeColList(ixRange);
		if(colList.length == 0)
			// empty encoder -> sub range encoder does not exist
			return null;
		boolean[] rmRows = _rmRows;
		if(_rmRows.length > 0)
			rmRows = Arrays.copyOfRange(rmRows, (int) ixRange.rowStart - 1, (int) ixRange.rowEnd - 1);

		return new EncoderOmit(colList, (int) (ixRange.colSpan()), rmRows);
	}

	protected int[] subRangeColList(IndexRange ixRange) {
		List<Integer> cols = new ArrayList<>();
		for(int col : _colList) {
			if(ixRange.inColRange(col)) {
				// add the correct column, removed columns before start
				// colStart - 1 because colStart is 1-based
				cols.add(col);
			}
		}
		return cols.stream().mapToInt(i -> i).toArray();
	}

	public void mergeAt(EncoderOmit other, int row, int col) {
		mergeColumnInfo(other, col);
		_rmRows = Arrays.copyOf(_rmRows, Math.max(_rmRows.length, (row - 1) + other._rmRows.length));
		for(int i = 0; i < other._rmRows.length; i++)
			_rmRows[(row - 1) + 1] |= other._rmRows[i];
	}

	public void updateIndexRanges(long[] beginDims, long[] endDims) {
		// first update begin dims
		int numRowsToRemove = 0;
		for(int i = 0; i < beginDims[0] - 1 && i < _rmRows.length; i++)
			if(_rmRows[i])
				numRowsToRemove++;
		beginDims[0] -= numRowsToRemove;
		// update end dims
		for(int i = 0; i < endDims[0] - 1 && i < _rmRows.length; i++)
			if(_rmRows[i])
				numRowsToRemove++;
		endDims[0] -= numRowsToRemove;
	}

	public FrameBlock getMetaData(FrameBlock out) {
		// do nothing
		return out;
	}

	public void initMetaData(FrameBlock meta) {
		// do nothing
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeBoolean(_federated);
		out.writeInt(_rmRows.length);
		for(boolean r : _rmRows)
			out.writeBoolean(r);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		if(_rmRows.length == 0) {
			_federated = in.readBoolean();
			_rmRows = new boolean[in.readInt()];
			for(int i = 0; i < _rmRows.length; i++)
				_rmRows[i] = in.readBoolean();
		}
	}

	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o == null || getClass() != o.getClass())
			return false;
		EncoderOmit that = (EncoderOmit) o;
		return _federated == that._federated && Arrays.equals(_rmRows, that._rmRows);
	}

	public int hashCode() {
		int result = Objects.hash(_federated);
		result = 31 * result + Arrays.hashCode(_rmRows);
		return result;
	}
}
