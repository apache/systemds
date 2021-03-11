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

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;
import java.util.Objects;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class EncoderOmit extends Encoder 
{	
	private static final long serialVersionUID = 1978852120416654195L;

	private boolean _federated = true;
	private boolean[] _rmRows = new boolean[0];

	
	public EncoderOmit() {
		super(-1);
	}
	
	public EncoderOmit(boolean federated) {
		this();
		_federated = federated;
	}

	public EncoderOmit(int colID, boolean federated) {
		super(colID);
		_federated = federated;
	}
	
	private EncoderOmit(int colID, boolean federated, boolean[] rmRows) {
		this(colID, federated);
		_rmRows = rmRows;
	}

	public static int getNumRemovedRows(boolean[] rmRows) {
		int cnt = 0;
		for(boolean v : rmRows)
			cnt += v ? 1 : 0;
		return cnt;
	}

	public int getNumRemovedRows() {
		return getNumRemovedRows(_rmRows);
	}


	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		return apply(in, out);
	}
	
	@Override
	public void build(FrameBlock in) {
		if(_federated)
			_rmRows = computeRmRows(in);
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		// local rmRows for broadcasting encoder in spark
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

		return ret;
	}

	private boolean[] computeRmRows(FrameBlock in) {
		boolean[] rmRows = new boolean[in.getNumRows()];
		ValueType[] schema = in.getSchema();
		//TODO perf evaluate if column-wise scan more efficient
		//  (sequential but less impact of early abort)
		for(int i = 0; i < in.getNumRows(); i++) {
			Object val = in.get(i, _colID - 1);
			if (val == null || (schema[_colID - 1] == ValueType.STRING && val.toString().isEmpty())) {
				rmRows[i] = true;
				break; // early abort
			}
		}
		return rmRows;
	}

	
	@Override
	public void updateIndexRanges(long[] beginDims, long[] endDims) {
		// first update begin dims
		int numRowsToRemove = 0;
		for (int i = 0; i < beginDims[0] - 1 && i < _rmRows.length; i++)
			if (_rmRows[i])
				numRowsToRemove++;
		beginDims[0] -= numRowsToRemove;
		// update end dims
		for (int i = 0; i < endDims[0] - 1 && i < _rmRows.length; i++)
			if (_rmRows[i])
				numRowsToRemove++;
		endDims[0] -= numRowsToRemove;
	}
	
	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		//do nothing
		return out;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		//do nothing
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

	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o == null || getClass() != o.getClass())
			return false;
		EncoderOmit that = (EncoderOmit) o;
		return _federated == that._federated && Arrays.equals(_rmRows, that._rmRows);
	}

	@Override
	public int hashCode() {
		int result = Objects.hash(_federated);
		result = 31 * result + Arrays.hashCode(_rmRows);
		return result;
	}
}
