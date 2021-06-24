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
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.UtilFunctions;

public class ColumnEncoderPassThrough extends ColumnEncoder {
	private static final long serialVersionUID = -8473768154646831882L;

	protected ColumnEncoderPassThrough(int ptCols) {
		super(ptCols); // 1-based
	}

	public ColumnEncoderPassThrough() {
		this(-1);
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

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol) {
		return apply(in, out, outputCol, 0, -1);
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		int col = _colID - 1; // 1-based
		ValueType vt = in.getSchema()[col];
		for(int i = rowStart; i < getEndIndex(in.getNumRows(), rowStart, blk); i++) {
			Object val = in.get(i, col);
			double v = (val == null ||
				(vt == ValueType.STRING && val.toString().isEmpty())) 
					? Double.NaN : UtilFunctions.objectToDouble(vt, val);
			out.quickSetValueThreadSafe(i, outputCol, v);
		}
		return out;
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		// only transfer from in to out
		int end = (blk <= 0) ? in.getNumRows() : in.getNumRows() < rowStart + blk ? in.getNumRows() : rowStart + blk;
		int col = _colID - 1; // 1-based
		for(int i = rowStart; i < end; i++) {
			double val = in.quickGetValueThreadSafe(i, col);
			out.quickSetValueThreadSafe(i, outputCol, val);
		}
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
}
