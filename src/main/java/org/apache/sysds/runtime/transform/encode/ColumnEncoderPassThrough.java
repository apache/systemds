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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Simple composite encoder that applies a list of encoders 
 * in specified order. By implementing the default encoder API
 * it can be used as a drop-in replacement for any other encoder. 
 * 
 */
public class ColumnEncoderPassThrough extends ColumnEncoder
{
	private static final long serialVersionUID = -8473768154646831882L;
	
	protected ColumnEncoderPassThrough(int ptCols) {
		super(ptCols); //1-based
	}

	public ColumnEncoderPassThrough() {
		this(-1);
	}

	@Override
	public void build(FrameBlock in) {
		//do nothing
	}
	
	@Override 
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol) {
		int col = _colID - 1; // 1-based
		ValueType vt = in.getSchema()[col];
		for( int i=0; i<in.getNumRows(); i++ ) {
			Object val = in.get(i, col);
			out.quickSetValue(i, outputCol, (val==null||(vt==ValueType.STRING
				&& val.toString().isEmpty())) ? Double.NaN :
				UtilFunctions.objectToDouble(vt, val));
		}
		return out;
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol) {
		// only transfer from in to out
		int col = _colID - 1; // 1-based
		for( int i=0; i<in.getNumRows(); i++ ) {
			double val = in.quickGetValue(i, col);
			out.quickSetValue(i, outputCol, val);
		}
		return out;
	}


	@Override
	public void mergeAt(ColumnEncoder other, int row) {
		if(other instanceof ColumnEncoderPassThrough) {
			return;
		}
		super.mergeAt(other, row);
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		//do nothing
		return meta;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		//do nothing
	}
}
