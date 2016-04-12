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

package org.apache.sysml.runtime.transform.decode;

import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * Simple atomic decoder for passing through numeric columns to the output.
 * This is required for block-wise decoding. 
 *  
 */
public class DecoderPassThrough extends Decoder
{
	private int[] _ptCols = null; //0-based
	
	protected DecoderPassThrough(List<ValueType> schema, int[] ptCols) {
		super(schema);
		_ptCols = ptCols;
	}

	@Override
	public Object[] decode(double[] in, Object[] out) {
		for( int j=0; j<_ptCols.length; j++ )
			out[_ptCols[j]] = in[_ptCols[j]];
		return out;
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		out.ensureAllocatedColumns(in.getNumRows());
		for( int i=0; i<in.getNumRows(); i++ ) {
			for( int j=0; j<_ptCols.length; j++ ) {
				double val = in.quickGetValue(i, _ptCols[j]);
				out.set(i, _ptCols[j], UtilFunctions.doubleToObject(
						_schema.get(_ptCols[j]), val));
			}
		}
		return out;
	}
}
