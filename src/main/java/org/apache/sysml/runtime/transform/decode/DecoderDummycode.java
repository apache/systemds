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
 * Simple atomic decoder for dummycoded columns. This decoder builds internally
 * inverted column mappings from the given frame meta data. 
 *  
 */
public class DecoderDummycode extends Decoder
{
	private static final long serialVersionUID = 4758831042891032129L;
	
	private int[] _clPos = null;
	private int[] _cuPos = null;
	
	protected DecoderDummycode(List<ValueType> schema, int[] dcCols) {
		//dcCols refers to column IDs in output (non-dc)
		super(schema, dcCols);
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		out.ensureAllocatedColumns(in.getNumRows());
		for( int i=0; i<in.getNumRows(); i++ )
			for( int j=0; j<_colList.length; j++ )
				for( int k=_clPos[j]; k<_cuPos[j]; k++ )
					if( in.quickGetValue(i, k-1) != 0 ) {
						int col = _colList[j] - 1;
						out.set(i, col, UtilFunctions.doubleToObject(
								out.getSchema().get(col), k-_clPos[j]+1));
					}		
		return out;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		_clPos = new int[_colList.length]; //col lower pos 
		_cuPos = new int[_colList.length]; //col upper pos 
		for( int j=0, off=0; j<_colList.length; j++ ) {
			int colID = _colList[j];
			int ndist = (int)meta.getColumnMetadata()
					.get(colID-1).getNumDistinct();
			_clPos[j] = off + colID;
			_cuPos[j] = _clPos[j] + ndist;
			off += ndist - 1;
		}
	}
}
