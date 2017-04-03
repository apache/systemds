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
	private static final long serialVersionUID = -8525203889417422598L;
	
	private int[] _dcCols = null;
	private int[] _srcCols = null;
	
	protected DecoderPassThrough(ValueType[] schema, int[] ptCols, int[] dcCols) {
		super(schema, ptCols);
		_dcCols = dcCols;
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		out.ensureAllocatedColumns(in.getNumRows());
		for( int i=0; i<in.getNumRows(); i++ ) {
			for( int j=0; j<_colList.length; j++ ) {
				int srcColID = _srcCols[j];
				int tgtColID = _colList[j];
				double val = in.quickGetValue(i, srcColID-1);
				out.set(i, tgtColID-1, UtilFunctions.doubleToObject(
						_schema[tgtColID-1], val));
			}
		}
		return out;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		if( _dcCols.length > 0 ) {
			//prepare source column id mapping w/ dummy coding
			_srcCols = new int[_colList.length];
			int ix1 = 0, ix2 = 0, off = 0;
			while( ix1<_colList.length ) {
				if( ix2>=_dcCols.length || _colList[ix1] < _dcCols[ix2] ) {
					_srcCols[ix1] = _colList[ix1] + off;
					ix1 ++;
				}
				else { //_colList[ix1] > _dcCols[ix2]
					off += (int)meta.getColumnMetadata()[_dcCols[ix2]-1]
							.getNumDistinct() - 1;
					ix2 ++;
				}
			}
		}
		else {
			//prepare direct source column mapping
			_srcCols = _colList;
		}
	}
}
