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

package org.apache.sysds.runtime.transform.decode;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

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
	
	protected DecoderDummycode(ValueType[] schema, int[] dcCols) {
		//dcCols refers to column IDs in output (non-dc)
		super(schema, dcCols);
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		//TODO perf (exploit sparse representation for better asymptotic behavior)
		out.ensureAllocatedColumns(in.getNumRows());
		decode(in, out, 0, in.getNumRows());
		return out;
	}

	@Override
	public void decode(MatrixBlock in, FrameBlock out, int rl, int ru) {
		//TODO perf (exploit sparse representation for better asymptotic behavior)
		// out.ensureAllocatedColumns(in.getNumRows());
		for( int i=rl; i<ru; i++ )
			for( int j=0; j<_colList.length; j++ )
				for( int k=_clPos[j]; k<_cuPos[j]; k++ )
					if( in.get(i, k-1) != 0 ) {
						int col = _colList[j] - 1;
						out.set(i, col, UtilFunctions.doubleToObject(
							out.getSchema()[col], k-_clPos[j]+1));
					}
		System.out.println("dummy"+out);

	}
	
	@Override
	public Decoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
		List<Integer> dcList = new ArrayList<>();
		List<Integer> clPosList = new ArrayList<>();
		List<Integer> cuPosList = new ArrayList<>();
		
		// get the column IDs for the sub range of the dummycode columns and their destination positions,
		// where they will be decoded to
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j];
			if (colID >= colStart && colID < colEnd) {
				dcList.add(colID - (colStart - 1));
				clPosList.add(_clPos[j] - dummycodedOffset);
				cuPosList.add(_cuPos[j] - dummycodedOffset);
			}
		}
		if (dcList.isEmpty())
			return null;
		// create sub-range decoder
		int[] colList = dcList.stream().mapToInt(i -> i).toArray();
		DecoderDummycode subRangeDecoder = new DecoderDummycode(
			Arrays.copyOfRange(_schema, colStart - 1, colEnd - 1), colList);
		subRangeDecoder._clPos = clPosList.stream().mapToInt(i -> i).toArray();
		subRangeDecoder._cuPos = cuPosList.stream().mapToInt(i -> i).toArray();
		return subRangeDecoder;
	}
	
	@Override
	public void updateIndexRanges(long[] beginDims, long[] endDims) {
		if(_colList == null)
			return;
		
		long lowerColDest = beginDims[1];
		long upperColDest = endDims[1];
		for(int i = 0; i < _colList.length; i++) {
			long numDistinct = _cuPos[i] - _clPos[i];
			
			if(_cuPos[i] <= beginDims[1] + 1)
				if(numDistinct > 0)
					lowerColDest -= numDistinct - 1;
			
			if(_cuPos[i] <= endDims[1] + 1)
				if(numDistinct > 0)
					upperColDest -= numDistinct - 1;
		}
		beginDims[1] = lowerColDest;
		endDims[1] = upperColDest;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		_clPos = new int[_colList.length]; //col lower pos 
		_cuPos = new int[_colList.length]; //col upper pos 
		for( int j=0, off=0; j<_colList.length; j++ ) {
			int colID = _colList[j];
			ColumnMetadata d = meta.getColumnMetadata()[colID-1];
			int ndist = d.isDefault() ? 0 : (int)d.getNumDistinct();
			ndist = ndist < -1 ? 0: ndist;
			_clPos[j] = off + colID;
			_cuPos[j] = _clPos[j] + ndist;
			off += ndist - 1;
		}
	}

	@Override
	public void writeExternal(ObjectOutput os) throws IOException {
		super.writeExternal(os);
		os.writeInt(_clPos.length);
		for(int i = 0; i < _clPos.length; i++) {
			os.writeInt(_clPos[i]);
			os.writeInt(_cuPos[i]);
		}
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		int size = in.readInt();
		_clPos = new int[size];
		_cuPos = new int[size];
		for(int i = 0; i < size; i++) {
			_clPos[i] = in.readInt();
			_cuPos[i] = in.readInt();
		}
	}
}
