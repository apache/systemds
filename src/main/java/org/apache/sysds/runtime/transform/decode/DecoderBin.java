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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Simple atomic decoder for binned columns. This decoder builds internally
 * arrays of lower/upper bin boundaries, accesses these boundaries in
 * constant time for incoming values and  
 *  
 */
public class DecoderBin extends Decoder {

	private static final long serialVersionUID = -3784249774608228805L;

	// a) column bin boundaries
	private int[] _dcCols = null;
	private int[] _srcCols = null;
	private double[][] _binMins = null;
	private double[][] _binMaxs = null;

	public DecoderBin() {
		super(null, null);
	}

	protected DecoderBin(ValueType[] schema, int[] binCols, int[] dcCols) {
		this(schema, binCols, dcCols, null);
	}

	protected DecoderBin(ValueType[] schema, int[] binCols, int[] dcCols, int[] hashCols) {
		super(schema, binCols);
		_dcCols = dcCols;
		_dcHashCols = hashCols;
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		out.ensureAllocatedColumns(in.getNumRows());
		decode(in, out, 0, in.getNumRows());
		return out;
	}

	@Override
	public void decode(MatrixBlock in, FrameBlock out, int rl, int ru) {
		for( int i=rl; i< ru; i++ ) {
			for( int j=0; j<_colList.length; j++ ) {
				final Array<?> a = out.getColumn(_colList[j] - 1);
				final double val = in.get(i, _srcCols[j] - 1);
				if(!Double.isNaN(val)){
					final int key = (int) Math.round(val);
					if(key == 0){
						a.set(i, _binMins[j][key]);
					}
					else{
						double bmin = _binMins[j][key - 1];
						double bmax = _binMaxs[j][key - 1];
						double oval = bmin + (bmax - bmin) / 2 // bin center
							+ (val - key) * (bmax - bmin); // bin fractions
						a.set(i, oval);
					}
				}
				else 
					a.set(i, val); // NaN
			}
		}
	}

	@Override
	public Decoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
		// federated not supported yet
		throw new NotImplementedException();
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		//initialize bin boundaries
		_binMins = new double[_colList.length][];
		_binMaxs = new double[_colList.length][];
		
		//parse and insert bin boundaries
		for( int j=0; j<_colList.length; j++ ) {
			int numBins = (int)meta.getColumnMetadata(_colList[j]-1).getNumDistinct();
			_binMins[j] = new double[numBins];
			_binMaxs[j] = new double[numBins];
			for( int i=0; i<meta.getNumRows() & i<numBins; i++ ) {
				if( meta.get(i, _colList[j]-1)==null  ) {
					if( i+1 < numBins )
						throw new DMLRuntimeException("Did not reach number of bins: "+(i+1)+"/"+numBins);
					break; //reached end of bins
				}
				String[] parts = UtilFunctions.splitRecodeEntry(
					meta.get(i, _colList[j]-1).toString());
				_binMins[j][i] = Double.parseDouble(parts[0]);
				_binMaxs[j][i] = Double.parseDouble(parts[1]);
			}
		}


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
					int dcCol = _dcCols[ix2];
					off += getNumDummycodeDistinct(meta, dcCol, isHashCol(dcCol)) - 1;
					ix2 ++;
				}
			}
		}
		else {
			//prepare direct source column mapping
			_srcCols = _colList;
		}
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		// bin boundaries; the per-column bin count is the length of the boundary arrays
		for( int i=0; i<_colList.length; i++ ) {
			int len = _binMins[i].length;
			out.writeInt(len);
			for(int j=0; j<len; j++) {
				out.writeDouble(_binMins[i][j]);
				out.writeDouble(_binMaxs[i][j]);
			}
		}
		// source-column mapping (rebuilt in initMetaData, but persisted for Spark broadcast)
		out.writeInt(_srcCols.length);
		for(int i = 0; i < _srcCols.length; i++)
			out.writeInt(_srcCols[i]);

		out.writeInt(_dcCols == null ? 0 : _dcCols.length);
		for(int i = 0; _dcCols != null && i < _dcCols.length; i++)
			out.writeInt(_dcCols[i]);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		_binMins = new double[_colList.length][];
		_binMaxs = new double[_colList.length][];
		for( int i=0; i<_colList.length; i++ ) {
			int len = in.readInt();
			_binMins[i] = new double[len];
			_binMaxs[i] = new double[len];
			for(int j=0; j<len; j++) {
				_binMins[i][j] = in.readDouble();
				_binMaxs[i][j] = in.readDouble();
			}
		}
		_srcCols = new int[in.readInt()];
		for(int i = 0; i < _srcCols.length; i++)
			_srcCols[i] = in.readInt();

		_dcCols = new int[in.readInt()];
		for(int i = 0; i < _dcCols.length; i++)
			_dcCols[i] = in.readInt();
	}
}
