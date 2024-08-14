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
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Simple atomic decoder for recoded columns. This decoder builds internally
 * inverted recode maps from the given frame meta data. 
 *  
 */
public class DecoderRecode extends Decoder
{
	private static final long serialVersionUID = -3784249774608228805L;

	private HashMap<Long, Object>[] _rcMaps = null;
	private boolean _onOut = false;

	// public DecoderRecode() {
	// 	super(null, null);
	// }

	protected DecoderRecode(ValueType[] schema, boolean onOut, int[] rcCols) {
		super(schema, rcCols);
		_onOut = onOut;
	}
	
	public Object getRcMapValue(int i, long key) {
		return _rcMaps[i].get(key);
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		decode(in, out, 0, in.getNumRows());
		return out;
	}

	@Override
	public void decode(MatrixBlock in, FrameBlock out, int rl, int ru) {
		if( _onOut ) { //recode on output (after dummy)
			for( int i=rl; i<ru; i++ ) {
				for( int j=0; j<_colList.length; j++ ) {
					int colID = _colList[j];
					double val = UtilFunctions.objectToDouble(
							out.getSchema()[colID-1], out.get(i, colID-1));
					long key = UtilFunctions.toLong(val);
					out.set(i, colID-1, getRcMapValue(j, key));
				}
			}
		}
		else { //recode on input (no dummy)
			out.ensureAllocatedColumns(in.getNumRows());
			for( int i=rl; i<ru; i++ ) {
				for( int j=0; j<_colList.length; j++ ) {
					double val = in.get(i, _colList[j]-1);
					long key = UtilFunctions.toLong(val);
					out.set(i, _colList[j]-1, getRcMapValue(j, key));
				}
			}
		}
	}

	@Override
	@SuppressWarnings("unchecked")
	public Decoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
		List<Integer> cols = new ArrayList<>();
		List<HashMap<Long, Object>> rcMaps = new ArrayList<>();
		for(int i = 0; i < _colList.length; i++) {
			int col = _colList[i];
			if(col >= colStart && col < colEnd) {
				// add the correct column, removed columns before start
				// colStart - 1 because colStart is 1-based
				int corrColumn = col - (colStart - 1);
				cols.add(corrColumn);
				rcMaps.add(new HashMap<>(_rcMaps[i]));
			}
		}
		if(cols.isEmpty())
			// empty encoder -> sub range encoder does not exist
			return null;

		int[] colList = cols.stream().mapToInt(i -> i).toArray();
		DecoderRecode subRangeDecoder = new DecoderRecode(
			Arrays.copyOfRange(_schema, colStart - 1, colEnd - 1), _onOut, colList);
		subRangeDecoder._rcMaps = rcMaps.toArray(new HashMap[0]);
		return subRangeDecoder;
	}

	@Override
	@SuppressWarnings("unchecked")
	public void initMetaData(FrameBlock meta) {
		//initialize recode maps according to schema
		_rcMaps = new HashMap[_colList.length];
		long[] max = new long[_colList.length];
		for( int j=0; j<_colList.length; j++ ) {
			HashMap<Long, Object> map = new HashMap<>();
			for( int i=0; i<meta.getNumRows(); i++ ) {
				try{

					if( meta.get(i, _colList[j]-1)==null )
						break; //reached end of recode map
					String[] tmp = ColumnEncoderRecode.splitRecodeMapEntry(meta.get(i, _colList[j]-1).toString());
					Object obj = UtilFunctions.stringToObject(_schema[_colList[j]-1], tmp[0]);
					long lval = Long.parseLong(tmp[1]);
					map.put(lval, obj);
					max[j] = Math.max(lval, max[j]);
				}
				catch(Exception e){
					throw new DMLRuntimeException("Failed to reinitialize recode map from: " + (meta.getColumn(_colList[j]-1)), e);
				}
			}
			_rcMaps[j] = map;
		}
		
		//convert to direct lookup arrays
		// if( Arrays.stream(max).allMatch(v -> v < Integer.MAX_VALUE) ) {
		// 	_rcMapsDirect = new Object[_rcMaps.length][];
		// 	for( int i=0; i<_rcMaps.length; i++ ) {
		// 		Object[] arr = new Object[(int)max[i]];
		// 		for(Entry<Long,Object> e1 : _rcMaps[i].entrySet())
		// 			arr[e1.getKey().intValue()-1] = e1.getValue();
		// 		_rcMapsDirect[i] = arr;
		// 	}
		// }
	}
	
	/**
	 * Parses a line of &lt;token, ID, count&gt; into &lt;token, ID&gt; pairs, where 
	 * quoted tokens (potentially including separators) are supported.
	 * 
	 * @param entry entry line (token, ID, count)
	 * @param pair token-ID pair
	 */
	public static void parseRecodeMapEntry(String entry, Pair<String,String> pair) {
		int ixq = entry.lastIndexOf('"');
		String token = UtilFunctions.unquote(entry.substring(0,ixq+1));
		int idx = ixq+2;
		while(entry.charAt(idx) != TfUtils.TXMTD_SEP.charAt(0))
			idx++;
		String id = entry.substring(ixq+2,idx); 
		pair.set(token, id);
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeBoolean(_onOut);
		out.writeInt(_rcMaps.length);
		for(int i = 0; i < _rcMaps.length; i++) {
			out.writeInt(_rcMaps[i].size());
			for(Entry<Long,Object> e1 : _rcMaps[i].entrySet()) {
				out.writeLong(e1.getKey());
				out.writeUTF(e1.getValue().toString());
			}
		}
	}

	@Override
	@SuppressWarnings("unchecked")
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		_onOut = in.readBoolean();
		_rcMaps = new HashMap[in.readInt()];
		for(int i = 0; i < _rcMaps.length; i++) {
			HashMap<Long, Object> maps = new HashMap<>();
			int size = in.readInt();
			for(int j = 0; j < size; j++)
				maps.put(in.readLong(), in.readUTF());
			_rcMaps[i] = maps;
		}
	}
}
