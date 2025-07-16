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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.*;

public class ColumnDecoderRecode extends ColumnDecoder {

    private static final long serialVersionUID = -3784249774608228805L;

    private HashMap<Long, Object> _rcMap = null;
    private Object[] _rcMapDirect = null;
    private boolean _onOut = false;

    public ColumnDecoderRecode() {
        super(null, -1, -1);
    }

    protected ColumnDecoderRecode(ValueType schema, boolean onOut, int rcCol, int offset) {
        super(schema, rcCol, offset);
        _onOut = onOut;
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        long t0 = System.nanoTime();
        out.ensureAllocatedColumns(in.getNumRows());
        for (int i = 0; i < in.getNumRows(); i++) {
            double val = in.get(i, _offset);
            Object obj = _rcMapDirect[(int)val-1];
            out.set(i, _colID, obj);
        }
        long t1 = System.nanoTime();
        System.out.println(this.getClass() + " time: " + (t1 - t0) / 1e6 + " ms");
        return out;
    }

    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        for (int i = rl; i < ru; i++) {
            long val = UtilFunctions.toLong(in.get(i, _offset)); // 修复类型错误
            Object obj = getRcMapValue(val);
            out.set(i, _colID, obj);
        }

    }
    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
        return null;

        //List<Integer> cols = new ArrayList<>();
        //List<HashMap<Long, Object>> rcMaps = new ArrayList<>();
        //List<Object[]> rcMapsDirect = (_rcMapsDirect != null) ? new ArrayList<>() : null;
        //for(int i = 0; i < _colList.length; i++) {
        //    int col = _colList[i];
        //    if(col >= colStart && col < colEnd) {
        //        cols.add(col - (colStart - 1));
        //        //rcMaps.add(new HashMap<>(_rcMaps[i]));
        //        rcMaps.add(_rcMaps[i]);
        //        if(rcMapsDirect != null)
        //            rcMapsDirect.add(_rcMapsDirect[i]);
        //    }
        //}
        //if(cols.isEmpty())
        //    return null;
//
        //int[] colList = cols.stream().mapToInt(v -> v).toArray();
        //ColumnDecoderRecode dec = new ColumnDecoderRecode(
        //        Arrays.copyOfRange(_multiSchema, colStart - 1, colEnd - 1), _onOut, colList);
        //dec._rcMaps = rcMaps.toArray(new HashMap[0]);
        //if(rcMapsDirect != null)
        //    dec._rcMapsDirect = rcMapsDirect.toArray(new Object[0][]);
        //return dec;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void initMetaData(FrameBlock meta) {
        long t0 = System.nanoTime();
        int col = _colID; // already 0-based
        _rcMap = new HashMap<>();
        long max = 0;
        for(int i=0; i<meta.getNumRows(); i++) {
            Object val = meta.get(i, col);
            if(val == null)
                break;
            String[] tmp = ColumnEncoderRecode.splitRecodeMapEntry(val.toString());
            Object obj = UtilFunctions.stringToObject(_schema, tmp[0]);
            long lval = Long.parseLong(tmp[1]);
            _rcMap.put(lval, obj);
            max = Math.max(max, lval);
        }
        if(max < Integer.MAX_VALUE) {
            _rcMapDirect = new Object[(int)max];
            for(Map.Entry<Long,Object> e : _rcMap.entrySet())
                _rcMapDirect[e.getKey().intValue()-1] = e.getValue();
        }

        long t1 = System.nanoTime();
        System.out.println(this.getClass() + " meta time: " + (t1 - t0) / 1e6 + " ms");
        //initialize recode maps according to schema
        //_rcMaps = new HashMap[_colList.length];
        //long[] max = new long[_colList.length];
        //for( int j=0; j<_colList.length; j++ ) {
        //    HashMap<Long, Object> map = new HashMap<>();
        //    for( int i=0; i<meta.getNumRows(); i++ ) {
        //        if( meta.get(i, _colList[j]-1)==null )
        //            break; //reached end of recode map
        //        String[] tmp = ColumnEncoderRecode.splitRecodeMapEntry(meta.get(i, _colList[j]-1).toString());
        //        Object obj = UtilFunctions.stringToObject(_multiSchema[_colList[j]-1], tmp[0]);
        //        long lval = Long.parseLong(tmp[1]);
        //        map.put(lval, obj);
        //        max[j] = Math.max(lval, max[j]);
        //    }
        //    _rcMaps[j] = map;
        //}
//
        ////convert to direct lookup arrays
        //if( Arrays.stream(max).allMatch(v -> v < Integer.MAX_VALUE) ) {
        //    _rcMapsDirect = new Object[_rcMaps.length][];
        //    for( int i=0; i<_rcMaps.length; i++ ) {
        //        Object[] arr = new Object[(int)max[i]];
        //        for(Map.Entry<Long,Object> e1 : _rcMaps[i].entrySet())
        //            arr[e1.getKey().intValue()-1] = e1.getValue();
        //        _rcMapsDirect[i] = arr;
        //    }
        //}
    }
    public Object getRcMapValue(long key) {
        return (_rcMapDirect != null && key > 0 && key <= _rcMapDirect.length) ?
                _rcMapDirect[(int)key-1] : _rcMap.get(key);
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

        //super.writeExternal(out);
        //out.writeBoolean(_onOut);
        //out.writeInt(_rcMaps.length);
        //for(int i = 0; i < _rcMaps.length; i++) {
        //    out.writeInt(_rcMaps[i].size());
        //    for(Map.Entry<Long,Object> e1 : _rcMaps[i].entrySet()) {
        //        out.writeLong(e1.getKey());
        //        out.writeUTF(e1.getValue().toString());
        //    }
        //}
        super.writeExternal(out);
        out.writeBoolean(_onOut);
        out.writeInt(_rcMap.size());
        for(Map.Entry<Long,Object> e : _rcMap.entrySet()) {
            out.writeLong(e.getKey());
            out.writeUTF(e.getValue().toString());
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void readExternal(ObjectInput in) throws IOException {
        //super.readExternal(in);
        //_onOut = in.readBoolean();
        //_rcMaps = new HashMap[in.readInt()];
        //for(int i = 0; i < _rcMaps.length; i++) {
        //    HashMap<Long, Object> maps = new HashMap<>();
        //    int size = in.readInt();
        //    for(int j = 0; j < size; j++)
        //        maps.put(in.readLong(), in.readUTF());
        //    _rcMaps[i] = maps;
        //}
        super.readExternal(in);
        _onOut = in.readBoolean();
        int size = in.readInt();
        _rcMap = new HashMap<>();
        long max = 0;
        for(int i = 0; i < size; i++) {
            long key = in.readLong();
            String val = in.readUTF();
            _rcMap.put(key, val);
            max = Math.max(max, key);
        }
        if(max < Integer.MAX_VALUE) {
            _rcMapDirect = new Object[(int)max];
            for(Map.Entry<Long,Object> e : _rcMap.entrySet())
                _rcMapDirect[e.getKey().intValue()-1] = e.getValue();
        }
    }
}
