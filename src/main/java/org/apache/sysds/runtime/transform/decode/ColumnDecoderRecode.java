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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class ColumnDecoderRecode extends ColumnDecoder {

    private static final long serialVersionUID = -3784249774608228805L;

    private HashMap<Long, Object>[] _rcMaps = null;
    private Object[][] _rcMapsDirect = null;
    private boolean _onOut = false;

    public ColumnDecoderRecode() {
        super(null, null);
    }

    protected ColumnDecoderRecode(ValueType[] schema, boolean onOut, int[] rcCols) {
        super(schema, rcCols);
        _onOut = onOut;
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        // TODO
        return null;
    }

    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        // TODO
    }
    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
        // TODO
        return null;
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
                if( meta.get(i, _colList[j]-1)==null )
                    break; //reached end of recode map
                String[] tmp = ColumnEncoderRecode.splitRecodeMapEntry(meta.get(i, _colList[j]-1).toString());
                Object obj = UtilFunctions.stringToObject(_schema[_colList[j]-1], tmp[0]);
                long lval = Long.parseLong(tmp[1]);
                map.put(lval, obj);
                max[j] = Math.max(lval, max[j]);
            }
            _rcMaps[j] = map;
        }

        //convert to direct lookup arrays
        if( Arrays.stream(max).allMatch(v -> v < Integer.MAX_VALUE) ) {
            _rcMapsDirect = new Object[_rcMaps.length][];
            for( int i=0; i<_rcMaps.length; i++ ) {
                Object[] arr = new Object[(int)max[i]];
                for(Map.Entry<Long,Object> e1 : _rcMaps[i].entrySet())
                    arr[e1.getKey().intValue()-1] = e1.getValue();
                _rcMapsDirect[i] = arr;
            }
        }
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
            for(Map.Entry<Long,Object> e1 : _rcMaps[i].entrySet()) {
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
