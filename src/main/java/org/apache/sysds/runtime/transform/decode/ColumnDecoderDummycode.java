package org.apache.sysds.runtime.transform.decode;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ColumnDecoderDummycode extends ColumnDecoder {

    private static final long serialVersionUID = 4758831042891032129L;

    private int[] _clPos = null;
    private int[] _cuPos = null;
    // category index for dedicated single-column decoders (-1 if not used)
    private int _category = -1;

    protected ColumnDecoderDummycode(Types.ValueType[] schema, int[] colList) {
        super(schema, colList);
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        out.ensureAllocatedColumns(in.getNumRows());
        columnDecode(in, out, 0, in.getNumRows());
        return out;
    }

    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        if(_category >= 0) {
            int col = _colList[0] - 1;
            Object val = UtilFunctions.doubleToObject(out.getSchema()[col], _category);
            for(int i = rl; i < ru; i++)
                if(in.get(i, _clPos[0]-1) == 1)
                    synchronized(out) { out.set(i, col, val); }
        }
        else {
            for( int i=rl; i<ru; i++ )
                for( int j=0; j<_colList.length; j++ )
                    for( int k=_clPos[j]; k<_cuPos[j]; k++ )
                        if( in.get(i, k-1) != 0 ) {
                            int col = _colList[j] - 1;
                            Object val = UtilFunctions.doubleToObject(out.getSchema()[col], k-_clPos[j]+1);
                            synchronized(out) { out.set(i, col, val); }
                        }
        }
    }

    @Override
    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
        // special case: request for exactly one encoded column
        if(colEnd - colStart == 1) {
            int encCol = colStart;
            for(int j=0; j<_clPos.length; j++)
                if(encCol >= _clPos[j] && encCol < _cuPos[j]) {
                    ColumnDecoderDummycode dec = new ColumnDecoderDummycode(
                            new Types.ValueType[]{_schema[_colList[j]-1]},
                            new int[]{_colList[j]});
                    dec._clPos = new int[]{1};
                    dec._cuPos = new int[]{2};
                    dec._category = encCol - _clPos[j] + 1;
                    return dec;
                }
            return null;
        }
        else {
            List<Integer> dcList = new ArrayList<>();
            List<Integer> clPosList = new ArrayList<>();
            List<Integer> cuPosList = new ArrayList<>();

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

            ColumnDecoderDummycode dec = new ColumnDecoderDummycode(
                    Arrays.copyOfRange(_schema, colStart - 1, colEnd - 1),
                    dcList.stream().mapToInt(i -> i).toArray());
            dec._clPos = clPosList.stream().mapToInt(i -> i).toArray();
            dec._cuPos = cuPosList.stream().mapToInt(i -> i).toArray();
            return dec;
        }
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
