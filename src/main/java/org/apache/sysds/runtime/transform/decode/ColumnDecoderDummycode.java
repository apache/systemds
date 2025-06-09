package org.apache.sysds.runtime.transform.decode;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

public class ColumnDecoderDummycode extends ColumnDecoder {

    private static final long serialVersionUID = 4758831042891032129L;

    private int[] _clPos = null;
    private int[] _cuPos = null;

    protected ColumnDecoderDummycode(Types.ValueType[] schema, int[] colList) {
        super(schema, colList);
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

    @Override
    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
        // TODO
        return null;
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
