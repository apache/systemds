package org.apache.sysds.runtime.transform.decode;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.common.Types.ValueType;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

public class ColumnDecoderPassThrough extends ColumnDecoder {

    private static final long serialVersionUID = -8525203889417422598L;

    private int[] _dcCols = null;
    private int[] _srcCols = null;

    protected ColumnDecoderPassThrough(ValueType[] schema, int[] ptCols, int[] dcCols) {
        super(schema, ptCols);
        _dcCols = dcCols;
    }

    public ColumnDecoderPassThrough() {
        super(null, null);
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        // TODO:先去搞Bin的，那边有提示
        return null;
    }

    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        // TODO:同
    }

    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset){
        // TODO:不知道是干嘛的，毛就完了，稍微研究一下
        return null;
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
                    ColumnMetadata d =meta.getColumnMetadata()[_dcCols[ix2]-1];
                    off += d.isDefault() ? -1 : d.getNumDistinct() - 1;
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
    public void writeExternal(ObjectOutput os)
            throws IOException
    {
        super.writeExternal(os);
        os.writeInt(_srcCols.length);
        for(int i = 0; i < _srcCols.length; i++)
            os.writeInt(_srcCols[i]);

        os.writeInt(_dcCols.length);
        for(int i = 0; i < _dcCols.length; i++)
            os.writeInt(_dcCols[i]);
    }

    @Override
    public void readExternal(ObjectInput in)
            throws IOException
    {
        super.readExternal(in);
        _srcCols = new int[in.readInt()];
        for(int i = 0; i < _srcCols.length; i++)
            _srcCols[i] = in.readInt();

        _dcCols = new int[in.readInt()];
        for(int i = 0; i < _dcCols.length; i++)
            _dcCols[i] = in.readInt();
    }
}
