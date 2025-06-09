package org.apache.sysds.runtime.transform.decode;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.List;

public class ColumnDecoderComposite extends ColumnDecoder {
    private static final long serialVersionUID = 5790600547144743716L;

    private List<ColumnDecoder> _decoders = null;
    protected ColumnDecoderComposite(Types.ValueType[] schema, List<ColumnDecoder> decoders) {
        super(schema, null);
        _decoders = decoders;
    }

    public ColumnDecoderComposite() { super(null, null); }

    /**
     * Helper function to slice original MatrixBlock into pieces with same decoder
     * */
    public List<ColumnInput> extractColumnInputs(MatrixBlock mb) {
        List<ColumnInput> inputs = new ArrayList<>();

        for (ColumnDecoder decoder : _decoders) {
            int[] colIDs = decoder.getColList();

            int colStart = colIDs[0] - 1;
            int colEnd = colIDs[colIDs.length - 1] - 1;

            MatrixBlock subMb = mb.slice(0, mb.getNumRows() - 1, colStart, colEnd, new MatrixBlock());

            ColumnBlock columnBlock = new ColumnBlock();
            columnBlock.data = subMb;
            columnBlock.targetCols = colIDs;

            ColumnInput columnInput = new ColumnInput();
            columnInput.columnBlock = columnBlock;
            columnInput.schema = _schema;
            columnInput.decoder = decoder;

            inputs.add(columnInput);
        }
        return inputs;
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        List<ColumnInput> inputs = extractColumnInputs(in);
        for (ColumnInput columnInput : inputs) {
            out = columnInput.decoder.columnDecode(columnInput.columnBlock.data, out);// 很重要！！在decode的方法中将拆开的column放到正确的位置上去！！
        }
        return out;
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out, final int k) {
        // TODO Parallelization Degree?
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
    public void updateIndexRanges(long[] beginDims, long[] endDims) {
        for(ColumnDecoder dec : _decoders)
            dec.updateIndexRanges(beginDims, endDims);
    }

    @Override
    public void initMetaData(FrameBlock meta) {
        for( ColumnDecoder decoder : _decoders )
            decoder.initMetaData(meta);
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        super.writeExternal(out);
        out.writeInt(_decoders.size());
        out.writeInt(_schema == null ? 0:_schema.length); //write #columns
        for(ColumnDecoder decoder : _decoders) {
            out.writeByte(ColumnDecoderFactory.getDecoderType(decoder));
            decoder.writeExternal(out);
        }
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException {
        super.readExternal(in);
        int decodersSize = in.readInt();
        int nCols = in.readInt();
        if (nCols > 0 && decodersSize > nCols*2)
            throw new IOException("Too many decoders");
        _decoders = new ArrayList<>();
        for(int i = 0; i < decodersSize; i++) {
            ColumnDecoder decoder = ColumnDecoderFactory.createInstance(in.readByte());
            decoder.readExternal(in);
            _decoders.add(decoder);
        }
    }
}
