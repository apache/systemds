package org.apache.sysds.runtime.transform.decode;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class ColumnDecoderComposite extends ColumnDecoder {
    private static final long serialVersionUID = 5790600547144743716L;

    private List<ColumnDecoder> _decoders = null;
    protected ColumnDecoderComposite(Types.ValueType[] schema, List<ColumnDecoder> decoders) {
        super(schema, null);
        _decoders = decoders;
    }

    public ColumnDecoderComposite() { super(null, null); }

    private List<MatrixBlock> sliceColumns(MatrixBlock mb, int[] cols) {
        List<MatrixBlock> list = new ArrayList<>();
        MatrixBlock ret = new MatrixBlock(mb.getNumRows(), 1, false);
        for (int col : cols) {
            for (int i = 0; i < mb.getNumRows(); i++) {
                ret.set(i, 0, mb.get(i, col - 1));
            }
            list.add(ret);
        }
        return list;
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        out.ensureAllocatedColumns(in.getNumRows());
        for (ColumnDecoder dec : _decoders) {
            List<MatrixBlock> slices = sliceColumns(in, dec.getColList());
            for (int c = 0; c < slices.size(); c++) {
                ColumnDecoder sub = dec.getColList().length == 1 ? dec :
                        dec.subRangeDecoder(dec.getColList()[c], dec.getColList()[c] + 1, 0);
                if (sub == null)
                    throw new RuntimeException("Decoder does not support column slicing: " + dec.getClass());
                sub.columnDecode(slices.get(c), out);
            }
        }
        return out;
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out, final int k) {
        //final ExecutorService pool = CommonThreadPool.get(k);
        //List<Future<ColumnInput>> futures = new ArrayList<>();
        //try {
        //    for (ColumnInput columnInput : inputs) {
//
        //    }
        //}
        //catch (Exception e) {
        //    throw new RuntimeException(e);
        //}
//
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
