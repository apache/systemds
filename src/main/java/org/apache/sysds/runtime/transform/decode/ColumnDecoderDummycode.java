package org.apache.sysds.runtime.transform.decode;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

public class ColumnDecoderDummycode extends ColumnDecoder {
    private static final long serialVersionUID = 4758831042891032129L;

    private int _cl = -1;  // dummy start col
    private int _cu = -1;  // dummy end col
    private int _category = -1;  // used for single category optimization (not used here)

    public ColumnDecoderDummycode(Types.ValueType schema, int colID, int offset) {
        super(schema, colID, offset); // _colID = colID
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        long t0 = System.nanoTime(); // 开始计时

        out.ensureAllocatedColumns(in.getNumRows());
        columnDecode(in, out, 0, in.getNumRows());

        long t1 = System.nanoTime(); // 结束计时
        System.out.println(this.getClass() + " time: " + (t1 - t0) / 1e6 + " ms");
        return out;
    }

    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        int col = _colID; // already 0-based
        for (int i = rl; i < ru; i++) {
            for (int k = _cl; k < _cu; k++) {
                if (in.get(i, k - 1) != 0) {
                    Object val = UtilFunctions.doubleToObject(out.getSchema()[col], k - _cl + 1);
                    out.set(i, col, val);
                    break;
                }
            }
        }
    }

    @Override
    public void initMetaData(FrameBlock meta) {
        int col = _colID; // already 0-based
        ColumnMetadata d = meta.getColumnMetadata()[col];
        int ndist = d.isDefault() ? 0 : (int) d.getNumDistinct();
        ndist = ndist < -1 ? 0 : ndist;
        _cl = col + 1;
        _cu = _cl + ndist;
    }

    @Override
    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
        // Not applicable in single-column version
        return null;
    }

    @Override
    public void writeExternal(ObjectOutput os) throws IOException {
        super.writeExternal(os);
        os.writeInt(_cl);
        os.writeInt(_cu);
        os.writeInt(_category);
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException {
        super.readExternal(in);
        _cl = in.readInt();
        _cu = in.readInt();
        _category = in.readInt();
    }
}
