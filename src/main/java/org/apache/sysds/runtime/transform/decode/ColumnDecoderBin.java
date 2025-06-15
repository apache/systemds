package org.apache.sysds.runtime.transform.decode;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

public class ColumnDecoderBin extends ColumnDecoder {
    private static final long serialVersionUID = -3784249774608228805L;

    private int[] _numBins;
    private double[][] _binMins = null;
    private double[][] _binMaxs = null;

    public ColumnDecoderBin() {
        super(null, null);
    }

    protected ColumnDecoderBin(ValueType[] schema, int[] binCols) {
        super(schema, binCols);
    }


    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        out.ensureAllocatedColumns(in.getNumRows());
        columnDecode(in, out, 0, in.getNumRows());
        return out;

        // in.ColumnBlock.targetCols里面有每列在原来Block中的位置，位置最后写回的时候调用out.set(r, targetCols[xxx], value);就可以
    }

    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        for (int i = rl; i < ru; i++) {
            for (int j = 0; j < _colList.length; j++) {
                double val = in.get(i, j);
                if (!Double.isNaN(val)) {
                    int key = (int) Math.round(val);
                    double bmin = _binMins[j][key - 1];
                    double bmax = _binMaxs[j][key - 1];
                    double oval = bmin + (bmax - bmin) / 2 + (val - key) * (bmax - bmin);
                    out.getColumn(_colList[j] - 1).set(i, oval);
                } else {
                    out.getColumn(_colList[j] - 1).set(i, val);
                }
            }
        }
    }

    @Override
    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
        if (colEnd - colStart != 1)
            throw new NotImplementedException();

        for (int i = 0; i < _colList.length; i++) {
            if (_colList[i] == colStart) {
                ValueType[] schema = (_schema != null) ? new ValueType[]{_schema[colStart - 1]} : null;
                ColumnDecoderBin sub = new ColumnDecoderBin(schema, new int[]{colStart});
                sub._numBins = new int[]{_numBins[i]};
                sub._binMins = new double[][]{_binMins[i]};
                sub._binMaxs = new double[][]{_binMaxs[i]};
                return sub;
            }
        }
        return null;
    }

    @Override
    public void initMetaData(FrameBlock meta) {
        //initialize bin boundaries
        _numBins = new int[_colList.length];
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
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        super.writeExternal(out);
        for( int i=0; i<_colList.length; i++ ) {
            int len = _numBins[i];
            out.writeInt(len);
            for(int j=0; j<len; j++) {
                out.writeDouble(_binMins[i][j]);
                out.writeDouble(_binMaxs[i][j]);
            }
        }
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException {
        super.readExternal(in);
        _numBins = new int[_colList.length];
        _binMins = new double[_colList.length][];
        _binMaxs = new double[_colList.length][];
        for( int i=0; i<_colList.length; i++ ) {
            int len = in.readInt();
            _numBins[i] = len;
            for(int j=0; j<len; j++) {
                _binMins[i][j] = in.readDouble();
                _binMaxs[i][j] = in.readDouble();
            }
        }
    }
}
