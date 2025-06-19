package org.apache.sysds.runtime.transform.decode;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;


import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

public abstract class ColumnDecoder implements Externalizable {
    protected static final Log LOG = LogFactory.getLog(Decoder.class.getName());
    private static final long serialVersionUID = -1732411001366177787L;

    protected ValueType[] _schema;
    protected int[] _colList;
    protected String[] _colnames = null;
    protected ColumnDecoder(ValueType[] schema, int[] colList) {
        _schema = schema;
        _colList = colList;
    }

    public ValueType[] getSchema() {
        return _schema;
    }

    public void setColnames(String[] colnames) {
        _colnames = colnames;
    }

    public String[] getColnames() {
        return _colnames;
    }

    public int[] getColList() {return _colList;}
    /**
     * Block decode API converting a matrix block into a frame block.
     *
     * @param in  Input matrix block
     * @param out Output frame block
     * @return returns given output frame block for convenience
     */
    public abstract FrameBlock columnDecode(MatrixBlock in, FrameBlock out);

    /**
     * Block decode API converting a matrix block into a frame block in parallel.
     *
     * @param in  Input matrix block
     * @param out Output frame block
     * @param k   Parallelization degree
     * @return returns the given output frame block for convenience
     */
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out, int k) {
        return columnDecode(in, out);
    }

    /**
     * Block decode row block
     *
     * @param in  input Matrix Block
     * @param out output FrameBlock
     * @param rl  row start to decode
     * @param ru  row end to decode (not inclusive)
     */
    public abstract void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru);

    /**
     * Returns a new Decoder that only handles a sub range of columns. The sub-range refers to the columns after
     * decoding.
     *
     * @param colStart         the start index of the sub-range (1-based, inclusive)
     * @param colEnd           the end index of the sub-range (1-based, exclusive)
     * @param dummycodedOffset the offset of dummycoded segments before colStart
     * @return a decoder of the same type, just for the sub-range
     */
    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
        throw new DMLRuntimeException(
                getClass().getSimpleName() + " does not support the creation of a sub-range decoder");
    }

    /**
     * Update index-ranges to after decoding. Note that only Dummycoding changes the ranges.
     *
     * @param beginDims the begin indexes before encoding
     * @param endDims   the end indexes before encoding
     */
    public void updateIndexRanges(long[] beginDims, long[] endDims) {
        // do nothing - default
    }

    public abstract void initMetaData(FrameBlock meta);

    /**
     * Redirects the default java serialization via externalizable to our default
     * hadoop writable serialization for efficient broadcast/rdd serialization.
     *
     * @param os object output
     * @throws IOException if IOException occurs
     */
    @Override
    public void writeExternal(ObjectOutput os)
            throws IOException
    {
        int size1 = (_colList == null) ? 0 : _colList.length;
        os.writeInt(size1);
        for(int i = 0; i < size1; i++)
            os.writeInt(_colList[i]);

        int size2 = (_colnames == null) ? 0 : _colnames.length;
        os.writeInt(size2);
        for(int j = 0; j < size2; j++)
            os.writeUTF(_colnames[j]);

        int size3 = (_schema == null) ? 0 : _schema.length;
        os.writeInt(size3);
        for(int j = 0; j < size3; j++)
            os.writeByte(_schema[j].ordinal());
    }

    /**
     * Redirects the default java serialization via externalizable to our default
     * hadoop writable serialization for efficient broadcast/rdd deserialization.
     *
     * @param in object input
     * @throws IOException if IOException occur
     */
    @Override
    public void readExternal(ObjectInput in)
            throws IOException
    {
        int size1 = in.readInt();
        _colList = (size1 == 0) ? null : new int[size1];
        for(int i = 0; i < size1; i++)
            _colList[i] = in.readInt();

        int size2 = in.readInt();
        _colnames = (size2 == 0) ? null : new String[size2];
        for(int j = 0; j < size2; j++) {
            _colnames[j] = in.readUTF();
        }

        int size3 = in.readInt();
        _schema = (size3 == 0) ? null : new ValueType[size3];
        for(int j = 0; j < size3; j++) {
            _schema[j] = ValueType.values()[in.readByte()];
        }
    }
}
