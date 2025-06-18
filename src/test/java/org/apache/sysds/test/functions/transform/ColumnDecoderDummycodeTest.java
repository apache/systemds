package org.apache.sysds.test.functions.transform;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.decode.ColumnDecoder;
import org.apache.sysds.runtime.transform.decode.ColumnDecoderFactory;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ColumnDecoderDummycodeTest extends AutomatedTestBase {
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    @Test
    public void testColumnDecoderDummycode() {
        try {
            int rows = 20;
            double[][] arr = new double[rows][1];
            for (int i = 0; i < rows; i++)
                arr[i][0] = (i % 3) + 1;
            MatrixBlock mb = DataConverter.convertToMatrixBlock(arr);
            FrameBlock data = DataConverter.convertToFrameBlock(mb);
            String spec = "{ids:true, dummycode:[1]}";

            MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, null);
            MatrixBlock encoded = enc.encode(data);
            FrameBlock meta = enc.getMetaData(new FrameBlock(1, ValueType.STRING));

            Decoder dec = DecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta, encoded.getNumColumns());
            FrameBlock expected = new FrameBlock(data.getSchema());
            dec.decode(encoded, expected);

            ColumnDecoder cdec = ColumnDecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta);
            FrameBlock actual = new FrameBlock(data.getSchema());
            cdec.columnDecode(encoded, actual);

            TestUtils.compareFrames(expected, actual, false);
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}