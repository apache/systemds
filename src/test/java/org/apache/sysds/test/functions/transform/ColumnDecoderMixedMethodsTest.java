
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

public class ColumnDecoderMixedMethodsTest extends AutomatedTestBase {
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    @Test
    public void testColumnDecoderMixedMethods() {
        try {
            int rows = 3;
            double[][] arr = new double[rows][5];
            for (int i = 0; i < rows; i++) {
                arr[i][0] = (i % 4) + 1; // recode column
                arr[i][1] = i + 1;       // bin column
                arr[i][2] = 100 + i;     // recode column
                arr[i][3] = i + 1;       // bin column
                arr[i][4] = 100 + i;     // recode column

            }
            MatrixBlock mb = DataConverter.convertToMatrixBlock(arr);
            FrameBlock data = DataConverter.convertToFrameBlock(mb);
            String spec = "{ids:true, recode:[1,3], bin:[{id:2, method:equi-width, numbins:4},{id:4, method:equi-width, numbins:4}]}";

            MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(), null);
            MatrixBlock encoded = enc.encode(data);
            FrameBlock meta = enc.getMetaData(new FrameBlock(data.getNumColumns(), ValueType.STRING));

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