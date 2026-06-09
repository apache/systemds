package org.apache.sysds.test.component.frame.transform;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoder;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderComposite;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderDummycode;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderPassThrough;
import org.apache.sysds.runtime.transform.encode.CompressedEncode;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class TransformDummySeparatedTest extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(TransformDummySeparatedTest.class.getName());

	final FrameBlock data;

	public TransformDummySeparatedTest() {
		data = TestUtils.generateRandomFrameBlock(100, new org.apache.sysds.common.Types.ValueType[] {
			org.apache.sysds.common.Types.ValueType.UINT8 }, 231);
		data.setSchema(new org.apache.sysds.common.Types.ValueType[] {
			org.apache.sysds.common.Types.ValueType.INT32 });
	}

	@Test
	public void testDummySeparatedBasic() {

        test("{ids:true, dummycode:[1]}");

	}

	public void test(String spec) {
		try {
			FrameBlock meta = null;
			MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(), meta);
			
			MatrixBlock out = encoder.encode(data);
			meta = encoder.getMetaData(new FrameBlock(data.getNumColumns(), org.apache.sysds.common.Types.ValueType.STRING));
			MatrixBlock out2 = encoder.apply(data);

			// Compare consistency
			TestUtils.compareMatrices(out, out2, 0, "Not Equal after apply");

			// Print output
			System.out.println("== Encoded MatrixBlock ==");
			System.out.println(out.toString());

			System.out.println("== Metadata FrameBlock ==");
			System.out.println(meta.toString());

		} catch (Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

    @Override
    public void setUp() {
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'setUp'");
    }
}
