package org.apache.sysds.test.component.compress.indexes;

import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex.ColIndexType;
import org.junit.Test;

public class NegativeIndexTest {
	@Test(expected = DMLCompressionException.class)
	public void notValidRead() {
		try {

			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			fos.writeByte(ColIndexType.UNKNOWN.ordinal());
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);
			ColIndexFactory.read(fis);
		}
		catch(IOException e) {
			fail("Wrong type of exception");
		}
	}
}
