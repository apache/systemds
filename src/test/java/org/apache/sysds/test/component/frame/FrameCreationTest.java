package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.junit.Test;

public class FrameCreationTest {
	@Test
	public void createEmptyRow() {
		FrameBlock a = createEmptyRow(new ValueType[] {ValueType.STRING, ValueType.STRING});
		assertEquals(null, a.get(0, 0));
		assertEquals(null, a.get(0, 1));
	}

	private FrameBlock createEmptyRow(ValueType[] schema) {
		String[][] arr = new String[1][];
		arr[0] = new String[schema.length];
		return new FrameBlock(schema, null, arr);
	}
}
