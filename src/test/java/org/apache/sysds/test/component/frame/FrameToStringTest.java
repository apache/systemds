/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Test;

public class FrameToStringTest {
	@Test
	public void testDefault() {
		FrameBlock f = createFrameBlock();
		assertTrue(DataConverter.toString(f).length() < 75);
	}
	
	@Test
	public void test100x100() {
		FrameBlock f = createFrameBlock();
		assertTrue(DataConverter.toString(f, false, " ", "\n", 100, 100, 3).length() < 75);
	}

	@Test
	public void testDecimalClampsFractionDigits() {
		FrameBlock f = new FrameBlock(new ValueType[]{ValueType.FP64}, new String[]{"C1"});
		f.ensureAllocatedColumns(1);
		f.set(0, 0, 5.244058388023880);
		// decimal=2 must print exactly two fraction digits, not DecimalFormat's default max of 3
		String out = DataConverter.toString(f, false, " ", "\n", 1, 1, 2);
		assertTrue("expected value clamped to 5.24, got: " + out, out.contains("5.24\n"));
		assertFalse("decimal=2 must not print three digits: " + out, out.contains("5.244"));
	}

	@Test
	public void testDecimalPadsAndRounds() {
		FrameBlock f = new FrameBlock(new ValueType[]{ValueType.FP64}, new String[]{"C1"});
		f.ensureAllocatedColumns(2);
		f.set(0, 0, 22.0);                // integer-valued: padded up to the requested digits
		f.set(1, 0, 5.244058388023880);   // rounded at the last requested digit
		String out = DataConverter.toString(f, false, " ", "\n", 2, 1, 4);
		assertTrue("expected 22.0000 padded: " + out, out.contains("22.0000\n"));
		assertTrue("expected 5.2441 rounded: " + out, out.contains("5.2441\n"));
	}

	@Test
	public void testNegativeDecimalUsesDefaultFormatting() {
		FrameBlock f = new FrameBlock(new ValueType[]{ValueType.FP64}, new String[]{"C1"});
		f.ensureAllocatedColumns(2);
		f.set(0, 0, 22.0);                // integer-valued: no fraction digits when unconstrained
		f.set(1, 0, 5.244058388023880);   // default cap of three fraction digits
		// decimal < 0 leaves DecimalFormat unconstrained (no min/max fraction digits set)
		String out = DataConverter.toString(f, false, " ", "\n", 2, 1, -1);
		assertTrue("expected unpadded 22: " + out, out.contains("22\n"));
		assertFalse("integer value must not be padded: " + out, out.contains("22.0"));
		assertTrue("expected default 5.244: " + out, out.contains("5.244\n"));
		assertFalse("must not print a fourth digit: " + out, out.contains("5.2441"));
	}
	
	private FrameBlock createFrameBlock() {
		FrameBlock f = new FrameBlock(new ValueType[]{ValueType.STRING, ValueType.STRING});
		for(int i=0; i<5; i++)
			f.appendRow(new String[] {"a","b"});
		return f;
	}
}
