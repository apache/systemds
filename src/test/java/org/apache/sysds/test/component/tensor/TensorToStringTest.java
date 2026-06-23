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

package org.apache.sysds.test.component.tensor;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Test;

public class TensorToStringTest {
	@Test
	public void testDecimalClampsFractionDigits() {
		TensorBlock tb = new TensorBlock(ValueType.FP64, new int[]{1, 1});
		tb.allocateBlock();
		tb.set(0, 0, 5.244058388023880);
		// decimal=2 must print exactly two fraction digits, not DecimalFormat's default max of 3
		String out = DataConverter.toString(tb, false, " ", "\n", "[", "]", 1, 1, 2);
		assertTrue("expected value clamped to 5.24, got: " + out, out.contains("5.24"));
		assertFalse("decimal=2 must not print three digits: " + out, out.contains("5.244"));
	}

	@Test
	public void testDecimalPadsAndRounds() {
		TensorBlock tb = new TensorBlock(ValueType.FP64, new int[]{1, 2});
		tb.allocateBlock();
		tb.set(0, 0, 22.0);                // integer-valued: padded up to the requested digits
		tb.set(0, 1, 5.244058388023880);   // rounded at the last requested digit
		String out = DataConverter.toString(tb, false, " ", "\n", "[", "]", 1, 2, 4);
		assertTrue("expected 22.0000 padded: " + out, out.contains("22.0000"));
		assertTrue("expected 5.2441 rounded: " + out, out.contains("5.2441"));
	}
}
