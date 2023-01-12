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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Random;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibApplySchema;
import org.junit.Test;

public class FrameApplySchema {

	@Test
	public void testApplySchemaStringToBoolean() {
		try {

			FrameBlock fb = genStringContainingBoolean(10, 2);
			ValueType[] schema = new ValueType[] {ValueType.BOOLEAN, ValueType.BOOLEAN};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema);
			assertTrue(ret.getColumn(0).getValueType() == ValueType.BOOLEAN);
			assertTrue(ret.getColumn(1).getValueType() == ValueType.BOOLEAN);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToInt() {
		try {
			FrameBlock fb = genStringContainingInteger(10, 2);
			ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema);
			assertTrue(ret.getColumn(0).getValueType() == ValueType.INT32);
			assertTrue(ret.getColumn(1).getValueType() == ValueType.INT32);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntSingleCol() {
		try {
			FrameBlock fb = genStringContainingInteger(10, 1);
			ValueType[] schema = new ValueType[] {ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema);
			assertTrue(ret.getColumn(0).getValueType() == ValueType.INT32);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntDirectCallSingleThread() {
		try {
			FrameBlock fb = genStringContainingInteger(10, 3);
			ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32, ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, 1);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntDirectCallMultiThread() {
		try {
			FrameBlock fb = genStringContainingInteger(10, 3);
			ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32, ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, 3);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntDirectCallMultiThreadSingleCol() {
		try {
			FrameBlock fb = genStringContainingInteger(10, 1);
			ValueType[] schema = new ValueType[] {ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, 3);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = DMLRuntimeException.class)
	public void testInvalidInput() {
		FrameBlock fb = genStringContainingInteger(10, 10);
		ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32, ValueType.INT32};
		FrameLibApplySchema.applySchema(fb, schema, 3);
	}

	@Test
	public void testUnkownColumnDefaultToString() {
		FrameBlock fb = genStringContainingInteger(10, 3);
		ValueType[] schema = new ValueType[] {ValueType.UNKNOWN, ValueType.INT32, ValueType.INT32};
		fb = FrameLibApplySchema.applySchema(fb, schema, 3);
		assertEquals(ValueType.UNKNOWN, fb.getSchema()[0]);
	}

	private FrameBlock genStringContainingInteger(int row, int col) {
		FrameBlock ret = new FrameBlock();
		Random r = new Random(31);
		for(int c = 0; c < col; c++) {
			String[] column = new String[row];
			for(int i = 0; i < row; i++)
				column[i] = "" + r.nextInt();

			ret.appendColumn(column);
		}
		return ret;
	}

	private FrameBlock genStringContainingBoolean(int row, int col) {
		FrameBlock ret = new FrameBlock();
		Random r = new Random(31);
		for(int c = 0; c < col; c++) {
			String[] column = new String[row];
			for(int i = 0; i < row; i++)
				column[i] = "" + r.nextBoolean();

			ret.appendColumn(column);
		}
		return ret;
	}
}
