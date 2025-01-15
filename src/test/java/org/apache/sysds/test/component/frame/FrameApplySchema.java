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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.DoubleArray;
import org.apache.sysds.runtime.frame.data.columns.IntegerArray;
import org.apache.sysds.runtime.frame.data.columns.OptionalArray;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.apache.sysds.runtime.frame.data.lib.FrameLibApplySchema;
import org.apache.sysds.runtime.frame.data.lib.FrameLibCompress;
import org.apache.sysds.runtime.frame.data.lib.FrameUtil;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.test.component.frame.array.FrameArrayTests;
import org.apache.sysds.test.component.frame.compress.FrameCompressTestUtils;
import org.junit.Test;
import org.mockito.Mockito;

public class FrameApplySchema {
	protected static final Log LOG = LogFactory.getLog(FrameApplySchema.class.getName());

	// static {
	// 	FrameLibApplySchema.PAR_ROW_THRESHOLD = 10;
	// }

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
			FrameBlock ret = FrameLibApplySchema.applySchema(fb.copyShallow(), schema, 3);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntWithNullParallel() {
		try {
			FrameBlock fb = genStringContainingOptInteger(1300, 1);
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

	@Test
	public void testApplySchemaStringToIntWithNullSingle() {
		try {
			FrameBlock fb = genStringContainingOptInteger(100, 1);
			ValueType[] schema = new ValueType[] {ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, new boolean[] {true}, 1);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntWithNullSingleMultiCol() {
		try {
			FrameBlock fb = genStringContainingOptInteger(1026, 4);
			ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32, ValueType.INT32, ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, new boolean[] {true, true, true, true}, 1);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntWithNullParallelMultiCol() {
		try {
			FrameBlock fb = genStringContainingOptInteger(1300, 4);
			ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32, ValueType.INT32, ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, new boolean[] {true, true, true, true}, 4);
			for(int i = 0; i < ret.getNumColumns(); i++) {
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);
				assertTrue(ret.getColumn(i) instanceof OptionalArray);
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntWithNullPartiallyParallelMultiCol() {
		try {
			FrameBlock fb = genStringContainingOptInteger(1300, 4);
			ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32, ValueType.INT32, ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, new boolean[] {true, true, false, true}, 4);
			for(int i = 0; i < ret.getNumColumns(); i++) {
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);
				if(i != 2)
					assertTrue(ret.getColumn(i) instanceof OptionalArray);
				else
					assertFalse(ret.getColumn(i) instanceof OptionalArray);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntSTRINGWithNullPartiallyParallelMultiCol() {
		try {
			FrameBlock fb = genStringContainingOptInteger(1300, 4);
			ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32, ValueType.STRING, ValueType.STRING};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, new boolean[] {true, true, false, true}, 4);
			for(int i = 0; i < ret.getNumColumns(); i++) {
				if(i < 2) {
					assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);
					assertTrue(ret.getColumn(i) instanceof OptionalArray);
				}
				else {
					assertTrue(ret.getColumn(i).getValueType() == ValueType.STRING);
					assertTrue(ret.getColumn(i) instanceof StringArray);
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaStringToIntStringPartiallyParallelMultiCol() {
		try {
			FrameBlock fb = genStringContainingOptInteger(1300, 4);
			ValueType[] schema = new ValueType[] {ValueType.INT32, ValueType.INT32, ValueType.STRING, ValueType.INT32};
			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema, 4);
			for(int i = 0; i < ret.getNumColumns(); i++) {
				if(i != 2) {
					assertTrue(ret.getColumn(i).getValueType() == ValueType.INT32);
					assertTrue(ret.getColumn(i) instanceof IntegerArray);
				}
				else {
					assertTrue(ret.getColumn(i).getValueType() == ValueType.STRING);
					assertTrue(ret.getColumn(i) instanceof StringArray);
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaFrameBlockArg() {
		try {
			FrameBlock fb = genStringContainingOptInteger(100, 4);
			FrameBlock schema = new FrameBlock(
				new Array<?>[] {ArrayFactory.create(new String[] {"int"}), ArrayFactory.create(new String[] {"int"}),
					ArrayFactory.create(new String[] {"int"}), ArrayFactory.create(new String[] {"int"}),});

			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT64);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaFrameBlockArgOpt() {
		try {
			FrameBlock fb = genStringContainingOptInteger(100, 4);
			FrameBlock schema = new FrameBlock(
				new Array<?>[] {ArrayFactory.create(new String[] {"int" + FrameUtil.SCHEMA_SEPARATOR + "n"}),
					ArrayFactory.create(new String[] {"int" + FrameUtil.SCHEMA_SEPARATOR + "n"}),
					ArrayFactory.create(new String[] {"int" + FrameUtil.SCHEMA_SEPARATOR + "n"}),
					ArrayFactory.create(new String[] {"int" + FrameUtil.SCHEMA_SEPARATOR + "n"}),});

			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT64);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testApplySchemaFrameBlockArgExtra() {
		try {
			FrameBlock fb = genStringContainingInteger(320, 4);
			FrameBlock schema = new FrameBlock(
				new Array<?>[] {ArrayFactory.create(new String[] {"int" + FrameUtil.SCHEMA_SEPARATOR + "some"}),
					ArrayFactory.create(new String[] {"int" + FrameUtil.SCHEMA_SEPARATOR + "nope"}),
					ArrayFactory.create(new String[] {"int" + FrameUtil.SCHEMA_SEPARATOR + "thing"}),
					ArrayFactory.create(new String[] {"int" + FrameUtil.SCHEMA_SEPARATOR + "else"}),});

			FrameBlock ret = FrameLibApplySchema.applySchema(fb, schema);
			for(int i = 0; i < ret.getNumColumns(); i++)
				assertTrue(ret.getColumn(i).getValueType() == ValueType.INT64);
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

	@Test
	public void testUnkownColumnDefaultToStringPar() {
		try {
			FrameBlock fb = genStringContainingInteger(100, 3);
			ValueType[] schema = new ValueType[] {ValueType.UNKNOWN, ValueType.INT32, ValueType.INT32};
			fb = FrameLibApplySchema.applySchema(fb, schema, 3);
			assertEquals(ValueType.UNKNOWN, fb.getSchema()[0]);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	protected static FrameBlock genStringContainingInteger(int row, int col) {
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

	protected static FrameBlock genStringContainingOptInteger(int row, int col) {
		FrameBlock ret = new FrameBlock();
		Random r = new Random(31);
		for(int c = 0; c < col; c++) {
			String[] column = new String[row];
			for(int i = 0; i < row; i++)
				if(r.nextBoolean())
					column[i] = "" + r.nextInt();

			ret.appendColumn(column);
			column[row - 1] = null;
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

	@Test
	public void testApplySameSchemaReturnSame() {
		FrameBlock fb = genStringContainingInteger(10, 1);
		FrameBlock fb2 = FrameLibApplySchema.applySchema(fb, fb.getSchema(), 3);
		assertEquals(fb, fb2);
	}

	@Test
	public void testApplySchemaCompressed() {
		FrameBlock fb = FrameCompressTestUtils.generateCompressableBlock(1000, 3, 13241, ValueType.INT32);
		FrameBlock comp = FrameLibCompress.compress(fb, 13);
		FrameBlock fbOther = FrameLibApplySchema.applySchema(comp,
			new ValueType[] {ValueType.INT64, ValueType.INT64, ValueType.INT64});

		FrameArrayTests.compare(fb.getColumn(0), fbOther.getColumn(0));
		FrameArrayTests.compare(fb.getColumn(1), fbOther.getColumn(1));
		FrameArrayTests.compare(fb.getColumn(2), fbOther.getColumn(2));

	}

	@Test
	public void testApplySchemaCompressedParallel() {
		FrameBlock fb = FrameCompressTestUtils.generateCompressableBlock(1000, 3, 13241, ValueType.INT32);
		FrameBlock comp = FrameLibCompress.compress(fb, 13);
		FrameBlock fbOther = FrameLibApplySchema.applySchema(comp,
			new ValueType[] {ValueType.INT64, ValueType.INT64, ValueType.INT64}, 132);

		FrameArrayTests.compare(fb.getColumn(0), fbOther.getColumn(0));
		FrameArrayTests.compare(fb.getColumn(1), fbOther.getColumn(1));
		FrameArrayTests.compare(fb.getColumn(2), fbOther.getColumn(2));

	}

	@Test(expected = RuntimeException.class)
	public void failingTransformation() {
		try {
			DoubleArray m = mock(DoubleArray.class);
			when(m.getValueType()).thenReturn(ValueType.FP64);
			when(m.size()).thenReturn(30);

			when(m.changeType(ValueType.INT32)).thenThrow(new RuntimeException());
			FrameBlock fb = new FrameBlock(new Array<?>[] {m});
			FrameLibApplySchema.applySchema(fb, new ValueType[] {ValueType.INT32});
		}
		catch(Exception e) {
			// e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void failingParallelPartialTransformation() {
		try {
			DoubleArray m = mock(DoubleArray.class);
			when(m.getValueType()).thenReturn(ValueType.FP64);
			when(m.size()).thenReturn(1030);
			when(m.analyzeValueType()).thenReturn(new Pair<>(ValueType.INT32, false));

			Array<?> i = new IntegerArray(new int[1030]);

			Mockito.<Array<?>>when(m.changeTypeWithNulls(any(), anyInt(), anyInt())).thenThrow(new RuntimeException())
				.thenThrow(new RuntimeException()).thenReturn(i);

			FrameBlock fb = new FrameBlock(new Array<?>[] {m, new DoubleArray(new double[1030])});
			FrameLibApplySchema.applySchema(fb, new ValueType[] {ValueType.INT32, ValueType.INT32}, 5);

		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}
}
