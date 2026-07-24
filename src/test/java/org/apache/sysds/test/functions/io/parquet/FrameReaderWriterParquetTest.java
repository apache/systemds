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

package org.apache.sysds.test.functions.io.parquet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Random-frame write/read round-trip test for the parquet frame reader/writer.
 */
public class FrameReaderWriterParquetTest {

	private static final String FILENAME = "target/testTemp/functions/io/parquet/FrameReaderWriterParquetTest/frame.parquet";

	// Parquet-supported value types
	private static final ValueType[] SCHEMA = {ValueType.FP64, ValueType.FP32, ValueType.INT32, ValueType.INT64,
		ValueType.BOOLEAN, ValueType.STRING};

	@Test
	public void testSingleRowSingleCol() throws IOException {
		runWriteReadRoundTrip(1, 1, 4669201);
	}

	@Test
	public void testSingleRowMultiCol() throws IOException {
		runWriteReadRoundTrip(1, 6, 4669201);
	}

	@Test
	public void testMultiRowSingleCol() throws IOException {
		runWriteReadRoundTrip(21, 1, 4669201);
	}

	@Test
	public void testMultiRowMultiCol() throws IOException {
		runWriteReadRoundTrip(42, 5, 4669201);
	}

	@Test
	public void testLargerFrame() throws IOException {
		runWriteReadRoundTrip(694, 6, 4669201);
	}

	@Test
	public void testValueTypeEdgeCases() throws IOException {
		// type min/max, empty string, special chars (comma/quote/newline/unicode), column name with space
		ValueType[] schema = {ValueType.FP32, ValueType.FP64, ValueType.INT32, ValueType.INT64, ValueType.BOOLEAN,
			ValueType.STRING};
		String[] names = {"f 32", "f64", "i32", "i64", "b", "s"};
		String[][] data = {
			{String.valueOf(Float.MAX_VALUE), String.valueOf(Double.MAX_VALUE), String.valueOf(Integer.MAX_VALUE),
				String.valueOf(Long.MAX_VALUE), "true", ""},
			{String.valueOf(-Float.MAX_VALUE), String.valueOf(-Double.MAX_VALUE), String.valueOf(Integer.MIN_VALUE),
				String.valueOf(Long.MIN_VALUE), "false", "a,b\"c\nd"},
			{"0.0", "0.0", "0", "0", "true", "unicode_é中"}};
		FrameBlock in = new FrameBlock(schema, names, data);

		new FrameWriterParquet().writeFrameToHDFS(in, FILENAME, 3, 6);
		FrameBlock out = new FrameReaderParquet().readFrameFromHDFS(FILENAME, schema, names, 3, 6);

		TestUtils.compareFrames(in, out, false);
	}

	@Test
	public void testNullsInStringColumn() throws IOException {
		// Numeric columns from String[][] convert null to 0. Only test string nulls here.
		// Numeric nulls are covered by the Spark-written userdata1/all files in ReadParquetTest.
		ValueType[] schema = {ValueType.STRING, ValueType.STRING};
		String[] names = {"a", "b"};
		String[][] data = {{"x", "y"}, {null, null}, {"p", null}};
		FrameBlock in = new FrameBlock(schema, names, data);

		new FrameWriterParquet().writeFrameToHDFS(in, FILENAME, 3, 2);
		FrameBlock out = new FrameReaderParquet().readFrameFromHDFS(FILENAME, schema, names, 3, 2);

		org.junit.Assert.assertNull(out.get(1, 0));
		org.junit.Assert.assertNull(out.get(1, 1));
		org.junit.Assert.assertNull(out.get(2, 1));
		org.junit.Assert.assertNotNull(out.get(0, 0));
		org.junit.Assert.assertNotNull(out.get(2, 0));
		TestUtils.compareFrames(in, out, false);
	}

	@Test
	public void testMismatchedSchemaRead() throws IOException {
		// requested frame types differ from the parquet physical types -> per-cell conversion path
		ValueType[] writeSchema = {ValueType.INT32, ValueType.FP32, ValueType.BOOLEAN, ValueType.FP64,
			ValueType.STRING};
		String[] names = {"i32", "f32", "b", "f64", "s"};
		String[][] data = {{"1", "1.5", "true", "2.5", "7"}, {"2", "2.5", "false", "3.5", null}};
		FrameBlock in = new FrameBlock(writeSchema, names, data);

		new FrameWriterParquet().writeFrameToHDFS(in, FILENAME, 2, 5);

		ValueType[] readSchema = {ValueType.INT64, ValueType.FP64, ValueType.STRING, ValueType.STRING, ValueType.FP64};
		FrameBlock out = new FrameReaderParquet().readFrameFromHDFS(FILENAME, readSchema, names, 2, 5);

		assertEquals(1L, out.get(0, 0));
		assertEquals(1.5, ((Number) out.get(0, 1)).doubleValue(), 0.0);
		assertEquals("true", out.get(0, 2));
		assertEquals("2.5", out.get(0, 3));
		assertEquals(7.0, ((Number) out.get(0, 4)).doubleValue(), 0.0);
		// numeric nulls keep the array default
		assertEquals(0.0, ((Number) out.get(1, 4)).doubleValue(), 0.0);
	}

	private void runWriteReadRoundTrip(int rows, int cols, long seed) throws IOException {
		try {
			ValueType[] schema = new ValueType[cols];
			for(int i = 0; i < cols; i++)
				schema[i] = SCHEMA[i % SCHEMA.length];

			FrameBlock writeBlock = TestUtils.generateRandomFrameBlock(rows, schema, seed);

			new FrameWriterParquet().writeFrameToHDFS(writeBlock, FILENAME, rows, cols);
			FrameBlock readBlock = new FrameReaderParquet().readFrameFromHDFS(FILENAME, schema,
				writeBlock.getColumnNames(), rows, cols);

			TestUtils.compareFrames(writeBlock, readBlock, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
