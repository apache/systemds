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

import org.junit.Test;

import org.apache.sysds.test.TestUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.runtime.io.FrameWriterParquetParallel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

import java.io.IOException;

/**
 * This test class verifies that a FrameBlock with different data types is correctly written and read from Parquet files. 
 * It tests both sequential and parallel implementations. In these tests a FrameBlock is created, populated with sample 
 * data, written to a Parquet file, and then read back into a new FrameBlock. The test compares the original and read 
 * data to ensure that schema information is preserved and that data conversion is performed correctly.
 */
public class FrameParquetSchemaTest extends AutomatedTestBase {

	private final static String TEST_NAME = "FrameParquetSchemaTest";
	private final static String TEST_DIR = "functions/io/parquet";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameParquetSchemaTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"Rout"}));
	}


	/**
	 * Test for sequential writer and reader
	 *
	 */
	@Test
	public void testParquetWriteReadAllSchemaTypes() throws IOException {
		runWriteReadAllSchemaTypes(new FrameWriterParquet(), new FrameReaderParquet(), output("Rout"));
	}

	/**
	 * Test for multithreaded writer and reader
	 *
	 */
	@Test
	public void testParquetWriteReadAllSchemaTypesParallel() throws IOException {
		runWriteReadAllSchemaTypes(new FrameWriterParquetParallel(), new FrameReaderParquetParallel(),
			output("Rout_parallel"));
	}

	private static void runWriteReadAllSchemaTypes(FrameWriter writer, FrameReader reader, String fname)
		throws IOException {
		// Define a schema with one column per type
		ValueType[] schema = new ValueType[] {ValueType.FP64, ValueType.FP32, ValueType.INT32, ValueType.INT64,
			ValueType.BOOLEAN, ValueType.STRING};

		// Create an empty frame block with the above schema
		FrameBlock fb = new FrameBlock(schema);

		// Populate frame block
		Object[][] rows = new Object[][] {{1.0, 1.1f, 10, 100L, true, "A"}, {2.0, 2.1f, 20, 200L, false, "B"},
			{3.0, 3.1f, 30, 300L, true, "C"}, {4.0, 4.1f, 40, 400L, false, "D"}, {5.0, 5.1f, 50, 500L, true, "E"}};

		for(Object[] row : rows)
			fb.appendRow(row);

		int numRows = fb.getNumRows();
		int numCols = fb.getNumColumns();

		writer.writeFrameToHDFS(fb, fname, numRows, numCols);

		String[] colNames = fb.getColumnNames();
		FrameBlock fbRead = reader.readFrameFromHDFS(fname, schema, colNames, numRows, numCols);

		// Compare the original and the read frame blocks
		TestUtils.compareFrames(fb, fbRead, false);
	}
}
