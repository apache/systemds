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

import java.io.File;
import java.nio.file.Files;
import java.util.Map;

import org.apache.sysds.test.TestUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.runtime.io.FrameWriterParquetParallel;
import org.apache.sysds.test.functions.io.parquet.ParquetTestUtils.ParquetMetadataInfo;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class WriteParquetTest {

	private static final String TEMP_FILE     = System.getProperty("java.io.tmpdir") + "/systemds_write_parquet_test.parquet";
	private static final String TEMP_PAR_PATH = System.getProperty("java.io.tmpdir") + "/systemds_write_parquet_test_par";

	// See ParquetTestUtils.generatePublicTestFiles(): these are generated with Spark's DataFrameWriter 
	private static File testFileDir;
	private static String[] PUBLIC_FILES;

	@BeforeClass
	public static void generateTestFiles() throws Exception {
		testFileDir = Files.createTempDirectory("systemds_parquet_public_test_files").toFile();
		Map<String, String> files = ParquetTestUtils.generatePublicTestFiles(testFileDir);
		PUBLIC_FILES = new String[] { files.get("userdata1"), files.get("alltypes_plain"), files.get("all") };
	}

	@AfterClass
	public static void cleanupTestFiles() {
		File[] children = testFileDir.listFiles();
		if (children != null)
			for (File f : children)
				f.delete();
		testFileDir.delete();
	}

	@After
	public void cleanup() {
		new File(TEMP_FILE).delete();
		deleteRecursive(new File(TEMP_PAR_PATH));
	}

	private static void deleteRecursive(File f) {
		if (f.isDirectory())
			for (File c : f.listFiles()) deleteRecursive(c);
		f.delete();
	}

	@Test
	public void testRoundtripPublicFiles() throws Exception {
		for (String filename : PUBLIC_FILES) {
			ParquetMetadataInfo info = ParquetTestUtils.inferMetadata(filename);

			FrameReaderParquet reader = new FrameReaderParquet();
			FrameBlock original = reader.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);

			FrameWriterParquet writer = new FrameWriterParquet();
			writer.writeFrameToHDFS(original, TEMP_FILE, original.getNumRows(), original.getNumColumns());

			FrameBlock result = reader.readFrameFromHDFS(TEMP_FILE, info.schema, info.names, info.rlen, info.clen);

			TestUtils.compareFrames(original, result, false);
		}
	}

	@Test
	public void testMultiPartFileRoundtrip() throws Exception {
		// Create two parquet part files and verify that the parallel reader
		// reads both files correctly and combines them into the expected result.
		ValueType[] schema = {
			ValueType.STRING,
			ValueType.INT32,
			ValueType.INT64,
			ValueType.FP32,
			ValueType.FP64,
			ValueType.BOOLEAN
		};
		String[] names = { "name", "age", "id", "score", "ratio", "active" };
		String[][] data = {
			{ "Alice", "30", "1000", "1.5", "0.75", "true"  },
			{ "Bob",   "25", "2000", "2.5", "0.50", "false" },
			{ "Carol", "40", "3000", "3.5", "0.25", "true"  },
			{ "Dave",  "35", "4000", "4.5", "0.10", "false" },
			{ "Eve",   "28", "5000", "5.5", "0.90", "true"  },
			{ "Frank", "45", "6000", "6.5", "0.60", "false" }
		};
		FrameBlock original = new FrameBlock(schema, names, data);

		new File(TEMP_PAR_PATH).mkdir();
		FrameWriterParquet writer = new FrameWriterParquet();
		writer.writeFrameToHDFS(original.slice(0, 2), TEMP_PAR_PATH + "/part-0.parquet", 3, schema.length);
		writer.writeFrameToHDFS(original.slice(3, 5), TEMP_PAR_PATH + "/part-1.parquet", 3, schema.length);

		FrameBlock result = new FrameReaderParquetParallel()
			.readFrameFromHDFS(TEMP_PAR_PATH, schema, names, 6, schema.length);

		TestUtils.compareFrames(original, result, false);
	}

	@Test
	public void testParallelRoundtrip() throws Exception {
		ValueType[] schema = {
			ValueType.STRING,
			ValueType.INT32,
			ValueType.INT64,
			ValueType.FP32,
			ValueType.FP64,
			ValueType.BOOLEAN
		};
		String[] names = { "name", "age", "id", "score", "ratio", "active" };
		String[][] data = {
			{ "Alice", "30", "1000", "1.5", "0.75", "true"  },
			{ "Bob",   "25", "2000", "2.5", "0.50", "false" },
			{ "Carol", "40", "3000", "3.5", "0.25", "true"  }
		};
		FrameBlock original = new FrameBlock(schema, names, data);

		new FrameWriterParquetParallel().writeFrameToHDFS(original, TEMP_PAR_PATH, original.getNumRows(), original.getNumColumns());
		FrameBlock result = new FrameReaderParquetParallel().readFrameFromHDFS(TEMP_PAR_PATH, schema, names, original.getNumRows(), original.getNumColumns());

		TestUtils.compareFrames(original, result, false);
	}

	@Test
	public void testRoundtrip() throws Exception {
		ValueType[] schema = {
			ValueType.STRING,
			ValueType.INT32,
			ValueType.INT64,
			ValueType.FP32,
			ValueType.FP64,
			ValueType.BOOLEAN
		};
		String[] names = { "name test", "age", "id", "score", "ratio", "active" };
		String[][] data = {
			{ "Alice", "30",  "1000", "1.5", "0.75",  "true"  },
			{ "Bob",   "25",  "2000", "2.5", "0.50",  "false" },
			{ "Carol", "40",  "3000", "3.5", "0.25",  "true"  }
		};

		FrameBlock original = new FrameBlock(schema, names, data);

		// Write
		FrameWriterParquet writer = new FrameWriterParquet();
		writer.writeFrameToHDFS(original, TEMP_FILE, original.getNumRows(), original.getNumColumns());

		// Read back
		FrameReaderParquet reader = new FrameReaderParquet();
		FrameBlock result = reader.readFrameFromHDFS(TEMP_FILE, schema, names, original.getNumRows(), original.getNumColumns());

		TestUtils.compareFrames(original, result, false);
	}
}
