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
import java.util.Arrays;
import java.util.Map;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.runtime.io.FrameWriterParquetParallel;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.io.parquet.ParquetTestUtils.ParquetMetadataInfo;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Verifies the parquet frame writers against Spark's parquet reader.
 *
 * Column 0 carries the row index so Spark reads of multi-file outputs can be compared order-independently (sorted by
 * id), since engines do not guarantee row order across files.
 */
public class WriteParquetTest {

	private static final String TEMP_FILE = System.getProperty("java.io.tmpdir")
		+ "/systemds_write_parquet_test.parquet";
	private static final String TEMP_PAR_PATH = System.getProperty("java.io.tmpdir")
		+ "/systemds_write_parquet_test_par";

	private static final ValueType[] SCHEMA = {ValueType.INT64, ValueType.STRING, ValueType.FP64, ValueType.BOOLEAN,
		ValueType.INT32, ValueType.FP32};
	private static final long SEED = 4669201;

	// See ParquetTestUtils.generatePublicTestFiles(): these are generated with Spark's DataFrameWriter
	private static File testFileDir;
	private static String[] PUBLIC_FILES;

	@BeforeClass
	public static void generateTestFiles() throws Exception {
		testFileDir = Files.createTempDirectory("systemds_parquet_public_test_files").toFile();
		Map<String, String> files = ParquetTestUtils.generatePublicTestFiles(testFileDir);
		PUBLIC_FILES = new String[] {files.get("userdata1"), files.get("alltypes_plain"), files.get("all")};
	}

	@AfterClass
	public static void cleanupTestFiles() {
		File[] children = testFileDir.listFiles();
		if(children != null)
			for(File f : children)
				f.delete();
		testFileDir.delete();
	}

	@After
	public void cleanup() {
		new File(TEMP_FILE).delete();
		deleteRecursive(new File(TEMP_PAR_PATH));
	}

	private static void deleteRecursive(File f) {
		if(f.isDirectory())
			for(File c : f.listFiles())
				deleteRecursive(c);
		f.delete();
	}

	@Test
	public void testSparkReadsMultiPartFiles() throws Exception {
		FrameBlock original = indexedRandomFrame(60);
		writeTwoParts(original);

		FrameBlock result = sparkReadSorted(TEMP_PAR_PATH, original.getColumnNames());
		TestUtils.compareFrames(original, result, false);
	}

	@Test
	public void testMultiPartFileRoundtrip() throws Exception {
		FrameBlock original = indexedRandomFrame(60);
		writeTwoParts(original);

		FrameBlock result = new FrameReaderParquetParallel().readFrameFromHDFS(TEMP_PAR_PATH, SCHEMA,
			original.getColumnNames(), 60, SCHEMA.length);

		TestUtils.compareFrames(original, result, false);
	}

	@Test
	public void testSequentialReaderMultiPartFileRoundtrip() throws Exception {
		// the serial reader must also expand a directory into its part files, not just the parallel one
		FrameBlock original = indexedRandomFrame(60);
		writeTwoParts(original);

		FrameBlock result = new FrameReaderParquet().readFrameFromHDFS(TEMP_PAR_PATH, SCHEMA, original.getColumnNames(),
			60, SCHEMA.length);

		TestUtils.compareFrames(original, result, false);
	}

	@Test
	public void testForcedParallelWriterRoundTrip() throws Exception {
		FrameBlock original = indexedRandomFrame(20);

		FrameWriterParquetParallel writer = new FrameWriterParquetParallel();
		writer.setForcedParallel(true);
		writer.writeFrameToHDFS(original, TEMP_PAR_PATH, 20, SCHEMA.length);

		FrameBlock result = new FrameReaderParquetParallel().readFrameFromHDFS(TEMP_PAR_PATH, SCHEMA,
			original.getColumnNames(), 20, SCHEMA.length);

		TestUtils.compareFrames(original, result, false);
	}

	@Test
	public void testRoundtripPublicFiles() throws Exception {
		for(String filename : PUBLIC_FILES) {
			ParquetMetadataInfo info = ParquetTestUtils.inferMetadata(filename);

			FrameReaderParquet reader = new FrameReaderParquet();
			FrameBlock original = reader.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);

			FrameWriterParquet writer = new FrameWriterParquet();
			writer.writeFrameToHDFS(original, TEMP_FILE, original.getNumRows(), original.getNumColumns());

			FrameBlock result = reader.readFrameFromHDFS(TEMP_FILE, info.schema, info.names, info.rlen, info.clen);

			TestUtils.compareFrames(original, result, false);

			FrameBlock sparkResult = ParquetTestUtils.sparkReadAsFrame(TEMP_FILE, info.schema, info.names);
			TestUtils.compareFrames(original, sparkResult, false);
		}
	}

	/** Random frame over all parquet-supported value types, with the row index in column 0. */
	private static FrameBlock indexedRandomFrame(int rows) {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(rows, SCHEMA, SEED);
		for(int r = 0; r < rows; r++)
			fb.set(r, 0, (long) r);
		return fb;
	}

	private static void writeTwoParts(FrameBlock fb) throws Exception {
		int rows = fb.getNumRows();
		new File(TEMP_PAR_PATH).mkdir();
		FrameWriterParquet writer = new FrameWriterParquet();
		writer.writeFrameToHDFS(fb.slice(0, rows / 2 - 1), TEMP_PAR_PATH + "/part-0.parquet", rows / 2, SCHEMA.length);
		writer.writeFrameToHDFS(fb.slice(rows / 2, rows - 1), TEMP_PAR_PATH + "/part-1.parquet", rows - rows / 2,
			SCHEMA.length);
	}

	private static FrameBlock sparkReadSorted(String path, String[] names) {
		Dataset<Row> df = ParquetTestUtils.sparkSession().read().parquet(path)
			.select(names[0], Arrays.copyOfRange(names, 1, names.length)).sort(names[0]);
		return ParquetTestUtils.toFrameBlock(df.collectAsList(), SCHEMA, names);
	}
}
