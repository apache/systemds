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
import java.io.IOException;
import java.nio.file.Files;
import java.util.Map;

import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.test.functions.io.parquet.ParquetTestUtils.ParquetMetadataInfo;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Verifies the parquet frame readers against Spark's parquet reader on Spark-written files.
 */
public class ReadParquetTest {

	// Generated once per test class with Spark's DataFrameWriter
	private static File testFileDir;
	private static String[] FILENAMES;

	@BeforeClass
	public static void generateTestFiles() throws Exception {
		testFileDir = Files.createTempDirectory("systemds_parquet_public_test_files").toFile();
		Map<String, String> files = ParquetTestUtils.generatePublicTestFiles(testFileDir);
		FILENAMES = new String[] {files.get("userdata1"), files.get("alltypes_plain"), files.get("all")};
	}

	@AfterClass
	public static void cleanupTestFiles() {
		deleteRecursive(testFileDir);
	}

	private static void deleteRecursive(File f) {
		File[] children = f.listFiles();
		if(children != null)
			for(File c : children)
				deleteRecursive(c);
		f.delete();
	}

	@Test
	public void testReadMatchesSpark() throws Exception {
		for(String filename : FILENAMES) {
			ParquetMetadataInfo info = ParquetTestUtils.inferMetadata(filename);

			FrameBlock expected = ParquetTestUtils.sparkReadAsFrame(filename, info.schema, info.names);
			FrameBlock actual = new FrameReaderParquet().readFrameFromHDFS(filename, info.schema, info.names, info.rlen,
				info.clen);

			TestUtils.compareFrames(expected, actual, false);
		}
	}

	@Test
	public void testColumnSubsetProjection() throws Exception {
		// request a reordered subset of columns (d, a, c; skip b)
		File temp = Files.createTempFile("systemds_subset_parquet", ".parquet").toFile();
		try {
			ValueType[] fullSchema = {ValueType.INT64, ValueType.STRING, ValueType.FP64, ValueType.BOOLEAN};
			String[] fullNames = {"a", "b", "c", "d"};
			FrameBlock original = new FrameBlock(fullSchema, fullNames,
				new String[][] {{"10", "x", "1.5", "true"}, {"20", "y", "2.5", "false"}, {"30", "z", "3.5", "true"}});
			new FrameWriterParquet().writeFrameToHDFS(original, temp.getPath(), 3, 4);

			ValueType[] subSchema = {ValueType.BOOLEAN, ValueType.INT64, ValueType.FP64};
			String[] subNames = {"d", "a", "c"};

			FrameBlock expected = ParquetTestUtils.sparkReadAsFrame(temp.getPath(), subSchema, subNames);
			FrameBlock actual = new FrameReaderParquet().readFrameFromHDFS(temp.getPath(), subSchema, subNames, 3, 3);

			TestUtils.compareFrames(expected, actual, false);
		}
		finally {
			temp.delete();
		}
	}

	@Test
	public void testInt96ColumnRejectedOthersReadable() throws Exception {
		String file = ParquetTestUtils.writeInt96TimestampFile(testFileDir);

		ValueType[] fullSchema = {ValueType.INT32, ValueType.INT64, ValueType.STRING};
		String[] fullNames = {"id", "ts", "name"};
		try {
			new FrameReaderParquet().readFrameFromHDFS(file, fullSchema, fullNames, 3, 3);
			Assert.fail("Expected rejection of the INT96 column");
		}
		catch(IOException e) {
			Assert.assertTrue(e.getMessage(), e.getMessage().contains("INT96"));
		}

		ValueType[] subSchema = {ValueType.INT32, ValueType.STRING};
		String[] subNames = {"id", "name"};
		FrameBlock expected = ParquetTestUtils.sparkReadAsFrame(file, subSchema, subNames);
		FrameBlock actual = new FrameReaderParquet().readFrameFromHDFS(file, subSchema, subNames, 3, 2);
		TestUtils.compareFrames(expected, actual, false);
	}

	@Test
	public void testEmptyFile() throws Exception {
		File temp = Files.createTempFile("systemds_empty_parquet", ".parquet").toFile();
		try {
			ValueType[] schema = {ValueType.INT64, ValueType.STRING};
			String[] names = {"a", "b"};
			FrameBlock empty = new FrameBlock(schema, names, new String[0][]);
			new FrameWriterParquet().writeFrameToHDFS(empty, temp.getPath(), 0, 2);

			FrameBlock result = new FrameReaderParquet().readFrameFromHDFS(temp.getPath(), schema, names, 0, 2);

			Assert.assertEquals("Empty file should yield 0 rows", 0, result.getNumRows());
			Assert.assertEquals(2, result.getNumColumns());
		}
		finally {
			temp.delete();
		}
	}

	@Test
	public void testReadSparkOutputDirectory() throws Exception {
		// raw Spark output layout: a directory with _SUCCESS marker, .crc files and uuid part names;
		// repartition shuffles rows across the part files, so values are verified keyed by id
		File dir = new File(testFileDir, "spark_dir_out");
		ValueType[] schema = {ValueType.INT64, ValueType.FP64, ValueType.STRING};
		String[] names = {"id", "val", "name"};
		SparkSession spark = ParquetTestUtils.sparkSession();
		spark.range(100).selectExpr("id", "id * 0.5d as val", "concat('r', id) as name").repartition(2).write()
			.mode(SaveMode.Overwrite).parquet(dir.getPath());
		Assert.assertTrue(new File(dir, "_SUCCESS").exists());

		FrameBlock serial = new FrameReaderParquet().readFrameFromHDFS(dir.getPath(), schema, names, 100, 3);
		Assert.assertEquals(100, serial.getNumRows());
		boolean[] seen = new boolean[100];
		for(int r = 0; r < 100; r++) {
			int id = ((Long) serial.get(r, 0)).intValue();
			Assert.assertFalse("duplicate id " + id, seen[id]);
			seen[id] = true;
			Assert.assertEquals(id * 0.5, (Double) serial.get(r, 1), 0);
			Assert.assertEquals("r" + id, serial.get(r, 2));
		}

		FrameBlock parallel = new FrameReaderParquetParallel().readFrameFromHDFS(dir.getPath(), schema, names, 100, 3);
		TestUtils.compareFrames(serial, parallel, false);
	}

	@Test
	public void testParallelReaderMatchesSequential() throws Exception {
		for(String filename : FILENAMES) {
			ParquetMetadataInfo info = ParquetTestUtils.inferMetadata(filename);

			FrameReaderParquet sequential = new FrameReaderParquet();
			FrameBlock expected = sequential.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);

			FrameReaderParquetParallel parallel = new FrameReaderParquetParallel();
			FrameBlock actual = parallel.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);

			TestUtils.compareFrames(expected, actual, false);
		}
	}
}
