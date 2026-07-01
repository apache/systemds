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
import java.util.HashSet;
import java.util.Set;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.runtime.io.FrameWriterParquetParallel;
import org.apache.sysds.test.functions.io.parquet.ParquetTestUtils.ParquetMetadataInfo;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class ReadParquetTest {

	private static final String[] FILENAMES = {
		"src/test/resources/datasets/parquet/userdata1.parquet",           // https://github.com/duckdb/duckdb/blob/main/data/parquet-testing/userdata1.parquet
		"src/test/resources/datasets/parquet/alltypes_plain.parquet",      // https://github.com/apache/parquet-testing/blob/master/data/alltypes_plain.parquet
		"src/test/resources/datasets/parquet/all.parquet"                  // https://huggingface.co/datasets/cardiffnlp/databench/blob/main/data/002_Titanic/all.parquet
	};

	@Test
	public void testReadParquet() throws Exception {
		for (String filename : FILENAMES) {
			ParquetMetadataInfo info = ParquetTestUtils.inferMetadata(filename);

			FrameReaderParquet reader = new FrameReaderParquet();
			FrameBlock frame = reader.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);

			Assert.assertEquals("Row count mismatch for " + filename, info.rlen, frame.getNumRows());
			Assert.assertEquals("Column count mismatch for " + filename, info.clen, frame.getNumColumns());
		}
	}

	@Test
	public void testInt96ColumnsDecodedCorrectly() throws Exception {
		assertColumnIsEpochMillis("src/test/resources/datasets/parquet/userdata1.parquet", 0);
		assertColumnIsEpochMillis("src/test/resources/datasets/parquet/alltypes_plain.parquet", 10);
	}

	private void assertColumnIsEpochMillis(String filename, int colIdx) throws Exception {
		ParquetMetadataInfo info = ParquetTestUtils.inferMetadata(filename);
		FrameReaderParquet reader = new FrameReaderParquet();
		FrameBlock frame = reader.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);
		int decoded = 0;
		for (int r = 0; r < frame.getNumRows(); r++) {
			Object val = frame.get(r, colIdx);
			if (val == null) continue;
			decoded++;
			Assert.assertTrue(
				"Expected Long (epoch millis) at row " + r + " col " + colIdx + ", got: " + val.getClass().getSimpleName(),
				val instanceof Long
			);
		}
		Assert.assertTrue("No INT96 values were decoded in " + filename, decoded > 0);
	}

	@Test
	public void testNullHandling() throws Exception {
		File temp = Files.createTempFile("systemds_null_parquet", ".parquet").toFile();
		try {
			ValueType[] schema = { ValueType.STRING, ValueType.STRING };
			String[] names = { "a", "b" };
			FrameBlock original = new FrameBlock(schema, names,
				new String[][] { { "x", "y" }, { null, null }, { "p", "q" } });

			new FrameWriterParquet().writeFrameToHDFS(original, temp.getPath(), 3, 2);

			FrameBlock result = new FrameReaderParquet()
				.readFrameFromHDFS(temp.getPath(), schema, names, 3, 2);

			Assert.assertNotNull("Row 0 col 0 should be non-null", result.get(0, 0));
			Assert.assertNotNull("Row 0 col 1 should be non-null", result.get(0, 1));
			Assert.assertNull("Row 1 col 0 should be null",        result.get(1, 0));
			Assert.assertNull("Row 1 col 1 should be null",        result.get(1, 1));
			Assert.assertNotNull("Row 2 col 0 should be non-null", result.get(2, 0));
			Assert.assertNotNull("Row 2 col 1 should be non-null", result.get(2, 1));
		} finally {
			temp.delete();
		}

	}

	@Test
	public void testColumnReaderMatchesLegacy() throws Exception {
		for (String filename : FILENAMES) {
			ParquetMetadataInfo info = ParquetTestUtils.inferMetadata(filename);

			FrameBlock legacy = new FrameReaderParquetLegacy()
				.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);
			FrameBlock current = new FrameReaderParquet()
				.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);

			TestUtils.compareFrames(legacy, current, false);
		}
	}

	@Test
	public void testColumnSubsetProjection() throws Exception {
		// request a reordered subset of columns (d, a, c; skip b)
		File temp = Files.createTempFile("systemds_subset_parquet", ".parquet").toFile();
		try {
			ValueType[] fullSchema = { ValueType.INT64, ValueType.STRING, ValueType.FP64, ValueType.BOOLEAN };
			String[] fullNames = { "a", "b", "c", "d" };
			FrameBlock original = new FrameBlock(fullSchema, fullNames, new String[][] {
				{ "10", "x", "1.5", "true"  },
				{ "20", "y", "2.5", "false" },
				{ "30", "z", "3.5", "true"  }
			});
			new FrameWriterParquet().writeFrameToHDFS(original, temp.getPath(), 3, 4);

			ValueType[] subSchema = { ValueType.BOOLEAN, ValueType.INT64, ValueType.FP64 };
			String[] subNames = { "d", "a", "c" };

			FrameBlock legacy = new FrameReaderParquetLegacy()
				.readFrameFromHDFS(temp.getPath(), subSchema, subNames, 3, 3);
			FrameBlock current = new FrameReaderParquet()
				.readFrameFromHDFS(temp.getPath(), subSchema, subNames, 3, 3);

			TestUtils.compareFrames(legacy, current, false);

			Assert.assertEquals(true, current.get(0, 0));   // d
			Assert.assertEquals(10L,  current.get(0, 1));   // a
			Assert.assertEquals(3.5,  ((Number) current.get(2, 2)).doubleValue(), 0.0); // c
		} finally {
			temp.delete();
		}
	}

	@Test
	public void testEmptyFile() throws Exception {
		File temp = Files.createTempFile("systemds_empty_parquet", ".parquet").toFile();
		try {
			ValueType[] schema = { ValueType.INT64, ValueType.STRING };
			String[] names = { "a", "b" };
			FrameBlock empty = new FrameBlock(schema, names, new String[0][]);
			new FrameWriterParquet().writeFrameToHDFS(empty, temp.getPath(), 0, 2);

			FrameBlock legacy = new FrameReaderParquetLegacy()
				.readFrameFromHDFS(temp.getPath(), schema, names, 0, 2);
			FrameBlock current = new FrameReaderParquet()
				.readFrameFromHDFS(temp.getPath(), schema, names, 0, 2);

			Assert.assertEquals("Empty file should yield 0 rows (legacy)", 0, legacy.getNumRows());
			Assert.assertEquals("Empty file should yield 0 rows (column API)", 0, current.getNumRows());
			Assert.assertEquals(2, current.getNumColumns());
		} finally {
			temp.delete();
		}
	}

	@Test
	public void testParallelReaderMatchesSequential() throws Exception {
		for (String filename : FILENAMES) {
			ParquetMetadataInfo info = ParquetTestUtils.inferMetadata(filename);

			FrameReaderParquet sequential = new FrameReaderParquet();
			FrameBlock expected = sequential.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);

			FrameReaderParquetParallel parallel = new FrameReaderParquetParallel();
			FrameBlock actual = parallel.readFrameFromHDFS(filename, info.schema, info.names, info.rlen, info.clen);

			TestUtils.compareFrames(expected, actual, false);
		}
	}

	@Test
	public void testParallelReaderMultiFileOffsets() throws Exception {
		// each file must use its own row range, else the parallel reader overwrites/loses rows
		File tempDir = Files.createTempDirectory("systemds_parallel_parquet").toFile();
		try {
			ValueType[] schema = { ValueType.STRING, ValueType.INT32 };
			String[] names = { "label", "value" };

			FrameBlock block1 = new FrameBlock(schema, names,
				new String[][] { { "a", "1" }, { "b", "2" }, { "c", "3" } });
			FrameBlock block2 = new FrameBlock(schema, names,
				new String[][] { { "d", "4" }, { "e", "5" }, { "f", "6" } });

			String path1 = tempDir + "/part-0.parquet";
			String path2 = tempDir + "/part-1.parquet";
			FrameWriterParquet writer = new FrameWriterParquet();
			writer.writeFrameToHDFS(block1, path1, 3, 2);
			writer.writeFrameToHDFS(block2, path2, 3, 2);

			FrameReaderParquet seq = new FrameReaderParquet();
			Set<String> expectedLabels = new HashSet<>();
			String[] parts = { path1, path2 };
			for (String p : parts) {
				FrameBlock fb = seq.readFrameFromHDFS(p, schema, names, 3, 2);
				for (int r = 0; r < fb.getNumRows(); r++)
					expectedLabels.add((String) fb.get(r, 0));
			}

			FrameReaderParquetParallel parallel = new FrameReaderParquetParallel();
			FrameBlock result = parallel.readFrameFromHDFS(tempDir.toString(), schema, names, 6, 2);

			Assert.assertEquals("Expected 6 total rows", 6, result.getNumRows());

			Set<String> actualLabels = new HashSet<>();
			for (int r = 0; r < result.getNumRows(); r++) {
				Object label = result.get(r, 0);
				Assert.assertNotNull("Row " + r + " is null, row-offset bug suspected", label);
				actualLabels.add((String) label);
			}
			Assert.assertEquals("Parallel result does not match sequential ground truth", expectedLabels, actualLabels);
		} finally {
			for (File f : tempDir.listFiles())
				f.delete();
			tempDir.delete();
		}
	}

	@Test
	public void testParallelWriterRoundTrip() throws Exception {
		File tempDir = Files.createTempDirectory("systemds_parallel_write").toFile();
		try {
			ValueType[] schema = { ValueType.INT64, ValueType.STRING, ValueType.FP64 };
			String[] names = { "id", "name", "val" };
			String[][] data = new String[20][];
			for (int i = 0; i < 20; i++)
				data[i] = new String[] { String.valueOf(i), "row" + i, String.valueOf(i + 0.5) };
			FrameBlock original = new FrameBlock(schema, names, data);

			FrameWriterParquetParallel writer = new FrameWriterParquetParallel();
			writer.setForcedParallel(true);
			writer.writeFrameToHDFS(original, tempDir.getPath(), 20, 3);

			FrameBlock result = new FrameReaderParquetParallel()
				.readFrameFromHDFS(tempDir.getPath(), schema, names, 20, 3);

			Assert.assertEquals("Row count mismatch after parallel write", 20, result.getNumRows());
			// Rows may be out of order in parallel reads, validate by comparing tuples, not row positions
			Set<String> expected = new HashSet<>();
			for (int i = 0; i < 20; i++)
				expected.add(i + "|row" + i + "|" + (i + 0.5));
			Set<String> actual = new HashSet<>();
			for (int r = 0; r < result.getNumRows(); r++)
				actual.add(result.get(r, 0) + "|" + result.get(r, 1) + "|" + result.get(r, 2));
			Assert.assertEquals("Parallel-written data does not round-trip", expected, actual);
		} finally {
			for (File f : tempDir.listFiles())
				f.delete();
			tempDir.delete();
		}
	}
}
