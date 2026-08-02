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

package org.apache.sysds.test.component.io;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderDelta;
import org.apache.sysds.runtime.io.FrameReaderDeltaParallel;
import org.apache.sysds.runtime.io.FrameWriterDelta;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Cross-engine interoperability tests for the native (Delta Kernel based) frame reader/writer against the reference
 * Delta implementation (Delta's Spark connector, {@code delta-spark}, pulled in test-only).
 *
 * <p>
 * The other Delta frame tests round-trip exclusively through SystemDS' own Kernel-based read/write paths, so they
 * cannot catch a table that SystemDS writes in a way other Delta engines reject (or vice versa). These tests close that
 * gap by routing a mixed-type (long/double/string/boolean) frame through two independent engines:
 * <ul>
 * <li>SystemDS writes -&gt; Spark/Delta reads (our output is spec-compliant), and</li>
 * <li>Spark/Delta writes -&gt; SystemDS reads, including a multi-file layout and a table with deletion vectors / a
 * second commit that the SystemDS writer never produces itself.</li>
 * </ul>
 *
 * <p>
 * Row order is never assumed: every table carries a unique id in column 0 and comparisons are keyed by that id, since
 * neither engine guarantees row order across files.
 */
@net.jcip.annotations.NotThreadSafe
public class DeltaFrameSparkInteropTest {

	// nonsense schema/dims handed to the reader to confirm it discovers everything from the table
	private static final ValueType[] NO_SCHEMA = new ValueType[] {ValueType.STRING};
	private static final String[] NO_NAMES = new String[] {"x"};

	private static SparkSession spark;

	@BeforeClass
	public static void startSpark() {
		// each test class runs in its own fork (surefire reuseForks=false), so this
		// is the only SparkSession in the JVM and gets the Delta extensions injected.
		SparkSession.clearActiveSession();
		SparkSession.clearDefaultSession();
		spark = SparkSession.builder().appName("sysds-delta-frame-interop").master("local[2]")
			.config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2")
			.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
			.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog").getOrCreate();
	}

	@AfterClass
	public static void stopSpark() {
		if(spark != null)
			spark.stop();
		SparkSession.clearActiveSession();
		SparkSession.clearDefaultSession();
		spark = null;
	}

	@Test
	public void systemdsWriteSparkReadMultiFile() throws Exception {
		// SystemDS writes a (forced) multi-file mixed-type frame Delta table; the
		// reference Delta engine (Spark) must read every data file back with
		// matching values across all four column types.
		int rows = 20_000, cols = 4;
		FrameBlock in = indexedFrame(rows);

		// small target file size -> multiple parquet data files (exercise that an
		// external reader stitches all of our data files, not just the first).
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(16L * 1024));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_s2s_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, rows, cols);
			assertTrue("writer should have produced a multi-file table", countParquet(tablePath) > 1);

			Dataset<Row> df = spark.read().format("delta").load(tablePath);
			assertEquals("rows", rows, df.count());
			assertEquals("cols", cols, df.schema().fields().length);

			List<Row> read = df.collectAsList();
			assertEquals(rows, read.size());
			boolean[] seen = new boolean[rows];
			for(Row r : read) {
				int id = (int) r.getLong(0);
				assertTrue("id in range and unique: " + id, id >= 0 && id < rows && !seen[id]);
				seen[id] = true;
				assertEquals("id" + id + " c1", dval(id), r.getDouble(1), 1e-9);
				assertEquals("id" + id + " c2", sval(id), r.getString(2));
				assertEquals("id" + id + " c3", Boolean.valueOf(bval(id)), Boolean.valueOf(r.getBoolean(3)));
			}
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void sparkWriteSystemdsReadMultiFile() throws Exception {
		// the reference Delta engine writes a multi-file mixed-type table; both the
		// serial and parallel SystemDS frame readers must reconstruct it cell-for-cell.
		int rows = 600;
		Dataset<Row> df = indexedDataFrame(rows).repartition(3); // -> multiple data files
		Path dir = Files.createTempDirectory("sysds_delta_frame_p2s_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			df.write().format("delta").save(tablePath);
			assertTrue("spark should have written a multi-file table", countParquet(tablePath) > 1);

			Set<Integer> expected = idRange(0, rows);
			assertFrameMatchesIds(new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1),
				expected, "serial");
			assertFrameMatchesIds(
				new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1), expected,
				"parallel");
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void sparkDeletionVectorsSystemdsRead() throws Exception {
		// a Delta table with deletion vectors + a second commit (the DELETE) is a
		// layout the SystemDS writer never emits; the frame readers must honor the DV
		// and return only the surviving rows. With DVs present hasExactRowCounts() is
		// false, so this drives the buffered (selection-mask) frame read path.
		int rows = 400, deleteBelow = 50;
		Path dir = Files.createTempDirectory("sysds_delta_frame_dv_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			// enable deletion vectors for tables created in this block, then delete a
			// row range so Delta records a DV rather than rewriting the data files.
			spark.conf().set(DV_DEFAULT, "true");
			indexedDataFrame(rows).write().format("delta").save(tablePath);
			spark.sql("DELETE FROM delta.`" + tablePath + "` WHERE c0 < " + deleteBelow);

			Set<Integer> expected = idRange(deleteBelow, rows);

			FrameBlock serial = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertEquals("surviving rows (serial)", rows - deleteBelow, serial.getNumRows());
			assertFrameMatchesIds(serial, expected, "serial-dv");

			FrameBlock parallel = new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1,
				-1);
			assertEquals("surviving rows (parallel)", rows - deleteBelow, parallel.getNumRows());
			assertFrameMatchesIds(parallel, expected, "parallel-dv");
		}
		finally {
			// fresh fork per test class, so simply clearing the override is enough
			spark.conf().unset(DV_DEFAULT);
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	private static final String DV_DEFAULT = "spark.databricks.delta.properties.defaults.enableDeletionVectors";

	// deterministic, exactly-representable cell values keyed by the row id in column 0
	private static double dval(int id) {
		return id * 0.5 - 1.0;
	}

	private static String sval(int id) {
		return "s" + id;
	}

	private static boolean bval(int id) {
		return id % 2 == 0;
	}

	/** Frame whose column 0 is the row id and the remaining columns are exact per-id values. */
	private static FrameBlock indexedFrame(int rows) {
		ValueType[] schema = {ValueType.INT64, ValueType.FP64, ValueType.STRING, ValueType.BOOLEAN};
		String[] names = {"c0", "c1", "c2", "c3"};
		FrameBlock fb = new FrameBlock(schema, names);
		fb.ensureAllocatedColumns(rows);
		for(int r = 0; r < rows; r++) {
			fb.set(r, 0, (long) r);
			fb.set(r, 1, dval(r));
			fb.set(r, 2, sval(r));
			fb.set(r, 3, bval(r));
		}
		return fb;
	}

	/** Spark DataFrame mirroring {@link #indexedFrame} with columns c0..c3 (long/double/string/boolean). */
	private Dataset<Row> indexedDataFrame(int rows) {
		StructType schema = DataTypes
			.createStructType(new StructField[] {DataTypes.createStructField("c0", DataTypes.LongType, false),
				DataTypes.createStructField("c1", DataTypes.DoubleType, false),
				DataTypes.createStructField("c2", DataTypes.StringType, false),
				DataTypes.createStructField("c3", DataTypes.BooleanType, false)});

		List<Row> data = new ArrayList<>(rows);
		for(int r = 0; r < rows; r++)
			data.add(RowFactory.create((long) r, dval(r), sval(r), bval(r)));
		return spark.createDataFrame(data, schema);
	}

	private static Set<Integer> idRange(int fromInclusive, int toExclusive) {
		Set<Integer> ids = new LinkedHashSet<>(toExclusive - fromInclusive);
		for(int id = fromInclusive; id < toExclusive; id++)
			ids.add(id);
		return ids;
	}

	/** Asserts every row of {@code out} (keyed by its column-0 id) is expected and carries the exact per-id values. */
	private static void assertFrameMatchesIds(FrameBlock out, Set<Integer> expectedIds, String tag) {
		assertEquals(tag + " rows", expectedIds.size(), out.getNumRows());
		assertEquals(tag + " cols", 4, out.getNumColumns());
		// discovered types: long->INT64, double->FP64, string->STRING, boolean->BOOLEAN
		assertEquals(tag + " c0 type", ValueType.INT64, out.getSchema()[0]);
		assertEquals(tag + " c1 type", ValueType.FP64, out.getSchema()[1]);
		assertEquals(tag + " c2 type", ValueType.STRING, out.getSchema()[2]);
		assertEquals(tag + " c3 type", ValueType.BOOLEAN, out.getSchema()[3]);
		Set<Integer> seen = new HashSet<>();
		for(int r = 0; r < out.getNumRows(); r++) {
			int id = ((Number) out.get(r, 0)).intValue();
			assertTrue(tag + ": unexpected/duplicate id " + id, expectedIds.contains(id) && seen.add(id));
			assertEquals(tag + " id" + id + " c1", dval(id), ((Number) out.get(r, 1)).doubleValue(), 1e-9);
			assertEquals(tag + " id" + id + " c2", sval(id), out.get(r, 2).toString());
			assertEquals(tag + " id" + id + " c3", Boolean.valueOf(bval(id)), out.get(r, 3));
		}
	}

	private static long countParquet(String tablePath) throws Exception {
		return DeltaFrameTestUtils.countParquet(tablePath);
	}
}
