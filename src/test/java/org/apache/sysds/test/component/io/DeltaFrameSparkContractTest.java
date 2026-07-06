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
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderDelta;
import org.apache.sysds.runtime.io.FrameReaderDeltaParallel;
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
 * Regression tests pinning the Delta Kernel {@code ParquetHandler} contract cases that a custom parquet decode path
 * (the column-API fast path, {@code sysds.io.delta.reader.columnapi}, on by default) must honor beyond plain flat
 * reads. Each case is a table layout the SystemDS writer never produces itself, so it must be created by the
 * reference engine (Spark/Delta).
 *
 * dvFeatureEnabledNoDeleteRead covers a table whose protocol carries the {@code deletionVectors} reader feature but
 * has no deleted rows - a distinct case from {@link DeltaFrameSparkInteropTest#sparkDeletionVectorsSystemdsRead},
 * which covers row filtering once rows are actually deleted. Here the kernel still appends the
 * {@code _metadata.row_index} metadata column to every read once the feature is enabled, which does not exist in the
 * data files and must not be requested from them.
 *
 * schemaEvolutionAddedColumnRead covers a column added after the first commit, so older data files lack it and the
 * handler must surface it as nulls rather than fail. partitionedTableRead covers partition values, which are not
 * stored in the data files and must be spliced back in. idColumnMappingRead covers a table using
 * {@code delta.columnMapping.mode = id}, where columns must be resolvable by parquet field id rather than logical
 * name.
 */
@net.jcip.annotations.NotThreadSafe
public class DeltaFrameSparkContractTest {

	// nonsense schema/dims handed to the reader to confirm it discovers everything from the table
	private static final ValueType[] NO_SCHEMA = new ValueType[] {ValueType.STRING};
	private static final String[] NO_NAMES = new String[] {"x"};

	private static final String DV_DEFAULT = "spark.databricks.delta.properties.defaults.enableDeletionVectors";

	private static SparkSession spark;

	@BeforeClass
	public static void startSpark() {
		// each test class runs in its own fork (surefire reuseForks=false), so this
		// is the only SparkSession in the JVM and gets the Delta extensions injected.
		SparkSession.clearActiveSession();
		SparkSession.clearDefaultSession();
		spark = SparkSession.builder().appName("sysds-delta-frame-contract").master("local[2]")
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
	public void dvFeatureEnabledNoDeleteRead() throws Exception {
		// enabling deletion vectors adds the feature to the table protocol, which
		// makes the kernel append the _metadata.row_index metadata column to the physical
		// read schema of every read, no row has to be deleted. The parquet handler must
		// populate that column (it is not stored in the data files) instead of failing.
		int rows = 500;
		Path dir = Files.createTempDirectory("sysds_delta_frame_dvfeat_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			spark.conf().set(DV_DEFAULT, "true");
			indexedDataFrame(rows).write().format("delta").save(tablePath);

			assertFrameMatchesIds(new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1),
				rows, "serial-dvfeat");
			assertFrameMatchesIds(
				new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1), rows,
				"parallel-dvfeat");
		}
		finally {
			spark.conf().unset(DV_DEFAULT);
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void schemaEvolutionAddedColumnRead() throws Exception {
		// column c4 is added by the second commit, so the data files of the first commit
		// do not contain it; the parquet handler must return nulls for it there (the
		// kernel hands every file the same table-level physical read schema).
		int oldRows = 200, allRows = 300;
		Path dir = Files.createTempDirectory("sysds_delta_frame_evo_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			indexedDataFrame(oldRows).write().format("delta").save(tablePath);
			evolvedDataFrame(oldRows, allRows).write().format("delta").mode("append").option("mergeSchema", "true")
				.save(tablePath);

			for(FrameBlock out : new FrameBlock[] {
				new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1),
				new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1)}) {
				assertEquals("rows", allRows, out.getNumRows());
				assertEquals("cols", 5, out.getNumColumns());
				assertEquals("c4 type", ValueType.STRING, out.getSchema()[4]);
				Set<Integer> seen = new HashSet<>();
				for(int r = 0; r < out.getNumRows(); r++) {
					int id = ((Number) out.get(r, 0)).intValue();
					assertTrue("unexpected/duplicate id " + id, id >= 0 && id < allRows && seen.add(id));
					assertEquals("id" + id + " c1", dval(id), ((Number) out.get(r, 1)).doubleValue(), 1e-9);
					assertEquals("id" + id + " c2", sval(id), out.get(r, 2).toString());
					Object c4 = out.get(r, 4);
					if(id < oldRows)
						assertNull("id" + id + " c4 must be null (file predates the column)", c4);
					else
						assertEquals("id" + id + " c4", vval(id), c4.toString());
				}
				assertEquals(allRows, seen.size());
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void partitionedTableRead() throws Exception {
		// partition values are not stored in the data files; the kernel splices them back
		// into every batch (ColumnarBatch.withNewColumn), so this pins the batch-reshaping
		// side of the contract that plain unpartitioned round-trips never touch.
		int rows = 300;
		Path dir = Files.createTempDirectory("sysds_delta_frame_part_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			indexedDataFrame(rows).write().format("delta").partitionBy("c3").save(tablePath);

			assertFrameMatchesIds(new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1),
				rows, "serial-part");
			assertFrameMatchesIds(
				new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1), rows,
				"parallel-part");
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void idColumnMappingRead() throws Exception {
		// with delta.columnMapping.mode=id the parquet columns carry field ids and
		// physical names; the handler must resolve columns through the
		// mapped physical schema, preferring field ids per the ParquetHandler contract.
		int rows = 400;
		Path dir = Files.createTempDirectory("sysds_delta_frame_idmap_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			spark.sql("CREATE TABLE delta.`" + tablePath + "` (c0 BIGINT, c1 DOUBLE, c2 STRING, c3 BOOLEAN) "
				+ "USING delta TBLPROPERTIES ('delta.columnMapping.mode'='id')");
			indexedDataFrame(rows).write().format("delta").mode("append").save(tablePath);

			assertFrameMatchesIds(new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1),
				rows, "serial-idmap");
			assertFrameMatchesIds(
				new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1), rows,
				"parallel-idmap");
		}
		finally {
			spark.sql("DROP TABLE IF EXISTS delta.`" + tablePath + "`");
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

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

	private static String vval(int id) {
		return "v" + id;
	}

	/** Spark DataFrame with columns c0..c3 (long/double/string/boolean) keyed by the row id in c0. */
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

	/** Like {@link #indexedDataFrame} for ids [from,to) but with an additional string column c4. */
	private Dataset<Row> evolvedDataFrame(int from, int to) {
		StructType schema = DataTypes
			.createStructType(new StructField[] {DataTypes.createStructField("c0", DataTypes.LongType, false),
				DataTypes.createStructField("c1", DataTypes.DoubleType, false),
				DataTypes.createStructField("c2", DataTypes.StringType, false),
				DataTypes.createStructField("c3", DataTypes.BooleanType, false),
				DataTypes.createStructField("c4", DataTypes.StringType, true)});
		List<Row> data = new ArrayList<>(to - from);
		for(int r = from; r < to; r++)
			data.add(RowFactory.create((long) r, dval(r), sval(r), bval(r), vval(r)));
		return spark.createDataFrame(data, schema);
	}

	/** Asserts {@code out} holds exactly ids [0,rows) with the exact per-id values in c1..c3. */
	private static void assertFrameMatchesIds(FrameBlock out, int rows, String tag) {
		assertEquals(tag + " rows", rows, out.getNumRows());
		assertEquals(tag + " cols", 4, out.getNumColumns());
		assertEquals(tag + " c0 type", ValueType.INT64, out.getSchema()[0]);
		assertEquals(tag + " c1 type", ValueType.FP64, out.getSchema()[1]);
		assertEquals(tag + " c2 type", ValueType.STRING, out.getSchema()[2]);
		assertEquals(tag + " c3 type", ValueType.BOOLEAN, out.getSchema()[3]);
		boolean[] seen = new boolean[rows];
		for(int r = 0; r < rows; r++) {
			int id = ((Number) out.get(r, 0)).intValue();
			assertTrue(tag + ": unexpected/duplicate id " + id, id >= 0 && id < rows && !seen[id]);
			seen[id] = true;
			assertEquals(tag + " id" + id + " c1", dval(id), ((Number) out.get(r, 1)).doubleValue(), 1e-9);
			assertEquals(tag + " id" + id + " c2", sval(id), out.get(r, 2).toString());
			assertEquals(tag + " id" + id + " c3", Boolean.valueOf(bval(id)), out.get(r, 3));
		}
	}
}
