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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.io.ReaderDelta;
import org.apache.sysds.runtime.io.ReaderDeltaParallel;
import org.apache.sysds.runtime.io.WriterDelta;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
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
 * Cross-engine interoperability tests for the native (Delta Kernel based) matrix
 * reader/writer against the reference Delta implementation (Delta's Spark
 * connector, {@code delta-spark}, pulled in test-only).
 *
 * <p>The other Delta matrix tests round-trip exclusively through SystemDS' own
 * Kernel-based read/write paths, so they cannot catch a table that SystemDS
 * writes in a way other Delta engines reject (or vice versa). These tests close
 * that gap by routing data through two independent engines:
 * <ul>
 *   <li>SystemDS writes -&gt; Spark/Delta reads (our output is spec-compliant), and</li>
 *   <li>Spark/Delta writes -&gt; SystemDS reads, including a multi-file layout and a
 *       table with deletion vectors / a second commit that the SystemDS writer
 *       never produces itself.</li>
 * </ul>
 *
 * <p>Row order is never assumed: every table carries a unique id in column 0 and
 * comparisons are keyed by that id, since neither engine guarantees row order
 * across files.
 */
@net.jcip.annotations.NotThreadSafe
public class DeltaMatrixSparkInteropTest {

	private static SparkSession spark;

	@BeforeClass
	public static void startSpark() {
		//each test class runs in its own fork (surefire reuseForks=false), so this
		//is the only SparkSession in the JVM and gets the Delta extensions injected.
		SparkSession.clearActiveSession();
		SparkSession.clearDefaultSession();
		spark = SparkSession.builder()
			.appName("sysds-delta-interop")
			.master("local[2]")
			.config("spark.ui.enabled", "false")
			.config("spark.sql.shuffle.partitions", "2")
			.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
			.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
			.getOrCreate();
	}

	@AfterClass
	public static void stopSpark() {
		if( spark != null )
			spark.stop();
		SparkSession.clearActiveSession();
		SparkSession.clearDefaultSession();
		spark = null;
	}

	@Test
	public void systemdsWriteSparkReadMultiFile() throws Exception {
		//SystemDS writes a (forced) multi-file Delta table; the reference Delta
		//engine (Spark) must read every data file back with matching values.
		int rows = 500, cols = 5;
		MatrixBlock in = indexedMatrix(rows, cols);

		//small target file size -> multiple parquet data files (exercise that an
		//external reader stitches all of our data files, not just the first).
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(16L * 1024));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_s2s_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new WriterDelta().writeMatrixToHDFS(in, tablePath, rows, cols, -1, in.getNonZeros());
			assertTrue("writer should have produced a multi-file table", countParquet(tablePath) > 1);

			Dataset<Row> df = spark.read().format("delta").load(tablePath);
			assertEquals("rows", rows, df.count());
			assertEquals("cols", cols, df.schema().fields().length);

			List<Row> read = df.collectAsList();
			assertEquals(rows, read.size());
			for( Row r : read ) {
				int id = (int) Math.round(r.getDouble(0));
				assertTrue("id in range: " + id, id >= 0 && id < rows);
				for( int c = 0; c < cols; c++ )
					assertEquals("r" + id + " c" + c, in.get(id, c), r.getDouble(c), 1e-9);
			}
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void sparkWriteSystemdsReadMultiFile() throws Exception {
		//the reference Delta engine writes a multi-file table; both the serial and
		//parallel SystemDS readers must reconstruct it (coercing long ids to double).
		int rows = 600, cols = 4;
		Dataset<Row> df = indexedDataFrame(rows, cols).repartition(3); //-> multiple data files
		Path dir = Files.createTempDirectory("sysds_delta_p2s_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			df.write().format("delta").save(tablePath);
			assertTrue("spark should have written a multi-file table", countParquet(tablePath) > 1);

			Map<Integer, double[]> expected = expectedById(rows, cols);
			assertMatchesById(new ReaderDelta()
				.readMatrixFromHDFS(tablePath, -1, -1, -1, -1), expected, cols, "serial");
			assertMatchesById(new ReaderDeltaParallel()
				.readMatrixFromHDFS(tablePath, -1, -1, -1, -1), expected, cols, "parallel");
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void sparkDeletionVectorsSystemdsRead() throws Exception {
		//a Delta table with deletion vectors + a second commit (the DELETE) is a
		//layout the SystemDS writer never emits; the readers must honor the DV and
		//return only the surviving rows. This exercises the hasDeletionVector path.
		int rows = 400, cols = 3, deleteBelow = 50;
		Path dir = Files.createTempDirectory("sysds_delta_dv_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			//enable deletion vectors for tables created in this block, then delete a
			//row range so Delta records a DV rather than rewriting the data files.
			spark.conf().set(DV_DEFAULT, "true");
			indexedDataFrame(rows, cols).write().format("delta").save(tablePath);
			spark.sql("DELETE FROM delta.`" + tablePath + "` WHERE c0 < " + deleteBelow);

			Map<Integer, double[]> expected = expectedById(rows, cols);
			expected.keySet().removeIf(id -> id < deleteBelow);

			MatrixBlock serial = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("surviving rows (serial)", rows - deleteBelow, serial.getNumRows());
			assertMatchesById(serial, expected, cols, "serial-dv");

			MatrixBlock parallel = new ReaderDeltaParallel().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("surviving rows (parallel)", rows - deleteBelow, parallel.getNumRows());
			assertMatchesById(parallel, expected, cols, "parallel-dv");
		}
		finally {
			//fresh fork per test class, so simply clearing the override is enough
			spark.conf().unset(DV_DEFAULT);
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	private static final String DV_DEFAULT =
		"spark.databricks.delta.properties.defaults.enableDeletionVectors";

	/** Matrix whose column 0 is the row index and remaining columns are exact doubles. */
	private static MatrixBlock indexedMatrix(int rows, int cols) {
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		for( int r = 0; r < rows; r++ ) {
			mb.set(r, 0, r);
			for( int c = 1; c < cols; c++ )
				mb.set(r, c, value(r, c));
		}
		mb.recomputeNonZeros();
		return mb;
	}

	/** Spark DataFrame mirroring {@link #indexedMatrix} with columns c0..c(cols-1) as doubles. */
	private static Dataset<Row> indexedDataFrame(int rows, int cols) {
		StructField[] fields = new StructField[cols];
		for( int c = 0; c < cols; c++ )
			fields[c] = DataTypes.createStructField("c" + c, DataTypes.DoubleType, false);
		StructType schema = DataTypes.createStructType(fields);

		List<Row> data = new ArrayList<>(rows);
		for( int r = 0; r < rows; r++ ) {
			Object[] vals = new Object[cols];
			vals[0] = (double) r;
			for( int c = 1; c < cols; c++ )
				vals[c] = value(r, c);
			data.add(RowFactory.create(vals));
		}
		return spark.createDataFrame(data, schema);
	}

	/** Deterministic, exactly-representable cell value for (row,col), col>=1. */
	private static double value(int row, int col) {
		return row * 0.5 - col;
	}

	private static Map<Integer, double[]> expectedById(int rows, int cols) {
		Map<Integer, double[]> exp = new HashMap<>(rows);
		for( int r = 0; r < rows; r++ ) {
			double[] row = new double[cols];
			row[0] = r;
			for( int c = 1; c < cols; c++ )
				row[c] = value(r, c);
			exp.put(r, row);
		}
		return exp;
	}

	/** Asserts every row of {@code out} (keyed by its column-0 id) matches {@code expected}. */
	private static void assertMatchesById(MatrixBlock out, Map<Integer, double[]> expected, int cols, String tag) {
		assertEquals(tag + " rows", expected.size(), out.getNumRows());
		assertEquals(tag + " cols", cols, out.getNumColumns());
		boolean[] seen = new boolean[expected.size() == 0 ? 0 : maxId(expected) + 1];
		for( int r = 0; r < out.getNumRows(); r++ ) {
			int id = (int) Math.round(out.get(r, 0));
			double[] exp = expected.get(id);
			assertTrue(tag + ": unexpected/duplicate id " + id, exp != null && id < seen.length && !seen[id]);
			seen[id] = true;
			for( int c = 0; c < cols; c++ )
				assertEquals(tag + " id" + id + " c" + c, exp[c], out.get(r, c), 1e-9);
		}
	}

	private static int maxId(Map<Integer, double[]> expected) {
		int m = 0;
		for( int id : expected.keySet() )
			m = Math.max(m, id);
		return m;
	}

	private static long countParquet(String tablePath) throws Exception {
		try( java.util.stream.Stream<Path> s = Files.walk(new File(tablePath).toPath()) ) {
			return s.filter(p -> p.toString().endsWith(".parquet")).count();
		}
	}
}
