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
import static org.junit.Assert.fail;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.NoSuchElementException;
import java.util.Optional;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.DeltaKernelUtils;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.io.ReaderDelta;
import org.apache.sysds.runtime.io.ReaderDeltaParallel;
import org.apache.sysds.runtime.io.WriterDelta;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.ColumnarBatch;
import io.delta.kernel.data.FilteredColumnarBatch;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.BooleanType;
import io.delta.kernel.types.ByteType;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.DoubleType;
import io.delta.kernel.types.FloatType;
import io.delta.kernel.types.IntegerType;
import io.delta.kernel.types.LongType;
import io.delta.kernel.types.ShortType;
import io.delta.kernel.types.StringType;
import io.delta.kernel.types.StructType;
import io.delta.kernel.utils.CloseableIterator;

/**
 * Direct (no DML) round-trip tests for the native Delta Kernel based matrix
 * reader/writer. Each test writes a MatrixBlock to a fresh local Delta table
 * directory and reads it back, asserting dimensions and values match.
 */
public class DeltaMatrixReadWriteTest {

	//small writer target file size (bytes) used to force a multi-file table
	//layout cheaply, instead of brute-forcing huge row counts.
	private static final long SMALL_TARGET_FILE_SIZE = 256L * 1024;
	private static final int ROWS_MULTI_FILE = 100_000;

	private static MatrixBlock writeThenRead(MatrixBlock in) throws Exception {
		Path dir = Files.createTempDirectory("sysds_delta_");
		//WriterDelta creates the table at the given (empty) directory
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			WriterDelta writer = new WriterDelta();
			writer.writeMatrixToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());

			MatrixReader reader = new ReaderDelta();
			return reader.readMatrixFromHDFS(tablePath, in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void parallelReadMatchesSerialMultiFile() throws Exception {
		//force a multi-file table cheaply via a small writer target file size
		//(rather than a huge row count), so the parallel per-file path is
		//actually exercised rather than falling back to serial.
		MatrixBlock in = TestUtils.generateTestMatrixBlock(ROWS_MULTI_FILE, 8, -10, 10, 1.0, 13);
		in.recomputeNonZeros();

		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_par_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new WriterDelta().writeMatrixToHDFS(in, tablePath,
				in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());

			//sanity: confirm the table really is split across multiple files
			long files;
			try( java.util.stream.Stream<Path> s = Files.walk(new File(tablePath).toPath()) ) {
				files = s.filter(p -> p.toString().endsWith(".parquet")).count();
			}
			assertTrue("expected a multi-file Delta table to exercise the parallel path, got " + files,
				files > 1);

			MatrixBlock serial = new ReaderDelta()
				.readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			MatrixBlock parallel = new ReaderDeltaParallel()
				.readMatrixFromHDFS(tablePath, -1, -1, -1, -1);

			assertEquals("rows", serial.getNumRows(), parallel.getNumRows());
			assertEquals("cols", serial.getNumColumns(), parallel.getNumColumns());
			assertEquals("nnz", serial.getNonZeros(), parallel.getNonZeros());
			TestUtils.compareMatrices(serial, parallel, 0, "serial-vs-parallel");
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void roundTripDenseSmall() throws Exception {
		MatrixBlock in = new MatrixBlock(3, 4, false);
		double v = 1.0;
		for( int i=0; i<3; i++ )
			for( int j=0; j<4; j++ )
				in.set(i, j, v++);
		in.recomputeNonZeros();

		MatrixBlock out = writeThenRead(in);
		assertEquals("rows", 3, out.getNumRows());
		assertEquals("cols", 4, out.getNumColumns());
		TestUtils.compareMatrices(in, out, 1e-12, "dense-small");
	}

	@Test
	public void roundTripDenseRandom() throws Exception {
		MatrixBlock in = TestUtils.generateTestMatrixBlock(500, 17, -10, 10, 1.0, 7);
		MatrixBlock out = writeThenRead(in);
		assertEquals("rows", in.getNumRows(), out.getNumRows());
		assertEquals("cols", in.getNumColumns(), out.getNumColumns());
		TestUtils.compareMatrices(in, out, 1e-12, "dense-random");
	}

	@Test
	public void roundTripSparseRandom() throws Exception {
		//values written are dense parquet, but exercise a sparse-ish input
		MatrixBlock in = TestUtils.generateTestMatrixBlock(1200, 9, -5, 5, 0.1, 13);
		in.recomputeNonZeros();
		MatrixBlock out = writeThenRead(in);
		assertEquals("rows", in.getNumRows(), out.getNumRows());
		assertEquals("cols", in.getNumColumns(), out.getNumColumns());
		TestUtils.compareMatrices(in, out, 1e-12, "sparse-random");
	}

	@Test
	public void roundTripMultiBatch() throws Exception {
		//more rows than the writer batch size (4096) to exercise chunking
		MatrixBlock in = TestUtils.generateTestMatrixBlock(10000, 5, 0, 100, 1.0, 1);
		MatrixBlock out = writeThenRead(in);
		assertEquals("rows", 10000, out.getNumRows());
		assertEquals("cols", 5, out.getNumColumns());
		TestUtils.compareMatrices(in, out, 1e-12, "multi-batch");
	}

	@Test
	public void readDiscoversUnknownDimensions() throws Exception {
		MatrixBlock in = TestUtils.generateTestMatrixBlock(123, 6, -1, 1, 1.0, 3);
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new WriterDelta().writeMatrixToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());
			//pass -1 dimensions: the reader must discover them from the table
			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("rows", 123, out.getNumRows());
			assertEquals("cols", 6, out.getNumColumns());
			TestUtils.compareMatrices(in, out, 1e-12, "unknown-dims");
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void emptyMatrixRoundTrip() throws Exception {
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new WriterDelta().writeEmptyMatrixToHDFS(tablePath, 0, 4, -1);
			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("rows", 0, out.getNumRows());
			assertEquals("cols", 4, out.getNumColumns());
			assertEquals("nnz", 0, out.getNonZeros());
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readNonDoubleNumericColumns() throws Exception {
		//tables produced by external tools (or the frame writer) can carry
		//long/int/boolean columns; the matrix reader must coerce them to double
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			double[] longVals = {1, -2, 1_000_000_000L, 0};
			double[] intVals  = {7, -8, 123456, 0};
			double[] boolVals = {1, 0, 1, 0};
			writeTypedColumns(tablePath,
				new DataType[] {LongType.LONG, IntegerType.INTEGER, BooleanType.BOOLEAN},
				new double[][] {longVals, intVals, boolVals});

			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("rows", 4, out.getNumRows());
			assertEquals("cols", 3, out.getNumColumns());
			for( int r=0; r<4; r++ ) {
				assertEquals("long col r" + r, longVals[r], out.get(r, 0), 0.0);
				assertEquals("int col r" + r, intVals[r], out.get(r, 1), 0.0);
				assertEquals("bool col r" + r, boolVals[r], out.get(r, 2), 0.0);
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void rewriteSamePathReplacesData() throws Exception {
		//writing to a path that already holds a Delta table must fully replace it
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			MatrixBlock first = TestUtils.generateTestMatrixBlock(50, 8, 0, 100, 1.0, 1);
			new WriterDelta().writeMatrixToHDFS(first, tablePath, 50, 8, -1, first.getNonZeros());

			//second write has different dimensions and values
			MatrixBlock second = TestUtils.generateTestMatrixBlock(20, 3, -5, 5, 1.0, 2);
			new WriterDelta().writeMatrixToHDFS(second, tablePath, 20, 3, -1, second.getNonZeros());

			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("rows", 20, out.getNumRows());
			assertEquals("cols", 3, out.getNumColumns());
			TestUtils.compareMatrices(second, out, 1e-12, "rewrite-replace");
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void parallelBufferedPathMatchesSerial() throws Exception {
		//the direct fast path is always taken for SystemDS-written tables (exact
		//row stats, no deletion vectors); force the buffered fallback to exercise
		//its per-file decode + serial concatenation and assert it matches serial.
		//force a multi-file table cheaply via a small writer target file size.
		MatrixBlock in = TestUtils.generateTestMatrixBlock(ROWS_MULTI_FILE, 8, -10, 10, 1.0, 23);
		in.recomputeNonZeros();

		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_buf_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new WriterDelta().writeMatrixToHDFS(in, tablePath,
				in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());

			long files;
			try( java.util.stream.Stream<Path> s = Files.walk(new File(tablePath).toPath()) ) {
				files = s.filter(p -> p.toString().endsWith(".parquet")).count();
			}
			assertTrue("expected a multi-file Delta table, got " + files, files > 1);

			MatrixBlock serial = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			//subclass that always declines the direct path -> readBuffered()
			MatrixBlock buffered = new ReaderDeltaParallel() {
				@Override protected boolean useDirectPath(DeltaKernelUtils.ScanHandle h) { return false; }
			}.readMatrixFromHDFS(tablePath, -1, -1, -1, -1);

			assertEquals("rows", serial.getNumRows(), buffered.getNumRows());
			assertEquals("cols", serial.getNumColumns(), buffered.getNumColumns());
			assertEquals("nnz", serial.getNonZeros(), buffered.getNonZeros());
			TestUtils.compareMatrices(serial, buffered, 0, "serial-vs-buffered");
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void writerTargetFileSizeConfigProducesMoreFiles() throws Exception {
		//a smaller configured target file size must make the writer roll more
		//data files for the same matrix (the lever the parallel reader relies on).
		MatrixBlock in = TestUtils.generateTestMatrixBlock(400_000, 16, -10, 10, 1.0, 7);
		in.recomputeNonZeros();

		//isolate the override in a fresh thread-local config (restored in finally)
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(1L * 1024 * 1024));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_cfg_");
		try {
			assertEquals("config getter reflects the override",
				1L * 1024 * 1024, ConfigurationManager.getDeltaWriterTargetFileSize());

			String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
			new WriterDelta().writeMatrixToHDFS(in, tablePath,
				in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());
			long files;
			try( java.util.stream.Stream<Path> s = Files.walk(new File(tablePath).toPath()) ) {
				files = s.filter(p -> p.toString().endsWith(".parquet")).count();
			}
			assertTrue("expected >1 data file with a 1MB target, got " + files, files > 1);

			//data still round-trips correctly with the custom layout
			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			TestUtils.compareMatrices(in, out, 1e-12, "small-target-roundtrip");
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readerBatchSizeConfigRoundTrips() throws Exception {
		//a non-default reader batch size must not change the result (more, smaller
		//batches exercise the per-batch extract/concatenate loop more often).
		MatrixBlock in = TestUtils.generateTestMatrixBlock(5000, 7, -10, 10, 1.0, 11);
		//isolate the override in a fresh thread-local config (restored in finally)
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_READER_BATCH_SIZE, "128");
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_bs_");
		try {
			assertEquals("config getter reflects the override",
				128, ConfigurationManager.getDeltaReaderBatchSize());

			String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
			new WriterDelta().writeMatrixToHDFS(in, tablePath,
				in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());
			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			TestUtils.compareMatrices(in, out, 1e-12, "small-batch-roundtrip");
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void factoryRoutesDeltaToParallelWhenEnabled() {
		//the factory must pick the parallel reader iff parallel CP read is enabled
		CompilerConfig cc = ConfigurationManager.getCompilerConfig();
		try {
			cc.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, true);
			ConfigurationManager.setLocalConfig(cc);
			MatrixReader par = MatrixReaderFactory.createMatrixReader(FileFormat.DELTA);
			assertTrue("expected ReaderDeltaParallel when parallel read enabled",
				par instanceof ReaderDeltaParallel);

			cc.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, false);
			ConfigurationManager.setLocalConfig(cc);
			MatrixReader ser = MatrixReaderFactory.createMatrixReader(FileFormat.DELTA);
			assertTrue("expected serial ReaderDelta when parallel read disabled",
				ser instanceof ReaderDelta && !(ser instanceof ReaderDeltaParallel));
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
		}
	}

	@Test
	public void readFloatColumnsCoercedToDouble() throws Exception {
		//float columns must be widened to double on read (exact-representable values)
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			double[] f0 = {1.5, -2.25, 0.0, 1024.5};
			double[] f1 = {-0.5, 3.75, 100.125, -7.0};
			writeTypedColumns(tablePath,
				new DataType[] {FloatType.FLOAT, FloatType.FLOAT},
				new double[][] {f0, f1});

			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("rows", 4, out.getNumRows());
			assertEquals("cols", 2, out.getNumColumns());
			for( int r=0; r<4; r++ ) {
				assertEquals("f0 r" + r, f0[r], out.get(r, 0), 0.0);
				assertEquals("f1 r" + r, f1[r], out.get(r, 1), 0.0);
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readShortByteColumnsCoercedToDouble() throws Exception {
		//short/byte columns must be coerced to double on read, exercising the
		//T_SHORT / T_BYTE branches of ReaderDelta.getDoubleValue.
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			double[] shortVals = {1, -2, 30000, 0};
			double[] byteVals  = {7, -8, 120, 0};
			writeTypedColumns(tablePath,
				new DataType[] {ShortType.SHORT, ByteType.BYTE},
				new double[][] {shortVals, byteVals});

			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("rows", 4, out.getNumRows());
			assertEquals("cols", 2, out.getNumColumns());
			for( int r=0; r<4; r++ ) {
				assertEquals("short col r" + r, shortVals[r], out.get(r, 0), 0.0);
				assertEquals("byte col r" + r, byteVals[r], out.get(r, 1), 0.0);
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void writerRejectsDimensionMismatch() throws Exception {
		//WriterDelta validates that the passed rlen/clen match the MatrixBlock
		//and rejects a mismatch with an IOException.
		MatrixBlock in = TestUtils.generateTestMatrixBlock(10, 4, -1, 1, 1.0, 5);
		in.recomputeNonZeros();
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new WriterDelta().writeMatrixToHDFS(in, tablePath, 11, 4, -1, in.getNonZeros());
			fail("expected an IOException for mismatched matrix dimensions");
		}
		catch(java.io.IOException ex) {
			assertTrue("message should mention the dimension mismatch, got: " + ex.getMessage(),
				ex.getMessage() != null && ex.getMessage().contains("dimensions mismatch"));
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readNullCellsBecomeZero() throws Exception {
		//nullable numeric columns with null cells must read back as 0.0
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			double[] vals    = {3.0, 7.0, 9.0, 11.0};
			boolean[] nulls  = {false, true, false, true};
			writeNullableDoubleColumn(tablePath, vals, nulls);

			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("rows", 4, out.getNumRows());
			assertEquals("cols", 1, out.getNumColumns());
			for( int r=0; r<4; r++ )
				assertEquals("r" + r, nulls[r] ? 0.0 : vals[r], out.get(r, 0), 0.0);
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readStringColumnRejected() throws Exception {
		//string columns cannot back an all-double matrix -> reader must reject them
		Path dir = Files.createTempDirectory("sysds_delta_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			writeStringColumn(tablePath, new String[] {"a", "b", "c"});
			try {
				new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
				fail("expected a DMLRuntimeException for a non-numeric (string) Delta column");
			}
			catch(DMLRuntimeException ex) {
				assertTrue("message should mention the non-numeric column, got: " + ex.getMessage(),
					ex.getMessage() != null && ex.getMessage().contains("non-numeric"));
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	/** Writes a single-batch Delta table with one column per given (type, values) pair. */
	private static void writeTypedColumns(String tablePath, DataType[] types, double[][] vals) throws Exception {
		Engine engine = DeltaKernelUtils.createEngine();
		StructType schema = new StructType();
		for( int c=0; c<types.length; c++ )
			schema = schema.add("c" + c, types[c], false);
		FilteredColumnarBatch fcb = new FilteredColumnarBatch(new TypedBatch(schema, types, vals, null), Optional.empty());
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(tablePath), schema, singleton(fcb));
	}

	/** Writes a single nullable double column with the given per-row null mask. */
	private static void writeNullableDoubleColumn(String tablePath, double[] vals, boolean[] nulls) throws Exception {
		Engine engine = DeltaKernelUtils.createEngine();
		StructType schema = new StructType().add("c0", DoubleType.DOUBLE, true);
		FilteredColumnarBatch fcb = new FilteredColumnarBatch(
			new TypedBatch(schema, new DataType[] {DoubleType.DOUBLE}, new double[][] {vals},
				new boolean[][] {nulls}), Optional.empty());
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(tablePath), schema, singleton(fcb));
	}

	/** Writes a single string column (used to assert the matrix reader rejects it). */
	private static void writeStringColumn(String tablePath, String[] vals) throws Exception {
		Engine engine = DeltaKernelUtils.createEngine();
		final StructType schema = new StructType().add("s", StringType.STRING, false);
		ColumnarBatch batch = new ColumnarBatch() {
			@Override public StructType getSchema() { return schema; }
			@Override public int getSize() { return vals.length; }
			@Override public ColumnVector getColumnVector(int ordinal) { return new StringVector(vals); }
		};
		FilteredColumnarBatch fcb = new FilteredColumnarBatch(batch, Optional.empty());
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(tablePath), schema, singleton(fcb));
	}

	private static CloseableIterator<FilteredColumnarBatch> singleton(FilteredColumnarBatch fcb) {
		return new CloseableIterator<FilteredColumnarBatch>() {
			private boolean _done = false;
			@Override public boolean hasNext() { return !_done; }
			@Override public FilteredColumnarBatch next() {
				if( _done ) throw new NoSuchElementException();
				_done = true;
				return fcb;
			}
			@Override public void close() {}
		};
	}

	/** Minimal in-memory columnar batch backed by per-column double[] values, with
	 *  an optional per-column null mask ({@code nulls==null} => no nulls). */
	private static class TypedBatch implements ColumnarBatch {
		private final StructType _schema;
		private final DataType[] _types;
		private final double[][] _vals;
		private final boolean[][] _nulls;
		TypedBatch(StructType schema, DataType[] types, double[][] vals, boolean[][] nulls) {
			_schema = schema; _types = types; _vals = vals; _nulls = nulls;
		}
		@Override public StructType getSchema() { return _schema; }
		@Override public int getSize() { return _vals[0].length; }
		@Override public ColumnVector getColumnVector(int ordinal) {
			return new TypedVector(_types[ordinal], _vals[ordinal],
				_nulls == null ? null : _nulls[ordinal]);
		}
	}

	/** Column view exposing a double[] as the requested Delta primitive type. */
	private static class TypedVector implements ColumnVector {
		private final DataType _type;
		private final double[] _vals;
		private final boolean[] _nulls;
		TypedVector(DataType type, double[] vals, boolean[] nulls) { _type = type; _vals = vals; _nulls = nulls; }
		@Override public DataType getDataType() { return _type; }
		@Override public int getSize() { return _vals.length; }
		@Override public boolean isNullAt(int rowId) { return _nulls != null && _nulls[rowId]; }
		@Override public double getDouble(int rowId) { return _vals[rowId]; }
		@Override public float getFloat(int rowId) { return (float) _vals[rowId]; }
		@Override public long getLong(int rowId) { return (long) _vals[rowId]; }
		@Override public int getInt(int rowId) { return (int) _vals[rowId]; }
		@Override public short getShort(int rowId) { return (short) _vals[rowId]; }
		@Override public byte getByte(int rowId) { return (byte) _vals[rowId]; }
		@Override public boolean getBoolean(int rowId) { return _vals[rowId] != 0; }
		@Override public void close() {}
	}

	/** Column view exposing a String[] as a Delta string column. */
	private static class StringVector implements ColumnVector {
		private final String[] _vals;
		StringVector(String[] vals) { _vals = vals; }
		@Override public DataType getDataType() { return StringType.STRING; }
		@Override public int getSize() { return _vals.length; }
		@Override public boolean isNullAt(int rowId) { return _vals[rowId] == null; }
		@Override public String getString(int rowId) { return _vals[rowId]; }
		@Override public void close() {}
	}
}
