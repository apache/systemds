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
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.DeltaKernelUtils;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderDelta;
import org.apache.sysds.runtime.io.FrameReaderDeltaParallel;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriterDelta;
import org.junit.Test;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.ColumnarBatch;
import io.delta.kernel.data.FilteredColumnarBatch;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.ByteType;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.DateType;
import io.delta.kernel.types.DoubleType;
import io.delta.kernel.types.LongType;
import io.delta.kernel.types.ShortType;
import io.delta.kernel.types.StringType;
import io.delta.kernel.types.StructType;
import io.delta.kernel.utils.CloseableIterator;

/**
 * Direct (no DML) round-trip tests for the native Delta Kernel based frame reader/writer. Each test writes a FrameBlock
 * to a fresh local Delta table directory and reads it back, asserting the discovered schema, column names, dimensions,
 * and per-cell values match. Several tests additionally assert that the parallel reader
 * ({@link FrameReaderDeltaParallel}) agrees with the serial reader cell-for-cell across a multi-file table (both its
 * direct and buffered paths).
 */
public class DeltaFrameReadWriteTest {

	// nonsense schema/dims handed to the reader to confirm it discovers everything
	private static final ValueType[] NO_SCHEMA = new ValueType[] {ValueType.STRING};
	private static final String[] NO_NAMES = new String[] {"x"};

	// small target file size + enough random rows so the writer rolls multiple
	// data files, exercising the per-file parallel read path rather than the
	// single-file serial fallback.
	private static final long SMALL_TARGET_FILE_SIZE = 512L * 1024;
	private static final int ROWS_MULTI_FILE = 150_000;

	private static FrameBlock writeThenRead(FrameBlock in) throws Exception {
		Path dir = Files.createTempDirectory("sysds_delta_frame_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			// pass nonsense schema/dims: the reader must discover everything from the table
			return new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	private static FrameBlock alloc(ValueType[] schema, String[] names, int nrow) {
		FrameBlock fb = new FrameBlock(schema, names);
		fb.ensureAllocatedColumns(nrow);
		return fb;
	}

	@Test
	public void roundTripMixedTypes() throws Exception {
		ValueType[] schema = {ValueType.STRING, ValueType.INT64, ValueType.FP64, ValueType.BOOLEAN, ValueType.INT32,
			ValueType.FP32};
		String[] names = {"name", "id", "score", "active", "count", "ratio"};
		int nrow = 5;
		FrameBlock in = alloc(schema, names, nrow);
		for(int r = 0; r < nrow; r++) {
			in.set(r, 0, "row" + r);
			in.set(r, 1, (long) (r * 1000L + 7));
			in.set(r, 2, r + 0.5);
			in.set(r, 3, (r % 2 == 0));
			in.set(r, 4, r * 3);
			in.set(r, 5, (float) (r / 4.0));
		}

		FrameBlock out = writeThenRead(in);

		assertEquals(nrow, out.getNumRows());
		assertEquals(schema.length, out.getNumColumns());
		// schema and names discovered from the table
		for(int c = 0; c < schema.length; c++) {
			assertEquals("schema col " + c, schema[c], out.getSchema()[c]);
			assertEquals("name col " + c, names[c], out.getColumnNames()[c]);
		}
		// values (compare as strings to be type-agnostic across boxed numerics)
		for(int r = 0; r < nrow; r++)
			for(int c = 0; c < schema.length; c++)
				assertEquals("cell (" + r + "," + c + ")", in.get(r, c).toString(), out.get(r, c).toString());
	}

	@Test
	public void roundTripMultiBatch() throws Exception {
		// more rows than the writer batch size (4096) to exercise chunking
		ValueType[] schema = {ValueType.INT64, ValueType.STRING};
		String[] names = {"k", "v"};
		int nrow = 10000;
		FrameBlock in = alloc(schema, names, nrow);
		for(int r = 0; r < nrow; r++) {
			in.set(r, 0, (long) r);
			in.set(r, 1, "v" + r);
		}

		FrameBlock out = writeThenRead(in);
		assertEquals(nrow, out.getNumRows());
		assertEquals(2, out.getNumColumns());
		for(int r = 0; r < nrow; r++) {
			assertEquals((long) r, ((Number) out.get(r, 0)).longValue());
			assertEquals("v" + r, out.get(r, 1).toString());
		}
	}

	@Test
	public void roundTripWithStringNulls() throws Exception {
		// nulls are only representable in object-backed (string) columns; numeric
		// frame columns store primitives and cannot carry a null.
		ValueType[] schema = {ValueType.STRING, ValueType.FP64};
		String[] names = {"s", "d"};
		int nrow = 4;
		FrameBlock in = alloc(schema, names, nrow);
		in.set(0, 0, "a");
		in.set(0, 1, 1.0);
		in.set(1, 0, null);
		in.set(1, 1, 2.0);
		in.set(2, 0, "c");
		in.set(2, 1, 3.0);
		in.set(3, 0, null);
		in.set(3, 1, 4.0);

		FrameBlock out = writeThenRead(in);
		assertEquals(nrow, out.getNumRows());
		assertEquals(2, out.getNumColumns());
		assertEquals("a", out.get(0, 0).toString());
		assertEquals(1.0, ((Number) out.get(0, 1)).doubleValue(), 1e-12);
		assertNull(out.get(1, 0));
		assertEquals(2.0, ((Number) out.get(1, 1)).doubleValue(), 1e-12);
		assertEquals("c", out.get(2, 0).toString());
		assertEquals(3.0, ((Number) out.get(2, 1)).doubleValue(), 1e-12);
		assertNull(out.get(3, 0));
		assertEquals(4.0, ((Number) out.get(3, 1)).doubleValue(), 1e-12);
	}

	@Test
	public void parallelReadMatchesSerialMultiFile() throws Exception {
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_par_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			FrameBlock in = genMixedFrame(ROWS_MULTI_FILE, 13);
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			assertMultiFile(tablePath);

			FrameBlock serial = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			FrameBlock parallel = new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1,
				-1);

			assertFramesEqual(serial, parallel);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void parallelBufferedPathMatchesSerialMultiFile() throws Exception {
		// the direct fast path is always taken for SystemDS-written tables (exact
		// row stats, no deletion vectors); force the buffered fallback to exercise
		// its per-file decode + serial concatenation and assert it matches serial.
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_buf_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			FrameBlock in = genMixedFrame(ROWS_MULTI_FILE, 23);
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			assertMultiFile(tablePath);

			FrameBlock serial = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			// subclass that always declines the direct path -> readBuffered()
			FrameBlock buffered = new FrameReaderDeltaParallel() {
				@Override
				protected boolean useDirectPath(DeltaKernelUtils.ScanHandle h) {
					return false;
				}
			}.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);

			assertFramesEqual(serial, buffered);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void serialBufferedPathMatchesDirectMultiFile() throws Exception {
		// the direct (pre-sized, metadata-driven) path is always taken for SystemDS-
		// written tables; force the serial buffered fallback (per-batch extract +
		// concatenate) to exercise it and assert it matches the direct read.
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_sbuf_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			FrameBlock in = genMixedFrame(ROWS_MULTI_FILE, 29);
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			assertMultiFile(tablePath);

			FrameBlock direct = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			// subclass that always declines the direct path -> buffered extract+concat
			FrameBlock buffered = new FrameReaderDelta() {
				@Override
				protected boolean useDirectPath(DeltaKernelUtils.ScanHandle h) {
					return false;
				}
			}.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);

			assertFramesEqual(direct, buffered);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void adaptiveTargetFileSizeClampsAndRespectsFlag() {
		// cap chosen above the 4MB floor so both clamp directions are observable
		final long cap = 64L * 1024 * 1024;
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(cap));
		conf.setTextValue(DMLConfig.DELTA_WRITER_ADAPTIVE_FILE_SIZE, "true");
		ConfigurationManager.setLocalConfig(conf);
		try {
			assertEquals("estimatedBytes<=0 -> configured cap", cap, DeltaKernelUtils.adaptiveWriterTargetFileSize(0));
			assertEquals("negative estimate -> configured cap", cap, DeltaKernelUtils.adaptiveWriterTargetFileSize(-1));
			assertEquals("huge table -> never above the configured cap", cap,
				DeltaKernelUtils.adaptiveWriterTargetFileSize(Long.MAX_VALUE / 2));
			assertEquals("tiny table -> never below the floor", DeltaKernelUtils.ADAPTIVE_WRITER_MIN_FILE_SIZE,
				DeltaKernelUtils.adaptiveWriterTargetFileSize(1));

			conf.setTextValue(DMLConfig.DELTA_WRITER_ADAPTIVE_FILE_SIZE, "false");
			assertEquals("flag OFF -> always the configured cap regardless of size", cap,
				DeltaKernelUtils.adaptiveWriterTargetFileSize(1));
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
		}
	}

	@Test
	public void factoryRoutesDeltaToParallelWhenEnabled() {
		// the factory must pick the parallel frame reader iff parallel CP read is enabled
		CompilerConfig cc = ConfigurationManager.getCompilerConfig();
		try {
			cc.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, true);
			ConfigurationManager.setLocalConfig(cc);
			FrameReader par = FrameReaderFactory.createFrameReader(FileFormat.DELTA);
			assertTrue("expected FrameReaderDeltaParallel when parallel read enabled",
				par instanceof FrameReaderDeltaParallel);

			cc.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, false);
			ConfigurationManager.setLocalConfig(cc);
			FrameReader ser = FrameReaderFactory.createFrameReader(FileFormat.DELTA);
			assertTrue("expected serial FrameReaderDelta when parallel read disabled",
				ser instanceof FrameReaderDelta && !(ser instanceof FrameReaderDeltaParallel));
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
		}
	}

	@Test
	public void readerBatchSizeConfigRoundTrips() throws Exception {
		// a non-default reader batch size must not change the result (more, smaller
		// batches exercise the per-batch extract/concatenate loop more often).
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_READER_BATCH_SIZE, "128");
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_bs_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			assertEquals("config getter reflects the override", 128, ConfigurationManager.getDeltaReaderBatchSize());

			FrameBlock in = genMixedFrame(5000, 31);
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			FrameBlock out = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertFramesEqual(in, out);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void writerTargetFileSizeConfigProducesMoreFiles() throws Exception {
		// a smaller configured target file size must make the writer roll more
		// data files for the same frame (the lever the parallel reader relies on).
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_cfg_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			assertEquals("config getter reflects the override", SMALL_TARGET_FILE_SIZE,
				ConfigurationManager.getDeltaWriterTargetFileSize());

			FrameBlock in = genMixedFrame(ROWS_MULTI_FILE, 41);
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			assertMultiFile(tablePath);

			// data still round-trips correctly with the custom layout
			FrameBlock out = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertFramesEqual(in, out);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void emptyFrameRoundTrip() throws Exception {
		// a schema-only Delta table (no data files, 0 rows); the reader must
		// rebuild empty typed columns and discover the schema/names from the table.
		ValueType[] schema = {ValueType.STRING, ValueType.FP64, ValueType.INT64};
		String[] names = {"s", "d", "k"};
		DataType[] dtypes = {StringType.STRING, DoubleType.DOUBLE, LongType.LONG};

		Path dir = Files.createTempDirectory("sysds_delta_frame_empty_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			writeEmptyTable(tablePath, names, dtypes);
			FrameBlock out = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertEquals("rows", 0, out.getNumRows());
			assertEquals("cols", schema.length, out.getNumColumns());
			for(int c = 0; c < schema.length; c++) {
				assertEquals("schema col " + c, schema[c], out.getSchema()[c]);
				assertEquals("name col " + c, names[c], out.getColumnNames()[c]);
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readDiscoversSchemaAndDims() throws Exception {
		// reader handed -1 dims and a nonsense schema must discover both from the table
		ValueType[] schema = {ValueType.INT32, ValueType.FP32, ValueType.BOOLEAN, ValueType.STRING};
		String[] names = {"a", "b", "c", "d"};
		int nrow = 321;
		FrameBlock in = alloc(schema, names, nrow);
		Random rnd = new Random(7);
		for(int r = 0; r < nrow; r++) {
			in.set(r, 0, rnd.nextInt());
			in.set(r, 1, rnd.nextFloat());
			in.set(r, 2, rnd.nextBoolean());
			in.set(r, 3, "s" + r);
		}

		Path dir = Files.createTempDirectory("sysds_delta_frame_disc_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, nrow, schema.length);
			FrameBlock out = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertEquals("rows", nrow, out.getNumRows());
			assertEquals("cols", schema.length, out.getNumColumns());
			for(int c = 0; c < schema.length; c++) {
				assertEquals("schema col " + c, schema[c], out.getSchema()[c]);
				assertEquals("name col " + c, names[c], out.getColumnNames()[c]);
			}
			assertFramesEqual(in, out);
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readNonMappableColumnRejected() throws Exception {
		// a Delta column type that does not map to a frame value type (date) must
		// be rejected by the reader rather than silently mis-read.
		Path dir = Files.createTempDirectory("sysds_delta_frame_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			writeDateColumn(tablePath, new int[] {0, 1, 100, 18000});
			try {
				new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
				fail("expected a DMLRuntimeException for a non-mappable (date) Delta column");
			}
			catch(DMLRuntimeException ex) {
				assertTrue("message should mention the non-mappable column, got: " + ex.getMessage(),
					ex.getMessage() != null && ex.getMessage().contains("non-mappable"));
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readShortByteColumnsCoercedToInt32() throws Exception {
		// the kernel can store short/byte columns; the frame reader has no narrower
		// integer value type, so both must surface as INT32 with the values intact.
		short[] shorts = {0, 1, -1, Short.MAX_VALUE, Short.MIN_VALUE};
		byte[] bytes = {0, 7, -7, Byte.MAX_VALUE, Byte.MIN_VALUE};
		Path dir = Files.createTempDirectory("sysds_delta_frame_sb_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			writeShortByteColumns(tablePath, shorts, bytes);
			FrameBlock out = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertEquals("rows", shorts.length, out.getNumRows());
			assertEquals("cols", 2, out.getNumColumns());
			assertEquals("short column coerced to INT32", ValueType.INT32, out.getSchema()[0]);
			assertEquals("byte column coerced to INT32", ValueType.INT32, out.getSchema()[1]);
			for(int r = 0; r < shorts.length; r++) {
				assertEquals("short cell (" + r + ")", shorts[r], ((Number) out.get(r, 0)).intValue());
				assertEquals("byte cell (" + r + ")", bytes[r], ((Number) out.get(r, 1)).intValue());
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void writerRejectsDimensionMismatch() throws Exception {
		ValueType[] schema = {ValueType.STRING, ValueType.INT64};
		String[] names = {"s", "k"};
		int nrow = 3;
		FrameBlock fb = alloc(schema, names, nrow);
		for(int r = 0; r < nrow; r++) {
			fb.set(r, 0, "r" + r);
			fb.set(r, 1, (long) r);
		}
		Path dir = Files.createTempDirectory("sysds_delta_frame_dim_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			// declare one more row than the frame actually has -> writer must reject
			new FrameWriterDelta().writeFrameToHDFS(fb, tablePath, fb.getNumRows() + 1, fb.getNumColumns());
			fail("expected an IOException for a frame/metadata dimension mismatch");
		}
		catch(IOException ex) {
			assertTrue("message should mention the dimension mismatch, got: " + ex.getMessage(),
				ex.getMessage() != null && ex.getMessage().contains("dimensions mismatch"));
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readFromInputStreamUnsupported() throws Exception {
		// Delta is a directory-based table format; stream reads are not supported
		try {
			new FrameReaderDelta().readFrameFromInputStream(null, NO_SCHEMA, NO_NAMES, -1, -1);
			fail("expected UnsupportedOperationException for a Delta input-stream read");
		}
		catch(UnsupportedOperationException ex) {
			// expected: must throw before touching the (null) stream
		}
	}

	@Test
	public void parallelReadStringNullsMatchSerialMultiFile() throws Exception {
		// string nulls across a multi-file table: the parallel direct path must
		// reproduce the serial read cell-for-cell (assertFramesEqual uses
		// Objects.equals, so nulls are compared faithfully).
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_parnull_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			ValueType[] schema = {ValueType.STRING, ValueType.INT64};
			String[] names = {"s", "k"};
			int nrow = ROWS_MULTI_FILE;
			FrameBlock in = alloc(schema, names, nrow);
			for(int r = 0; r < nrow; r++) {
				// interspersed string nulls (every 7th row) plus a numeric column
				in.set(r, 0, (r % 7 == 0) ? null : "s" + r);
				in.set(r, 1, (long) r);
			}
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			assertMultiFile(tablePath);

			FrameBlock serial = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			FrameBlock parallel = new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1,
				-1);

			assertFramesEqual(serial, parallel);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	private static FrameBlock genMixedFrame(int nrow, int seed) {
		ValueType[] schema = {ValueType.STRING, ValueType.INT64, ValueType.FP64, ValueType.BOOLEAN, ValueType.INT32,
			ValueType.FP32};
		String[] names = {"name", "id", "score", "active", "count", "ratio"};
		FrameBlock fb = alloc(schema, names, nrow);
		Random rnd = new Random(seed);
		for(int r = 0; r < nrow; r++) {
			fb.set(r, 0, "row" + rnd.nextInt(1_000_000));
			fb.set(r, 1, (long) rnd.nextInt());
			fb.set(r, 2, rnd.nextDouble() * 200 - 100);
			fb.set(r, 3, rnd.nextBoolean());
			fb.set(r, 4, rnd.nextInt());
			fb.set(r, 5, rnd.nextFloat());
		}
		return fb;
	}

	private static void assertMultiFile(String tablePath) throws Exception {
		long files;
		try(java.util.stream.Stream<Path> s = Files.walk(new File(tablePath).toPath())) {
			files = s.filter(p -> p.toString().endsWith(".parquet")).count();
		}
		assertTrue("expected a multi-file Delta table to exercise the parallel path, got " + files, files > 1);
	}

	private static void assertFramesEqual(FrameBlock expected, FrameBlock actual) {
		assertEquals("rows", expected.getNumRows(), actual.getNumRows());
		assertEquals("cols", expected.getNumColumns(), actual.getNumColumns());
		int ncol = expected.getNumColumns();
		for(int c = 0; c < ncol; c++) {
			assertEquals("schema col " + c, expected.getSchema()[c], actual.getSchema()[c]);
			assertEquals("name col " + c, expected.getColumnNames()[c], actual.getColumnNames()[c]);
		}
		int nrow = expected.getNumRows();
		for(int r = 0; r < nrow; r++)
			for(int c = 0; c < ncol; c++)
				assertTrue("cell (" + r + "," + c + ")", Objects.equals(expected.get(r, c), actual.get(r, c)));
	}

	/** Commits a schema-only Delta table (no data files) to exercise the 0-row read path. */
	private static void writeEmptyTable(String tablePath, String[] names, DataType[] dtypes) throws Exception {
		Engine engine = DeltaKernelUtils.createEngine();
		StructType schema = new StructType();
		for(int c = 0; c < dtypes.length; c++)
			schema = schema.add(names[c], dtypes[c], true);
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(tablePath), schema, empty());
	}

	private static CloseableIterator<FilteredColumnarBatch> empty() {
		return new CloseableIterator<FilteredColumnarBatch>() {
			@Override
			public boolean hasNext() {
				return false;
			}

			@Override
			public FilteredColumnarBatch next() {
				throw new NoSuchElementException();
			}

			@Override
			public void close() {
			}
		};
	}

	/**
	 * Writes a single date column (kernel stores dates as INT32 days) used to assert the frame reader rejects a
	 * non-mappable column type.
	 */
	private static void writeDateColumn(String tablePath, int[] days) throws Exception {
		Engine engine = DeltaKernelUtils.createEngine();
		final StructType schema = new StructType().add("d", DateType.DATE, false);
		ColumnarBatch batch = new ColumnarBatch() {
			@Override
			public StructType getSchema() {
				return schema;
			}

			@Override
			public int getSize() {
				return days.length;
			}

			@Override
			public ColumnVector getColumnVector(int ordinal) {
				return new DateVector(days);
			}
		};
		FilteredColumnarBatch fcb = new FilteredColumnarBatch(batch, Optional.empty());
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(tablePath), schema, singleton(fcb));
	}

	/**
	 * Writes a short column and a byte column (kernel stores these as 16/8-bit integers) used to assert the frame
	 * reader coerces both to INT32.
	 */
	private static void writeShortByteColumns(String tablePath, short[] shorts, byte[] bytes) throws Exception {
		Engine engine = DeltaKernelUtils.createEngine();
		final StructType schema = new StructType().add("sh", ShortType.SHORT, false).add("by", ByteType.BYTE, false);
		ColumnarBatch batch = new ColumnarBatch() {
			@Override
			public StructType getSchema() {
				return schema;
			}

			@Override
			public int getSize() {
				return shorts.length;
			}

			@Override
			public ColumnVector getColumnVector(int ordinal) {
				return (ordinal == 0) ? new ShortVector(shorts) : new ByteVector(bytes);
			}
		};
		FilteredColumnarBatch fcb = new FilteredColumnarBatch(batch, Optional.empty());
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(tablePath), schema, singleton(fcb));
	}

	private static CloseableIterator<FilteredColumnarBatch> singleton(FilteredColumnarBatch fcb) {
		return new CloseableIterator<FilteredColumnarBatch>() {
			private boolean _done = false;

			@Override
			public boolean hasNext() {
				return !_done;
			}

			@Override
			public FilteredColumnarBatch next() {
				if(_done)
					throw new NoSuchElementException();
				_done = true;
				return fcb;
			}

			@Override
			public void close() {
			}
		};
	}

	/** Column view exposing an int[] as a Delta date column. */
	private static class DateVector implements ColumnVector {
		private final int[] _days;

		DateVector(int[] days) {
			_days = days;
		}

		@Override
		public DataType getDataType() {
			return DateType.DATE;
		}

		@Override
		public int getSize() {
			return _days.length;
		}

		@Override
		public boolean isNullAt(int rowId) {
			return false;
		}

		@Override
		public int getInt(int rowId) {
			return _days[rowId];
		}

		@Override
		public void close() {
		}
	}

	/** Column view exposing a short[] as a Delta short column. */
	private static class ShortVector implements ColumnVector {
		private final short[] _vals;

		ShortVector(short[] vals) {
			_vals = vals;
		}

		@Override
		public DataType getDataType() {
			return ShortType.SHORT;
		}

		@Override
		public int getSize() {
			return _vals.length;
		}

		@Override
		public boolean isNullAt(int rowId) {
			return false;
		}

		@Override
		public short getShort(int rowId) {
			return _vals[rowId];
		}

		@Override
		public void close() {
		}
	}

	/** Column view exposing a byte[] as a Delta byte column. */
	private static class ByteVector implements ColumnVector {
		private final byte[] _vals;

		ByteVector(byte[] vals) {
			_vals = vals;
		}

		@Override
		public DataType getDataType() {
			return ByteType.BYTE;
		}

		@Override
		public int getSize() {
			return _vals.length;
		}

		@Override
		public boolean isNullAt(int rowId) {
			return false;
		}

		@Override
		public byte getByte(int rowId) {
			return _vals[rowId];
		}

		@Override
		public void close() {
		}
	}
}
