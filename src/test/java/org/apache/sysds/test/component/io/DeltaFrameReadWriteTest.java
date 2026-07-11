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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
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
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import io.delta.kernel.DataWriteContext;
import io.delta.kernel.Operation;
import io.delta.kernel.Table;
import io.delta.kernel.Transaction;
import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.ColumnarBatch;
import io.delta.kernel.data.FilteredColumnarBatch;
import io.delta.kernel.data.Row;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.internal.util.Utils;
import io.delta.kernel.types.ByteType;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.DateType;
import io.delta.kernel.types.DoubleType;
import io.delta.kernel.types.IntegerType;
import io.delta.kernel.types.LongType;
import io.delta.kernel.types.ShortType;
import io.delta.kernel.types.StringType;
import io.delta.kernel.types.StructType;
import io.delta.kernel.utils.CloseableIterable;
import io.delta.kernel.utils.CloseableIterator;
import io.delta.kernel.utils.DataFileStatus;

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

	// mixed-type schema used by the multi-file round-trip tests; random data is
	// generated via TestUtils rather than a bespoke per-test generator.
	private static final ValueType[] MIXED_SCHEMA = {ValueType.STRING, ValueType.INT64, ValueType.FP64,
		ValueType.BOOLEAN, ValueType.INT32, ValueType.FP32};

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

	@FunctionalInterface
	private interface TableTest {
		void accept(FrameBlock in, String tablePath) throws Exception;
	}

	/**
	 * Write {@code in} to a fresh temp Delta table with a small target file size (so the writer rolls multiple data
	 * files), assert the layout really is multi-file, then run {@code body} against the table. Local config and the
	 * temp directory are always cleaned up.
	 */
	private static void withSmallTargetTable(FrameBlock in, TableTest body) throws Exception {
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_mf_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			assertMultiFile(tablePath);
			body.accept(in, tablePath);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
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
		FrameBlock in = TestUtils.generateRandomFrameBlock(ROWS_MULTI_FILE, MIXED_SCHEMA, 13);
		withSmallTargetTable(in, (frame, tablePath) -> {
			FrameBlock serial = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			FrameBlock parallel = new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1,
				-1);
			assertFramesEqual(serial, parallel);
		});
	}

	@Test
	public void parallelBufferedPathMatchesSerialMultiFile() throws Exception {
		// the direct fast path is always taken for SystemDS-written tables (exact
		// row stats, no deletion vectors); force the buffered fallback to exercise
		// its per-file decode + serial concatenation and assert it matches serial.
		FrameBlock in = TestUtils.generateRandomFrameBlock(ROWS_MULTI_FILE, MIXED_SCHEMA, 23);
		withSmallTargetTable(in, (frame, tablePath) -> {
			FrameBlock serial = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			// subclass that always declines the direct path -> readBuffered()
			FrameBlock buffered = new FrameReaderDeltaParallel() {
				@Override
				protected boolean useDirectPath(DeltaKernelUtils.ScanHandle h) {
					return false;
				}
			}.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertFramesEqual(serial, buffered);
		});
	}

	@Test
	public void serialBufferedPathMatchesDirectMultiFile() throws Exception {
		// the direct (pre-sized, metadata-driven) path is always taken for SystemDS-
		// written tables; force the serial buffered fallback (per-batch extract +
		// concatenate) to exercise it and assert it matches the direct read.
		FrameBlock in = TestUtils.generateRandomFrameBlock(ROWS_MULTI_FILE, MIXED_SCHEMA, 29);
		withSmallTargetTable(in, (frame, tablePath) -> {
			FrameBlock direct = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			// subclass that always declines the direct path -> buffered extract+concat
			FrameBlock buffered = new FrameReaderDelta() {
				@Override
				protected boolean useDirectPath(DeltaKernelUtils.ScanHandle h) {
					return false;
				}
			}.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertFramesEqual(direct, buffered);
		});
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
			FrameBlock in = TestUtils.generateRandomFrameBlock(5000, MIXED_SCHEMA, 31);
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
		// data files for the same frame (the lever the parallel reader relies on);
		// the multi-file layout is asserted inside withSmallTargetTable.
		FrameBlock in = TestUtils.generateRandomFrameBlock(ROWS_MULTI_FILE, MIXED_SCHEMA, 41);
		withSmallTargetTable(in, (frame, tablePath) -> {
			// data still round-trips correctly with the custom layout
			FrameBlock out = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			assertFramesEqual(frame, out);
		});
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
	public void readTypeWidenedIntFilesAsLongColumn() throws Exception {
		// what Delta type widening leaves behind: the schema declares bigint while an
		// older data file still physically stores INT32. The direct decode must detect
		// the narrower physical type and fall back to the kernel engine for that file
		// only (the INT64 file stays on the direct path), in both readers.
		long[] oldInts = {1, -2, 0, Integer.MAX_VALUE, Integer.MIN_VALUE};
		long[] newLongs = {10L, -20L, 5_000_000_000L, Long.MAX_VALUE, Long.MIN_VALUE};
		Path dir = Files.createTempDirectory("sysds_delta_frame_tw_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			writeWidenedLongTable(tablePath, oldInts, newLongs);
			// pin the fixture: stats present (so the pre-sized direct path is what runs,
			// not the buffered fallback) and exactly one file physically narrower than
			// the schema (so the per-file kernel fallback really triggers)
			DeltaKernelUtils.ScanHandle handle = DeltaKernelUtils.openScan(DeltaKernelUtils.createEngine(),
				DeltaKernelUtils.qualify(tablePath));
			assertTrue("fixture must carry exact row counts", handle.hasExactRowCounts());
			assertEquals("fixture must span two data files", 2, handle.scanFiles.size());
			assertEquals("fixture must contain exactly one INT32-physical data file", 1, countInt32Files(tablePath));
			FrameReader[] readers = {new FrameReaderDelta(), new FrameReaderDeltaParallel()};
			for(FrameReader reader : readers) {
				FrameBlock out = reader.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
				assertEquals("rows", oldInts.length + newLongs.length, out.getNumRows());
				assertEquals("cols", 1, out.getNumColumns());
				assertEquals("widened column surfaces as INT64", ValueType.INT64, out.getSchema()[0]);
				for(int r = 0; r < oldInts.length; r++)
					assertEquals("int-file cell " + r, oldInts[r], ((Number) out.get(r, 0)).longValue());
				for(int r = 0; r < newLongs.length; r++)
					assertEquals("long-file cell " + r, newLongs[r],
						((Number) out.get(oldInts.length + r, 0)).longValue());
			}
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void readSchemaEvolvedFilesMissingAndReorderedColumns() throws Exception {
		// schema evolution leftovers: one data file written before column 'w' existed
		// (its cells must keep the column default, 0 for numerics), and one whose
		// parquet column order is reversed relative to the table schema (values must
		// land by name, not by position), identically on all three read paths.
		StructType tableSchema = new StructType().add("v", LongType.LONG, true).add("w", LongType.LONG, true);
		StructType vOnly = new StructType().add("v", LongType.LONG, true);
		StructType reversed = new StructType().add("w", LongType.LONG, true).add("v", LongType.LONG, true);
		long[] v1 = {1L, 2L, 3L};
		long[] v2 = {10L, 20L};
		long[] w2 = {100L, 200L};

		Path dir = Files.createTempDirectory("sysds_delta_frame_se_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			commitFiles(tablePath, tableSchema, batchOf(vOnly, new LongBackedVector(LongType.LONG, v1)),
				batchOf(reversed, new LongBackedVector(LongType.LONG, w2), new LongBackedVector(LongType.LONG, v2)));
			FrameReader[] readers = {new FrameReaderDelta(), new FrameReaderDeltaParallel(), new FrameReaderDelta() {
				@Override
				protected boolean useDirectPath(DeltaKernelUtils.ScanHandle h) {
					return false;
				}
			}};
			for(FrameReader reader : readers) {
				FrameBlock out = reader.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
				assertEquals("rows", v1.length + v2.length, out.getNumRows());
				assertEquals("cols", 2, out.getNumColumns());
				for(int r = 0; r < v1.length; r++) {
					assertEquals("old-file v " + r, v1[r], ((Number) out.get(r, 0)).longValue());
					assertEquals("old-file w (missing -> default) " + r, 0L, ((Number) out.get(r, 1)).longValue());
				}
				for(int r = 0; r < v2.length; r++) {
					assertEquals("reordered-file v " + r, v2[r], ((Number) out.get(v1.length + r, 0)).longValue());
					assertEquals("reordered-file w " + r, w2[r], ((Number) out.get(v1.length + r, 1)).longValue());
				}
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
			// must throw before touching the (null) stream, for the documented reason
			assertTrue("message should mention input stream, got: " + ex.getMessage(),
				ex.getMessage() != null && ex.getMessage().contains("input stream"));
		}
	}

	@Test
	public void parallelReadStringNullsMatchSerialMultiFile() throws Exception {
		// string nulls across a multi-file table: the parallel direct path must
		// reproduce the serial read cell-for-cell (assertFramesEqual uses
		// assertEquals, so nulls are compared faithfully).
		ValueType[] schema = {ValueType.STRING, ValueType.INT64};
		String[] names = {"s", "k"};
		int nrow = ROWS_MULTI_FILE;
		FrameBlock in = alloc(schema, names, nrow);
		for(int r = 0; r < nrow; r++) {
			// interspersed string nulls (every 7th row) plus a numeric column
			in.set(r, 0, (r % 7 == 0) ? null : "s" + r);
			in.set(r, 1, (long) r);
		}
		withSmallTargetTable(in, (frame, tablePath) -> {
			FrameBlock serial = new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			FrameBlock parallel = new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1,
				-1);
			assertFramesEqual(serial, parallel);
		});
	}

	private static void assertMultiFile(String tablePath) throws Exception {
		long files = DeltaFrameTestUtils.countParquet(tablePath);
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
				assertEquals("cell (" + r + "," + c + ")", expected.get(r, c), actual.get(r, c));
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

	/**
	 * Creates what Delta type widening leaves behind: a table whose schema declares {@code bigint} while its first data
	 * file physically stores INT32 (written before the widen) and its second INT64.
	 */
	private static void writeWidenedLongTable(String tablePath, long[] oldInts, long[] newLongs) throws Exception {
		StructType tableSchema = new StructType().add("v", LongType.LONG, true);
		StructType intSchema = new StructType().add("v", IntegerType.INTEGER, true);
		commitFiles(tablePath, tableSchema, batchOf(intSchema, new LongBackedVector(IntegerType.INTEGER, oldInts)),
			batchOf(tableSchema, new LongBackedVector(LongType.LONG, newLongs)));
	}

	/**
	 * Creates a table with the given schema and commits one physically-written data file per batch. The files are
	 * written through the kernel's parquet handler directly (bypassing the logical-data transform, which would reject
	 * batch schemas deviating from the table schema) and committed with their statistics, so reads take the pre-sized
	 * direct path.
	 */
	private static void commitFiles(String tablePath, StructType tableSchema, ColumnarBatch... batches)
		throws Exception {
		Engine engine = DeltaKernelUtils.createEngine();
		Table table = Table.forPath(engine, DeltaKernelUtils.qualify(tablePath));
		Transaction txn = table.createTransactionBuilder(engine, "SystemDS-test", Operation.CREATE_TABLE)
			.withSchema(engine, tableSchema).build(engine);
		Row txnState = txn.getTransactionState(engine);
		DataWriteContext ctx = Transaction.getWriteContext(engine, txnState, Collections.emptyMap());

		List<DataFileStatus> files = new ArrayList<>();
		for(ColumnarBatch batch : batches)
			drain(engine.getParquetHandler().writeParquetFiles(ctx.getTargetDirectory(),
				singleton(new FilteredColumnarBatch(batch, Optional.empty())), ctx.getStatisticsColumns()), files);

		CloseableIterator<Row> actions = Transaction.generateAppendActions(engine, txnState,
			Utils.toCloseableIterator(files.iterator()), ctx);
		txn.commit(engine, CloseableIterable.inMemoryIterable(actions));
	}

	/** Count the parquet data files whose single column is physically stored as INT32. */
	private static int countInt32Files(String tablePath) throws Exception {
		Configuration conf = new Configuration();
		int n = 0;
		try(java.util.stream.Stream<Path> s = Files.walk(new File(tablePath).toPath())) {
			for(Path p : (Iterable<Path>) s.filter(f -> f.toString().endsWith(".parquet"))::iterator) {
				try(ParquetFileReader r = ParquetFileReader
					.open(HadoopInputFile.fromPath(new org.apache.hadoop.fs.Path(p.toString()), conf))) {
					PrimitiveTypeName t = r.getFooter().getFileMetaData().getSchema().getType(0).asPrimitiveType()
						.getPrimitiveTypeName();
					if(t == PrimitiveTypeName.INT32)
						n++;
				}
			}
		}
		return n;
	}

	private static void drain(CloseableIterator<DataFileStatus> it, List<DataFileStatus> into) throws IOException {
		try(CloseableIterator<DataFileStatus> i = it) {
			while(i.hasNext())
				into.add(i.next());
		}
	}

	/** Batch view over per-column vectors (positionally matching the given schema). */
	private static ColumnarBatch batchOf(StructType schema, ColumnVector... cols) {
		return new ColumnarBatch() {
			@Override
			public StructType getSchema() {
				return schema;
			}

			@Override
			public int getSize() {
				return cols[0].getSize();
			}

			@Override
			public ColumnVector getColumnVector(int ordinal) {
				return cols[ordinal];
			}
		};
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

	/** Column view exposing a long[] as a Delta integer or long column ({@code getInt} narrows). */
	private static class LongBackedVector implements ColumnVector {
		private final DataType _dt;
		private final long[] _vals;

		LongBackedVector(DataType dt, long[] vals) {
			_dt = dt;
			_vals = vals;
		}

		@Override
		public DataType getDataType() {
			return _dt;
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
		public int getInt(int rowId) {
			return (int) _vals[rowId];
		}

		@Override
		public long getLong(int rowId) {
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
