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

import org.apache.commons.io.FileUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.DeltaKernelUtils;
import org.apache.sysds.runtime.io.FrameReaderDelta;
import org.apache.sysds.runtime.io.FrameReaderDeltaParallel;
import org.apache.sysds.runtime.io.FrameWriterDelta;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Systematic shape/size coverage for the native Delta frame readers: the direct parquet column decode across input
 * scales (100 rows to 1 million rows, 1 column to 1000 columns) plus the edge shapes around them: single-cell tables,
 * writer batch boundaries (4096 rows), prime/odd dimensions, multi-file layouts, all-null columns, and adversarial cell
 * values (NaN/infinities, signed zeros, integer extremes, empty/unicode/very long strings).
 *
 * Every shape is verified on all three read paths (serial direct decode as the default, parallel direct decode, and the
 * forced kernel-engine buffered fallback) against the in-memory input, cell for cell. Column types cycle through all
 * six writable frame value types so each typed decode loop is exercised at every shape, and every string column carries
 * interspersed nulls, empty strings and multi-byte unicode so the definition-level (null) handling and UTF-8 decode are
 * stressed at every size as well.
 */
public class DeltaFrameShapeCoverageTest {

	// nonsense schema/dims handed to the readers to confirm discovery from the table
	private static final ValueType[] NO_SCHEMA = new ValueType[] {ValueType.STRING};
	private static final String[] NO_NAMES = new String[] {"x"};

	// small target file size so large frames roll multiple data files and the
	// per-file parallel path really splits (mirrors DeltaFrameReadWriteTest)
	private static final long SMALL_TARGET_FILE_SIZE = 512L * 1024;

	// all value types the Delta frame writer can emit, cycled across columns so
	// every typed decode loop (string/long/double/boolean/int/float) is hit at
	// every covered shape
	private static final ValueType[] TYPE_CYCLE = {ValueType.STRING, ValueType.INT64, ValueType.FP64, ValueType.BOOLEAN,
		ValueType.INT32, ValueType.FP32};

	@Test
	public void tinyShapesRoundTrip() throws Exception {
		// the smallest possible tables, including a 1x1 whose only (string) cell is
		// null: a parquet page carrying no data bytes at all
		assertRoundTrip(1, 1, 101, false, true);
		assertRoundTrip(2, 1, 102, false, true);
		assertRoundTrip(3, 2, 103, false, true);
		assertRoundTrip(7, 6, 104, false, true);
		assertRoundTrip(100, 1, 105, false, true);
		assertRoundTrip(100, 3, 106, false, true);
		assertRoundTrip(100, 6, 107, false, true);
	}

	@Test
	public void writerBatchBoundariesRoundTrip() throws Exception {
		// the writer chunks frames into 4096-row batches; hit that boundary exactly,
		// one below and one above, plus a multiple of it
		assertRoundTrip(4095, 2, 201, false, true);
		assertRoundTrip(4096, 2, 202, false, true);
		assertRoundTrip(4097, 2, 203, false, true);
		assertRoundTrip(8192, 5, 204, false, true);
	}

	@Test
	public void columnScalingRoundTrip() throws Exception {
		// 1 -> 1000 columns at fixed row counts
		assertRoundTrip(1000, 1, 301, false, true);
		assertRoundTrip(100, 10, 302, false, true);
		assertRoundTrip(100, 100, 303, false, true);
		assertRoundTrip(1000, 64, 304, false, true);
		assertRoundTrip(1000, 100, 305, false, true);
		assertRoundTrip(1000, 256, 306, false, true);
		assertRoundTrip(100, 1000, 307, false, true);
		assertRoundTrip(1000, 1000, 308, false, true);
	}

	@Test
	public void rowScalingRoundTrip() throws Exception {
		// 100 -> 1M rows (the 100-row points are in tinyShapesRoundTrip); prime/odd
		// dimensions and multi-file layouts included
		assertRoundTrip(997, 7, 401, false, true);
		assertRoundTrip(10_000, 3, 402, false, true);
		assertRoundTrip(100_000, 6, 403, true, true);
		assertRoundTrip(250_000, 2, 404, true, true);
	}

	@Test
	public void millionRowsRoundTrip() throws Exception {
		// the 1M-row end of the row scaling, as multi-file tables so the parallel
		// reader really splits across files; the kernel-path parity at this scale
		// is covered via the forced-buffered read of the 1Mx3 table
		assertRoundTrip(1_000_000, 1, 501, true, false);
		assertRoundTrip(1_000_000, 3, 502, true, true);
	}

	@Test
	public void allNullStringColumnRoundTrip() throws Exception {
		// a string column that is entirely null (definition level 0 for every row,
		// no data bytes in any page) next to a fully-live numeric column
		int nrow = 10_000;
		FrameBlock in = TestUtils.generateRandomFrameBlock(nrow, new ValueType[] {ValueType.STRING, ValueType.FP64},
			61);
		for(int r = 0; r < nrow; r++)
			in.set(r, 0, null);
		roundTripAllReaders("allNullString 10000x2", in, false, true);
	}

	@Test
	public void extremeValuesRoundTrip() throws Exception {
		// adversarial cell values for every type: NaN, infinities, signed zeros,
		// subnormals, MIN/MAX of each numeric width, and empty / whitespace /
		// multi-byte unicode / control-character / very long strings
		ValueType[] schema = {ValueType.FP64, ValueType.FP32, ValueType.INT64, ValueType.INT32, ValueType.BOOLEAN,
			ValueType.STRING};
		String[] names = {"d", "f", "l", "i", "b", "s"};
		double[] d = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.MAX_VALUE,
			-Double.MAX_VALUE, Double.MIN_VALUE, 0.0, -0.0};
		float[] f = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.MAX_VALUE, -Float.MAX_VALUE,
			Float.MIN_VALUE, 0.0f, -0.0f};
		long[] l = {Long.MAX_VALUE, Long.MIN_VALUE, 0L, -1L, 1L, 1L << 40, -(1L << 40), 42L};
		int[] i = {Integer.MAX_VALUE, Integer.MIN_VALUE, 0, -1, 1, 1 << 20, -(1 << 20), 42};
		String longStr = new String(new char[10_000]).replace('\0', 'x');
		String[] s = {null, "", " ", "ü€𐍈", longStr, "line\nbreak", "tab\tsep", "quote\"'"};

		int nrow = d.length;
		FrameBlock in = new FrameBlock(schema, names);
		in.ensureAllocatedColumns(nrow);
		for(int r = 0; r < nrow; r++) {
			in.set(r, 0, d[r]);
			in.set(r, 1, f[r]);
			in.set(r, 2, l[r]);
			in.set(r, 3, i[r]);
			in.set(r, 4, r % 2 == 0);
			in.set(r, 5, s[r]);
		}
		roundTripAllReaders("extremes 8x6", in, false, true);
	}

	// ------------------------------------------
	// helpers
	// ------------------------------------------

	private static ValueType[] cycleSchema(int ncol) {
		ValueType[] schema = new ValueType[ncol];
		for(int c = 0; c < ncol; c++)
			schema[c] = TYPE_CYCLE[c % TYPE_CYCLE.length];
		return schema;
	}

	/**
	 * Deterministic test frame for a covered shape: random data over the cycled schema plus adversarial patterns
	 * injected into every string column (nulls every 7th row, empty strings every 11th, multi-byte unicode every 13th)
	 * so the null definition-level handling and UTF-8 decode are covered at every size, including across file and batch
	 * boundaries of the larger shapes.
	 */
	private static FrameBlock genFrame(int nrow, int ncol, long seed) {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(nrow, cycleSchema(ncol), seed);
		ValueType[] schema = fb.getSchema();
		for(int c = 0; c < ncol; c++) {
			if(schema[c] != ValueType.STRING)
				continue;
			for(int r = 0; r < nrow; r++) {
				if(r % 7 == 0)
					fb.set(r, c, null);
				else if(r % 11 == 0)
					fb.set(r, c, "");
				else if(r % 13 == 0)
					fb.set(r, c, "ü€𐍈" + r);
			}
		}
		return fb;
	}

	@FunctionalInterface
	private interface TableBody {
		void accept(String tablePath) throws Exception;
	}

	/**
	 * Write {@code in} to a fresh temp Delta table and run {@code body} against it. With {@code multiFile} a small
	 * target file size is configured and the resulting layout is asserted to really span multiple data files. Local
	 * config and the temp directory are always cleaned up.
	 */
	private static void withTable(FrameBlock in, boolean multiFile, TableBody body) throws Exception {
		if(multiFile) {
			DMLConfig conf = new DMLConfig();
			conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(SMALL_TARGET_FILE_SIZE));
			ConfigurationManager.setLocalConfig(conf);
		}
		Path dir = Files.createTempDirectory("sysds_delta_shape_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			if(multiFile)
				assertTrue("expected a multi-file Delta table for this shape",
					DeltaFrameTestUtils.countParquet(tablePath) > 1);
			body.accept(tablePath);
		}
		finally {
			if(multiFile)
				ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	private static void assertRoundTrip(int nrow, int ncol, long seed, boolean multiFile, boolean checkBuffered)
		throws Exception {
		roundTripAllReaders(nrow + "x" + ncol, genFrame(nrow, ncol, seed), multiFile, checkBuffered);
	}

	/**
	 * Write the frame and assert that the serial direct read (the default path), the parallel direct read, and (when
	 * {@code checkBuffered}) the forced kernel-engine buffered fallback each reproduce the input cell for cell.
	 */
	private static void roundTripAllReaders(String label, FrameBlock in, boolean multiFile, boolean checkBuffered)
		throws Exception {
		withTable(in, multiFile, tablePath -> {
			assertFramesEqual(label + " serial", in,
				new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
			assertFramesEqual(label + " parallel", in,
				new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
			if(checkBuffered)
				assertFramesEqual(label + " buffered", in,
					newBufferedReader().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
		});
	}

	/** Serial reader that always declines the direct path, forcing the kernel-engine buffered read. */
	private static FrameReaderDelta newBufferedReader() {
		return new FrameReaderDelta() {
			@Override
			protected boolean useDirectPath(DeltaKernelUtils.ScanHandle h) {
				return false;
			}
		};
	}

	private static void assertFramesEqual(String label, FrameBlock expected, FrameBlock actual) {
		assertEquals(label + ": rows", expected.getNumRows(), actual.getNumRows());
		assertEquals(label + ": cols", expected.getNumColumns(), actual.getNumColumns());
		int ncol = expected.getNumColumns();
		for(int c = 0; c < ncol; c++) {
			assertEquals(label + ": schema col " + c, expected.getSchema()[c], actual.getSchema()[c]);
			assertEquals(label + ": name col " + c, expected.getColumnNames()[c], actual.getColumnNames()[c]);
		}
		int nrow = expected.getNumRows();
		for(int r = 0; r < nrow; r++)
			for(int c = 0; c < ncol; c++)
				assertEquals(label + ": cell (" + r + "," + c + ")", expected.get(r, c), actual.get(r, c));
	}
}
