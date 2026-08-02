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
 * Shape/size coverage for the native Delta frame readers: single-cell tables, the writer batch boundary (4096 rows), 1
 * to 1000 columns, a multi-file layout, all-null columns, and adversarial cell values (NaN/infinities, signed zeros,
 * integer extremes, empty/unicode/very long strings).
 *
 * Every shape is verified on all three read paths (serial direct decode as the default, parallel direct decode, and the
 * forced kernel-engine buffered fallback) against the in-memory input, cell for cell. Column types cycle through all
 * six writable frame value types so each typed decode loop is exercised at every shape; string nulls, empty strings and
 * multi-byte unicode are covered by the all-null column and extreme value tests.
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
		assertRoundTrip(1, 1, 101, false);
		assertRoundTrip(7, 6, 102, false);
	}

	@Test
	public void writerBatchBoundaryRoundTrip() throws Exception {
		assertRoundTrip(4096, 2, 201, false);
	}

	@Test
	public void columnScalingRoundTrip() throws Exception {
		// the 1-column and 1000-column endpoints at small row counts
		assertRoundTrip(1000, 1, 301, false);
		assertRoundTrip(100, 1000, 302, false);
	}

	@Test
	public void multiFileRoundTrip() throws Exception {
		// a table large enough to roll multiple data files so the per-file slicing
		// of the direct and parallel paths is covered
		assertRoundTrip(100_000, 6, 401, true);
	}

	@Test
	public void allNullStringColumnRoundTrip() throws Exception {
		// a string column that is entirely null (definition level 0 for every row,
		// no data bytes in any page) next to a fully-live numeric column
		int nrow = 1000;
		FrameBlock in = TestUtils.generateRandomFrameBlock(nrow, new ValueType[] {ValueType.STRING, ValueType.FP64},
			61);
		for(int r = 0; r < nrow; r++)
			in.set(r, 0, null);
		roundTripAllReaders(in, false);
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
		roundTripAllReaders(in, false);
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

	private static void assertRoundTrip(int nrow, int ncol, long seed, boolean multiFile) throws Exception {
		roundTripAllReaders(TestUtils.generateRandomFrameBlock(nrow, cycleSchema(ncol), seed), multiFile);
	}

	/**
	 * Write the frame and assert that the serial direct read (the default path), the parallel direct read, and the
	 * forced kernel-engine buffered fallback each reproduce the input cell for cell.
	 */
	private static void roundTripAllReaders(FrameBlock in, boolean multiFile) throws Exception {
		withTable(in, multiFile, tablePath -> {
			TestUtils.compareFrames(in,
				new FrameReaderDelta().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1), true);
			TestUtils.compareFrames(in,
				new FrameReaderDeltaParallel().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1), true);
			TestUtils.compareFrames(in, newBufferedReader().readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1),
				true);
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
}
