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

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.DeltaKernelUtils;
import org.apache.sysds.runtime.io.FrameReaderDelta;
import org.apache.sysds.runtime.io.FrameReaderDeltaParallel;
import org.apache.sysds.runtime.io.FrameWriterDelta;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Manual micro-benchmark comparing the serial {@link FrameReaderDelta} against the parallel
 * {@link FrameReaderDeltaParallel} on multi-file Delta frame tables. Not a correctness test (those live in
 * {@link DeltaFrameReadWriteTest}); it just prints timing/throughput numbers and is {@link Ignore}d so it does not run
 * in the normal build.
 *
 * <p>
 * The parallel reader decodes one task per parquet data file, so the speedup scales with the number of files
 * (controlled here via the writer target file size). Run it on a JVM with a realistically sized heap; under a tiny
 * young generation (e.g. the Surefire fork's {@code -Xmn300m}) the concurrent decode's higher allocation rate is
 * dominated by young-GC pauses and the numbers are not representative of a normal SystemDS process.
 * </p>
 *
 * <p>
 * Run explicitly (remove {@link Ignore} or run the compiled class directly), e.g.
 * {@code mvn -q test -Dtest=DeltaFrameReadPerf -DfailIfNoTests=false}.
 * </p>
 */
public class DeltaFrameReadPerf {

	private static final ValueType[] NO_SCHEMA = new ValueType[] {ValueType.STRING};
	private static final String[] NO_NAMES = new String[] {"x"};

	private static final long MB = 1024L * 1024;
	private static final int WARMUP = 2;
	private static final int REPS = 7;

	/** Entry point so the (otherwise {@code @Ignore}d) benchmarks can be run directly. */
	public static void main(String[] args) throws Exception {
		DeltaFrameReadPerf p = new DeltaFrameReadPerf();
		p.serialDirectVsBuffered();
	}

	/**
	 * Isolates the serial-reader change: compares the new direct (pre-sized, metadata-driven, single-pass) read against
	 * the old buffered (per-batch extract + concatenate) read on the SAME single-file table, so the only difference is
	 * the extra allocation + concatenation copy. Single file => no file-level parallelism involved, pure serial decode
	 * cost.
	 */
	@Test
	@Ignore("manual benchmark; remove @Ignore (or run the compiled class directly) to run")
	public void serialDirectVsBuffered() throws Exception {
		System.out.println("\n=== serial direct vs buffered (single file, 4M rows) ===");
		System.out.printf("%-9s %11s %11s %9s%n", "schema", "direct(ms)", "buffered(ms)", "speedup");
		for(String kind : new String[] {"numeric", "mixed", "string"}) {
			// force a single data file: disable adaptive sizing, huge target
			DMLConfig c = new DMLConfig();
			c.setTextValue(DMLConfig.DELTA_WRITER_ADAPTIVE_FILE_SIZE, "false");
			c.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(4L * 1024 * MB));
			ConfigurationManager.setLocalConfig(c);
			Path dir = Files.createTempDirectory("sysds_delta_frame_ab_");
			String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
			try {
				FrameBlock in = genFrame(kind, 4_000_000, 7);
				new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
				// direct = default serial reader; buffered = force the fallback path
				FrameReaderDelta direct = new FrameReaderDelta();
				FrameReaderDelta buffered = new FrameReaderDelta() {
					@Override
					protected boolean useDirectPath(DeltaKernelUtils.ScanHandle h) {
						return false;
					}
				};
				for(int i = 0; i < WARMUP; i++) {
					direct.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
					buffered.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
				}
				double[] td = new double[REPS], tb = new double[REPS];
				for(int i = 0; i < REPS; i++) {
					td[i] = time(() -> direct.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
					tb[i] = time(() -> buffered.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
				}
				double md = median(td), mb = median(tb);
				long ad = allocBytes(() -> direct.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
				long ab = allocBytes(() -> buffered.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
				System.out.printf("%-9s %11.2f %11.2f %8.2fx   alloc: %6.0f / %6.0f MB (%.2fx)%n", kind, md, mb,
					mb / md, ad / (double) MB, ab / (double) MB, ab / (double) ad);
			}
			finally {
				ConfigurationManager.clearLocalConfigs();
				FileUtils.deleteQuietly(dir.toFile());
			}
		}
	}

	/**
	 * End-to-end check of adaptive writer file sizing with NO explicit target size configured (the real default): the
	 * table should now be split into ~one file per reader and read fast, versus the single/few-file layout the fixed
	 * 64MB default produced.
	 */
	@Test
	@Ignore("manual benchmark; remove @Ignore (or run the compiled class directly) to run")
	public void adaptiveCheck() throws Exception {
		System.out.println("\n=== adaptive writer file sizing (default config, no target set) ===");
		System.out.println("threads = " + OptimizerUtils.getParallelBinaryReadParallelism());
		System.out.printf("%-9s %-8s %-6s %11s %11s %9s%n", "rows", "adaptive", "files", "serial(ms)", "par(ms)",
			"speedup");
		for(int rows : new int[] {1_000_000, 4_000_000}) {
			// default config => 64MB cap, adaptive sizing enabled
			ConfigurationManager.setLocalConfig(new DMLConfig());
			Path dir = Files.createTempDirectory("sysds_delta_frame_adp_");
			String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
			try {
				FrameBlock in = genFrame("mixed", rows, 7);
				long est = in.getInMemorySize();
				long target = DeltaKernelUtils.adaptiveWriterTargetFileSize(est);
				new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
				long files = countParquet(tablePath);
				double[] r = measure(tablePath);
				System.out.printf("%-9d %-8s %-6d %11.2f %11.2f %8.2fx%n", rows, (target / MB) + "MB", files, r[0],
					r[1], r[0] / r[1]);
			}
			finally {
				ConfigurationManager.clearLocalConfigs();
				FileUtils.deleteQuietly(dir.toFile());
			}
		}
	}

	/**
	 * Sweep the writer target file size ({@link DMLConfig#DELTA_WRITER_TARGET_FILE_SIZE}, the one public Delta knob
	 * that affects read parallelism) to find where the per-file parallel read stops improving, i.e. a good default for
	 * read-heavy use.
	 */
	@Test
	@Ignore("manual benchmark; remove @Ignore (or run the compiled class directly) to run")
	public void targetSizeSweep() throws Exception {
		final int rows = 4_000_000;
		final long[] sizesMB = {128, 64, 32, 16, 8, 4, 2};
		System.out.println("\n=== writer target-size sweep (mixed, " + rows + " rows, "
			+ OptimizerUtils.getParallelBinaryReadParallelism() + " threads) ===");
		System.out.printf("%-9s %-6s %11s %11s %9s%n", "targetMB", "files", "serial(ms)", "par(ms)", "speedup");
		for(long mb : sizesMB) {
			DMLConfig c = new DMLConfig();
			c.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(mb * MB));
			ConfigurationManager.setLocalConfig(c);
			Path dir = Files.createTempDirectory("sysds_delta_frame_ts_");
			String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
			try {
				FrameBlock in = genFrame("mixed", rows, 7);
				new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
				long files = countParquet(tablePath);
				double[] r = measure(tablePath);
				System.out.printf("%-9d %-6d %11.2f %11.2f %8.2fx%n", mb, files, r[0], r[1], r[0] / r[1]);
			}
			finally {
				ConfigurationManager.clearLocalConfigs();
				FileUtils.deleteQuietly(dir.toFile());
			}
		}
	}

	/**
	 * Sweep the parquet reader batch size ({@link DMLConfig#DELTA_READER_BATCH_SIZE}, a public Delta Kernel knob) on a
	 * fixed multi-file table, with and without quieting the parquet/delta loggers. Pure "how we call the public API"
	 * tuning.
	 */
	@Test
	@Ignore("manual benchmark; remove @Ignore (or run the compiled class directly) to run")
	public void batchSizeSweep() throws Exception {
		final int rows = 2_000_000;
		final long fileSize = 8 * MB;
		final int[] batches = {1024, 4096, 8192, 16384, 32768, 65536, 131072};
		System.out.println("\n=== reader batch-size sweep (mixed, " + rows + " rows, 8MB files) ===");

		// write the table ONCE; the batch size only affects the read path
		DMLConfig wconf = new DMLConfig();
		wconf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(fileSize));
		ConfigurationManager.setLocalConfig(wconf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_bs_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			FrameBlock in = genFrame("mixed", rows, 7);
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			System.out.println("files = " + countParquet(tablePath));

			for(boolean quietLog : new boolean[] {false, true}) {
				if(quietLog)
					silenceParquetLogging();
				System.out.println(quietLog ? "-- parquet/delta logging -> ERROR --" : "-- default logging --");
				System.out.printf("%-9s %11s %11s%n", "batch", "serial(ms)", "par(ms)");
				for(int bs : batches) {
					DMLConfig c = new DMLConfig();
					c.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(fileSize));
					c.setTextValue(DMLConfig.DELTA_READER_BATCH_SIZE, String.valueOf(bs));
					ConfigurationManager.setLocalConfig(c);
					double[] r = measure(tablePath);
					System.out.printf("%-9d %11.2f %11.2f%n", bs, r[0], r[1]);
				}
			}
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	/** Median serial and parallel read time (ms) for a fixed table under the current config. */
	private double[] measure(String tablePath) throws Exception {
		FrameReaderDelta serial = new FrameReaderDelta();
		FrameReaderDeltaParallel parallel = new FrameReaderDeltaParallel();
		for(int i = 0; i < WARMUP; i++) {
			serial.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			parallel.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
		}
		double[] ts = new double[REPS], tp = new double[REPS];
		for(int i = 0; i < REPS; i++) {
			ts[i] = time(() -> serial.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
			tp[i] = time(() -> parallel.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
		}
		return new double[] {median(ts), median(tp)};
	}

	@Test
	@Ignore("manual benchmark; remove @Ignore (or run the compiled class directly) to run")
	public void benchmark() throws Exception {
		System.out.println("\n=== Delta frame reader benchmark ===");
		System.out.println("parallel read threads = " + OptimizerUtils.getParallelBinaryReadParallelism()
			+ ", processors = " + Runtime.getRuntime().availableProcessors());
		System.out.printf("%-9s %-7s %-6s %11s %11s %9s%n", "rows", "fileMB", "files", "serial(ms)", "par(ms)",
			"speedup");
		runCase(1_000_000, 4 * MB);
		runCase(1_000_000, 16 * MB);
		runCase(4_000_000, 8 * MB);
		runCase(4_000_000, 64 * MB);
	}

	@Test
	@Ignore("manual benchmark; remove @Ignore (or run the compiled class directly) to run")
	public void schemaBreakdown() throws Exception {
		System.out.println("\n=== schema composition breakdown (2M rows, 8MB files) ===");
		System.out.printf("%-10s %-6s %11s %11s %9s%n", "schema", "files", "serial(ms)", "par(ms)", "speedup");
		int rows = 2_000_000;
		for(boolean quietLog : new boolean[] {false, true}) {
			if(quietLog)
				silenceParquetLogging();
			System.out.println(quietLog ? "-- parquet/delta logging -> ERROR --" : "-- default logging --");
			runSchema("numeric", rows, 8 * MB);
			runSchema("mixed", rows, 8 * MB);
			runSchema("string", rows, 8 * MB);
		}
	}

	private static void silenceParquetLogging() {
		org.apache.log4j.Logger.getLogger("org.apache.parquet").setLevel(org.apache.log4j.Level.ERROR);
		org.apache.log4j.Logger.getLogger("io.delta").setLevel(org.apache.log4j.Level.ERROR);
		org.apache.log4j.Logger.getLogger("shaded.parquet").setLevel(org.apache.log4j.Level.ERROR);
	}

	private void runSchema(String kind, int rows, long targetFileSize) throws Exception {
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(targetFileSize));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_perf_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			FrameBlock in = genFrame(kind, rows, 7);
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			long files = countParquet(tablePath);
			FrameReaderDelta serial = new FrameReaderDelta();
			FrameReaderDeltaParallel parallel = new FrameReaderDeltaParallel();
			for(int i = 0; i < WARMUP; i++) {
				serial.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
				parallel.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			}
			double[] ts = new double[REPS], tp = new double[REPS];
			for(int i = 0; i < REPS; i++) {
				ts[i] = time(() -> serial.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
				tp[i] = time(() -> parallel.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
			}
			double ms = median(ts), mp = median(tp);
			System.out.printf("%-10s %-6d %11.2f %11.2f %8.2fx%n", kind, files, ms, mp, ms / mp);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	private void runCase(int rows, long targetFileSize) throws Exception {
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(targetFileSize));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_frame_perf_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			FrameBlock in = genMixedFrame(rows, 7);
			new FrameWriterDelta().writeFrameToHDFS(in, tablePath, in.getNumRows(), in.getNumColumns());
			long files = countParquet(tablePath);

			FrameReaderDelta serial = new FrameReaderDelta();
			FrameReaderDeltaParallel parallel = new FrameReaderDeltaParallel();

			for(int i = 0; i < WARMUP; i++) {
				serial.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
				parallel.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1);
			}

			double[] ts = new double[REPS], tp = new double[REPS];
			for(int i = 0; i < REPS; i++) {
				ts[i] = time(() -> serial.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
				tp[i] = time(() -> parallel.readFrameFromHDFS(tablePath, NO_SCHEMA, NO_NAMES, -1, -1));
			}
			double ms = median(ts), mp = median(tp);
			System.out.printf("%-9d %-7d %-6d %11.2f %11.2f %8.2fx%n", rows, targetFileSize / MB, files, ms, mp,
				ms / mp);
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	private static FrameBlock genFrame(String kind, int nrow, int seed) {
		ValueType[] schema;
		switch(kind) {
			case "numeric":
				schema = new ValueType[] {ValueType.INT64, ValueType.FP64, ValueType.INT32, ValueType.FP32,
					ValueType.BOOLEAN, ValueType.INT64};
				break;
			case "string":
				schema = new ValueType[] {ValueType.STRING, ValueType.STRING, ValueType.STRING, ValueType.STRING,
					ValueType.STRING, ValueType.STRING};
				break;
			default: // mixed
				schema = new ValueType[] {ValueType.STRING, ValueType.INT64, ValueType.FP64, ValueType.BOOLEAN,
					ValueType.INT32, ValueType.FP32};
		}
		String[] names = {"c0", "c1", "c2", "c3", "c4", "c5"};
		FrameBlock fb = new FrameBlock(schema, names);
		fb.ensureAllocatedColumns(nrow);
		Random rnd = new Random(seed);
		for(int r = 0; r < nrow; r++)
			for(int c = 0; c < schema.length; c++)
				fb.set(r, c, randVal(schema[c], rnd, r));
		return fb;
	}

	private static Object randVal(ValueType vt, Random rnd, int r) {
		switch(vt) {
			case STRING:
				return "row" + rnd.nextInt(1_000_000);
			case INT64:
				return (long) rnd.nextInt();
			case FP64:
				return rnd.nextDouble() * 200 - 100;
			case INT32:
				return rnd.nextInt();
			case FP32:
				return rnd.nextFloat();
			case BOOLEAN:
				return rnd.nextBoolean();
			default:
				return null;
		}
	}

	private interface IORun {
		FrameBlock run() throws Exception;
	}

	/** Bytes allocated by the calling (single) thread during one read. */
	private static long allocBytes(IORun r) throws Exception {
		com.sun.management.ThreadMXBean tb = (com.sun.management.ThreadMXBean) java.lang.management.ManagementFactory
			.getThreadMXBean();
		long id = Thread.currentThread().getId();
		long a0 = tb.getThreadAllocatedBytes(id);
		FrameBlock fb = r.run();
		long alloc = tb.getThreadAllocatedBytes(id) - a0;
		if(fb.getNumRows() <= 0)
			throw new IllegalStateException("empty read");
		return alloc;
	}

	private static double time(IORun r) throws Exception {
		long t0 = System.nanoTime();
		FrameBlock fb = r.run();
		long t1 = System.nanoTime();
		if(fb.getNumRows() <= 0)
			throw new IllegalStateException("empty read");
		return (t1 - t0) / 1e6;
	}

	private static double median(double[] v) {
		double[] c = v.clone();
		Arrays.sort(c);
		return c[c.length / 2];
	}

	private static long countParquet(String tablePath) throws Exception {
		try(java.util.stream.Stream<Path> s = Files.walk(new File(tablePath).toPath())) {
			return s.filter(p -> p.toString().endsWith(".parquet")).count();
		}
	}

	private static FrameBlock genMixedFrame(int nrow, int seed) {
		ValueType[] schema = {ValueType.STRING, ValueType.INT64, ValueType.FP64, ValueType.BOOLEAN, ValueType.INT32,
			ValueType.FP32};
		String[] names = {"name", "id", "score", "active", "count", "ratio"};
		FrameBlock fb = new FrameBlock(schema, names);
		fb.ensureAllocatedColumns(nrow);
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
}
