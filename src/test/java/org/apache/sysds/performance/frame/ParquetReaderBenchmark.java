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
package org.apache.sysds.performance.frame;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.junit.After;
import org.junit.Assume;

/**
 * Parquet reader benchmark comparing the sequential and parallel column-API readers on the TPC-H lineitem dataset.
 *
 * Results report median and min/max across runs and are appended to temp/benchmark_results.csv for plotting.
 *
 * The benchmark methods are disabled by default; uncomment the Test annotations to run manually. If the dataset is not
 * present the benchmark is skipped with instructions.
 */
public class ParquetReaderBenchmark {

	private static final String TPCH_FILE = "temp/lineitem.tbl";
	private static final String RESULTS_CSV = "temp/benchmark_results.csv";
	private static final String TEMP_FILE = System.getProperty("java.io.tmpdir") + "/systemds_read_bench.parquet";
	private static final int RUNS = 7;
	// override on larger machine with -DmaxRows=... (e.g. -DmaxRows=20000000) to test larger row groups.
	private static final int MAX_ROWS = Integer.getInteger("maxRows", 2_000_000);

	// TPC-H lineitem schema
	private static final ValueType[] LINEITEM_SCHEMA = {ValueType.INT64, ValueType.INT64, ValueType.INT64,
		ValueType.INT32, ValueType.FP64, ValueType.FP64, ValueType.FP64, ValueType.FP64, ValueType.STRING,
		ValueType.STRING, ValueType.STRING, ValueType.STRING, ValueType.STRING, ValueType.STRING, ValueType.STRING,
		ValueType.STRING};
	private static final String[] LINEITEM_NAMES = {"orderkey", "partkey", "suppkey", "linenumber", "quantity",
		"extendedprice", "discount", "tax", "returnflag", "linestatus", "shipdate", "commitdate", "receiptdate",
		"shipinstruct", "shipmode", "comment"};

	private PrintWriter csv;

	@After
	public void cleanup() {
		new File(TEMP_FILE).delete();
		if(csv != null)
			csv.close();
	}

	// @Test
	public void benchmarkReadTpch() throws Exception {
		ReadSpec spec = writeTempParquet(loadLineitemOrSkip());
		runReadBenchmark("read_tpch", spec);
	}

	private static final class ReadSpec {
		final ValueType[] schema;
		final String[] names;
		final int rows;
		final int cols;

		ReadSpec(ValueType[] schema, String[] names, int rows, int cols) {
			this.schema = schema;
			this.names = names;
			this.rows = rows;
			this.cols = cols;
		}
	}

	private ReadSpec writeTempParquet(FrameBlock data) throws Exception {
		int rows = data.getNumRows(), cols = data.getNumColumns();
		ReadSpec spec = new ReadSpec(data.getSchema(), data.getColumnNames(), rows, cols);
		new File(TEMP_FILE).delete();
		new FrameWriterParquet().writeFrameToHDFS(data, TEMP_FILE, rows, cols);
		return spec;
	}

	/**
	 * Writes a frame to Parquet once, then times reading it back with both readers.
	 */
	private void runReadBenchmark(String category, ReadSpec spec) throws Exception {
		final ValueType[] schema = spec.schema;
		final String[] names = spec.names;
		final int rows = spec.rows, cols = spec.cols;

		FrameBlock probe = new FrameReaderParquet().readFrameFromHDFS(TEMP_FILE, schema, names, rows, cols);
		org.junit.Assert.assertEquals("Row count mismatch", rows, probe.getNumRows());
		org.junit.Assert.assertEquals("Column count mismatch", cols, probe.getNumColumns());
		probe = null;

		openCsv();
		System.out.println(
			"\n=== Parquet Read Benchmark [" + category + "] (" + rows + " rows, median of " + RUNS + " runs) ===\n");
		System.out.printf("%-24s  %12s  %15s  %s%n", "Reader", "Median (ms)", "Rows/sec", "[runs]");
		System.out.println("-".repeat(72));

		timeRead(category, "Sequential",
			() -> new FrameReaderParquet().readFrameFromHDFS(TEMP_FILE, schema, names, rows, cols), rows);
		timeRead(category, "Parallel",
			() -> new FrameReaderParquetParallel().readFrameFromHDFS(TEMP_FILE, schema, names, rows, cols), rows);
		System.out.println();
	}

	private interface ReadAction {
		FrameBlock run() throws Exception;
	}

	private void timeRead(String category, String label, ReadAction action, int rows) throws Exception {
		action.run(); // warmup

		long[] times = new long[RUNS];
		for(int run = 0; run < RUNS; run++) {
			long start = System.currentTimeMillis();
			action.run();
			times[run] = System.currentTimeMillis() - start;
		}
		long med = median(times);
		long min = Arrays.stream(times).min().orElse(med);
		long max = Arrays.stream(times).max().orElse(med);
		System.out.printf("%-24s  %12d  %15.0f  %14s  %s%n", label, med, rows * 1000.0 / med, meanStd(times),
			Arrays.toString(times));
		// columns: benchmark,label,time_ms(median),rows_per_sec,min_ms,max_ms
		csv.printf("%s,%s,%d,%.0f,%d,%d%n", category, label, med, rows * 1000.0 / med, min, max);
	}

	private static long median(long[] times) {
		long[] sorted = times.clone();
		Arrays.sort(sorted);
		return sorted[sorted.length / 2];
	}

	private static String meanStd(long[] times) {
		double mean = Arrays.stream(times).average().orElse(0);
		double var = Arrays.stream(times).mapToDouble(t -> (t - mean) * (t - mean)).average().orElse(0);
		return String.format("%.0f+-%.0f ms", mean, Math.sqrt(var));
	}

	private void openCsv() throws Exception {
		new File("temp").mkdirs();
		boolean exists = new File(RESULTS_CSV).exists();
		csv = new PrintWriter(new FileWriter(RESULTS_CSV, true));
		if(!exists)
			csv.println("benchmark,label,time_ms,rows_per_sec,size_mb,compression_ratio");
		csv.flush();
	}

	private FrameBlock loadLineitemOrSkip() throws Exception {
		File f = new File(TPCH_FILE);
		if(!f.exists()) {
			System.out.println("=== TPC-H read benchmark skipped, dataset not found at " + TPCH_FILE + " ===");
			Assume.assumeTrue("TPC-H dataset not found at " + TPCH_FILE, false);
		}
		System.out.print("Loading " + f.getPath() + " ... ");
		List<String[]> rows = new ArrayList<>();
		try(BufferedReader br = new BufferedReader(new FileReader(f))) {
			String line;
			while((line = br.readLine()) != null && rows.size() < MAX_ROWS) {
				if(line.isEmpty())
					continue;
				if(line.endsWith("|"))
					line = line.substring(0, line.length() - 1);
				rows.add(line.split("\\|", -1));
			}
		}
		String[][] arr = rows.toArray(new String[0][]);
		System.out.println(arr.length + " rows loaded.");
		return new FrameBlock(LINEITEM_SCHEMA, LINEITEM_NAMES, arr);
	}

}
