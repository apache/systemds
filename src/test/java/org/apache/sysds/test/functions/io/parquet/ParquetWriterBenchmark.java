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
package org.apache.sysds.test.functions.io.parquet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.runtime.io.FrameWriterParquet.DictEncoding;
import org.junit.After;
import org.junit.Assume;

/**
 * Parquet writer benchmark using the TPC-H lineitem dataset.
 * Writes results to temp/benchmark_results.csv for plotting.
 *
 * The benchmark methods are disabled by default; uncomment to run manually.
 * If a dataset is not present inside the temp dir the corresponding test is skipped with instructions.
 *
 * The default maxRows=2_000_000, to benchmark row-group
 * size properly, run on a machine with enough RAM for a larger frame:
 *   # 1. Generate a larger TPC-H lineitem into temp/lineitem.tbl
 *   # 2. Raise the maxRows cap, then run the benchmark:
 *   mvn test -Dtest=ParquetWriterBenchmark#benchmarkRowGroupSizes \
 *            -DmaxRows=60000000 -DargLine="-Xms24g -Xmx24g" -DfailIfNoTests=false
 */
public class ParquetWriterBenchmark {

	private static final String TPCH_FILE   = "temp/lineitem.tbl";
	private static final String RESULTS_CSV = "temp/benchmark_results.csv";
	private static final String TEMP_FILE   = System.getProperty("java.io.tmpdir") + "/systemds_tpch_bench.parquet";
	private static final int    RUNS        = 3;
	private static final int    MAX_ROWS    = Integer.getInteger("maxRows", 2_000_000);
	private static final int[]  BATCH_SIZES = { 1, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 200_000 };
	private static final long[] ROW_GROUP_SIZES = {
		1024 * 1024,        // 1 MB
		8L * 1024 * 1024,   // 8 MB
		16L * 1024 * 1024,  // 16 MB
		32L * 1024 * 1024,  // 32 MB
		64L * 1024 * 1024,  // 64 MB
		128L * 1024 * 1024, // 128 MB (Parquet default)
		256L * 1024 * 1024, // 256 MB
		512L * 1024 * 1024  // 512 MB 
	};

	// TPC-H lineitem schema
	private static final ValueType[] LINEITEM_SCHEMA = {
		ValueType.INT64,  // orderkey
		ValueType.INT64,  // partkey
		ValueType.INT64,  // suppkey
		ValueType.INT32,  // linenumber
		ValueType.FP64,   // quantity
		ValueType.FP64,   // extendedprice
		ValueType.FP64,   // discount
		ValueType.FP64,   // tax
		ValueType.STRING, // returnflag      (3 unique values)
		ValueType.STRING, // linestatus      (2 unique values)
		ValueType.STRING, // shipdate
		ValueType.STRING, // commitdate
		ValueType.STRING, // receiptdate
		ValueType.STRING, // shipinstruct    (4 unique values)
		ValueType.STRING, // shipmode        (7 unique values)
		ValueType.STRING  // comment         
	};

	private static final String[] LINEITEM_NAMES = {
		"orderkey", "partkey", "suppkey", "linenumber",
		"quantity", "extendedprice", "discount", "tax",
		"returnflag", "linestatus", "shipdate", "commitdate",
		"receiptdate", "shipinstruct", "shipmode", "comment"
	};

	private PrintWriter csv;

	@After
	public void cleanup() {
		new File(TEMP_FILE).delete();
		if (csv != null) csv.close();
	}

	// @Test
	public void benchmarkWriters() throws Exception {
		FrameBlock data = loadOrSkip();
		int rows = data.getNumRows();

		openCsv();
		System.out.println("\n=== TPC-H Writer Benchmark (" + rows + " rows, median of " + RUNS + " runs) ===\n");
		System.out.printf("%-38s  %12s  %15s%n", "Configuration", "Time (ms)", "Rows/sec");
		System.out.println("-".repeat(70));

		FrameWriterParquet newWriter = new FrameWriterParquet(CompressionCodecName.UNCOMPRESSED, DictEncoding.ALL_ON);

		// Warmup
		new File(TEMP_FILE).delete();
		new FrameWriterParquetLegacy(1000).writeFrameToHDFS(data, TEMP_FILE, rows, data.getNumColumns());
		new File(TEMP_FILE).delete();
		newWriter.writeFrameToHDFS(data, TEMP_FILE, rows, data.getNumColumns());

		for (int batchSize : BATCH_SIZES) {
			long[] times = new long[RUNS];
			for (int run = 0; run < RUNS; run++) {
				new File(TEMP_FILE).delete();
				long start = System.currentTimeMillis();
				new FrameWriterParquetLegacy(batchSize).writeFrameToHDFS(data, TEMP_FILE, rows, data.getNumColumns());
				times[run] = System.currentTimeMillis() - start;
			}
			long med = median(times);
			String label = "Legacy batchSize=" + batchSize;
			System.out.printf("%-38s  %12d  %15.0f%n", label, med, rows * 1000.0 / med);
			csv.printf("batch_sizes,%s,%d,%.0f,,%n", label, med, rows * 1000.0 / med);
		}

		System.out.println("-".repeat(70));

		long[] times = new long[RUNS];
		for (int run = 0; run < RUNS; run++) {
			new File(TEMP_FILE).delete();
			long start = System.currentTimeMillis();
			newWriter.writeFrameToHDFS(data, TEMP_FILE, rows, data.getNumColumns());
			times[run] = System.currentTimeMillis() - start;
		}
		long med = median(times);
		System.out.printf("%-38s  %12d  %15.0f%n", "New WriteSupport", med, rows * 1000.0 / med);
		csv.printf("batch_sizes,New WriteSupport,%d,%.0f,,%n", med, rows * 1000.0 / med);
		System.out.println();
	}

	// @Test
	public void benchmarkDictionaryEncoding() throws Exception {
		FrameBlock data = loadOrSkip();
		int rows = data.getNumRows();

		openCsv();
		System.out.println("\n=== TPC-H Dictionary Encoding Benchmark (" + rows + " rows, median of " + RUNS + " runs) ===\n");
		System.out.printf("%-20s  %12s  %15s%n", "Strategy", "Time (ms)", "Rows/sec");
		System.out.println("-".repeat(52));

		time("encoding", "ALL_ON",      new FrameWriterParquet(CompressionCodecName.UNCOMPRESSED, DictEncoding.ALL_ON),      data, rows);
		time("encoding", "ALL_OFF",     new FrameWriterParquet(CompressionCodecName.UNCOMPRESSED, DictEncoding.ALL_OFF),     data, rows);
		time("encoding", "STRING_ONLY", new FrameWriterParquet(CompressionCodecName.UNCOMPRESSED, DictEncoding.STRING_ONLY), data, rows);
		System.out.println();
	}


	// @Test
	public void benchmarkRowGroupSizes() throws Exception {
		FrameBlock data = loadOrSkip();
		int rows = data.getNumRows();

		openCsv();
		System.out.println("\n=== TPC-H Row Group Size Benchmark (" + rows + " rows, median of " + RUNS + " runs) ===\n");
		System.out.printf("%-20s  %12s  %15s%n", "Row Group Size", "Time (ms)", "Rows/sec");
		System.out.println("-".repeat(52));

		for (long rowGroupSize : ROW_GROUP_SIZES) {
			String label = (rowGroupSize / (1024 * 1024)) + "MB";
			time("row_group_sizes", label,
				new FrameWriterParquet(CompressionCodecName.ZSTD, DictEncoding.ALL_ON, rowGroupSize), data, rows);
		}
		System.out.println();
	}

	private FrameBlock loadOrSkip() throws Exception {
		File f = new File(TPCH_FILE);
		if (!f.exists()) {
			System.out.println();
			System.out.println("===================================================");
			System.out.println("TPC-H benchmark skipped, dataset not found");
			System.out.println("To reproduce:");
			System.out.println("  1. Install DuckDB:  https://duckdb.org");
			System.out.println("  2. Open shell:      duckdb");
			System.out.println("  3. Run in DuckDB:   INSTALL tpch;");
			System.out.println("                      LOAD tpch;");
			System.out.println("                      CALL dbgen(sf=1);");
			System.out.println("                      COPY lineitem TO '<systemds>/temp/lineitem.tbl'");
			System.out.println("                        (DELIMITER '|', HEADER false);");
			System.out.println("===================================================");
			Assume.assumeTrue("TPC-H dataset not found at " + TPCH_FILE, false);
		}
		return loadLineitem(f);
	}

	private void openCsv() throws Exception {
		new File("temp").mkdirs();
		boolean exists = new File(RESULTS_CSV).exists();
		csv = new PrintWriter(new FileWriter(RESULTS_CSV, true));
		if (!exists)
			csv.println("benchmark,label,time_ms,rows_per_sec,size_mb,compression_ratio");
		csv.flush();
	}

	private void time(String category, String label, FrameWriterParquet writer, FrameBlock data, int rows) throws Exception {
		new File(TEMP_FILE).delete();
		writer.writeFrameToHDFS(data, TEMP_FILE, rows, data.getNumColumns()); // warmup

		long[] times = new long[RUNS];
		for (int run = 0; run < RUNS; run++) {
			new File(TEMP_FILE).delete();
			long start = System.currentTimeMillis();
			writer.writeFrameToHDFS(data, TEMP_FILE, rows, data.getNumColumns());
			times[run] = System.currentTimeMillis() - start;
		}
		long med = median(times);
		System.out.printf("%-20s  %12d  %15.0f%n", label, med, rows * 1000.0 / med);
		csv.printf("%s,%s,%d,%.0f,,%n", category, label, med, rows * 1000.0 / med);
	}

	private long timeWithSize(String category, String label, FrameWriterParquet writer, FrameBlock data, int rows, long baseSize) throws Exception {
		new File(TEMP_FILE).delete();
		writer.writeFrameToHDFS(data, TEMP_FILE, rows, data.getNumColumns()); // warmup

		long[] times = new long[RUNS];
		for (int run = 0; run < RUNS; run++) {
			new File(TEMP_FILE).delete();
			long start = System.currentTimeMillis();
			writer.writeFrameToHDFS(data, TEMP_FILE, rows, data.getNumColumns());
			times[run] = System.currentTimeMillis() - start;
		}
		long med = median(times);
		long size = new File(TEMP_FILE).length();
		double mb = size / (1024.0 * 1024.0);
		String ratio = baseSize > 0 ? String.format("%.2fx", (double) baseSize / size) : "baseline";
		System.out.printf("%-20s  %12d  %15.0f  %12.2f  %10s%n", label, med, rows * 1000.0 / med, mb, ratio);
		csv.printf("%s,%s,%d,%.0f,%.2f,%s%n", category, label, med, rows * 1000.0 / med, mb, ratio);
		return size;
	}

	private static long median(long[] times) {
		long[] sorted = times.clone();
		Arrays.sort(sorted);
		return sorted[sorted.length / 2];
	}

	private static FrameBlock loadLineitem(File f) throws Exception {
		System.out.print("Loading " + f.getPath() + " ... ");
		List<String[]> rows = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(f))) {
			String line;
			while ((line = br.readLine()) != null && rows.size() < MAX_ROWS) {
				if (line.isEmpty()) continue;
				if (line.endsWith("|")) line = line.substring(0, line.length() - 1);
				rows.add(line.split("\\|", -1));
			}
		}
		String[][] data = rows.toArray(new String[0][]);
		System.out.println(data.length + " rows loaded.");
		return new FrameBlock(LINEITEM_SCHEMA, LINEITEM_NAMES, data);
	}
}
