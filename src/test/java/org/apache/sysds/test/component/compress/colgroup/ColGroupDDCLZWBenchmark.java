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

package org.apache.sysds.test.component.compress.colgroup;

import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDCLZW;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.junit.Test;

import java.util.Arrays;
import java.util.stream.IntStream;

public class ColGroupDDCLZWBenchmark {
	private static final int BENCHMARK_ITERATIONS = 10;

	private static final int[] DATA_SIZES = {1, 10, 100, 1000, 10000, 100_000};

	private static class BenchmarkResult {
		int dataSize;

		long ddcMemoryBytes;
		long ddcCompressionTimeNs;
		long ddcDecompressionTimeNs;

		long ddclzwMemoryBytes;
		long ddclzwCompressionTimeNs;
		long ddclzwDecompressionTimeNs;

		// Comparison info
		double memoryReduction;
		double compressionSpeedup;
		double decompressionSpeedup;

		void calculateMetrics() {
			memoryReduction = (double) ddclzwMemoryBytes / ddcMemoryBytes;
			compressionSpeedup = (double) ddcCompressionTimeNs / ddclzwCompressionTimeNs;
			decompressionSpeedup = (double) ddcDecompressionTimeNs / ddclzwDecompressionTimeNs;
		}

		/// Pretty-print a colorful percent text
		String formatPercent(double ratio) {
			double percent = (100.0 * (1.0 - ratio));
			String ansiColor = percent > 0 ? "\u001B[32m" : "\u001B[31m";
			return ansiColor + String.format("%6.2f%%", percent) + "\u001B[0m";
		}

		@Override
		public String toString() {
			return String.format("Size: %7d | DDC: %8d bytes | DDCLZW: %8d bytes | " +
					"Memory reduction: %s | De-/Compression speedup: %.2f/%.2f times", dataSize, ddcMemoryBytes,
				ddclzwMemoryBytes, formatPercent(memoryReduction), decompressionSpeedup, compressionSpeedup);
		}
	}

	// Pattern generators (array)
	private int[] genPatternRepeating(int size, int... pattern) {
		int[] result = new int[size];
		for(int i = 0; i < size; i++) {
			result[i] = pattern[i % pattern.length];
		}
		return result;
	}

	/**
	 * Args (10, 5) Generates a pattern like: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
	 */
	private int[] genPatternDistributed(int size, int nUnique) {
		int[] result = new int[size];
		int runLength = size / nUnique;
		int pos = 0;
		for(int i = 0; i < nUnique && pos < size; i++) {
			int endPos = Math.min(pos + runLength, size);
			Arrays.fill(result, pos, endPos, i);
			pos = endPos;
		}
		return result;
	}

	private int[] genPatternRandom(int size, int nUnique, long seed) {
		int[] result = new int[size];
		java.util.Random rand = new java.util.Random(seed);
		for(int i = 0; i < size; i++) {
			result[i] = rand.nextInt(nUnique);
		}
		return result;
	}

	private void printBenchmarkTitle() {
		String callerMethodName = StackWalker.getInstance().walk(stream -> stream.skip(1).findFirst().get())
			.getMethodName();

		System.out.println();
		System.out.println("=".repeat(80));
		System.out.println("Benchmark: " + callerMethodName);
		System.out.println("=".repeat(80));
		System.out.println();
	}

	private ColGroupDDC createBenchmarkDDC(int[] mapping, int nUnique, int nCols) {
		IColIndex colIndexes = ColIndexFactory.create(nCols);

		double[] dictValues = new double[nUnique * nCols];
		for(int i = 0; i < nUnique; i++) {
			for(int c = 0; c < nCols; c++) {
				dictValues[i * nCols + c] = (i + 1) * 10.0 + c;
			}
		}
		Dictionary dict = Dictionary.create(dictValues);

		AMapToData data = MapToFactory.create(mapping.length, nUnique);
		for(int i = 0; i < mapping.length; i++) {
			data.set(i, mapping[i]);
		}

		return (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
	}

	private BenchmarkResult runBenchmark(int[] mapping, int nUnique, int nCols) {
		BenchmarkResult result = new BenchmarkResult();
		result.dataSize = mapping.length;

		ColGroupDDC ddc = createBenchmarkDDC(mapping, nUnique, nCols);

		// Measure DDC memory (though the method calculates how much storage it would take if the data structure were written to disk)
		result.ddcMemoryBytes = ddc.getExactSizeOnDisk();

		// Measure DDC decompression time (it's already decompressed, so measure access time)
		long startTime = System.nanoTime();
		for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
			AMapToData mapping_copy = ddc.getMapToData();
			mapping_copy.getIndex(mapping.length / 2);
		}
		long endTime = System.nanoTime();
		result.ddcDecompressionTimeNs = (endTime - startTime) / BENCHMARK_ITERATIONS;

		// Measure DDCLZW compression time
		startTime = System.nanoTime();
		ColGroupDDCLZW ddclzw = null;
		for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
			ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();
		}
		endTime = System.nanoTime();
		result.ddclzwCompressionTimeNs = (endTime - startTime) / BENCHMARK_ITERATIONS;

		// Measure DDCLZW memory
		result.ddclzwMemoryBytes = ddclzw.getExactSizeOnDisk();

		// Measure DDCLZW decompression time
		startTime = System.nanoTime();
		for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
			ColGroupDDC decompressed = (ColGroupDDC) ddclzw.convertToDDC();
			AMapToData mapping_copy = decompressed.getMapToData();
			mapping_copy.getIndex(mapping.length / 2);
		}
		endTime = System.nanoTime();
		result.ddclzwDecompressionTimeNs = (endTime - startTime) / BENCHMARK_ITERATIONS;

		result.calculateMetrics();
		return result;
	}

	@Test
	public void benchmarkRepeatingPatterns() {
		printBenchmarkTitle();
		for(int size : DATA_SIZES) {
			int[] mapping = genPatternRepeating(size, 0, 1, 2);
			BenchmarkResult result = runBenchmark(mapping, 3, 1);
			System.out.println(result);
		}
	}

	@Test
	public void benchmarkDistributed() {
		printBenchmarkTitle();
		for(int size : DATA_SIZES) {
			int[] mapping = genPatternDistributed(size, 3);
			BenchmarkResult result = runBenchmark(mapping, 3, 1);
			System.out.println(result);
		}
	}

	@Test
	public void benchmarkRandomData() {
		printBenchmarkTitle();
		for(int size : DATA_SIZES) {
			int[] mapping = genPatternRandom(size, 5, 42);
			BenchmarkResult result = runBenchmark(mapping, 5, 1);
			System.out.println(result);
		}
	}

	@Test
	public void benchmarkMultiColumn() {
		printBenchmarkTitle();
		for(int size : DATA_SIZES) {
			int[] mapping = genPatternRepeating(size, 0, 1, 2, 1, 0);
			BenchmarkResult result = runBenchmark(mapping, 3, 3);
			System.out.println(result);
		}
	}

	@Test
	public void benchmarkUniques() {
		printBenchmarkTitle();
		int size = 10000;
		for(int nUnique : new int[] {2, 5, 10, 20, 50}) {
			int[] mapping = genPatternRepeating(size, IntStream.range(0, nUnique).toArray());
			BenchmarkResult result = runBenchmark(mapping, nUnique, 1);
			System.out.println(result);
		}
	}

	@Test
	public void benchmarkGetIdx() { // TODO: is this benchmark useful when the time complexity is completely different?
		printBenchmarkTitle();

		final int[] DATA_SIZES_GET_IDX = {10, 50, 100};
		for(int size : DATA_SIZES_GET_IDX) {
			int[] mapping = genPatternRepeating(size, 0, 1, 2);
			ColGroupDDC ddc = createBenchmarkDDC(mapping, 3, 2);
			ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

			// Benchmark DDC
			long startTime = System.nanoTime();
			for(int iter = 0; iter < BENCHMARK_ITERATIONS * 100; iter++) {
				ddc.getIdx(size / 2, 0);
			}
			long ddcTime = System.nanoTime() - startTime;

			// Benchmark DDCLZW
			startTime = System.nanoTime();
			for(int iter = 0; iter < BENCHMARK_ITERATIONS * 100; iter++) {
				ddclzw.getIdx(size / 2, 0);
			}
			long ddclzwTime = System.nanoTime() - startTime;

			System.out.printf("Size: %7d | DDC: %6.2f ms | DDCLZW: %6d ms | Slowdown: %.2f times\n", size,
				(double) ddcTime / 1_000_000, ddclzwTime / 1_000_000, (double) ddclzwTime / ddcTime);
		}
	}

	@Test
	public void benchmarkSlice() {
		printBenchmarkTitle();

		for(int size : DATA_SIZES) {
			int[] mapping = genPatternRepeating(size, 0, 1, 2);
			ColGroupDDC ddc = createBenchmarkDDC(mapping, 3, 1);
			ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

			int sliceStart = size / 4;
			int sliceEnd = 3 * size / 4;

			// Benchmark DDC
			long startTime = System.nanoTime();
			for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
				ddc.sliceRows(sliceStart, sliceEnd);
			}
			long ddcTime = System.nanoTime() - startTime;

			// Benchmark DDCLZW
			startTime = System.nanoTime();
			for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
				ddclzw.sliceRows(sliceStart, sliceEnd);
			}
			long ddclzwTime = System.nanoTime() - startTime;

			System.out.printf("Size: %7d | Slice[%5d:%5d] | DDC: %6d ms | DDCLZW: %6d ms | Slowdown: %.2f times\n",
				size, sliceStart, sliceEnd, ddcTime / 1_000_000, ddclzwTime / 1_000_000, (double) ddclzwTime / ddcTime);
		}
	}
}
