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

	private static class BenchmarkResult {
		int dataSize;
		int nUnique;
		double entropy;

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

		/// Format a percent for pretty-printing
		String formatPercent(double ratio) {
			double percent = (100.0 * (1.0 - ratio));
			String ansiColor = percent > 0 ? "\u001B[32m" : "\u001B[31m";
			return ansiColor + String.format("%7.2f%%", percent) + "\u001B[0m";
		}

		/// Calculate entropy, and format a percent for pretty-printing
		String formatEntropyPercent(double entropy, int nUnique) {
			double maxEntropy = Math.log(nUnique) / Math.log(2); // log_2{nUnique}
			double percent = (entropy / maxEntropy) * 100;

			String ansiColor;
			if(percent < 33)
				ansiColor = "\u001B[32m";
			else if(percent < 66)
				ansiColor = "\u001B[33m";
			else
				ansiColor = "\u001B[31m";

			return String.format("%s%6.2f%%%s", ansiColor, percent, "\u001B[0m");
		}

		@Override
		public String toString() {
			return String.format("Size: %7d | nUnique: %4d | Entropy: %s | DDC: %7d bytes | DDCLZW: %7d bytes | " +
					"Memory reduction: %s | De-/Compression speedup: %.2f/%.2f times", dataSize, nUnique,
				formatEntropyPercent(entropy, nUnique), ddcMemoryBytes, ddclzwMemoryBytes,
				formatPercent(memoryReduction), decompressionSpeedup, compressionSpeedup);
		}
	}

	/// Calculates the entropy of the given array. Returns value between 0 (predictable) and log_2{nUnique}
	private double calculateEntropy(int[] arr, int nUnique) {
		int[] freq = new int[nUnique];
		for(int val : arr) {
			if(val >= 0 && val < nUnique) {
				freq[val]++;
			}
		}
		double entropy = 0.0;
		int total = arr.length;
		for(int f : freq) {
			if(f > 0) {
				double p = (double) f / total;
				entropy -= p * (Math.log(p) / Math.log(2));
			}
		}
		return entropy;
	}

	// Pattern generators (array)
	private int[] genPatternRepeating(int size, int... pattern) {
		int[] result = new int[size];
		for(int i = 0; i < size; i++) {
			result[i] = pattern[i % pattern.length];
		}
		return result;
	}

	/// Args (10, 5) Generates a pattern like: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
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

	private int[] genPatternLZWOptimal(int size, int nUnique) {
		int[] result = new int[size];
		int pos = 0;
		int patternLen = 10; // TODO: calculate based on nUnique?

		while(pos < size) {
			// Repeat the same pattern twice
			for(int i = 0; i < patternLen && pos < size; i++) {
				result[pos++] = i % nUnique;
			}
			for(int i = 0; i < patternLen && pos < size; i++) {
				result[pos++] = i % nUnique;
			}
			patternLen++;
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
		result.nUnique = nUnique;
		result.entropy = calculateEntropy(mapping, nUnique);

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
	public void benchmarkLZWOptimalScaling() {
		printBenchmarkTitle();

		for(int size : new int[] {100, 1000, 10_000, 40_000}) {
			System.out.println(".".repeat(35) + " Size: " + size + " " + ".".repeat(35));
			for(int nUnique : new int[] {2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 10_000, 20_000}) {
				if(nUnique > size)
					continue;
				int[] mapping = genPatternLZWOptimal(size, nUnique);
				BenchmarkResult result = runBenchmark(mapping, nUnique, 1);
				System.out.println(result);
			}
		}

		System.out.println("\nExpected memory growths for size n: DDC: O(n), DDCLZW: O(sqrt(n))");
	}

	@Test
	public void benchmarkDistributed() {
		printBenchmarkTitle();

		for(int size : new int[] {100, 1000, 10_000, 40_000}) {
			System.out.println(".".repeat(35) + " Size: " + size + " " + ".".repeat(35));
			for(int nUnique : new int[] {2, 3, 5, 10, 20, 50, 100, 200, 500, 1000}) {
				if(nUnique > size)
					continue;
				int[] mapping = genPatternDistributed(size, nUnique);
				BenchmarkResult result = runBenchmark(mapping, nUnique, 1);
				System.out.println(result);
			}
		}
	}

	@Test
	public void benchmarkUniquesRepeating() {
		printBenchmarkTitle();
		for(int size : new int[] {100, 1000, 10_000, 40_000}) {
			System.out.println(".".repeat(35) + " Size: " + size + " " + ".".repeat(35));
			for(int nUnique : new int[] {2, 3, 5, 10, 20, 50, 100, 200, 500, 1000}) {
				if(nUnique > size)
					continue;
				int[] mapping = genPatternRepeating(size, IntStream.range(0, nUnique).toArray());
				BenchmarkResult result = runBenchmark(mapping, nUnique, 1);
				System.out.println(result);
			}
		}
	}

	@Test
	public void benchmarkUniquesLZWOptimal() {
		printBenchmarkTitle();
		for(int size : new int[] {100, 1000, 10_000, 40_000}) {
			System.out.println(".".repeat(35) + " Size: " + size + " " + ".".repeat(35));
			for(int nUnique : new int[] {2, 3, 5, 10, 20, 50, 100, 200, 500, 1000}) {
				if(nUnique > size)
					continue;
				int[] mapping = genPatternLZWOptimal(size, nUnique);
				BenchmarkResult result = runBenchmark(mapping, nUnique, 1);
				System.out.println(result);
			}
		}
	}

	@Test
	public void benchmarkGetIdx() { // TODO: benchmark a different, efficient method instead
		printBenchmarkTitle();

		final int[] DATA_SIZES_GET_IDX = {10, 50, 100};
		for(int size : DATA_SIZES_GET_IDX) {
			int[] mapping = genPatternRepeating(size, 0, 1, 2, 2, 2, 1, 0, 0, 1);
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

	//	@Test
	//	public void benchmarkSlice() {
	//		printBenchmarkTitle();
	//
	//		for(int size : DATA_SIZES) {
	//			int[] mapping = genPatternRepeating(size, 0, 1, 2);
	//			ColGroupDDC ddc = createBenchmarkDDC(mapping, 3, 1);
	//			ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();
	//
	//			int sliceStart = size / 4;
	//			int sliceEnd = 3 * size / 4;
	//
	//			// Benchmark DDC
	//			long startTime = System.nanoTime();
	//			for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
	//				ddc.sliceRows(sliceStart, sliceEnd);
	//			}
	//			long ddcTime = System.nanoTime() - startTime;
	//
	//			// Benchmark DDCLZW
	//			startTime = System.nanoTime();
	//			for(int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
	//				ddclzw.sliceRows(sliceStart, sliceEnd);
	//			}
	//			long ddclzwTime = System.nanoTime() - startTime;
	//
	//			System.out.printf("Size: %7d | Slice[%5d:%5d] | DDC: %6d ms | DDCLZW: %6d ms | Slowdown: %.2f times\n",
	//				size, sliceStart, sliceEnd, ddcTime / 1_000_000, ddclzwTime / 1_000_000, (double) ddclzwTime / ddcTime);
	//		}
	//	}
}
