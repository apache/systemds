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

package org.apache.sysds.performance.matrix;

import java.util.Arrays;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.LibMatrixSketch;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Runtime comparison of unique() against the k=1 baseline, across different shares of unique values, for all three
 * directions (RowCol, Row, Col) and a small and a large input, at a fixed degree of parallelism.
 *
 * <p>
 * The share of unique values is defined per direction, matching what that direction deduplicates: for RowCol it is the
 * number of distinct scalar values in the whole input, for Row the number of distinct values within each row, and for
 * Col the number within each column.
 *
 * <p>
 * Each configuration is warmed up and then measured several times; the median is reported and the order of the k=1 and
 * k=N runs is alternated between iterations to reduce ordering effects. Input generation is excluded from the
 * measurements.
 *
 * <p>
 * Usage: {@code UniquePerf [numThreads] [reps]}
 */
public class UniquePerf {
	private static final int WARMUP = 3;
	private static final double[] SHARES = {0.01, 0.10, 0.50, 1.00};

	/** Per-direction input shapes, chosen so that each direction has a meaningful small and large case. */
	private static final Case[] CASES = {new Case(Types.Direction.RowCol, "small", 100000, 1),
		new Case(Types.Direction.RowCol, "large", 2000000, 1), new Case(Types.Direction.Row, "small", 10000, 64),
		new Case(Types.Direction.Row, "large", 200000, 64), new Case(Types.Direction.Col, "small", 64, 10000),
		new Case(Types.Direction.Col, "large", 64, 200000)};

	public static void main(String[] args) {
		int k = args.length > 0 ? Integer.parseInt(args[0]) : 4;
		int reps = args.length > 1 ? Integer.parseInt(args[1]) : 5;

		System.out.println("# unique() runtime vs. share of unique values");
		System.out.printf(
			"# threads k=%d vs. k=1 baseline, %d warmup + %d measured runs (median), JVM maxMemory=%d MiB%n", k, WARMUP,
			reps, Runtime.getRuntime().maxMemory() / 1024 / 1024);

		for(Types.Direction dir : new Types.Direction[] {Types.Direction.RowCol, Types.Direction.Row,
			Types.Direction.Col}) {
			System.out.printf("%n## %s%n%n", name(dir));
			System.out.printf("| size | input | unique share | %s | k=1 [ms] | k=%d [ms] | speedup |%n",
				distinctLabel(dir), k);
			System.out.println("|---|---|---|---|---|---|---|");

			for(Case c : CASES) {
				if(c._dir != dir)
					continue;
				for(double share : SHARES) {
					int distinct = distinctCount(dir, c._rlen, c._clen, share);
					MatrixBlock in = generate(dir, c._rlen, c._clen, distinct);

					double[] res = measure(in, dir, k, reps);
					System.out.printf("| %s | %dx%d | %.0f%% | %d | %.3f | %.3f | %.2f |%n", c._label, c._rlen, c._clen,
						share * 100, distinct, res[0], res[1], res[0] / res[1]);
				}
			}
		}
	}

	/**
	 * Warms up and then measures one configuration, alternating the order of the sequential and parallel runs.
	 *
	 * @return {median sequential time, median parallel time} in milliseconds
	 */
	private static double[] measure(MatrixBlock in, Types.Direction dir, int k, int reps) {
		for(int w = 0; w < WARMUP; w++) {
			LibMatrixSketch.getUniqueValues(in, dir, 1);
			LibMatrixSketch.getUniqueValues(in, dir, k);
		}

		double[] seq = new double[reps];
		double[] par = new double[reps];
		for(int i = 0; i < reps; i++) {
			if(i % 2 == 0) { // alternate order to reduce ordering effects
				seq[i] = time(in, dir, 1);
				par[i] = time(in, dir, k);
			}
			else {
				par[i] = time(in, dir, k);
				seq[i] = time(in, dir, 1);
			}
		}
		return new double[] {median(seq), median(par)};
	}

	private static double time(MatrixBlock in, Types.Direction dir, int k) {
		long t0 = System.nanoTime();
		LibMatrixSketch.getUniqueValues(in, dir, k);
		return (System.nanoTime() - t0) / 1e6;
	}

	private static String name(Types.Direction dir) {
		return dir == Types.Direction.RowCol ? "RowCol" : dir == Types.Direction.Row ? "Row" : "Col";
	}

	private static String distinctLabel(Types.Direction dir) {
		return dir == Types.Direction.RowCol ? "unique values" : dir == Types.Direction.Row ? "unique values per row" : "unique values per column";
	}

	/**
	 * Number of distinct values the given share corresponds to, per deduplicated unit.
	 */
	private static int distinctCount(Types.Direction dir, int rlen, int clen, double share) {
		long unit = dir == Types.Direction.RowCol ? (long) rlen * clen : dir == Types.Direction.Row ? clen : rlen;
		return (int) Math.max(1, Math.round(share * unit));
	}

	/**
	 * Builds an input whose per-unit distinct count matches the requested share.
	 */
	private static MatrixBlock generate(Types.Direction dir, int rlen, int clen, int distinct) {
		MatrixBlock ret = new MatrixBlock(rlen, clen, false).allocateBlock();
		for(int i = 0; i < rlen; i++) {
			for(int j = 0; j < clen; j++) {
				double val;
				if(dir == Types.Direction.RowCol)
					val = ((long) i * clen + j) % distinct;
				else if(dir == Types.Direction.Row)
					val = (i % 7) * (long) distinct + (j % distinct);
				else
					val = (j % 7) * (long) distinct + (i % distinct);
				ret.set(i, j, val);
			}
		}
		ret.recomputeNonZeros();
		return ret;
	}

	private static double median(double[] values) {
		double[] sorted = values.clone();
		Arrays.sort(sorted);
		return sorted[sorted.length / 2];
	}

	/** One benchmarked input shape for a given direction. */
	private static class Case {
		private final Types.Direction _dir;
		private final String _label;
		private final int _rlen;
		private final int _clen;

		private Case(Types.Direction dir, String label, int rlen, int clen) {
			_dir = dir;
			_label = label;
			_rlen = rlen;
			_clen = clen;
		}
	}
}
