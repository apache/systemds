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
import java.util.Locale;

import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

/**
 * Benchmarks LibMatrixReorg.transpose for fixed input/output formats (dense or sparse) and prints CSV results.
 * <p>
 * Dispatched from {@link org.apache.sysds.performance.Main} (program id 1010).
 * Arguments: [reps] [warmup] [k], with k=-1 for all cores.
 */
public class TransposePerf {

	public enum Mode {
		DENSE_DENSE(false, false), DENSE_SPARSE(false, true);

		public final boolean sparseIn;
		public final boolean sparseOut;

		Mode(boolean sparseIn, boolean sparseOut) {
			this.sparseIn = sparseIn;
			this.sparseOut = sparseOut;
		}
	}

	private static final int[][] SHAPES = {{1000, 1000}, {3000, 3000}};
	private static final double[] SPARSITIES = {0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001};
	private static final Mode[] MODES = {Mode.DENSE_DENSE, Mode.DENSE_SPARSE};

	private final int reps;
	private final int warmup;

	public TransposePerf(int reps, int warmup) {
		this.reps = reps;
		this.warmup = warmup;
	}

	public void sweep(int[][] shapes, double[] sparsities, Mode[] modes, int k) {
		System.out.println("mode,rows,cols,sparsity,nnz,k,in_sparse,out_sparse,median_ms,mean_ms,min_ms");
		for(int[] shape : shapes)
			for(double sp : sparsities) {
				MatrixBlock base = TestUtils.generateTestMatrixBlock(shape[0], shape[1], -1, 1, sp, 7);
				for(Mode mode : modes) {
					MatrixBlock in = new MatrixBlock();
					in.copy(base, mode.sparseIn);
					run(mode, in, k);
				}
			}
	}

	public void run(Mode mode, MatrixBlock in, int k) {
		final MatrixBlock[] out = new MatrixBlock[1];
		TimingUtils.F f = () -> {
			out[0] = new MatrixBlock(in.getNumColumns(), in.getNumRows(), mode.sparseOut);
			LibMatrixReorg.transpose(in, out[0], k);
		};
		TimingUtils.time(f, warmup);
		double[] t = TimingUtils.time(f, reps);
		Arrays.sort(t);

		double median = t[t.length / 2];
		double mean = Arrays.stream(t).average().orElse(0);
		double min = t[0];

		System.out.printf(Locale.ROOT, "%s,%d,%d,%.2e,%d,%d,%b,%b,%.4f,%.4f,%.4f%n",
			mode, in.getNumRows(), in.getNumColumns(), in.getSparsity(), in.getNonZeros(), k,
			in.isInSparseFormat(), out[0].isInSparseFormat(),
			median, mean, min);
	}

	public static void main(String[] args) {
		int reps = args.length > 1 ? Integer.parseInt(args[1]) : 20;
		int warmup = args.length > 2 ? Integer.parseInt(args[2]) : 5;
		int k = args.length > 3 ? Integer.parseInt(args[3]) : 1;
		if(k == -1)
			k = InfrastructureAnalyzer.getLocalParallelism();
		new TransposePerf(reps, warmup).sweep(SHAPES, SPARSITIES, MODES, k);
	}
}
