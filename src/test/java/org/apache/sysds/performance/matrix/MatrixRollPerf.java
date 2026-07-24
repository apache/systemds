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

import org.apache.sysds.performance.compression.APerfTest;
import org.apache.sysds.performance.generators.ConstMatrix;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.RollIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

import java.util.Random;

public class MatrixRollPerf extends APerfTest<Object, MatrixBlock> {

	private final int rows;
	private final int cols;
	private final int shift;
	private final int k;

	private final ReorgOperator reorg;
	private MatrixBlock out;

	public MatrixRollPerf(int N, int W, IGenerate<MatrixBlock> gen, int rows, int cols, int shift, int k) {
		super(N, W, gen);
		this.rows = rows;
		this.cols = cols;
		this.shift = shift;
		this.k = k;

		IndexFunction op = new RollIndex(shift);
		this.reorg = new ReorgOperator(op, k);
	}

	public void run() throws Exception {
		MatrixBlock mb = gen.take();
		logInfos(rows, cols, shift, mb.getSparsity(), k);


		String info = String.format("rows: %5d cols: %5d sp: %.4f shift: %4d k: %2d",
			rows, cols, mb.getSparsity(), shift, k);


		warmup(this::rollOnce, W);

		execute(this::rollOnce, info);
	}

	private void logInfos(int rows, int cols, int shift, double sparsity, int k) {
		String matrixType = sparsity == 1 ? "Dense" : "Sparse";
		if (k == 1) {
			System.out.println("---------------------------------------------------------------------------------------------------------");
			System.out.printf("%s Experiment for rows %d columns %d and shift %d \n", matrixType, rows, cols, shift);
			System.out.println("---------------------------------------------------------------------------------------------------------");
		}
	}

	private void rollOnce() {
		MatrixBlock in = gen.take();

		if (out == null)
			out = new MatrixBlock(rows, cols, in.isInSparseFormat());

		out.reset(rows, cols, in.isInSparseFormat());

		in.reorgOperations(reorg, out, 0, 0, 0);

		ret.add(null);
	}

	@Override
	protected String makeResString() {
		return "";
	}

	public static void main(String[] args) throws Exception {
		int kMulti = InfrastructureAnalyzer.getLocalParallelism();
		int reps = 2000;
		int warmup = 200;

		//int minRows = 2017;
		//int minCols = 1001;
		double spSparse = 0.01;
		int minShift = -50;
		int maxShift = 1022;
		int iterations = 10;

		Random rand = new Random(42);

		for (int i = 0; i < iterations; i++) {
			int rows = 10_000_000;
			int cols = 10;
			int shift = rand.nextInt((maxShift - minShift) + 1) + minShift;

			MatrixBlock denseIn = TestUtils.generateTestMatrixBlock(rows, cols, -100, 100, 1.0, 42);
			MatrixBlock sparseIn = TestUtils.generateTestMatrixBlock(rows, cols, -100, 100, spSparse, 42);

			// Run Dense Case (Single vs Multi-threaded)
			new MatrixRollPerf(reps, warmup, new ConstMatrix(denseIn, -1), rows, cols, shift, 1).run();
			new MatrixRollPerf(reps, warmup, new ConstMatrix(denseIn, -1), rows, cols, shift, kMulti).run();

			// Run Sparse Case (Single vs Multi-threaded)
			new MatrixRollPerf(reps, warmup, new ConstMatrix(sparseIn, -1), rows, cols, shift, 1).run();
			new MatrixRollPerf(reps, warmup, new ConstMatrix(sparseIn, -1), rows, cols, shift, kMulti).run();
		}
	}
}
