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
import org.apache.sysds.performance.generators.GenPair;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class MatrixBinaryCellPerf extends APerfTest<Object, Pair<MatrixBlock, MatrixBlock>> {

	private final int k;

	public MatrixBinaryCellPerf(int N, IGenerate<Pair<MatrixBlock, MatrixBlock>> gen, int k) {
		super(N, gen);
		this.k = k;
	}

	public void run() throws Exception {

		final BinaryOperator plus = new BinaryOperator(Plus.getPlusFnObject(), k);
		warmup(() -> task(plus), 10);
		execute(() -> task(plus), "plus");
		final BinaryOperator mult = new BinaryOperator(Multiply.getMultiplyFnObject(), k);
		warmup(() -> task(mult), 10);
		execute(() -> task(mult), "mult");
		

	}

	private void task(BinaryOperator op) {
		Pair<MatrixBlock, MatrixBlock> in = gen.take();
		in.getKey().binaryOperations(op, in.getValue());
		ret.add(null);
	}

	@Override
	protected String makeResString() {
		return "";
	}

	public static void main(String[] args) throws Exception {
		// Matrix Blocks:
		MatrixBlock a_d = TestUtils.ceil(TestUtils.generateTestMatrixBlock(10000, 10000, 0, 100, 1.0, 42));
		MatrixBlock b_d = TestUtils.ceil(TestUtils.generateTestMatrixBlock(10000, 10000, 0, 100, 1.0, 32));
		MatrixBlock a_s = TestUtils.ceil(TestUtils.generateTestMatrixBlock(10000, 10000, 0, 100, 0.1, 42));
		MatrixBlock b_s = TestUtils.ceil(TestUtils.generateTestMatrixBlock(10000, 10000, 0, 100, 0.1, 32));
		int N = Integer.parseInt(args[1]);
		
		GenPair<MatrixBlock> gen;
		int k = InfrastructureAnalyzer.getLocalParallelism();

		gen = new GenPair<>(new ConstMatrix(a_s, -1), new ConstMatrix(b_s, -1));
		System.out.println("Sparse Sparse 0.1");
		new MatrixBinaryCellPerf(N, gen, k).run();

		gen = new GenPair<>(new ConstMatrix(a_d, -1), new ConstMatrix(b_s, -1));
		System.out.println("Dense Sparse 0.1");
		new MatrixBinaryCellPerf(N, gen, k).run();

		gen = new GenPair<>(new ConstMatrix(a_s, -1), new ConstMatrix(b_d, -1));
		System.out.println("Sparse Dense 0.1");
		new MatrixBinaryCellPerf(N, gen, k).run();

		gen = new GenPair<>(new ConstMatrix(a_d, -1), new ConstMatrix(b_d, -1));
		System.out.println("Dense Dense");
		new MatrixBinaryCellPerf(N, gen, k).run();
	}
}
