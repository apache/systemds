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
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;

public class ReshapePerf extends APerfTest<Object, MatrixBlock> {

	private final int k;

	public ReshapePerf(int N, IGenerate<MatrixBlock> gen, int k) {
		super(N, gen);
		this.k = k;
	}

	public void run() throws Exception {
		MatrixBlock mb = gen.take();
		System.out.println(
			String.format("Input Size: %d x %d , sparsity: %f ", mb.getNumRows(), mb.getNumColumns(), mb.getSparsity()));
		warmup(() -> reshape_Div(k, 2), 1000);

		execute(() -> reshape_Div(k, 1), "same");
		for(int i = 2; i < 51; i++) {
			double d = ((double) mb.getNumRows() / i);
			final int ii = i;
			if((int) d == d) {
				execute(() -> reshape_Div(k, ii), "replace_div " + i + " Parallel: " + k);
			}

		}
	}

	private void reshape_Div(int k, int div) {
		MatrixBlock mb = gen.take();
		LibMatrixReorg.reshape(mb, null, mb.getNumRows() / div, mb.getNumColumns() * div, true, k);
		ret.add(null);
	}

	@Override
	protected String makeResString() {
		return "";
	}

	public static void main(String[] args) throws Exception {
		MatrixBlock a = TestUtils.ceil(TestUtils.generateTestMatrixBlock(10000, 10000, 0, 100, 0.1, 42));
		// to CSR
		// System.out.println("MCSR to CSR");
		// new ReshapePerf(100, new ConstMatrix(a), 1).run();
		// new ReshapePerf(1000, new ConstMatrix(a), 16).run();

		// System.out.println("MCSR to MCSR parallel");
		// MatrixBlock spy = spy(a);
		// when(spy.getNonZeros()).thenReturn((long)Integer.MAX_VALUE + 42L);
		// new ReshapePerf(100, new ConstMatrix(spy), 1).run();
		// new ReshapePerf(100, new ConstMatrix(spy), 16).run();

		System.out.println("CSR to CSR");
		a.setSparseBlock(new SparseBlockCSR(a.getSparseBlock()));
		new ReshapePerf(100, new ConstMatrix(a), 1).run();

	}

}
