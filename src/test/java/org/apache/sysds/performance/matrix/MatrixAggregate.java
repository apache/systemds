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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class MatrixAggregate extends APerfTest<Object, MatrixBlock> {

	private final int k;

	public MatrixAggregate(int N, IGenerate<MatrixBlock> gen, int k) {
		super(N, gen);
		this.k = k;
	}

	public void run() throws Exception {
		MatrixBlock mb = gen.take();

		String info = String.format("rows: %5d cols: %5d sp: %5.3f par: %2d", mb.getNumRows(), mb.getNumColumns(),
			mb.getSparsity(), k);
		warmup(() -> sum(), 100);
		execute(() -> sum(), info + " sum");
	}

	private void sum() {
		MatrixBlock in = gen.take();
		in.sum(k);
		ret.add(null);
	}

	@Override
	protected String makeResString() {
		return "";
	}

	public static void main(String[] args) throws Exception {
		// Matrix Blocks:

		int k = InfrastructureAnalyzer.getLocalParallelism();
		for(double i = 2.0d; i < 3.0; i += 0.15) {
			MatrixBlock a = TestUtils
				.ceil(TestUtils.generateTestMatrixBlock((int)Math.pow(10, i), Math.min((int)Math.pow(10, i), 1000), 0, 100, 1.0, 42));
			new MatrixAggregate(3000, new ConstMatrix(a, -1), k).run();
			new MatrixAggregate(3000, new ConstMatrix(a, -1), 1).run();
		}

	}
}
