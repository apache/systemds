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
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.matrix.data.LibMatrixReplace;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class MatrixReplacePerf extends APerfTest<Object, MatrixBlock> {

	private final int k;

	public MatrixReplacePerf(int N, IGenerate<MatrixBlock> gen, int k) {
		super(N, gen);
		this.k = k;
	}

	public void run() throws Exception {

		warmup(() -> replaceZeroTask(k), 10);
		execute(() -> replaceZeroTask(k), "replaceZero");
		execute(() -> replaceOneTask(k), "replaceOne");
		execute(() -> replaceNaNTask(k), "replaceNaN");
	}

	private void replaceZeroTask(int k) {
		MatrixBlock mb = gen.take();
		LibMatrixReplace.replaceOperations(mb, null, 0, 1, k);
		ret.add(null);
	}


	private void replaceOneTask(int k) {
		MatrixBlock mb = gen.take();
		LibMatrixReplace.replaceOperations(mb, null, 1, 2, k);
		ret.add(null);
	}


	private void replaceNaNTask(int k) {
		MatrixBlock mb = gen.take();
		LibMatrixReplace.replaceOperations(mb, null, Double.NaN, 2, k);
		ret.add(null);
	}

	@Override
	protected String makeResString() {
		return "";
	}

}
