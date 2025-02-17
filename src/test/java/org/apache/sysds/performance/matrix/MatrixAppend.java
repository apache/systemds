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

import org.apache.sysds.performance.TimingUtils.StatsType;
import org.apache.sysds.performance.compression.APerfTest;
import org.apache.sysds.performance.generators.ConstMatrix;
import org.apache.sysds.performance.generators.GenMatrices;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class MatrixAppend extends APerfTest<Object, MatrixBlock> {

	final boolean cbind;

	final MatrixBlock base;
	final MatrixBlock[] others;
	final int n;

	public MatrixAppend(int N, int n, IGenerate<MatrixBlock> gen, boolean cbind) {
		super(N, 0, new ConstMatrix(gen.take()), StatsType.MEAN_STD_Q1);
		this.cbind = cbind;
		this.n = n;
		base = gen.take();

		others = new MatrixBlock[n];
		for(int i = 0; i < n; i++) {
			others[i] = gen.take();
		}

	}

	public void run() throws Exception {

		execute(() -> append(base, others), //
			String.format("appending:  rows:%5d cols:%5d sp:%3.1f  Blocks:%4d  rep:%6d  ", //
				base.getNumRows(), base.getNumColumns(), base.getSparsity(), n, N));

	}

	private void append(MatrixBlock a, MatrixBlock[] others) {
		a.append(others, null, cbind);
	}

	@Override
	protected String makeResString() {
		return "";
	}

	public static void main(String[] args) throws Exception {
		IGenerate<MatrixBlock> in;
		int nBlocks;
		int nRepeats;
		if(args.length == 0) {
			in = new GenMatrices(1000, 10, 10, 1.0);
			nBlocks = 10;
			nRepeats = 100;
		}
		else {
			in = new GenMatrices(Integer.parseInt(args[1]), Integer.parseInt(args[2]), 10, Double.parseDouble(args[3]));
			nBlocks = Integer.parseInt(args[4]);
			nRepeats = Integer.parseInt(args[5]);
		}
		in.generate(nBlocks + 2);

		new MatrixAppend(nRepeats, nBlocks, in, false).run();
	}

}
