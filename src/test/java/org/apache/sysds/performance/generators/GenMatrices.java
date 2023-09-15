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

package org.apache.sysds.performance.generators;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.TestUtils;

public class GenMatrices implements IGenerate<MatrixBlock> {

	/** A Task que that guarantee that the execution is not to long */
	protected final BlockingQueue<MatrixBlock> tasks;
	/** The number of rows in each task block */
	protected final int r;
	/** The number of cols in each task block */
	protected final int c;
	/** The number of max unique values */
	protected final int nVal;
	/** The sparsity of the generated matrices */
	protected final double s;
	/** The initial seed */
	protected final int seed;

	public GenMatrices(int r, int c, int nVal, double s) {
		// Make a thread pool if not already there
		CommonThreadPool.get();
		tasks = new ArrayBlockingQueue<>(8);
		this.r = r;
		this.c = c;
		this.nVal = nVal;
		this.s = s;
		this.seed = 42;
	}

	@Override
	public MatrixBlock take() {
		try {
			return tasks.take();
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void generate(int N) throws InterruptedException {
		CompletableFuture.runAsync(() -> {
			try {
				for(int i = 0; i < N; i++) {
					tasks.put(TestUtils.ceil(TestUtils.generateTestMatrixBlock(r, c, 0, nVal, s, i + seed)));
				}
			}
			catch(InterruptedException e) {
				e.printStackTrace();
			}
		});
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" rand(").append(r).append(", ").append(c).append(", ").append(nVal).append(", ").append(s).append(")");
		sb.append(" Seed: ").append(seed);
		return sb.toString();
	}

	@Override
	public boolean isEmpty() {
		return tasks.isEmpty();
	}

	@Override
	public int defaultWaitTime() {
		return 100;
	}

}
