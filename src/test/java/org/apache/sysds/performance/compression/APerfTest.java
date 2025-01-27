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

package org.apache.sysds.performance.compression;

import java.util.ArrayList;

import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.performance.TimingUtils.F;
import org.apache.sysds.performance.generators.IGenerate;

public abstract class APerfTest<T, G> {

	/** The Result array that all the results of the individual executions is producing */
	protected final ArrayList<T> ret;

	/** A Task que that guarantee that the execution is not to long */
	protected final IGenerate<G> gen;

	/** Default Repetitions */
	protected final int N;

	/** Warmup iterations */
	protected final int W;

	protected APerfTest(int N, IGenerate<G> gen) {
		ret = new ArrayList<>(N);
		this.gen = gen;
		this.N = N;
		this.W = 10;
	}


	protected APerfTest(int N, int W, IGenerate<G> gen) {
		ret = new ArrayList<>(N);
		this.gen = gen;
		this.N = N;
		this.W = 10;
	}

	protected void execute(F f, String name) throws InterruptedException {
		N n = new N();
		execute(f, n, n, name);
	}

	protected void execute(F f, F c, String name) throws InterruptedException {
		N n = new N();
		execute(f, c, n, name);
	}

	protected void execute(F f, F c, F b, String name) throws InterruptedException {
		warmup(f, W);
		gen.generate(N);
		ret.clear();
		double[] times = TimingUtils.time(f, c, b, N, gen);
		String retS = makeResString(times);
		System.out.println(String.format("%35s, %s, %10s", name, TimingUtils.stats(times), retS));
	}

	protected void warmup(F f, int n) throws InterruptedException {
		gen.generate(N);
		ret.clear();
	}

	protected void execute(F f, String name, int N) throws InterruptedException {
		N none = new N();
		execute(f, none, none, name, N);
	}

	protected void execute(F f, F c, String name, int N) throws InterruptedException {
		execute(f, c, new N(), name, N);
	}

	protected void execute(F f, F c, F b, String name, int N) throws InterruptedException {
		gen.generate(N);
		ret.clear();
		double[] times = TimingUtils.time(f, c, b, N, gen);
		String retS = makeResString(times);
		System.out.println(String.format("%35s, %s, %10s", name, TimingUtils.stats(times), retS));
	}

	protected abstract String makeResString();

	protected String makeResString(double[] times) {
		return makeResString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%20s ", this.getClass().getSimpleName()));
		sb.append(" Repetitions: ").append(N).append("\n");
		sb.append(String.format("%20s ","Generator:"));
		sb.append(gen);
		sb.append("\n");
		return sb.toString();
	}

	private class N implements F {

		@Override
		public void run() {
			// co nothing
		}

	}
}
