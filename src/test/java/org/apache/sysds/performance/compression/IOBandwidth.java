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

import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.scheme.CompressionScheme;
import org.apache.sysds.runtime.compress.lib.CLALibScheme;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;

public class IOBandwidth extends APerfTest<IOBandwidth.InOut, MatrixBlock> {

	final int k;

	public IOBandwidth(int N, IGenerate<MatrixBlock> gen) {
		super(N, gen);
		k = 1;
	}

	public IOBandwidth(int N, IGenerate<MatrixBlock> gen, int k) {
		super(N, gen);
		this.k = k;
	}

	public void run() throws Exception, InterruptedException {
		System.out.println(this);
		warmup(() -> sumTask(k), N);
		execute(() -> sumTask(k), "Sum");
		execute(() -> maxTask(k), "Max");
		final MatrixBlock v = genVector();
		execute(() -> matrixVector(v, k), "MV mult");

		final CompressionScheme sch2 = CLALibScheme.getScheme(getC());
		execute(() -> updateAndApplyScheme(sch2, k), "Update&Apply Scheme");
		execute(() -> updateAndApplySchemeFused(sch2, k), "Update&Apply Scheme Fused");
		execute(() -> applyScheme(sch2, k), "Apply Scheme");
		execute(() -> fromEmptySchemeDoNotKeep(k), "Update&Apply from Empty");
		execute(() -> compressTask(k), "Normal Compression");

	}

	public void runVector() throws Exception, InterruptedException {
		System.out.println(this);
		final MatrixBlock v = genVector();
		execute(() -> matrixVector(v, k), "MV mult");
		execute(() -> sumTask(k), "Sum");
		execute(() -> maxTask(k), "Max");
	}

	private void matrixVector(MatrixBlock v, int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock r = LibMatrixMult.matrixMult(mb, v, k);
		long out = r.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void sumTask(int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock r = mb.sum(k);
		long out = r.getInMemorySize();
		ret.add(new InOut(in, out));
	}
	
	private void maxTask(int k){

		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock r = mb.max(k);
		long out = r.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void compressTask(int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = CompressedMatrixBlockFactory.compress(mb, k).getLeft();
		long out = cmb.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void applyScheme(CompressionScheme sch, int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = sch.encode(mb, k);
		long out = cmb.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void updateAndApplyScheme(CompressionScheme sch, int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		sch.update(mb, k);
		MatrixBlock cmb = sch.encode(mb, k);
		long out = cmb.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void updateAndApplySchemeFused(CompressionScheme sch, int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		MatrixBlock cmb = sch.updateAndEncode(mb, k);
		long out = cmb.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void fromEmptySchemeDoNotKeep(int k) {
		MatrixBlock mb = gen.take();
		long in = mb.getInMemorySize();
		CompressionScheme sch = CLALibScheme.genScheme(CompressionType.EMPTY, mb.getNumColumns());
		MatrixBlock cmb = sch.updateAndEncode(mb, k);
		long out = cmb.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private CompressedMatrixBlock getC() throws InterruptedException {
		gen.generate(1);
		MatrixBlock mb = gen.take();
		return (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb).getLeft();
	}

	private MatrixBlock genVector() throws InterruptedException {
		gen.generate(1);
		MatrixBlock mb = gen.take();
		MatrixBlock vector = TestUtils.generateTestMatrixBlock(mb.getNumColumns(), 1, -1.0, 1.0, 1.0, 324);
		return vector;
	}

	@Override
	protected String makeResString() {
		throw new RuntimeException("Do not call");
	}

	@Override
	protected String makeResString(double[] times) {
		double totalIn = 0;
		double totalOut = 0;
		double totalTime = 0.0;
		for(int i = 0; i < ret.size(); i++) // set times
			ret.get(i).time = times[i] / 1000; // ms to sec

		ret.sort(IOBandwidth::compare);

		final int l = ret.size();
		final int remove = (int) Math.floor((double) l * 0.05);

		final int el = l - remove * 2;

		for(int i = remove; i < ret.size() - remove; i++) {
			InOut e = ret.get(i);
			totalIn += e.in;
			totalOut += e.out;
			totalTime += e.time;
		}

		double bytePerMsIn = totalIn / totalTime;
		double bytePerMsOut = totalOut / totalTime;
		// double meanTime = totalTime / el;

		double varIn = 0;
		double varOut = 0;
		// double varTime = 0;

		for(int i = remove; i < ret.size() - remove; i++) {
			InOut e = ret.get(i);
			varIn += Math.pow(e.in / e.time - bytePerMsIn, 2);
			varOut += Math.pow(e.out / e.time - bytePerMsOut, 2);
		}

		double stdIn = Math.sqrt(varIn / el);
		double stdOut = Math.sqrt(varOut / el);

		return String.format("%12.0f+-%12.0f Byte/s, %12.0f+-%12.0f Byte/s", bytePerMsIn, stdIn, bytePerMsOut, stdOut);
	}

	public static int compare(InOut a, InOut b) {
		if(a.time == b.time)
			return 0;
		else if(a.time < b.time)
			return -1;
		else
			return 1;
	}

	@Override
	public String toString() {
		return super.toString() + " threads: " + k;
	}

	protected class InOut {
		protected long in;
		protected long out;
		protected double time;

		protected InOut(long in, long out) {
			this.in = in;
			this.out = out;
		}

	}

}
