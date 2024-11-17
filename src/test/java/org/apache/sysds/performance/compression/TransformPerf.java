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

import org.apache.sysds.performance.PerfUtil;
import org.apache.sysds.performance.compression.Serialize.InOut;
import org.apache.sysds.performance.generators.ConstFrame;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibApplySchema;
import org.apache.sysds.runtime.frame.data.lib.FrameLibDetectSchema;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;

public class TransformPerf extends APerfTest<Serialize.InOut, FrameBlock> {

	private final String file;
	private final String spec;
	private final String specPath;
	private final int k;

	public TransformPerf(int n, int k, IGenerate<FrameBlock> gen, String specPath) throws Exception {
		super(n, gen);
		this.file = "tmp/perf-tmp.bin";
		this.k = k;
		this.spec = PerfUtil.readSpec(specPath);
		this.specPath = specPath;
	}

	public void run() throws Exception {
		System.out.println(this);
		CompressedMatrixBlock.debug = true;

		System.out.println(String.format("Unknown mem size: %30d", gen.take().getInMemorySize()));

		execute(() -> detectSchema(k), "Detect Schema");
		execute(() -> detectAndApply(k), "Detect&Apply Frame Schema");
		execute(() -> transformEncode(k), "TransformEncode Def");
		execute(() -> transformEncodeCompressed(k), "TransformEncode Comp");

		updateGen();

		System.out.println(String.format("Known mem size:   %30d", gen.take().getInMemorySize()));
		System.out.println(gen.take().slice(0, 10));
		execute(() -> transformEncode(k), "TransformEncode Def");
		execute(() -> transformEncodeCompressed(k), "TransformEncode Comp");

	}

	private void updateGen() {
		if(gen instanceof ConstFrame) {
			FrameBlock fb = gen.take();
			FrameBlock r = FrameLibDetectSchema.detectSchema(fb, k);
			FrameBlock out = FrameLibApplySchema.applySchema(fb, r, k);
			((ConstFrame) gen).change(out);
		}
	}

	private void detectSchema(int k) {
		FrameBlock fb = gen.take();
		long in = fb.getInMemorySize();
		FrameBlock r = FrameLibDetectSchema.detectSchema(fb, k);
		long out = r.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void detectAndApply(int k) {
		FrameBlock fb = gen.take();
		long in = fb.getInMemorySize();
		FrameBlock r = FrameLibDetectSchema.detectSchema(fb, k);
		FrameBlock out = FrameLibApplySchema.applySchema(fb, r, k);
		long outS = out.getInMemorySize();
		ret.add(new InOut(in, outS));
	}

	private void transformEncode(int k) {
		FrameBlock fb = gen.take();
		long in = fb.getInMemorySize();
		MultiColumnEncoder e = EncoderFactory.createEncoder(spec, fb.getNumColumns());
		MatrixBlock r = e.encode(fb, k);
		long out = r.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	private void transformEncodeCompressed(int k) {
		FrameBlock fb = gen.take();
		long in = fb.getInMemorySize();
		MultiColumnEncoder e = EncoderFactory.createEncoder(spec, fb.getNumColumns());
		MatrixBlock r = e.encode(fb, k, true);
		long out = r.getInMemorySize();
		ret.add(new InOut(in, out));
	}

	@Override
	protected String makeResString() {
		throw new RuntimeException("Do not call");
	}

	@Override
	protected String makeResString(double[] times) {
		return Serialize.makeResString(ret, times);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(" File: ");
		sb.append(file);
		sb.append(" Spec: ");
		sb.append(specPath);
		return sb.toString();
	}
}
