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

package org.apache.sysds.performance.frame;

import java.util.Arrays;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.performance.compression.APerfTest;
import org.apache.sysds.performance.generators.ConstFrame;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.lib.FrameLibCompress;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class Transform extends APerfTest<Object, FrameBlock> {

	private final int k;
	private final String spec;

	public Transform(int N, IGenerate<FrameBlock> gen, int k, String spec) {
		super(N,2, gen);
		this.k = k;
		this.spec = spec;
		FrameBlock in = gen.take();
		System.out
			.println("Transform Encode Perf: rows: " + in.getNumRows() + " schema:" + Arrays.toString(in.getSchema()));
		System.out.println(spec);
	}

	public void run() throws Exception {
		execute(() -> te(), () -> clear(), "Normal");
		execute(() -> tec(), () -> clear(), "Compressed");
	}

	private void te() {
		FrameBlock in = gen.take();
		MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, in.getNumColumns());
		enc.encode(in, k);
		ret.add(null);
	}

	private void tec() {
		FrameBlock in = gen.take();
		MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, in.getNumColumns());
		enc.encode(in, k, true);
		ret.add(null);
	}

	private void clear() {
		clearRDCCache(gen.take());
	}

	@Override
	protected String makeResString() {
		return "";
	}

	/**
	 * Forcefully clear recode cache of underlying arrays
	 */
	public void clearRDCCache(FrameBlock f) {
		for(Array<?> a : f.getColumns())
			a.setCache(null);
	}

	public static void main(String[] args) throws Exception {
		int k = InfrastructureAnalyzer.getLocalParallelism();
		FrameBlock in;

		// for(int i = 1; i < 1000; i *= 10) {
			int rows = 100000 * 100;
			in = TestUtils.generateRandomFrameBlock(rows, new ValueType[] {ValueType.UINT4}, 32);

			System.out.println("Without null");
			run(k, in);

			System.out.println("Compressed without null");
			in = FrameLibCompress.compress(in, k);
			run(k, in);

			in = TestUtils.generateRandomFrameBlock(rows, new ValueType[] {ValueType.UINT4}, 32, 0.5);

			System.out.println("With null");

			run(k, in);
			System.out.println("Compressed with null");
			in = FrameLibCompress.compress(in, k);
			run(k, in);

			in = TestUtils.generateRandomFrameBlock(
				rows, new ValueType[] {ValueType.UINT4, ValueType.UINT4, ValueType.UINT4, ValueType.UINT4,
					ValueType.UINT4, ValueType.UINT4, ValueType.UINT4, ValueType.UINT4, ValueType.UINT4, ValueType.UINT4},
				32);

			System.out.println("10 col without null");
			run10(k, in);
			System.out.println("10 col compressed without null");
			in = FrameLibCompress.compress(in, k);
			run10(k, in);

			in = TestUtils.generateRandomFrameBlock(
				rows, new ValueType[] {ValueType.UINT4, ValueType.UINT4, ValueType.UINT4, ValueType.UINT4,
					ValueType.UINT4, ValueType.UINT4, ValueType.UINT4, ValueType.UINT4, ValueType.UINT4, ValueType.UINT4},
				32, 0.5);

			System.out.println("10 col with null");
			run10(k, in);
			System.out.println("10 col Compressed with null");
			in = FrameLibCompress.compress(in, k);
			run10(k, in);
		// }

		System.exit(0); // forcefully stop.
	}

	private static void run10(int k, FrameBlock in) throws Exception {
		ConstFrame gen = new ConstFrame(in);
		new Transform(20, gen, k, "{}").run();
		new Transform(10, gen, k, "{ids:true, recode:[1,2,3,4,5,6,7,8,9,10]}").run();
		new Transform(10, gen, k, "{ids:true, bin:[" //
			+ "\n{id:1, method:equi-width, numbins:4}," //
			+ "\n{id:2, method:equi-width, numbins:4}," //
			+ "\n{id:3, method:equi-width, numbins:4}," //
			+ "\n{id:4, method:equi-width, numbins:4}," //
			+ "\n{id:5, method:equi-width, numbins:4}," //
			+ "\n{id:6, method:equi-width, numbins:4}," //
			+ "\n{id:7, method:equi-width, numbins:4}," //
			+ "\n{id:8, method:equi-width, numbins:4}," //
			+ "\n{id:9, method:equi-width, numbins:4}," //
			+ "\n{id:10, method:equi-width, numbins:4}," //
			+ "]}").run();
		new Transform(10, gen, k, "{ids:true, bin:[" //
			+ "\n{id:1, method:equi-width, numbins:4}," //
			+ "\n{id:2, method:equi-width, numbins:4}," //
			+ "\n{id:3, method:equi-width, numbins:4}," //
			+ "\n{id:4, method:equi-width, numbins:4}," //
			+ "\n{id:5, method:equi-width, numbins:4}," //
			+ "\n{id:6, method:equi-width, numbins:4}," //
			+ "\n{id:7, method:equi-width, numbins:4}," //
			+ "\n{id:8, method:equi-width, numbins:4}," //
			+ "\n{id:9, method:equi-width, numbins:4}," //
			+ "\n{id:10, method:equi-width, numbins:4}," //
			+ "],  dummycode:[1,2,3,4,5,6,7,8,9,10]}").run();
		new Transform(10, gen, k, "{ids:true, hash:[1,2,3,4,5,6,7,8,9,10], K:10}").run();
		new Transform(10, gen, k, "{ids:true, hash:[1,2,3,4,5,6,7,8,9,10], K:10, dummycode:[1,2,3,4,5,6,7,8,9,10]}")
			.run();
	}

	private static void run(int k, FrameBlock in) throws Exception {
		ConstFrame gen = new ConstFrame(in);
		// // passthrough
		new Transform(10, gen, k, "{}").run();
		new Transform(10, gen, k, "{ids:true, recode:[1]}").run();
		new Transform(10, gen, k, "{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}").run();
		new Transform(10, gen, k, "{ids:true, bin:[{id:1, method:equi-width, numbins:4}], dummycode:[1]}").run();
		new Transform(10, gen, k, "{ids:true, hash:[1], K:10}").run();
		new Transform(10, gen, k, "{ids:true, hash:[1], K:10, dummycode:[1]}").run();
	}

}
