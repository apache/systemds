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

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

import org.apache.sysds.performance.Util;
import org.apache.sysds.performance.Util.F;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.TestUtils;

public class SteamCompressTest {

	private static BlockingQueue<MatrixBlock> tasks = new ArrayBlockingQueue<>(8);
	private static ArrayList<Double> ret;

	public static void P1() throws Exception, InterruptedException {
		System.out.println("Running Steam Compression Test");
		CommonThreadPool.get(2);

		execute(() -> sumTask(), "Sum Task -- Warmup");
		execute(() -> blockSizeTask(), "In Memory Block Size");
		execute(() -> writeSteam(), "Write Blocks Stream");
		execute(() -> writeSteamDeflaterOutputStreamDef(), "Write Stream Deflate");
		execute(() -> writeSteamDeflaterOutputStreamSpeed(), "Write Stream Deflate Speedy");
		execute(() -> compressTask(), "In Memory Compress Individual (CI)");
		execute(() -> writeStreamCompressTask(), "Write CI Stream");
		execute(() -> writeStreamCompressDeflaterOutputStreamTask(), "Write CI Deflate Stream");
		execute(() -> writeStreamCompressDeflaterOutputStreamTaskSpeedy(), "Write CI Deflate Stream Speedy");

	}

	private static void execute(F f, String name) throws InterruptedException {
		final int N = 100;
		fillTasks(N);
		if(ret == null)
			ret = new ArrayList<Double>();
		else
			ret.clear();
		double[] times = Util.time(f, N, tasks);
		Double avgRes = ret.stream().mapToDouble(a -> a).average().getAsDouble();
		System.out.println(String.format("%35s, %50s, %10.2f", name, Util.stats(times), avgRes));

	}

	private static void sumTask() {
		try {
			ret.add(tasks.take().sum());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed sum");
		}
	}

	private static void blockSizeTask() {
		try {
			ret.add((double)tasks.take().getInMemorySize());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed sum");
		}
	}

	private static void compressTask() {
		try {
			MatrixBlock mb = CompressedMatrixBlockFactory.compress(tasks.take()).getLeft();
			ret.add((double) mb.getInMemorySize());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed compress");
		}
	}

	private static void writeStreamCompressTask() {
		try {
			MatrixBlock mb = CompressedMatrixBlockFactory.compress(tasks.take()).getLeft();
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			mb.write(fos);
			ret.add((double) bos.size());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed compress");
		}
	}

	private static void writeStreamCompressDeflaterOutputStreamTask() {
		try {
			MatrixBlock mb = CompressedMatrixBlockFactory.compress(tasks.take()).getLeft();
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DeflaterOutputStream decorator = new DeflaterOutputStream(bos);
			DataOutputStream fos = new DataOutputStream(decorator);
			mb.write(fos);
			ret.add((double) bos.size());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed compress");
		}
	}

	private static void writeStreamCompressDeflaterOutputStreamTaskSpeedy() {
		try {
			MatrixBlock mb = CompressedMatrixBlockFactory.compress(tasks.take()).getLeft();
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DeflaterOutputStream decorator = new DeflaterOutputStream(bos, new Deflater(Deflater.BEST_SPEED));
			DataOutputStream fos = new DataOutputStream(decorator);
			mb.write(fos);
			ret.add((double) bos.size());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed compress");
		}
	}

	private static void writeSteam() {
		try {
			MatrixBlock mb = tasks.take();
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			mb.write(fos);
			ret.add((double) bos.size());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("failed Write Stream");
		}
	}

	private static void writeSteamDeflaterOutputStreamDef() {
		try {
			MatrixBlock mb = tasks.take();
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DeflaterOutputStream decorator = new DeflaterOutputStream(bos);
			DataOutputStream fos = new DataOutputStream(decorator);
			mb.write(fos);
			ret.add((double) bos.size());

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed compress");
		}
	}

	private static void writeSteamDeflaterOutputStreamSpeed() {
		try {
			MatrixBlock mb = tasks.take();
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DeflaterOutputStream decorator = new DeflaterOutputStream(bos, new Deflater(Deflater.BEST_SPEED));
			DataOutputStream fos = new DataOutputStream(decorator);
			mb.write(fos);
			ret.add((double) bos.size());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed compress");
		}
	}

	private static void fillTasks(int nBlocks) {
		CompletableFuture.runAsync(() -> {

			for(int i = 0; i < nBlocks; i++) {
				MatrixBlock mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1000, 100, 0, 32, 0.2, i));
				try {
					tasks.put(mb);
				}
				catch(InterruptedException e) {
					e.printStackTrace();
				}
			}
		});
	}
}
