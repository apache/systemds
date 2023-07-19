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
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;

import org.apache.hadoop.io.compress.zlib.ZlibCompressor;
import org.apache.sysds.performance.Util;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.TestUtils;

public class SteamCompressTest {

	private static BlockingQueue<MatrixBlock> tasks = new ArrayBlockingQueue<>(8);

	public static void P1() throws Exception, InterruptedException {
		System.out.println("Running Steam Compression Test");
		CommonThreadPool.get(2);
		final int N = 10;
		double[] times;
		// fillTasks(N);
		// times = Util.time(() -> compressTask(), N, tasks);
		// Util.printStats(times);
		// fillTasks(N);
		// times = Util.time(() -> compressTask(), N, tasks);
		// Util.printStats(times);
		// fillTasks(N);
		// times = Util.time(() -> compressTask(), N, tasks);
		// Util.printStats(times);
		// fillTasks(N);
		// times = Util.time(() -> sumTask(), N, tasks);
		// Util.printStats(times);

		// fillTasks(N);
		// times = Util.time(() -> compressZLibTask(), N, tasks);
		// Util.printStats(times);

		fillTasks(N);

		// while(tasks.size() < 8) {
		// System.out.println("Starting");
		// Thread.sleep(100);
		// }
		times = Util.time(() -> compressZLibTask(), N, tasks);
		Util.printStats(times);
	}

	private static double sumTask() {
		try {
			return tasks.take().sum();
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed sum");
		}
	}

	private static MatrixBlock compressTask() {
		try {
			return CompressedMatrixBlockFactory.compress(tasks.take()).getLeft();
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed compress");
		}
	}

	private static Object compressZLibTask() {
		try {
			MatrixBlock mb = tasks.take();
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			mb.write(fos);
			// return null;
			byte[] data = bos.toByteArray();
			ZlibCompressor a = new ZlibCompressor();
			int l = a.compress(data, 0, data.length);
			return new Pair<Integer, byte[]>(l, data);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed compress");
		}
	}

	private static void fillTasks(int nBlocks) {
		CompletableFuture.runAsync(() -> {

			System.out.println("Generating " + nBlocks);
			for(int i = 0; i < nBlocks; i++) {
				MatrixBlock mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1000, 100, 0, 100, 0.2, i));
				try {
					tasks.put(mb);
					System.out.println("Put 1");
				}
				catch(InterruptedException e) {
					e.printStackTrace();
				}
			}
		});
	}
}
