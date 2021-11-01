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

package org.apache.sysds.utils.stats;

import java.util.concurrent.atomic.LongAdder;

import org.apache.sysds.utils.NativeHelper;

public class NativeStatistics {
	private static LongAdder numFailures = new LongAdder();
	private static LongAdder numLibMatrixMultCalls = new LongAdder();
	private static LongAdder libMatrixMultTime = new LongAdder();
	private static LongAdder numConv2dCalls = new LongAdder();
	private static LongAdder conv2dTime = new LongAdder();
	private static LongAdder numConv2dBwdDataCalls = new LongAdder();
	private static LongAdder conv2dBwdDataTime = new LongAdder();
	private static LongAdder numConv2dBwdFilterCalls = new LongAdder();
	private static LongAdder conv2dBwdFilterTime = new LongAdder();
	private static LongAdder numSparseConv2dCalls = new LongAdder();
	private static LongAdder numSparseConv2dBwdFilterCalls = new LongAdder();
	private static LongAdder numSparseConv2dBwdDataCalls = new LongAdder();

	public static void incrementFailuresCounter() {
		numFailures.increment();
		// This is very rare and am not sure it is possible at all. Our initial experiments never encountered this case.
		// Note: all the native calls have a fallback to Java; so if the user wants she can recompile SystemDS by
		// commenting this exception and everything should work fine.
		throw new RuntimeException("Unexpected ERROR: OOM caused during JNI transfer. Please disable native BLAS by setting enviroment variable: SYSTEMDS_BLAS=none");
	}

	public static void incrementNumLibMatrixMultCalls() {
		numLibMatrixMultCalls.increment();
	}

	public static void incrementLibMatrixMultTime(long time) {
		libMatrixMultTime.add(time);
	}

	public static void incrementNumConv2dCalls() {
		numConv2dCalls.increment();
	}

	public static void incrementConv2dTime(long time) {
		conv2dTime.add(time);
	}

	public static void incrementNumConv2dBwdDataCalls() {
		numConv2dBwdDataCalls.increment();
	}

	public static void incrementConv2dBwdDataTime(long time) {
		conv2dBwdDataTime.add(time);
	}

	public static void incrementNumConv2dBwdFilterCalls() {
		numConv2dBwdFilterCalls.increment();
	}

	public static void incrementConv2dBwdFilterTime(long time) {
		conv2dBwdFilterTime.add(time);
	}

	public static void incrementNumSparseConv2dCalls() {
		numSparseConv2dCalls.increment();
	}

	public static void incrementNumSparseConv2dBwdFilterCalls() {
		numSparseConv2dBwdFilterCalls.increment();
	}

	public static void incrementNumSparseConv2dBwdDataCalls() {
		numSparseConv2dBwdDataCalls.increment();
	}

	public static void reset() {
		numLibMatrixMultCalls.reset();
		numSparseConv2dCalls.reset();
		numSparseConv2dBwdDataCalls.reset();
		numSparseConv2dBwdFilterCalls.reset();
		numConv2dCalls.reset();
		numConv2dBwdDataCalls.reset();
		numConv2dBwdFilterCalls.reset();
		numFailures.reset();
		libMatrixMultTime.reset();
		conv2dTime.reset();
		conv2dBwdFilterTime.reset();
		conv2dBwdDataTime.reset();
	}

	public static String displayStatistics() {
		StringBuilder sb = new StringBuilder();
		String blas = NativeHelper.getCurrentBLAS();
		sb.append("Native " + blas + " calls (dense mult/conv/bwdF/bwdD):\t" + numLibMatrixMultCalls.longValue()  + "/"
			+ numConv2dCalls.longValue() + "/" + numConv2dBwdFilterCalls.longValue()
			+ "/" + numConv2dBwdDataCalls.longValue() + ".\n");
		sb.append("Native " + blas + " calls (sparse conv/bwdF/bwdD):\t"
			+ numSparseConv2dCalls.longValue() + "/" + numSparseConv2dBwdFilterCalls.longValue()
			+ "/" + numSparseConv2dBwdDataCalls.longValue() + ".\n");
		sb.append("Native " + blas + " times (dense mult/conv/bwdF/bwdD):\t"
			+ String.format("%.3f", libMatrixMultTime.longValue()*1e-9) + "/"
			+ String.format("%.3f", conv2dTime.longValue()*1e-9) + "/"
			+ String.format("%.3f", conv2dBwdFilterTime.longValue()*1e-9) + "/"
			+ String.format("%.3f", conv2dBwdDataTime.longValue()*1e-9) + ".\n");
		return sb.toString();
	}
}
