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
	private static LongAdder numNativeFailures = new LongAdder();
	private static LongAdder numNativeLibMatrixMultCalls = new LongAdder();
	private static LongAdder nativeLibMatrixMultTime = new LongAdder();
	private static LongAdder numNativeConv2dCalls = new LongAdder();
	private static LongAdder nativeConv2dTime = new LongAdder();
	private static LongAdder numNativeConv2dBwdDataCalls = new LongAdder();
	private static LongAdder nativeConv2dBwdDataTime = new LongAdder();
	private static LongAdder numNativeConv2dBwdFilterCalls = new LongAdder();
	private static LongAdder nativeConv2dBwdFilterTime = new LongAdder();
	private static LongAdder numNativeSparseConv2dCalls = new LongAdder();
	private static LongAdder numNativeSparseConv2dBwdFilterCalls = new LongAdder();
	private static LongAdder numNativeSparseConv2dBwdDataCalls = new LongAdder();

	public static void incrementNativeFailuresCounter() {
		numNativeFailures.increment();
		// This is very rare and am not sure it is possible at all. Our initial experiments never encountered this case.
		// Note: all the native calls have a fallback to Java; so if the user wants she can recompile SystemDS by
		// commenting this exception and everything should work fine.
		throw new RuntimeException("Unexpected ERROR: OOM caused during JNI transfer. Please disable native BLAS by setting enviroment variable: SYSTEMDS_BLAS=none");
	}

	public static void incrementNumNativeLibMatrixMultCalls() {
		numNativeLibMatrixMultCalls.increment();
	}

	public static void incrementNativeLibMatrixMultTime(long time) {
		nativeLibMatrixMultTime.add(time);
	}

	public static void incrementNumNativeConv2dCalls() {
		numNativeConv2dCalls.increment();
	}

	public static void incrementNativeConv2dTime(long time) {
		nativeConv2dTime.add(time);
	}

	public static void incrementNumNativeConv2dBwdDataCalls() {
		numNativeConv2dBwdDataCalls.increment();
	}

	public static void incrementNativeConv2dBwdDataTime(long time) {
		nativeConv2dBwdDataTime.add(time);
	}

	public static void incrementNumNativeConv2dBwdFilterCalls() {
		numNativeConv2dBwdFilterCalls.increment();
	}

	public static void incrementNativeConv2dBwdFilterTime(long time) {
		nativeConv2dBwdFilterTime.add(time);
	}

	public static void incrementNumNativeSparseConv2dCalls() {
		numNativeSparseConv2dCalls.increment();
	}

	public static void incrementNumNativeSparseConv2dBwdFilterCalls() {
		numNativeSparseConv2dBwdFilterCalls.increment();
	}

	public static void incrementNumNativeSparseConv2dBwdDataCalls() {
		numNativeSparseConv2dBwdDataCalls.increment();
	}

	public static void reset() {
		numNativeLibMatrixMultCalls.reset();
		numNativeSparseConv2dCalls.reset();
		numNativeSparseConv2dBwdDataCalls.reset();
		numNativeSparseConv2dBwdFilterCalls.reset();
		numNativeConv2dCalls.reset();
		numNativeConv2dBwdDataCalls.reset();
		numNativeConv2dBwdFilterCalls.reset();
		numNativeFailures.reset();
		nativeLibMatrixMultTime.reset();
		nativeConv2dTime.reset();
		nativeConv2dBwdFilterTime.reset();
		nativeConv2dBwdDataTime.reset();
	}

	public static String displayNativeStatistics() {
		StringBuilder sb = new StringBuilder();
		String blas = NativeHelper.getCurrentBLAS();
		sb.append("Native " + blas + " calls (dense mult/conv/bwdF/bwdD):\t" + numNativeLibMatrixMultCalls.longValue()  + "/"
			+ numNativeConv2dCalls.longValue() + "/" + numNativeConv2dBwdFilterCalls.longValue()
			+ "/" + numNativeConv2dBwdDataCalls.longValue() + ".\n");
		sb.append("Native " + blas + " calls (sparse conv/bwdF/bwdD):\t"
			+ numNativeSparseConv2dCalls.longValue() + "/" + numNativeSparseConv2dBwdFilterCalls.longValue()
			+ "/" + numNativeSparseConv2dBwdDataCalls.longValue() + ".\n");
		sb.append("Native " + blas + " times (dense mult/conv/bwdF/bwdD):\t"
			+ String.format("%.3f", nativeLibMatrixMultTime.longValue()*1e-9) + "/"
			+ String.format("%.3f", nativeConv2dTime.longValue()*1e-9) + "/"
			+ String.format("%.3f", nativeConv2dBwdFilterTime.longValue()*1e-9) + "/"
			+ String.format("%.3f", nativeConv2dBwdDataTime.longValue()*1e-9) + ".\n");
		return sb.toString();
	}
}
