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

import java.util.Random;

import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.data.SparseBlockMCSR;

public class SparseAppend {

	public SparseAppend(String[] args) {
		// Ignore args 1 it is containing the Id to run this test.

		final int rep = Integer.parseInt(args[1]);
		final int size = Integer.parseInt(args[2]);
		final Random r = new Random(42);

		System.out.println("Appending rep: " + rep + " of " + size + " distinct append calls (including random and allocations)");
		double[] times;
		// Warmup 
		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, r.nextDouble());
			}
		}, rep);

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, r.nextDouble());
			}
		}, rep);

		System.out.println("Append all dense:          " + TimingUtils.stats(times));

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, 0);
			}
		}, rep);

		System.out.println("Append all zero on empty:  " + TimingUtils.stats(times));

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < 100; i ++){
				sb.append(i, i, 1);
			}
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, 0);
			}
		}, rep);

		System.out.println("Append all zero on Scalar: " + TimingUtils.stats(times));

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < 100; i ++){
				sb.append(i, i, 1);
				sb.append(i, i, 1);
			}
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, 0);
			}
		}, rep);

		System.out.println("Append all zero on Array:  " + TimingUtils.stats(times));

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < 100; i ++){
				sb.append(i, i, 1);
				sb.append(i, i, 1);
			}
			for(int i = 0; i < size; i++) {
				double d = r.nextDouble();
				d = d > 0.5 ? d : 0;
				sb.append(i % 100, i, d);
			}
		}, rep);

		System.out.println("Append half zero on Array: " + TimingUtils.stats(times));
	}

	public static void main(String[] args) {
		new SparseAppend(new String[] {"1004", "10000", "10000"});
	}
}
