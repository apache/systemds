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

package org.apache.sysds.performance;

import java.util.Arrays;
import java.util.concurrent.BlockingQueue;

import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;

public interface Util {
	public static double time(F f) {
		Timing time = new Timing(true);
		f.run();
		return time.stop();
	}

	public static void time(F f, String p) {
		Timing time = new Timing(true);
		f.run();
		System.out.print(p);
		System.out.println(time.stop());
	}

	public static void time(F f, double[] times, int i) {
		Timing time = new Timing(true);
		f.run();
		times[i] = time.stop();
	}

	public static double[] time(F f, int rep, BlockingQueue<?> bq) throws InterruptedException {
		double[] times = new double[rep];
		for(int i = 0; i < rep; i++) {
			while(bq.isEmpty())
				Thread.sleep(100);
			Util.time(f, times, i);
		}
		return times;
	}

	public static String stats(double[] v) {

		Arrays.sort(v);
		final int l = v.length;

		double min = v[0];
		double max = v[l - 1];
		double q25 = v[(int) (l / 4)];
		double q50 = v[(int) (l / 2)];
		double q75 = v[(int) ((l / 4) * 3)];

		return String.format("[%.3f, %.3f, %.3f, %.3f, %.3f]", min, q25, q50, q75, max);
	}

	interface F {
		void run();
	}
}
