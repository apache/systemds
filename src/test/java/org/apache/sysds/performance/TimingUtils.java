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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.utils.stats.Timing;

/**
 * Util methods for the Performance suite
 */
public interface TimingUtils {

	/** A specification enum for the type of statistics to gather from the time measurements */
	public enum StatsType {
		MEAN_STD, MEAN_STD_Q1;
	}

	/** The specified measurement to use in this case. Can be set from any of the programs */
	// public static StatsType st = StatsType.MEAN_STD;

	/**
	 * Time the function given assuming that it should put result into the given time array at index i.
	 * 
	 * @param f   The function to time
	 * @param rep the number of repetitions
	 */
	public static double[] time(F f, int rep) {
		double[] times = new double[rep];
		for(int i = 0; i < rep; i++)
			time(f, times, i);

		return times;
	}

	/**
	 * Time the function given assuming that it should put result into the given time array at index i.
	 * 
	 * @param f     The function to time
	 * @param times The time array to put the time result into
	 * @param i     The index to put it into
	 */
	public static void time(F f, double[] times, int i) {
		Timing time = new Timing(true);
		f.run();
		times[i] = time.stop();
	}

	/**
	 * Time the given function a number of time using the generator to populate the input allocations without including
	 * it in the timing of the operation
	 * 
	 * @param f   The function to time
	 * @param c   A cleanup function or part that should not be timed.
	 * @param b   A setup function that should not be timed.
	 * @param rep The number of repetitions to make
	 * @param bq  The generator for the input
	 * @return A list of the individual repetitions execution time
	 * @throws InterruptedException An exception in case the job gets interrupted
	 */
	public static double[] time(F f, F c, F b, int rep, IGenerate<?> bq) throws InterruptedException {
		double[] times = new double[rep];
		for(int i = 0; i < rep; i++) {
			b.run();
			while(bq.isEmpty())
				Thread.sleep(bq.defaultWaitTime());
			DMLScript.SEED = i + 1000;
			time(f, times, i);
			c.run();
		}
		return times;
	}

	/**
	 * Calculate the statistics of the times executed
	 * <p>
	 * The default is to calculate the mean and standard deviation and return that as a string
	 * 
	 * @param v  The times observed
	 * @param st The type of stats to print
	 * @return The status string.
	 */
	public static String stats(double[] v) {
		return statsMeanSTD(v);
	}

	/**
	 * Calculate the statistics of the times executed, given the stats type provided
	 * <p>
	 * The default is to calculate the mean and standard deviation and return that as a string
	 * 
	 * @param v  The times observed
	 * @param st The type of stats to print
	 * @return The status string.
	 */
	public static String stats(double[] v, StatsType st) {
		switch(st) {
			case MEAN_STD_Q1:
				return statsMeanSTDQ1(v);
			case MEAN_STD:
			default:
				return statsMeanSTD(v);
		}
	}

	private static String statsMeanSTD(double[] v) {
		final int l = v.length;
		final int remove = (int) Math.floor(l * 0.05);
		Arrays.sort(v);

		double total = 0;
		final int el = v.length - remove * 2;
		for(int i = remove; i < l - remove; i++)
			total += v[i];

		double mean = total / el;

		double var = 0;
		for(int i = remove; i < l - remove; i++)
			var += Math.pow(Math.abs(v[i] - mean), 2);

		double std = Math.sqrt(var / el);

		return String.format("%8.3f+-%7.3f ms", mean, std);
	}

	private static String statsMeanSTDQ1(double[] v) {
		final int l = v.length;
		final int remove = (int) Math.floor(l * 0.05);
		Arrays.sort(v);

		double q1 = v[v.length - 1 - (int) (Math.floor((double) v.length / 100))];
		double q2p5 = v[v.length - 1 - (int) (Math.floor((double) v.length / 40))];
		double q5 = v[v.length - 1 - (int) (Math.floor((double) v.length / 20))];
		double q10 = v[v.length - 1 - (int) (Math.floor((double) v.length / 10))];

		double total = 0;
		final int el = v.length - remove * 2;
		for(int i = remove; i < l - remove; i++)
			total += v[i];

		double mean = total / el;

		double var = 0;
		for(int i = remove; i < l - remove; i++)
			var += Math.pow(Math.abs(v[i] - mean), 2);

		double std = Math.sqrt(var / el);

		return String.format("%8.3f+-%7.3f ms [q1:%7.3f, q2.5:%7.3f, q5:%7.3f, q10:%7.3f]", mean, std, q1, q2p5, q5, q10);
	}

	/**
	 * Interface method to enable timed calling from other Classes
	 */
	interface F {
		void run();
	}
}
