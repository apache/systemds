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

import org.apache.sysds.utils.Statistics;

public class ParForStatistics {
	//PARFOR optimization stats (low frequency updates)
	private static final LongAdder optTime = new LongAdder(); //in milli sec
	private static final LongAdder optCount = new LongAdder(); //count
	private static final LongAdder initTime = new LongAdder(); //in milli sec
	private static final LongAdder mergeTime = new LongAdder(); //in milli sec

	public static synchronized void incrementOptimCount(){
		optCount.increment();
	}

	public static synchronized void incrementOptimTime( long time ) {
		optTime.add(time);
	}

	public static synchronized void incrementInitTime( long time ) {
		initTime.add(time);
	}

	public static synchronized void incrementMergeTime( long time ) {
		mergeTime.add(time);
	}

	public static long getOptCount(){
		return optCount.longValue();
	}

	public static long getOptTime(){
		return optTime.longValue();
	}

	public static long getInitTime(){
		return initTime.longValue();
	}

	public static long getMergeTime(){
		return mergeTime.longValue();
	}

	public static void reset() {
		optCount.reset();
		optTime.reset();
		initTime.reset();
		mergeTime.reset();
	}

	public static String displayStatistics() {
		if( optCount.longValue() > 0 ){
			StringBuilder sb = new StringBuilder();
			sb.append("ParFor loops optimized:\t\t" + getOptCount() + ".\n");
			sb.append("ParFor optimize time:\t\t" + String.format("%.3f", ((double)getOptTime())/1000) + " sec.\n");
			sb.append("ParFor initialize time:\t\t" + String.format("%.3f", ((double)getInitTime())/1000) + " sec.\n");
			sb.append("ParFor result merge time:\t" + String.format("%.3f", ((double)getMergeTime())/1000) + " sec.\n");
			sb.append("ParFor total update in-place:\t" + Statistics.getTotalUIPVar() + "/"
				+ Statistics.getTotalLixUIP() + "/" + Statistics.getTotalLix() + "\n");
			return sb.toString();
		}
		return "";
	}
}
