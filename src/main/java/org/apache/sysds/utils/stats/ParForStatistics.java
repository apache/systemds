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
	private static final LongAdder parforOptTime = new LongAdder(); //in milli sec
	private static final LongAdder parforOptCount = new LongAdder(); //count
	private static final LongAdder parforInitTime = new LongAdder(); //in milli sec
	private static final LongAdder parforMergeTime = new LongAdder(); //in milli sec

	public static synchronized void incrementParForOptimCount(){
		parforOptCount.increment();
	}

	public static synchronized void incrementParForOptimTime( long time ) {
		parforOptTime.add(time);
	}

	public static synchronized void incrementParForInitTime( long time ) {
		parforInitTime.add(time);
	}

	public static synchronized void incrementParForMergeTime( long time ) {
		parforMergeTime.add(time);
	}

	public static long getParforOptCount(){
		return parforOptCount.longValue();
	}

	public static long getParforOptTime(){
		return parforOptTime.longValue();
	}

	public static long getParforInitTime(){
		return parforInitTime.longValue();
	}

	public static long getParforMergeTime(){
		return parforMergeTime.longValue();
	}

	public static void reset() {
		parforOptCount.reset();
		parforOptTime.reset();
		parforInitTime.reset();
		parforMergeTime.reset();
	}

	public static String displayParForStatistics() {
		if( parforOptCount.longValue() > 0 ){
			StringBuilder sb = new StringBuilder();
			sb.append("ParFor loops optimized:\t\t" + getParforOptCount() + ".\n");
			sb.append("ParFor optimize time:\t\t" + String.format("%.3f", ((double)getParforOptTime())/1000) + " sec.\n");
			sb.append("ParFor initialize time:\t\t" + String.format("%.3f", ((double)getParforInitTime())/1000) + " sec.\n");
			sb.append("ParFor result merge time:\t" + String.format("%.3f", ((double)getParforMergeTime())/1000) + " sec.\n");
			sb.append("ParFor total update in-place:\t" + Statistics.getTotalUIPVar() + "/"
				+ Statistics.getTotalLixUIP() + "/" + Statistics.getTotalLix() + "\n");
			return sb.toString();
		}
		return "";
	}
}
