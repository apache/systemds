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

package org.apache.sysds.hops.fedplanner;

import org.apache.sysds.utils.Statistics;

import java.util.ArrayList;
import java.util.List;

public class FederatedCompilationTimer {
	private static final List<TimeEntry> times = new ArrayList<>();
	private static TimeEntry privProcessTime;
	private static TimeEntry enumerationTime;
	private static TimeEntry selectPlanTime;
	private static boolean activated = false;

	public static class TimeEntry {
		private final long startTime;
		private long stopTime;
		private long duration;
		private String name;

		public TimeEntry(String name){
			this.name = name;
			this.startTime = System.nanoTime();
		}

		public void stopTime(){
			this.stopTime = System.nanoTime();
			this.duration = stopTime-startTime;
		}

		public boolean is(String searchName){
			return name.contains(searchName);
		}

		public long getDuration(){
			return duration;
		}
	}

	public static TimeEntry startPrivProcessTimer(){
		privProcessTime = new TimeEntry("PrivProcess");
		times.add(privProcessTime);
		return privProcessTime;
	}

	public static TimeEntry stopPrivProcessTimer(){
		privProcessTime.stopTime();
		return privProcessTime;
	}

	public static TimeEntry startPrivFetchTimer(long hopID){
		TimeEntry privFetchTimer = new TimeEntry("PrivFetch"+hopID);
		times.add(privFetchTimer);
		return privFetchTimer;
	}

	public static void startEnumerationTimer(){
		enumerationTime = new TimeEntry("Enumeration");
		times.add(enumerationTime);
	}

	public static void stopEnumerationTimer(){
		enumerationTime.stopTime();
	}

	public static void startSelectPlanTimer(){
		selectPlanTime = new TimeEntry("Selection");
		times.add(selectPlanTime);
	}

	public static void stopSelectPlanTimer(){
		selectPlanTime.stopTime();
	}

	private static long getTotalFetchTime(){
		return times.stream().filter(t -> t.is("PrivFetch")).map(TimeEntry::getDuration)
			.reduce(0L, Long::sum);
	}

	private static long getBasicCompileTime(){
		return Statistics.getCompileTime() - privProcessTime.getDuration()
			- enumerationTime.getDuration() - selectPlanTime.getDuration();
	}

	private static String nanoToSeconds(long nanoSeconds){
		return (String.format("%.3f", nanoSeconds*1e-9) + " sec.");
	}

	public static String getStringRepresentation(){
		if (activated && timesNotNull()){
			long totalFetchTime = getTotalFetchTime();
			long privPropagationTime = privProcessTime.getDuration()-totalFetchTime;
			long basicCompileTime = getBasicCompileTime();
			StringBuilder sb = new StringBuilder();
			sb.append("Basic Compilation Time:\t\t").append(nanoToSeconds(basicCompileTime)).append("\n");
			sb.append("Total Privacy Fetch Time:\t").append(nanoToSeconds(totalFetchTime)).append("\n");
			sb.append("Privacy Propagation Time:\t").append(nanoToSeconds(privPropagationTime)).append("\n");
			sb.append("Plan Enumeration Time:\t\t").append(nanoToSeconds(enumerationTime.getDuration())).append("\n");
			sb.append("Plan Selection Time:\t\t").append(nanoToSeconds(selectPlanTime.getDuration())).append("\n");
			return sb.toString();
		}
		else return "";
	}

	private static boolean timesNotNull(){
		return privProcessTime != null && enumerationTime != null && selectPlanTime != null;
	}

	public static void activate(){
		activated = true;
	}

	public static void display(){
		System.out.println(getStringRepresentation());
	}
}
