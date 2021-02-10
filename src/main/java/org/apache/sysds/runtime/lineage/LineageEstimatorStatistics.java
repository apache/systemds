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

package org.apache.sysds.runtime.lineage;

import java.util.concurrent.atomic.LongAdder;

import org.apache.commons.lang3.tuple.MutableTriple;
import org.apache.sysds.utils.Statistics;

public class LineageEstimatorStatistics {
	private static final LongAdder _ctimeSaved      = new LongAdder(); //in nano sec
	private static int INSTCOUNT = 10;
	
	public static void reset() {
		_ctimeSaved.reset();
	}

	public static void incrementSavedComputeTime(long delta) {
		// Total time saved by reusing.
		// TODO: Handle overflow
		_ctimeSaved.add(delta);
	}
	
	public static String displayComputeTime() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%.3f", ((double)Statistics.getRunTime())*1e-9)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeSaved.longValue())/1000000000)); //in sec
		return sb.toString();
	}
	
	public static String displaySize() {
		//size of all cached reusable intermediates/size of reused intermediates/cache size
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%.3f", ((double)LineageEstimator._totReusableSize)/(1024*1024))); //in MB
		sb.append("/");
		sb.append(String.format("%.3f", ((double)LineageEstimator._totReusedSize)/(1024*1024))); //in MB
		sb.append("/");
		sb.append(String.format("%.3f", ((double)LineageEstimator.CACHE_LIMIT)/(1024*1024))); //in MB
		return sb.toString();
	}
	
	public static String displayReusableInsts() {
		// Total time saved and reuse counts per opcode, ordered by saved time
		StringBuilder sb = new StringBuilder();
		sb.append("# Instrunction\t" + "  "+"Time(s)  Count \n");
		int instCount = Math.min(INSTCOUNT, LineageEstimator.computeSavingInst.size());
		for (int i=1; i<=instCount; i++) {
			MutableTriple<String, Long, Double> op = LineageEstimator.computeSavingInst.poll();
			int tl = String.valueOf(op.getRight()*1e-3).indexOf(".");
			if (op != null && op.getRight() > 0)
				sb.append(String.valueOf(i) 
					+ String.format("%"+(4-String.valueOf(i).length())+"s", "") // 4-length(i) spaces
					+ op.getLeft() 
					+ String.format("%"+(15-op.getLeft().length())+"s", "") // 15 - length(opcode) spaces
					+ String.format("%.3f", op.getRight()*1e-3)
					+ String.format("%"+(8-(tl+3))+"s", "") // 8 - length(time upto '.') spaces
					+ op.getMiddle()+ "\n");
		}
		return sb.toString();
	}
	
	public static String displayLineageEstimates() {
		StringBuilder sb = new StringBuilder();
		sb.append("Compute Time (Elapsed/Saved): \t" + displayComputeTime() + " sec.\n");
		sb.append("Space Used (C/R/L): \t\t" + displaySize() + " MB.\n"); // total cached/reused/cache limit
		sb.append("Cache Full Timestamp: \t\t" + LineageEstimator.computeCacheFullTime() + "% instructions.\n");
		sb.append(displayReusableInsts());
		return sb.toString();
	}
}
