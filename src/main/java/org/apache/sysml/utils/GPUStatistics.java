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

package org.apache.sysml.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

import org.apache.sysml.api.DMLScript;

/**
 * Measures performance numbers when GPU mode is enabled
 * Printed as part of {@link Statistics}.
 */
public class GPUStatistics {
	// Whether or not extra per-instruction statistics will be recorded and shown for the GPU
	public static boolean DISPLAY_STATISTICS = false;

	private static int iNoOfExecutedGPUInst = 0;

	public static long cudaInitTime = 0;
	public static long cudaLibrariesInitTime = 0;
	public static LongAdder cudaSparseToDenseTime = new LongAdder();		// time spent in converting sparse matrix block to dense
	public static LongAdder cudaDenseToSparseTime = new LongAdder();		// time spent in converting dense matrix block to sparse
	public static LongAdder cudaSparseConversionTime = new LongAdder();	// time spent in converting between sparse block types
	public static LongAdder cudaSparseToDenseCount = new LongAdder();
	public static LongAdder cudaDenseToSparseCount = new LongAdder();
	public static LongAdder cudaSparseConversionCount = new LongAdder();

	public static LongAdder cudaAllocTime = new LongAdder();             // time spent in allocating memory on the GPU
	public static LongAdder cudaDeAllocTime = new LongAdder();           // time spent in deallocating memory on the GPU
	public static LongAdder cudaMemSet0Time = new LongAdder();           // time spent in setting memory to 0 on the GPU (part of reusing and for new allocates)
	public static LongAdder cudaToDevTime = new LongAdder();             // time spent in copying data from host (CPU) to device (GPU) memory
	public static LongAdder cudaFromDevTime = new LongAdder();           // time spent in copying data from device to host
	public static LongAdder cudaAllocCount = new LongAdder();
	public static LongAdder cudaDeAllocCount = new LongAdder();
	public static LongAdder cudaMemSet0Count = new LongAdder();
	public static LongAdder cudaToDevCount = new LongAdder();
	public static LongAdder cudaFromDevCount = new LongAdder();
	public static LongAdder cudaEvictionCount = new LongAdder();

	// Per instruction miscellaneous timers.
	// Used to record events in a CP Heavy Hitter instruction and
	// provide a breakdown of how time was spent in that instruction
	private static HashMap<String, HashMap<String, Long>> _cpInstMiscTime = new HashMap<> ();
	private static HashMap<String, HashMap<String, Long>> _cpInstMiscCount = new HashMap<> ();

	/**
	 * Resets the miscellaneous timers & counters
	 */
	public static void resetMiscTimers(){
		_cpInstMiscTime.clear();
		_cpInstMiscCount.clear();
	}

	/**
	 * Resets all the cuda counters and timers, including the misc timers & counters
	 */
	public static void reset(){
		cudaInitTime = 0;
		cudaLibrariesInitTime = 0;
		cudaAllocTime.reset();
		cudaDeAllocTime.reset();
		cudaMemSet0Time.reset();
		cudaMemSet0Count.reset();
		cudaToDevTime.reset();
		cudaFromDevTime.reset();
		cudaAllocCount.reset();
		cudaDeAllocCount.reset();
		cudaToDevCount.reset();
		cudaFromDevCount.reset();
		cudaEvictionCount.reset();
		resetMiscTimers();
	}


	public static synchronized void setNoOfExecutedGPUInst(int numJobs) {
		iNoOfExecutedGPUInst = numJobs;
	}

	public static synchronized void incrementNoOfExecutedGPUInst() {
		iNoOfExecutedGPUInst ++;
	}

	public static synchronized int getNoOfExecutedGPUInst() {
		return iNoOfExecutedGPUInst;
	}

	/**
	 * "Maintains" or adds time to miscellaneous timers per instruction/op, also increments associated count
	 * @param instructionName	name of the instruction/op
	 * @param miscTimer				name of the miscellaneous timer
	 * @param timeNanos				time in nano seconds
	 * @param incrementCount	how much to increment the count of the miscTimer by
	 */
	public synchronized static void maintainCPMiscTimes( String instructionName, String miscTimer, long timeNanos, long incrementCount)
	{
		if (!(DISPLAY_STATISTICS || DMLScript.FINEGRAINED_STATISTICS))
			return;

		HashMap<String, Long> miscTimesMap = _cpInstMiscTime.get(instructionName);
		if (miscTimesMap == null) {
			miscTimesMap = new HashMap<>();
			_cpInstMiscTime.put(instructionName, miscTimesMap);
		}
		Long oldVal = miscTimesMap.get(miscTimer);
		Long newVal = timeNanos + ((oldVal!=null) ? oldVal : 0);
		miscTimesMap.put(miscTimer, newVal);

		HashMap<String, Long> miscCountMap = _cpInstMiscCount.get(instructionName);
		if (miscCountMap == null){
			miscCountMap = new HashMap<>();
			_cpInstMiscCount.put(instructionName, miscCountMap);
		}
		Long oldCnt = miscCountMap.get(miscTimer);
		Long newCnt = incrementCount + ((oldCnt!=null) ? oldCnt : 0);
		miscCountMap.put(miscTimer, newCnt);
	}

	/**
	 * "Maintains" or adds time to miscellaneous timers per instruction/op, also increments associated count by 1
	 * @param instructionName	name of the instruction/op
	 * @param miscTimer				name of the miscellaneous timer
	 * @param timeNanos				time in nano seconds
	 */
	public synchronized static void maintainCPMiscTimes( String instructionName, String miscTimer, long timeNanos){
		maintainCPMiscTimes(instructionName, miscTimer, timeNanos, 1);
	}

	/**
	 * Used to print misc timers (and their counts) for a given instruction/op
	 * @param instructionName name of the instruction/op
	 * @return  a formatted string of misc timers for a given instruction/op
	 */
	public static String getStringForCPMiscTimesPerInstruction(String instructionName) {
		StringBuffer sb = new StringBuffer();
		HashMap<String, Long> miscTimerMap = _cpInstMiscTime.get(instructionName);
		if (miscTimerMap != null) {
			List<Map.Entry<String, Long>> sortedList = new ArrayList<>(miscTimerMap.entrySet());
			// Sort the times to display by the most expensive first
			Collections.sort(sortedList, new Comparator<Map.Entry<String, Long>>() {
				@Override
				public int compare(Map.Entry<String, Long> o1, Map.Entry<String, Long> o2) {
					return (int) (o1.getValue() - o2.getValue());
				}
			});
			Iterator<Map.Entry<String, Long>> miscTimeIter = sortedList.iterator();
			HashMap<String, Long> miscCountMap = _cpInstMiscCount.get(instructionName);
			while (miscTimeIter.hasNext()) {
				Map.Entry<String, Long> e = miscTimeIter.next();
				String miscTimerName = e.getKey();
				Long miscTimerTime = e.getValue();
				Long miscCount = miscCountMap.get(miscTimerName);
				sb.append(miscTimerName + "[" + String.format("%.3f", (double) miscTimerTime / 1000000000.0) + "s," + miscCount + "]");
				if (miscTimeIter.hasNext())
					sb.append(", ");
			}
		}
		return sb.toString();
	}

	/**
	 * Used to print out cuda timers & counters
	 * @return a formatted string of cuda timers & counters
	 */
	public static String getStringForCudaTimers() {
		StringBuffer sb = new StringBuffer();
		sb.append("CUDA/CuLibraries init time:\t" + String.format("%.3f", cudaInitTime*1e-9) + "/"
				+ String.format("%.3f", cudaLibrariesInitTime*1e-9) + " sec.\n");
		sb.append("Number of executed GPU inst:\t" + getNoOfExecutedGPUInst() + ".\n");
		sb.append("GPU mem tx time  (alloc/dealloc/set0/toDev/fromDev):\t"
				+ String.format("%.3f", cudaAllocTime.longValue()*1e-9) + "/"
				+ String.format("%.3f", cudaDeAllocTime.longValue()*1e-9) + "/"
				+ String.format("%.3f", cudaMemSet0Time.longValue()*1e-9) + "/"
				+ String.format("%.3f", cudaToDevTime.longValue()*1e-9) + "/"
				+ String.format("%.3f", cudaFromDevTime.longValue()*1e-9)  + " sec.\n");
		sb.append("GPU mem tx count (alloc/dealloc/set0/toDev/fromDev/evict):\t"
				+ cudaAllocCount.longValue() + "/"
				+ cudaDeAllocCount.longValue() + "/"
				+ cudaMemSet0Count.longValue() + "/"
				+ cudaSparseConversionCount.longValue() + "/"
				+ cudaToDevCount.longValue() + "/"
				+ cudaFromDevCount.longValue() + "/"
				+ cudaEvictionCount.longValue() + ".\n");
		sb.append("GPU conversion time  (sparseConv/sp2dense/dense2sp):\t"
				+ String.format("%.3f", cudaSparseConversionTime.longValue()*1e-9) + "/"
				+ String.format("%.3f", cudaSparseToDenseTime.longValue()*1e-9) + "/"
				+ String.format("%.3f", cudaDenseToSparseTime.longValue()*1e-9) + " sec.\n");
		sb.append("GPU conversion count (sparseConv/sp2dense/dense2sp):\t"
				+ cudaSparseConversionCount.longValue() + "/"
				+ cudaSparseToDenseCount.longValue() + "/"
				+ cudaDenseToSparseCount.longValue() + ".\n");

		return sb.toString();
	}


}
