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

package org.apache.sysds.utils;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.fedplanner.FederatedCompilationTimer;
import org.apache.sysds.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageCacheStatistics;
import org.apache.sysds.runtime.privacy.CheckedConstraintsLog;
import org.apache.sysds.utils.stats.CodegenStatistics;
import org.apache.sysds.utils.stats.RecompileStatistics;
import org.apache.sysds.utils.stats.NativeStatistics;
import org.apache.sysds.utils.stats.ParamServStatistics;
import org.apache.sysds.utils.stats.ParForStatistics;
import org.apache.sysds.utils.stats.SparkStatistics;
import org.apache.sysds.utils.stats.TransformStatistics;

import java.lang.management.CompilationMXBean;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;

/**
 * This class captures all statistics.
 */
public class Statistics 
{
	private static class InstStats {
		private final LongAdder time = new LongAdder();
		private final LongAdder count = new LongAdder();
	}
	
	private static long compileStartTime = 0;
	private static long compileEndTime = 0;
	private static long execStartTime = 0;
	private static long execEndTime = 0;
	
	//heavy hitter counts and times 
	private static final ConcurrentHashMap<String,InstStats>_instStats = new ConcurrentHashMap<>();

	// number of compiled/executed SP instructions
	private static final LongAdder numExecutedSPInst = new LongAdder();
	private static final LongAdder numCompiledSPInst = new LongAdder();

	// number and size of pinned objects in scope
	private static final DoubleAdder sizeofPinnedObjects = new DoubleAdder();
	private static long maxNumPinnedObjects = 0;
	private static double maxSizeofPinnedObjects = 0;

	// Maps to keep track of CP memory objects for JMLC (e.g. in memory matrices and frames)
	private static final ConcurrentHashMap<String,Double> _cpMemObjs = new ConcurrentHashMap<>();
	private static final ConcurrentHashMap<Integer,Double> _currCPMemObjs = new ConcurrentHashMap<>();

	//JVM stats (low frequency updates)
	private static long jitCompileTime = 0; //in milli sec
	private static long jvmGCTime = 0; //in milli sec
	private static long jvmGCCount = 0; //count

	//Function recompile stats 
	private static final LongAdder funRecompileTime = new LongAdder(); //in nano sec
	private static final LongAdder funRecompiles = new LongAdder(); //count

	private static final LongAdder lTotalUIPVar = new LongAdder();
	private static final LongAdder lTotalLix = new LongAdder();
	private static final LongAdder lTotalLixUIP = new LongAdder();

	public static long recomputeNNZTime = 0;
	public static long examSparsityTime = 0;
	public static long allocateDoubleArrTime = 0;

	public static boolean allowWorkerStatistics = true;

	public static long getNoOfExecutedSPInst() {
		return numExecutedSPInst.longValue();
	}
	
	public static void incrementNoOfExecutedSPInst() {
		numExecutedSPInst.increment();
	}
	
	public static void decrementNoOfExecutedSPInst() {
		numExecutedSPInst.decrement();
	}

	public static long getNoOfCompiledSPInst() {
		return numCompiledSPInst.longValue();
	}

	public static void incrementNoOfCompiledSPInst() {
		numCompiledSPInst.increment();
	}

	public static long getTotalUIPVar() {
		return lTotalUIPVar.longValue();
	}

	public static void incrementTotalUIPVar() {
		lTotalUIPVar.increment();
	}

	public static long getTotalLixUIP() {
		return lTotalLixUIP.longValue();
	}

	public static void incrementTotalLixUIP() {
		lTotalLixUIP.increment();
	}

	public static long getTotalLix() {
		return lTotalLix.longValue();
	}

	public static void incrementTotalLix() {
		lTotalLix.increment();
	}

	public static void resetNoOfCompiledJobs( int count ) {
		//reset both mr/sp for multiple tests within one jvm
		numCompiledSPInst.reset();
		if( OptimizerUtils.isSparkExecutionMode() )
			numCompiledSPInst.add(count);
	}

	public static void resetNoOfExecutedJobs() {
		//reset both mr/sp for multiple tests within one jvm
		numExecutedSPInst.reset();
		if( DMLScript.USE_ACCELERATOR )
			GPUStatistics.setNoOfExecutedGPUInst(0);
	}
	
	public static synchronized void incrementJITCompileTime( long time ) {
		jitCompileTime += time;
	}
	
	public static synchronized void incrementJVMgcTime( long time ) {
		jvmGCTime += time;
	}
	
	public static synchronized void incrementJVMgcCount( long delta ) {
		jvmGCCount += delta;
	}

	public static void incrementFunRecompileTime( long delta ) {
		funRecompileTime.add(delta);
	}
	
	public static void incrementFunRecompiles() {
		funRecompiles.increment();
	}

	public static void startCompileTimer() {
		if( DMLScript.STATISTICS )
			compileStartTime = System.nanoTime();
	}

	public static void stopCompileTimer() {
		if( DMLScript.STATISTICS )
			compileEndTime = System.nanoTime();
	}

	public static long getCompileTime() {
		return compileEndTime - compileStartTime;
	}
	
	/**
	 * Starts the timer, should be invoked immediately before invoking
	 * Program.execute()
	 */
	public static void startRunTimer() {
		execStartTime = System.nanoTime();
	}

	/**
	 * Stops the timer, should be invoked immediately after invoking
	 * Program.execute()
	 */
	public static void stopRunTimer() {
		execEndTime = System.nanoTime();
	}

	/**
	 * Returns the total time of run in nanoseconds.
	 * 
	 * @return run time in nanoseconds
	 */
	public static long getRunTime() {
		return execEndTime - execStartTime;
	}
	
	public static void reset()
	{
		RecompileStatistics.reset();

		funRecompiles.reset();
		funRecompileTime.reset();

		CodegenStatistics.reset();
		ParForStatistics.reset();
		ParamServStatistics.reset();
		SparkStatistics.reset();
		TransformStatistics.reset();

		lTotalLix.reset();
		lTotalLixUIP.reset();
		lTotalUIPVar.reset();
		
		CacheStatistics.reset();
		LineageCacheStatistics.reset();
		
		resetJITCompileTime();
		resetJVMgcTime();
		resetJVMgcCount();
		resetCPHeavyHitters();

		GPUStatistics.reset();
		NativeStatistics.reset();
		DMLCompressionStatistics.reset();

		FederatedStatistics.reset();
	}

	public static void resetJITCompileTime(){
		jitCompileTime = -1 * getJITCompileTime();
	}
	
	public static void resetJVMgcTime(){
		jvmGCTime = -1 * getJVMgcTime();
	}
	
	public static void resetJVMgcCount(){
		jvmGCTime = -1 * getJVMgcCount();
	}

	public static void resetCPHeavyHitters(){
		_instStats.clear();
	}

	public static String getCPHeavyHitterCode( Instruction inst )
	{
		String opcode = null;
		
		if( inst instanceof SPInstruction ) {
			opcode = "SP_"+InstructionUtils.getOpCode(inst.toString());
			if( inst instanceof FunctionCallCPInstruction ) {
				FunctionCallCPInstruction extfunct = (FunctionCallCPInstruction)inst;
				opcode = extfunct.getFunctionName();
			}
		}
		else { //CPInstructions
			opcode = InstructionUtils.getOpCode(inst.toString());
			if( inst instanceof FunctionCallCPInstruction ) {
				FunctionCallCPInstruction extfunct = (FunctionCallCPInstruction)inst;
				opcode = extfunct.getFunctionName();
			}
		}
		
		return opcode;
	}

	public static void addCPMemObject(int hash, double sizeof) {
		double sizePrev = _currCPMemObjs.getOrDefault(hash, 0.0);
		_currCPMemObjs.put(hash, sizeof);
		sizeofPinnedObjects.add(sizeof - sizePrev);
		maintainMemMaxStats();
	}

	/**
	 * Helper method to keep track of the maximum number of pinned
	 * objects and total size yet seen
	 */
	private static void maintainMemMaxStats() {
		if (maxSizeofPinnedObjects < sizeofPinnedObjects.doubleValue())
			maxSizeofPinnedObjects = sizeofPinnedObjects.doubleValue();
		if (maxNumPinnedObjects < _currCPMemObjs.size())
			maxNumPinnedObjects = _currCPMemObjs.size();
	}

	/**
	 * Helper method to remove a memory object which has become unpinned
	 * @param hash hash of data object
	 */
	public static void removeCPMemObject( int hash ) {
		if (_currCPMemObjs.containsKey(hash)) {
			double sizeof = _currCPMemObjs.remove(hash);
			sizeofPinnedObjects.add(-1.0 * sizeof);
		}
	}

	/**
	 * Helper method which keeps track of the heaviest weight objects (by total memory used)
	 * throughout execution of the program. Only reported if JMLC memory statistics are enabled and
	 * finegrained statistics are enabled. We only keep track of the -largest- instance of data associated with a
	 * particular string identifier so no need to worry about multiple bindings to the same name
	 * @param name String denoting the variables name
	 * @param sizeof objects size (estimated bytes)
	 */
	public static void maintainCPHeavyHittersMem( String name, double sizeof ) {
		double prevSize = _cpMemObjs.getOrDefault(name, 0.0);
		if (prevSize < sizeof)
			_cpMemObjs.put(name, sizeof);
	}

	/**
	 * "Maintains" or adds time to per instruction/op timers, also increments associated count
	 * @param instName name of the instruction/op
	 * @param timeNanos time in nano seconds
	 */
	public static void maintainCPHeavyHitters( String instName, long timeNanos ) {
		//maintain instruction entry (w/ robustness for concurrent updates)
		InstStats tmp = _instStats.get(instName);
		if( tmp == null ) {
			InstStats tmp0 = new InstStats();
			InstStats tmp1 = _instStats.putIfAbsent(instName, tmp0);
			tmp = (tmp1 != null) ? tmp1 : tmp0;
		}
		
		//thread-local maintenance of instruction stats
		tmp.time.add(timeNanos);
		tmp.count.increment();
	}
	
	public static void maintainCPFuncCallStats(String instName) {
		InstStats tmp = _instStats.get(instName);
		if (tmp != null)  //tmp should never be null
			tmp.count.decrement();
	}

	public static Set<String> getCPHeavyHitterOpCodes() {
		return _instStats.keySet();
	}
	
	public static long getCPHeavyHitterCount(String opcode) {
		InstStats tmp = _instStats.get(opcode);
		return (tmp != null) ? tmp.count.longValue() : 0;
	}

	public static HashMap<String, Pair<Long, Double>> getHeavyHittersHashMap() {
		HashMap<String, Pair<Long, Double>> heavyHitters = new HashMap<>();
		for(String opcode : _instStats.keySet()) {
			InstStats val = _instStats.get(opcode);
			long count = val.count.longValue();
			double time = val.time.longValue() / 1000000000d; // in sec
			heavyHitters.put(opcode, new ImmutablePair<>(Long.valueOf(count), Double.valueOf(time)));
		}
		return heavyHitters;
	}

	/**
	 * Obtain a string tabular representation of the heavy hitter instructions
	 * that displays the time, instruction count, and optionally GPU stats about
	 * each instruction.
	 * 
	 * @param num
	 *            the maximum number of heavy hitters to display
	 * @return string representing the heavy hitter instructions in tabular
	 *         format
	 */
	@SuppressWarnings("unchecked")
	public static String getHeavyHitters(int num) {
		if (num <= 0 || _instStats.size() <= 0)
			return "-";

		// get top k via sort
		Entry<String, InstStats>[] tmp = _instStats.entrySet().toArray(Entry[]::new);
		Arrays.sort(tmp, new Comparator<Entry<String, InstStats>>() {
			@Override
			public int compare(Entry<String, InstStats> e1, Entry<String, InstStats> e2) {
				return Long.compare(e1.getValue().time.longValue(), e2.getValue().time.longValue());
			}
		});

		final String numCol = "#";
		final String instCol = "Instruction";
		final String timeSCol = "Time(s)";
		final String countCol = "Count";
		StringBuilder sb = new StringBuilder();
		int len = tmp.length;
		int numHittersToDisplay = Math.min(num, len);
		int maxNumLen = String.valueOf(numHittersToDisplay).length();
		int maxInstLen = instCol.length();
		int maxTimeSLen = timeSCol.length();
		int maxCountLen = countCol.length();
		DecimalFormat sFormat = new DecimalFormat("#,##0.000");
		for (int i = 0; i < numHittersToDisplay; i++) {
			Entry<String, InstStats> hh = tmp[len - 1 - i];
			String instruction = hh.getKey();
			long timeNs = hh.getValue().time.longValue();
			double timeS = timeNs / 1000000000d;

			maxInstLen = Math.max(maxInstLen, instruction.length());

			String timeSString = sFormat.format(timeS);
			maxTimeSLen = Math.max(maxTimeSLen, timeSString.length());

			maxCountLen = Math.max(maxCountLen, String.valueOf(hh.getValue().count.longValue()).length());
		}
		maxInstLen = Math.min(maxInstLen, DMLScript.STATISTICS_MAX_WRAP_LEN);
		sb.append(String.format( " %" + maxNumLen + "s  %-" + maxInstLen + "s  %"
			+ maxTimeSLen + "s  %" + maxCountLen + "s", numCol, instCol, timeSCol, countCol));
		sb.append("\n");
		for (int i = 0; i < numHittersToDisplay; i++) {
			String instruction = tmp[len - 1 - i].getKey();
			String [] wrappedInstruction = wrap(instruction, maxInstLen);

			long timeNs = tmp[len - 1 - i].getValue().time.longValue();
			double timeS = timeNs / 1000000000d;
			String timeSString = sFormat.format(timeS);

			long count = tmp[len - 1 - i].getValue().count.longValue();
			int numLines = wrappedInstruction.length;
			
			for(int wrapIter = 0; wrapIter < numLines; wrapIter++) {
				String instStr = (wrapIter < wrappedInstruction.length) ? wrappedInstruction[wrapIter] : "";
				if(wrapIter == 0) {
					// Display instruction count
					sb.append(String.format(
						" %" + maxNumLen + "d  %-" + maxInstLen + "s  %" + maxTimeSLen + "s  %" + maxCountLen + "d",
						(i + 1), instStr, timeSString, count));
				}
				else {
					sb.append(String.format(
						" %" + maxNumLen + "s  %-" + maxInstLen + "s  %" + maxTimeSLen + "s  %" + maxCountLen + "s",
						"", instStr, "", ""));
				}
				sb.append("\n");
			}
		}

		return sb.toString();
	}

	@SuppressWarnings("unchecked")
	public static String getCPHeavyHittersMem(int num) {
		if ((_cpMemObjs.size() <= 0) || (num <= 0))
			return "-";

		Entry<String,Double>[] entries = _cpMemObjs.entrySet().toArray(Entry[]::new);
		Arrays.sort(entries, new Comparator<Entry<String, Double>>() {
			@Override
			public int compare(Entry<String, Double> a, Entry<String, Double> b) {
				return b.getValue().compareTo(a.getValue());
			}
		});

		int n = entries.length;
		int numHittersToDisplay = Math.min(num, n);
		int numPadLen = String.format("%d", numHittersToDisplay).length();
		int maxNameLength = 0;
		for (String name : _cpMemObjs.keySet())
			maxNameLength = Math.max(name.length(), maxNameLength);

		maxNameLength = Math.max(maxNameLength, "Object".length());
		StringBuilder res = new StringBuilder();
		res.append(String.format("  %-" + numPadLen + "s" + "  %-" 
			+ maxNameLength + "s" + "  %s\n", "#", "Object", "Memory"));

		for (int ix = 1; ix <= numHittersToDisplay; ix++) {
			String objName = entries[ix-1].getKey();
			String objSize = byteCountToDisplaySize(entries[ix-1].getValue());
			String numStr = String.format("  %-" + numPadLen + "s", ix);
			String objNameStr = String.format("  %-" + maxNameLength + "s ", objName);
			res.append(numStr + objNameStr + String.format("  %s", objSize) + "\n");
		}

		return res.toString();
	}

	/**
	 * Helper method to create a nice representation of byte counts - this was copied from
	 * GPUMemoryManager and should eventually be refactored probably...
	 */
	private static String byteCountToDisplaySize(double numBytes) {
		if (numBytes < 1024) {
			return numBytes + " bytes";
		}
		else {
			int exp = (int) (Math.log(numBytes) / 6.931471805599453);
			return String.format("%.3f %sB", numBytes / Math.pow(1024, exp), "KMGTP".charAt(exp-1));
		}
	}

	/**
	 * Returns the total time of asynchronous JIT compilation in milliseconds.
	 * 
	 * @return JIT compile time
	 */
	public static long getJITCompileTime(){
		long ret = -1; //unsupported
		CompilationMXBean cmx = ManagementFactory.getCompilationMXBean();
		if( cmx.isCompilationTimeMonitoringSupported() ) {
			ret = cmx.getTotalCompilationTime();
			ret += jitCompileTime; //add from remote processes
		}
		return ret;
	}
	
	public static long getJVMgcTime(){
		long ret = 0; 
		
		List<GarbageCollectorMXBean> gcxs = ManagementFactory.getGarbageCollectorMXBeans();
		
		for( GarbageCollectorMXBean gcx : gcxs )
			ret += gcx.getCollectionTime();
		if( ret>0 )
			ret += jvmGCTime;
		
		return ret;
	}
	
	public static long getJVMgcCount(){
		long ret = 0; 
		
		List<GarbageCollectorMXBean> gcxs = ManagementFactory.getGarbageCollectorMXBeans();
		
		for( GarbageCollectorMXBean gcx : gcxs )
			ret += gcx.getCollectionCount();
		if( ret>0 )
			ret += jvmGCCount;
		
		return ret;
	}

	public static long getFunRecompileTime(){
		return funRecompileTime.longValue();
	}
	
	public static long getFunRecompiles(){
		return funRecompiles.longValue();
	}

	public static long getNumPinnedObjects() { return maxNumPinnedObjects; }

	public static double getSizeofPinnedObjects() { return maxSizeofPinnedObjects; }

	/**
	 * Returns statistics of the DML program that was recently completed as a string
	 * @return statistics as a string
	 */
	public static String display() {
		return display(DMLScript.STATISTICS_COUNT);
	}
	
	
	public static String [] wrap(String str, int wrapLength) {
		int numLines = (int) Math.ceil( ((double)str.length()) / wrapLength);
		int len = str.length();
		String [] ret = new String[numLines];
		for(int i = 0; i < numLines; i++) {
			ret[i] = str.substring(i*wrapLength, Math.min((i+1)*wrapLength, len));
		}
		return ret;
	}

	/**
	 * Returns statistics as a string
	 * @param maxHeavyHitters The maximum number of heavy hitters that are printed
	 * @return statistics as string
	 */
	public static String display(int maxHeavyHitters)
	{
		StringBuilder sb = new StringBuilder();

		sb.append("SystemDS Statistics:\n");
		if( DMLScript.STATISTICS ) {
			sb.append("Total elapsed time:\t\t" + String.format("%.3f", (getCompileTime()+getRunTime())*1e-9) + " sec.\n"); // nanoSec --> sec
			sb.append("Total compilation time:\t\t" + String.format("%.3f", getCompileTime()*1e-9) + " sec.\n"); // nanoSec --> sec
			sb.append(FederatedCompilationTimer.getStringRepresentation());
		}
		sb.append("Total execution time:\t\t" + String.format("%.3f", getRunTime()*1e-9) + " sec.\n"); // nanoSec --> sec
		if( OptimizerUtils.isSparkExecutionMode() ) {
			if( DMLScript.STATISTICS ) //moved into stats on Shiv's request
				sb.append("Number of compiled Spark inst:\t" + getNoOfCompiledSPInst() + ".\n");
			sb.append("Number of executed Spark inst:\t" + getNoOfExecutedSPInst() + ".\n");
		}

		if( DMLScript.USE_ACCELERATOR && DMLScript.STATISTICS)
			sb.append(GPUStatistics.getStringForCudaTimers());

		//show extended caching/compilation statistics
		if( DMLScript.STATISTICS )
		{
			if(NativeHelper.CURRENT_NATIVE_BLAS_STATE == NativeHelper.NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE)
				sb.append(NativeStatistics.displayStatistics());

			if(recomputeNNZTime != 0 || examSparsityTime != 0 || allocateDoubleArrTime != 0) {
				sb.append("MatrixBlock times (recomputeNNZ/examSparsity/allocateDoubleArr):\t" + String.format("%.3f", recomputeNNZTime*1e-9) + "/" +
						String.format("%.3f", examSparsityTime*1e-9) + "/" + String.format("%.3f", allocateDoubleArrTime*1e-9)  + ".\n");
			}

			sb.append("Cache hits (Mem/Li/WB/FS/HDFS):\t" + CacheStatistics.displayHits() + ".\n");
			sb.append("Cache writes (Li/WB/FS/HDFS):\t" + CacheStatistics.displayWrites() + ".\n");
			sb.append("Cache times (ACQr/m, RLS, EXP):\t" + CacheStatistics.displayTime() + " sec.\n");
			if (DMLScript.JMLC_MEM_STATISTICS)
				sb.append("Max size of live objects:\t" + byteCountToDisplaySize(getSizeofPinnedObjects()) + " ("  + getNumPinnedObjects() + " total objects)" + "\n");

			sb.append(RecompileStatistics.displayStatistics());

			if( getFunRecompiles()>0 ) {
				sb.append("Functions recompiled:\t\t" + getFunRecompiles() + ".\n");
				sb.append("Functions recompile time:\t" + String.format("%.3f", ((double)getFunRecompileTime())/1000000000) + " sec.\n");
			}
			if (DMLScript.LINEAGE && !ReuseCacheType.isNone()) {
				sb.append("LinCache hits (Mem/FS/Del): \t" + LineageCacheStatistics.displayHits() + ".\n");
				sb.append("LinCache MultiLevel (Ins/SB/Fn):" + LineageCacheStatistics.displayMultiLevelHits() + ".\n");
				sb.append("LinCache GPU (Hit/Async/Sync): \t" + LineageCacheStatistics.displayGpuStats() + ".\n");
				sb.append("LinCache writes (Mem/FS/Del): \t" + LineageCacheStatistics.displayWtrites() + ".\n");
				sb.append("LinCache FStimes (Rd/Wr): \t" + LineageCacheStatistics.displayFSTime() + " sec.\n");
				sb.append("LinCache Computetime (S/M): \t" + LineageCacheStatistics.displayComputeTime() + " sec.\n");
				sb.append("LinCache Rewrites:    \t\t" + LineageCacheStatistics.displayRewrites() + ".\n");
			}

			if( ConfigurationManager.isCodegenEnabled() )
				sb.append(CodegenStatistics.displayStatistics());

			if( OptimizerUtils.isSparkExecutionMode() )
				sb.append(SparkStatistics.displayStatistics());

			sb.append(ParamServStatistics.displayStatistics());

			sb.append(ParForStatistics.displayStatistics());

			sb.append(FederatedStatistics.displayFedIOExecStatistics());
			sb.append(FederatedStatistics.displayFedWorkerStats());

			sb.append(TransformStatistics.displayStatistics());

			if(ConfigurationManager.isCompressionEnabled()){
				DMLCompressionStatistics.display(sb);
			}

			sb.append("Total JIT compile time:\t\t" + ((double)getJITCompileTime())/1000 + " sec.\n");
			sb.append("Total JVM GC count:\t\t" + getJVMgcCount() + ".\n");
			sb.append("Total JVM GC time:\t\t" + ((double)getJVMgcTime())/1000 + " sec.\n");
			sb.append("Heavy hitter instructions:\n" + getHeavyHitters(maxHeavyHitters));
		}

		if (DMLScript.CHECK_PRIVACY)
			sb.append(CheckedConstraintsLog.display());

		if(DMLScript.FED_STATISTICS) {
			sb.append("\n");
			sb.append(FederatedStatistics.displayStatistics(DMLScript.FED_STATISTICS_COUNT));
			sb.append("\n");
			sb.append(ParamServStatistics.displayFloStatistics());
		}

		return sb.toString();
	}
}
