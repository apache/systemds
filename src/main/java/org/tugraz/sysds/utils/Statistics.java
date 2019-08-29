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

package org.tugraz.sysds.utils;

import java.lang.management.CompilationMXBean;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;

import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheStatistics;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.tugraz.sysds.runtime.instructions.spark.SPInstruction;
import org.tugraz.sysds.runtime.lineage.LineageCacheStatistics;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixDNN;

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
	
	//HOP DAG recompile stats (potentially high update frequency)
	private static final LongAdder hopRecompileTime = new LongAdder(); //in nano sec
	private static final LongAdder hopRecompilePred = new LongAdder(); //count
	private static final LongAdder hopRecompileSB = new LongAdder();   //count

	//CODEGEN
	private static final LongAdder codegenCompileTime = new LongAdder(); //in nano
	private static final LongAdder codegenClassCompileTime = new LongAdder(); //in nano
	private static final LongAdder codegenHopCompile = new LongAdder(); //count
	private static final LongAdder codegenCPlanCompile = new LongAdder(); //count
	private static final LongAdder codegenClassCompile = new LongAdder(); //count
	private static final LongAdder codegenEnumAll = new LongAdder(); //count
	private static final LongAdder codegenEnumAllP = new LongAdder(); //count
	private static final LongAdder codegenEnumEval = new LongAdder(); //count
	private static final LongAdder codegenEnumEvalP = new LongAdder(); //count
	private static final LongAdder codegenOpCacheHits = new LongAdder(); //count
	private static final LongAdder codegenOpCacheTotal = new LongAdder(); //count
	private static final LongAdder codegenPlanCacheHits = new LongAdder(); //count
	private static final LongAdder codegenPlanCacheTotal = new LongAdder(); //count
	
	//Function recompile stats 
	private static final LongAdder funRecompileTime = new LongAdder(); //in nano sec
	private static final LongAdder funRecompiles = new LongAdder(); //count
	
	//Spark-specific stats
	private static long sparkCtxCreateTime = 0; 
	private static final LongAdder sparkParallelize = new LongAdder();
	private static final LongAdder sparkParallelizeCount = new LongAdder();
	private static final LongAdder sparkCollect = new LongAdder();
	private static final LongAdder sparkCollectCount = new LongAdder();
	private static final LongAdder sparkBroadcast = new LongAdder();
	private static final LongAdder sparkBroadcastCount = new LongAdder();

	// Paramserv function stats (time is in milli sec)
	private static final LongAdder psNumWorkers = new LongAdder();
	private static final LongAdder psSetupTime = new LongAdder();
	private static final LongAdder psGradientComputeTime = new LongAdder();
	private static final LongAdder psAggregationTime = new LongAdder();
	private static final LongAdder psLocalModelUpdateTime = new LongAdder();
	private static final LongAdder psModelBroadcastTime = new LongAdder();
	private static final LongAdder psBatchIndexTime = new LongAdder();
	private static final LongAdder psRpcRequestTime = new LongAdder();

	//PARFOR optimization stats (low frequency updates)
	private static long parforOptTime = 0; //in milli sec
	private static long parforOptCount = 0; //count
	private static long parforInitTime = 0; //in milli sec
	private static long parforMergeTime = 0; //in milli sec

	private static final LongAdder lTotalUIPVar = new LongAdder();
	private static final LongAdder lTotalLix = new LongAdder();
	private static final LongAdder lTotalLixUIP = new LongAdder();
	
	private static LongAdder numNativeFailures = new LongAdder();
	public static LongAdder numNativeLibMatrixMultCalls = new LongAdder();
	public static LongAdder numNativeConv2dCalls = new LongAdder();
	public static LongAdder numNativeConv2dBwdDataCalls = new LongAdder();
	public static LongAdder numNativeConv2dBwdFilterCalls = new LongAdder();
	public static LongAdder numNativeSparseConv2dCalls = new LongAdder();
	public static LongAdder numNativeSparseConv2dBwdFilterCalls = new LongAdder();
	public static LongAdder numNativeSparseConv2dBwdDataCalls = new LongAdder();
	public static long nativeLibMatrixMultTime = 0;
	public static long nativeConv2dTime = 0;
	public static long nativeConv2dBwdDataTime = 0;
	public static long nativeConv2dBwdFilterTime = 0;
	
	public static long recomputeNNZTime = 0;
	public static long examSparsityTime = 0;
	public static long allocateDoubleArrTime = 0;
	
	public static void incrementNativeFailuresCounter() {
		numNativeFailures.increment();
		// This is very rare and am not sure it is possible at all. Our initial experiments never encountered this case.
		// Note: all the native calls have a fallback to Java; so if the user wants she can recompile SystemDS by
		// commenting this exception and everything should work fine.
		throw new RuntimeException("Unexpected ERROR: OOM caused during JNI transfer. Please disable native BLAS by setting enviroment variable: SYSTEMDS_BLAS=none");
	}

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
	
	public static boolean createdSparkContext() {
		return sparkCtxCreateTime > 0;
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
	
	public static void incrementHOPRecompileTime( long delta ) {
		hopRecompileTime.add(delta);
	}
	
	public static void incrementHOPRecompilePred() {
		hopRecompilePred.increment();
	}
	
	public static void incrementHOPRecompilePred(long delta) {
		hopRecompilePred.add(delta);
	}
	
	public static void incrementHOPRecompileSB() {
		hopRecompileSB.increment();
	}
	
	public static void incrementHOPRecompileSB(long delta) {
		hopRecompileSB.add(delta);
	}
	
	public static void incrementCodegenDAGCompile() {
		codegenHopCompile.increment();
	}
	
	public static void incrementCodegenCPlanCompile(long delta) {
		codegenCPlanCompile.add(delta);
	}
	
	public static void incrementCodegenEnumAll(long delta) {
		codegenEnumAll.add(delta);
	}
	public static void incrementCodegenEnumAllP(long delta) {
		codegenEnumAllP.add(delta);
	}
	public static void incrementCodegenEnumEval(long delta) {
		codegenEnumEval.add(delta);
	}
	public static void incrementCodegenEnumEvalP(long delta) {
		codegenEnumEvalP.add(delta);
	}
	
	public static void incrementCodegenClassCompile() {
		codegenClassCompile.increment();
	}
	
	public static void incrementCodegenCompileTime(long delta) {
		codegenCompileTime.add(delta);
	}
	
	public static void incrementCodegenClassCompileTime(long delta) {
		codegenClassCompileTime.add(delta);
	}
	
	public static void incrementCodegenOpCacheHits() {
		codegenOpCacheHits.increment();
	}
	
	public static void incrementCodegenOpCacheTotal() {
		codegenOpCacheTotal.increment();
	}
	
	public static void incrementCodegenPlanCacheHits() {
		codegenPlanCacheHits.increment();
	}
	
	public static void incrementCodegenPlanCacheTotal() {
		codegenPlanCacheTotal.increment();
	}
	
	public static long getCodegenDAGCompile() {
		return codegenHopCompile.longValue();
	}
	
	public static long getCodegenCPlanCompile() {
		return codegenCPlanCompile.longValue();
	}
	
	public static long getCodegenEnumAll() {
		return codegenEnumAll.longValue();
	}
	public static long getCodegenEnumAllP() {
		return codegenEnumAllP.longValue();
	}
	public static long getCodegenEnumEval() {
		return codegenEnumEval.longValue();
	}
	public static long getCodegenEnumEvalP() {
		return codegenEnumEvalP.longValue();
	}
	
	public static long getCodegenClassCompile() {
		return codegenClassCompile.longValue();
	}
	
	public static long getCodegenCompileTime() {
		return codegenCompileTime.longValue();
	}
	
	public static long getCodegenClassCompileTime() {
		return codegenClassCompileTime.longValue();
	}
	
	public static long getCodegenOpCacheHits() {
		return codegenOpCacheHits.longValue();
	}
	
	public static long getCodegenOpCacheTotal() {
		return codegenOpCacheTotal.longValue();
	}
	
	public static long getCodegenPlanCacheHits() {
		return codegenPlanCacheHits.longValue();
	}
	
	public static long getCodegenPlanCacheTotal() {
		return codegenPlanCacheTotal.longValue();
	}

	public static void incrementFunRecompileTime( long delta ) {
		funRecompileTime.add(delta);
	}
	
	public static void incrementFunRecompiles() {
		funRecompiles.increment();
	}
	
	public static synchronized void incrementParForOptimCount(){
		parforOptCount ++;
	}
	
	public static synchronized void incrementParForOptimTime( long time ) {
		parforOptTime += time;
	}
	
	public static synchronized void incrementParForInitTime( long time ) {
		parforInitTime += time;
	}
	
	public static synchronized void incrementParForMergeTime( long time ) {
		parforMergeTime += time;
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
		hopRecompileTime.reset();
		hopRecompilePred.reset();
		hopRecompileSB.reset();
		
		funRecompiles.reset();
		funRecompileTime.reset();
		
		codegenHopCompile.reset();
		codegenCPlanCompile.reset();
		codegenClassCompile.reset();
		codegenEnumAll.reset();
		codegenEnumAllP.reset();
		codegenEnumEval.reset();
		codegenEnumEvalP.reset();
		codegenCompileTime.reset();
		codegenClassCompileTime.reset();
		codegenOpCacheHits.reset();
		codegenOpCacheTotal.reset();
		codegenPlanCacheHits.reset();
		codegenPlanCacheTotal.reset();
		
		parforOptCount = 0;
		parforOptTime = 0;
		parforInitTime = 0;
		parforMergeTime = 0;
		
		sparkCtxCreateTime = 0;
		
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
		numNativeLibMatrixMultCalls.reset();
		numNativeSparseConv2dCalls.reset();
		numNativeSparseConv2dBwdDataCalls.reset();
		numNativeSparseConv2dBwdFilterCalls.reset();
		numNativeConv2dCalls.reset();
		numNativeConv2dBwdDataCalls.reset();
		numNativeConv2dBwdFilterCalls.reset();
		numNativeFailures.reset();
		nativeLibMatrixMultTime = 0;
		nativeConv2dTime = 0;
		nativeConv2dBwdFilterTime = 0;
		nativeConv2dBwdDataTime = 0;
		LibMatrixDNN.resetStatistics();
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

	public static void setSparkCtxCreateTime(long ns) {
		sparkCtxCreateTime = ns;
	}
	
	public static void accSparkParallelizeTime(long t) {
		sparkParallelize.add(t);
	}

	public static void incSparkParallelizeCount(long c) {
		sparkParallelizeCount.add(c);
	}

	public static void accSparkCollectTime(long t) {
		sparkCollect.add(t);
	}

	public static void incSparkCollectCount(long c) {
		sparkCollectCount.add(c);
	}

	public static void accSparkBroadCastTime(long t) {
		sparkBroadcast.add(t);
	}

	public static void incSparkBroadcastCount(long c) {
		sparkBroadcastCount.add(c);
	}

	public static void incWorkerNumber() {
		psNumWorkers.increment();
	}

	public static void incWorkerNumber(long n) {
		psNumWorkers.add(n);
	}

	public static void accPSSetupTime(long t) {
		psSetupTime.add(t);
	}

	public static void accPSGradientComputeTime(long t) {
		psGradientComputeTime.add(t);
	}

	public static void accPSAggregationTime(long t) {
		psAggregationTime.add(t);
	}

	public static void accPSLocalModelUpdateTime(long t) {
		psLocalModelUpdateTime.add(t);
	}

	public static void accPSModelBroadcastTime(long t) {
		psModelBroadcastTime.add(t);
	}

	public static void accPSBatchIndexingTime(long t) {
		psBatchIndexTime.add(t);
	}

	public static void accPSRpcRequestTime(long t) {
		psRpcRequestTime.add(t);
	}

	public static String getCPHeavyHitterCode( Instruction inst )
	{
		String opcode = null;
		
		if( inst instanceof SPInstruction )
		{
			opcode = "SP_"+InstructionUtils.getOpCode(inst.toString());
			if( inst instanceof FunctionCallCPInstruction ) {
				FunctionCallCPInstruction extfunct = (FunctionCallCPInstruction)inst;
				opcode = extfunct.getFunctionName();
			}	
		}
		else //CPInstructions
		{
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

	public static Set<String> getCPHeavyHitterOpCodes() {
		return _instStats.keySet();
	}
	
	public static long getCPHeavyHitterCount(String opcode) {
		InstStats tmp = _instStats.get(opcode);
		return (tmp != null) ? tmp.count.longValue() : 0;
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
		int len = _instStats.size();
		if (num <= 0 || len <= 0)
			return "-";

		// get top k via sort
		Entry<String, InstStats>[] tmp = _instStats.entrySet().toArray(new Entry[len]);
		Arrays.sort(tmp, new Comparator<Entry<String, InstStats>>() {
			public int compare(Entry<String, InstStats> e1, Entry<String, InstStats> e2) {
				return Long.compare(e1.getValue().time.longValue(), e2.getValue().time.longValue());
			}
		});

		final String numCol = "#";
		final String instCol = "Instruction";
		final String timeSCol = "Time(s)";
		final String countCol = "Count";
		StringBuilder sb = new StringBuilder();
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
			double timeS = (double) timeNs / 1000000000.0;

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
			double timeS = (double) timeNs / 1000000000.0;
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
		int n = _cpMemObjs.size();
		if ((n <= 0) || (num <= 0))
			return "-";

		Entry<String,Double>[] entries = _cpMemObjs.entrySet().toArray(new Entry[_cpMemObjs.size()]);
		Arrays.sort(entries, new Comparator<Entry<String, Double>>() {
			@Override
			public int compare(Entry<String, Double> a, Entry<String, Double> b) {
				return b.getValue().compareTo(a.getValue());
			}
		});

		int numHittersToDisplay = Math.min(num, n);
		int numPadLen = String.format("%d", numHittersToDisplay).length();
		int maxNameLength = 0;
		for (String name : _cpMemObjs.keySet())
			maxNameLength = Math.max(name.length(), maxNameLength);

		maxNameLength = Math.max(maxNameLength, "Object".length());
		StringBuilder res = new StringBuilder();
		res.append(String.format("  %-" + numPadLen + "s" + "  %-" + maxNameLength + "s" + "  %s\n",
				"#", "Object", "Memory"));

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
			return String.format("%.3f %sB", ((double)numBytes) / Math.pow(1024, exp), "KMGTP".charAt(exp-1));
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
		if( cmx.isCompilationTimeMonitoringSupported() )
		{
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
	
	public static long getHopRecompileTime(){
		return hopRecompileTime.longValue();
	}
	
	public static long getHopRecompiledPredDAGs(){
		return hopRecompilePred.longValue();
	}
	
	public static long getHopRecompiledSBDAGs(){
		return hopRecompileSB.longValue();
	}
	
	public static long getFunRecompileTime(){
		return funRecompileTime.longValue();
	}
	
	public static long getFunRecompiles(){
		return funRecompiles.longValue();
	}
		
	public static long getParforOptCount(){
		return parforOptCount;
	}
	
	public static long getParforOptTime(){
		return parforOptTime;
	}
	
	public static long getParforInitTime(){
		return parforInitTime;
	}
	
	public static long getParforMergeTime(){
		return parforMergeTime;
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
	
	
	private static String [] wrap(String str, int wrapLength) {
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
			if(NativeHelper.CURRENT_NATIVE_BLAS_STATE == NativeHelper.NativeBlasState.SUCCESSFULLY_LOADED_NATIVE_BLAS_AND_IN_USE) {
				String blas = NativeHelper.getCurrentBLAS();
				sb.append("Native " + blas + " calls (dense mult/conv/bwdF/bwdD):\t" + numNativeLibMatrixMultCalls.longValue()  + "/" +
						numNativeConv2dCalls.longValue() + "/" + numNativeConv2dBwdFilterCalls.longValue()
						+ "/" + numNativeConv2dBwdDataCalls.longValue() + ".\n");
				sb.append("Native " + blas + " calls (sparse conv/bwdF/bwdD):\t" +
						numNativeSparseConv2dCalls.longValue() + "/" + numNativeSparseConv2dBwdFilterCalls.longValue()
						+ "/" + numNativeSparseConv2dBwdDataCalls.longValue() + ".\n");
				sb.append("Native " + blas + " times (dense mult/conv/bwdF/bwdD):\t" + String.format("%.3f", nativeLibMatrixMultTime*1e-9) + "/" +
						String.format("%.3f", nativeConv2dTime*1e-9) + "/" + String.format("%.3f", nativeConv2dBwdFilterTime*1e-9) + "/" +
						String.format("%.3f", nativeConv2dBwdDataTime*1e-9) + ".\n");
			}
			if(recomputeNNZTime != 0 || examSparsityTime != 0 || allocateDoubleArrTime != 0) {
				sb.append("MatrixBlock times (recomputeNNZ/examSparsity/allocateDoubleArr):\t" + String.format("%.3f", recomputeNNZTime*1e-9) + "/" +
						String.format("%.3f", examSparsityTime*1e-9) + "/" + String.format("%.3f", allocateDoubleArrTime*1e-9)  + ".\n");
			}

			sb.append("Cache hits (Mem, WB, FS, HDFS):\t" + CacheStatistics.displayHits() + ".\n");
			sb.append("Cache writes (WB, FS, HDFS):\t" + CacheStatistics.displayWrites() + ".\n");
			sb.append("Cache times (ACQr/m, RLS, EXP):\t" + CacheStatistics.displayTime() + " sec.\n");
			if (DMLScript.JMLC_MEM_STATISTICS)
				sb.append("Max size of live objects:\t" + byteCountToDisplaySize(getSizeofPinnedObjects()) + " ("  + getNumPinnedObjects() + " total objects)" + "\n");
			sb.append("HOP DAGs recompiled (PRED, SB):\t" + getHopRecompiledPredDAGs() + "/" + getHopRecompiledSBDAGs() + ".\n");
			sb.append("HOP DAGs recompile time:\t" + String.format("%.3f", ((double)getHopRecompileTime())/1000000000) + " sec.\n");
			if( getFunRecompiles()>0 ) {
				sb.append("Functions recompiled:\t\t" + getFunRecompiles() + ".\n");
				sb.append("Functions recompile time:\t" + String.format("%.3f", ((double)getFunRecompileTime())/1000000000) + " sec.\n");
			}
			if (DMLScript.LINEAGE && DMLScript.LINEAGE_REUSE) {
				sb.append("LineageCache hits (Mem/FS/Del): " + LineageCacheStatistics.displayHits() + ".\n");
				sb.append("LineageCache writes (Mem/FS): \t" + LineageCacheStatistics.displayWtrites() + ".\n");
				sb.append("LineageCache FStimes (Rd/Wr): \t" + LineageCacheStatistics.displayTime() + " sec.\n");
				sb.append("LineageCache costing time: \t" + LineageCacheStatistics.displayCostingTime() + " sec.\n");
			}
			if( ConfigurationManager.isCodegenEnabled() ) {
				sb.append("Codegen compile (DAG,CP,JC):\t" + getCodegenDAGCompile() + "/"
						+ getCodegenCPlanCompile() + "/" + getCodegenClassCompile() + ".\n");
				sb.append("Codegen enum (ALLt/p,EVALt/p):\t" + getCodegenEnumAll() + "/" +
						getCodegenEnumAllP() + "/" + getCodegenEnumEval() + "/" + getCodegenEnumEvalP() + ".\n");
				sb.append("Codegen compile times (DAG,JC):\t" + String.format("%.3f", (double)getCodegenCompileTime()/1000000000) + "/" +
						String.format("%.3f", (double)getCodegenClassCompileTime()/1000000000)  + " sec.\n");
				sb.append("Codegen enum plan cache hits:\t" + getCodegenPlanCacheHits() + "/" + getCodegenPlanCacheTotal() + ".\n");
				sb.append("Codegen op plan cache hits:\t" + getCodegenOpCacheHits() + "/" + getCodegenOpCacheTotal() + ".\n");
			}
			if( OptimizerUtils.isSparkExecutionMode() ){
				String lazy = SparkExecutionContext.isLazySparkContextCreation() ? "(lazy)" : "(eager)";
				sb.append("Spark ctx create time "+lazy+":\t"+
						String.format("%.3f", ((double)sparkCtxCreateTime)*1e-9)  + " sec.\n" ); // nanoSec --> sec
				sb.append("Spark trans counts (par,bc,col):" +
						String.format("%d/%d/%d.\n", sparkParallelizeCount.longValue(),
								sparkBroadcastCount.longValue(), sparkCollectCount.longValue()));
				sb.append("Spark trans times (par,bc,col):\t" +
						String.format("%.3f/%.3f/%.3f secs.\n",
								((double)sparkParallelize.longValue())*1e-9,
								((double)sparkBroadcast.longValue())*1e-9,
								((double)sparkCollect.longValue())*1e-9));
			}
			if (psNumWorkers.longValue() > 0) {
				sb.append(String.format("Paramserv total num workers:\t%d.\n", psNumWorkers.longValue()));
				sb.append(String.format("Paramserv setup time:\t\t%.3f secs.\n", psSetupTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv grad compute time:\t%.3f secs.\n", psGradientComputeTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv model update time:\t%.3f/%.3f secs.\n",
						psLocalModelUpdateTime.doubleValue() / 1000, psAggregationTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv model broadcast time:\t%.3f secs.\n", psModelBroadcastTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv batch slice time:\t%.3f secs.\n", psBatchIndexTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv RPC request time:\t%.3f secs.\n", psRpcRequestTime.doubleValue() / 1000));
			}
			if( parforOptCount>0 ){
				sb.append("ParFor loops optimized:\t\t" + getParforOptCount() + ".\n");
				sb.append("ParFor optimize time:\t\t" + String.format("%.3f", ((double)getParforOptTime())/1000) + " sec.\n");
				sb.append("ParFor initialize time:\t\t" + String.format("%.3f", ((double)getParforInitTime())/1000) + " sec.\n");
				sb.append("ParFor result merge time:\t" + String.format("%.3f", ((double)getParforMergeTime())/1000) + " sec.\n");
				sb.append("ParFor total update in-place:\t" + lTotalUIPVar + "/" + lTotalLixUIP + "/" + lTotalLix + "\n");
			}

			sb.append("Total JIT compile time:\t\t" + ((double)getJITCompileTime())/1000 + " sec.\n");
			sb.append("Total JVM GC count:\t\t" + getJVMgcCount() + ".\n");
			sb.append("Total JVM GC time:\t\t" + ((double)getJVMgcTime())/1000 + " sec.\n");
			sb.append("Heavy hitter instructions:\n" + getHeavyHitters(maxHeavyHitters));
		}

		return sb.toString();
	}
}
