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
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.lineage.LineageCacheStatistics;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.utils.stats.CodegenStatistics;
import org.apache.sysds.utils.stats.NGramBuilder;
import org.apache.sysds.utils.stats.NativeStatistics;
import org.apache.sysds.utils.stats.ParForStatistics;
import org.apache.sysds.utils.stats.ParamServStatistics;
import org.apache.sysds.utils.stats.RecompileStatistics;
import org.apache.sysds.utils.stats.SparkStatistics;
import org.apache.sysds.utils.stats.TransformStatistics;

import java.lang.management.CompilationMXBean;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.text.DecimalFormat;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Consumer;

/**
 * This class captures all statistics.
 */
public class Statistics 
{
	private static class InstStats {
		private final LongAdder time = new LongAdder();
		private final LongAdder count = new LongAdder();
	}

	public static class NGramStats {

		public final long n;
		public final long cumTimeNanos;
		public final double m2;
		public final HashMap<String, Double> meta;

		public static <T> Comparator<NGramBuilder.NGramEntry<T, NGramStats>> getComparator() {
			return Comparator.comparingLong(entry -> entry.getCumStats().cumTimeNanos);
		}

		public static NGramStats merge(NGramStats stats1, NGramStats stats2) {
			// Using the algorithm from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
			long newN = stats1.n + stats2.n;
			long cumTimeNanos = stats1.cumTimeNanos + stats2.cumTimeNanos;

			// Ensure the calculation uses floating-point arithmetic
			double mean1 = (double) stats1.cumTimeNanos / 1000000000d / stats1.n;
			double mean2 = (double) stats2.cumTimeNanos / 1000000000d / stats2.n;
			double delta = mean2 - mean1;

			double newM2 = stats1.m2 + stats2.m2 + delta * delta * stats1.n * stats2.n / (double)newN;

			HashMap<String, Double> cpy = null;

			if (stats1.meta != null) {
				cpy = new HashMap<>(stats1.meta);
				final HashMap<String, Double> mCpy = cpy;
				if (stats2.meta != null)
					stats2.meta.forEach((key, value) -> mCpy.merge(key, value, Double::sum));
			} else if (stats2.meta != null) {
				cpy = new HashMap<>(stats2.meta);
			}

			return new NGramStats(newN, cumTimeNanos, newM2, cpy);
		}

		public NGramStats(final long n, final long cumTimeNanos, final double m2, HashMap<String, Double> meta) {
			this.n = n;
			this.cumTimeNanos = cumTimeNanos;
			this.m2 = m2;
			this.meta = meta;
		}

		public double getTimeVariance() {
			return m2 / Math.max(n-1, 1);
		}

		public String toString() {
			return String.format(Locale.US, "%.5f", (cumTimeNanos / 1000000000d));
		}

		public HashMap<String, Double> getMeta() {
			return meta;
		}
	}

	public static class LineageNGramExtension {
		private String _datatype;
		private String _valuetype;
		private long _execNanos;

		private HashMap<String, Double> _meta;

		public void setDataType(String dataType) {
			_datatype = dataType;
		}

		public String getDataType() {
			return _datatype == null ? "" : _datatype;
		}

		public void setValueType(String valueType) {
			_valuetype = valueType;
		}

		public String getValueType() {
			return _valuetype == null ? "" : _valuetype;
		}

		public void setExecNanos(long nanos) {
			_execNanos = nanos;
		}

		public long getExecNanos() {
			return _execNanos;
		}

		public void setMeta(String key, Double value) {
			if (_meta == null)
				_meta = new HashMap<>();
			_meta.put(key, value);
		}

		public Object getMeta(String key) {
			if (_meta == null)
				return null;
			return _meta.get(key);
		}
	}
	
	private static long compileStartTime = 0;
	private static long compileEndTime = 0;
	private static long execStartTime = 0;
	private static long execEndTime = 0;
	
	//heavy hitter counts and times 
	private static final ConcurrentHashMap<String,InstStats> _instStats = new ConcurrentHashMap<>();
	private static final ConcurrentHashMap<String, NGramBuilder<String, NGramStats>[]> _instStatsNGram = new ConcurrentHashMap<>();
	private static final ConcurrentHashMap<Long, Entry<String, LineageItem>> _instStatsLineageTracker = new ConcurrentHashMap<>();
	private static final ConcurrentHashMap<LineageItem, LineageNGramExtension> _lineageExtensions = new ConcurrentHashMap<>();

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

	// Out-of-core eviction metrics
	private static final ConcurrentHashMap<String, LongAdder> oocHeavyHitters = new  ConcurrentHashMap<>();
	private static final LongAdder oocGetCalls = new LongAdder();
	private static final LongAdder oocPutCalls = new LongAdder();
	private static final LongAdder oocLoadFromDiskCalls = new LongAdder();
	private static final LongAdder oocLoadFromDiskTimeNanos = new LongAdder();
	private static final LongAdder oocEvictionWriteCalls = new LongAdder();
	private static final LongAdder oocEvictionWriteTimeNanos = new LongAdder();
	private static final AtomicLong oocStatsStartTime = new AtomicLong(System.nanoTime());

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

	public static void resetOOCEvictionStats() {
		oocHeavyHitters.clear();
		oocGetCalls.reset();
		oocPutCalls.reset();
		oocLoadFromDiskCalls.reset();
		oocLoadFromDiskTimeNanos.reset();
		oocEvictionWriteCalls.reset();
		oocEvictionWriteTimeNanos.reset();
		oocStatsStartTime.set(System.nanoTime());
	}

	public static String getOOCHeavyHitters(int num) {
		if (num <= 0 || oocHeavyHitters == null || oocHeavyHitters.isEmpty())
			return "-";

		@SuppressWarnings("unchecked")
		Map.Entry<String, LongAdder>[] tmp =
			oocHeavyHitters.entrySet().toArray(new Map.Entry[0]);

		Arrays.sort(tmp, (e1, e2) ->
			Long.compare(e1.getValue().longValue(), e2.getValue().longValue())
		);

		final String numCol   = "#";
		final String instCol  = "Instruction";
		final String timeCol  = "Time(s)";

		DecimalFormat sFormat = new DecimalFormat("#,##0.000");

		StringBuilder sb = new StringBuilder();
		int len = tmp.length;
		int numHittersToDisplay = Math.min(num, len);

		int maxNumLen  = String.valueOf(numHittersToDisplay).length();
		int maxInstLen = instCol.length();
		int maxTimeLen = timeCol.length();

		// first pass: compute column widths
		for (int i = 0; i < numHittersToDisplay; i++) {
			Map.Entry<String, LongAdder> hh = tmp[len - 1 - i];

			String instruction = hh.getKey();
			double timeS = hh.getValue().longValue() / 1_000_000_000d;
			String timeStr = sFormat.format(timeS);

			maxInstLen = Math.max(maxInstLen, instruction.length());
			maxTimeLen = Math.max(maxTimeLen, timeStr.length());
		}

		maxInstLen = Math.min(maxInstLen, DMLScript.STATISTICS_MAX_WRAP_LEN);

		// header
		sb.append(String.format(
			" %" + maxNumLen + "s  %-" + maxInstLen + "s  %" + maxTimeLen + "s",
			numCol, instCol, timeCol));
		sb.append("\n");

		// rows
		for (int i = 0; i < numHittersToDisplay; i++) {
			Map.Entry<String, LongAdder> hh = tmp[len - 1 - i];

			String instruction = hh.getKey();
			double timeS = hh.getValue().longValue() / 1_000_000_000d;
			String timeStr = sFormat.format(timeS);

			String[] wrappedInstruction = wrap(instruction, maxInstLen);

			for (int w = 0; w < wrappedInstruction.length; w++) {
				if (w == 0) {
					sb.append(String.format(
						" %" + maxNumLen + "d  %-" + maxInstLen + "s  %"
							+ maxTimeLen + "s",
						(i + 1), wrappedInstruction[w], timeStr));
				} else {
					sb.append(String.format(
						" %" + maxNumLen + "s  %-" + maxInstLen + "s  %"
							+ maxTimeLen + "s",
						"", wrappedInstruction[w], ""));
				}
				sb.append("\n");
			}
		}

		return sb.toString();
	}

	public static void maintainOOCHeavyHitter(String op, long timeNanos) {
		LongAdder adder = oocHeavyHitters.computeIfAbsent(op, k -> new LongAdder());
		adder.add(timeNanos);
	}

	public static void incrementOOCEvictionGet() {
		oocGetCalls.increment();
	}

	public static void incrementOOCEvictionGet(int incr) {
		oocGetCalls.add(incr);
	}

	public static void incrementOOCEvictionPut() {
		oocPutCalls.increment();
	}

	public static void incrementOOCLoadFromDisk() {
		oocLoadFromDiskCalls.increment();
	}

	public static void incrementOOCEvictionWrite() {
		oocEvictionWriteCalls.increment();
	}

	public static void accumulateOOCLoadFromDiskTime(long nanos) {
		oocLoadFromDiskTimeNanos.add(nanos);
	}

	public static void accumulateOOCEvictionWriteTime(long nanos) {
		oocEvictionWriteTimeNanos.add(nanos);
	}

	public static String displayOOCEvictionStats() {
		long elapsedNanos = Math.max(1, System.nanoTime() - oocStatsStartTime.get());
		double elapsedSeconds = elapsedNanos / 1e9;
		double getThroughput = oocGetCalls.longValue() / elapsedSeconds;
		double putThroughput = oocPutCalls.longValue() / elapsedSeconds;

		StringBuilder sb = new StringBuilder();
		sb.append("OOC heavy hitters:\n");
		sb.append(getOOCHeavyHitters(DMLScript.OOC_STATISTICS_COUNT));
		sb.append('\n');
		sb.append(String.format(Locale.US, "  get calls:\t\t%d (%.2f/sec)\n",
			oocGetCalls.longValue(), getThroughput));
		sb.append(String.format(Locale.US, "  put calls:\t\t%d (%.2f/sec)\n",
			oocPutCalls.longValue(), putThroughput));
		sb.append(String.format(Locale.US, "  loadFromDisk:\t\t%d (time %.3f sec)\n",
			oocLoadFromDiskCalls.longValue(), oocLoadFromDiskTimeNanos.longValue() / 1e9));
		sb.append(String.format(Locale.US, "  evict writes:\t\t%d (time %.3f sec)\n",
			oocEvictionWriteCalls.longValue(), oocEvictionWriteTimeNanos.longValue() / 1e9));
		return sb.toString();
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
		resetOOCEvictionStats();
		
		resetJITCompileTime();
		resetJVMgcTime();
		resetJVMgcCount();
		resetCPHeavyHitters();

		GPUStatistics.reset();
		NativeStatistics.reset();
		DMLCompressionStatistics.reset();

		FederatedStatistics.reset();

		_instStatsNGram.clear();
		_instStatsLineageTracker.clear();
		_instStats.clear();
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

	public static void prepareNGramInst(Entry<String, LineageItem> li) {
		if (li == null)
			_instStatsLineageTracker.remove(Thread.currentThread().getId());
		else
			_instStatsLineageTracker.put(Thread.currentThread().getId(), li);
	}

	public static Optional<Entry<String, LineageItem>> getCurrentLineageItem() {
		Entry<String, LineageItem> item = _instStatsLineageTracker.get(Thread.currentThread().getId());
		return item == null ? Optional.empty() : Optional.of(item);
	}

	public static synchronized void clearNGramRecording() {
		NGramBuilder<String, NGramStats>[] bl = _instStatsNGram.get(Thread.currentThread().getName());
		for (NGramBuilder<String, NGramStats> b : bl)
			b.clearCurrentRecording();
	}

	public static synchronized void extendLineageItem(LineageItem li, LineageNGramExtension ext) {
		_lineageExtensions.put(li, ext);
	}

	public static synchronized LineageNGramExtension getExtendedLineage(LineageItem li) {
		return _lineageExtensions.get(li);
	}
	
	public static synchronized void maintainNGramsFromLineage(Instruction tmp, ExecutionContext ec, long t0) {
		final long nanoTime = System.nanoTime() - t0;
		if (DMLScript.STATISTICS_NGRAMS_USE_LINEAGE) {
			Statistics.getCurrentLineageItem().ifPresent(li -> {
				Data data = ec.getVariable(li.getKey());
				Statistics.LineageNGramExtension ext = new Statistics.LineageNGramExtension();
				if (data != null) {
					ext.setDataType(data.getDataType().toString());
					ext.setValueType(data.getValueType().toString());
					if (data instanceof CacheableData) {
						DataCharacteristics dc = ((CacheableData<?>)data).getDataCharacteristics();
						ext.setMeta("NDims", (double)dc.getNumDims());
						ext.setMeta("NumRows", (double)dc.getRows());
						ext.setMeta("NumCols", (double)dc.getCols());
						ext.setMeta("NonZeros", (double)dc.getNonZeros());
					}
				}
				ext.setExecNanos(nanoTime);
				Statistics.extendLineageItem(li.getValue(), ext);
				Statistics.maintainNGramsFromLineage(li.getValue());
			});
		} else
			Statistics.maintainNGrams(tmp.getExtendedOpcode(), nanoTime);
	}

	@SuppressWarnings("unchecked")
	public static synchronized void maintainNGramsFromLineage(LineageItem li) {
		NGramBuilder<String, NGramStats>[] tmp = _instStatsNGram.computeIfAbsent(Thread.currentThread().getName(), k -> {
			NGramBuilder<String, NGramStats>[] threadEntry = new NGramBuilder[DMLScript.STATISTICS_NGRAM_SIZES.length];
			for (int i = 0; i < threadEntry.length; i++) {
				threadEntry[i] = new NGramBuilder<String, NGramStats>(String.class, NGramStats.class, DMLScript.STATISTICS_NGRAM_SIZES[i], s -> s, NGramStats::merge);
			}
			return threadEntry;
		});
		addLineagePaths(li, new ArrayList<>(), new ArrayList<>(), tmp);
	}

	/**
	 * Adds the corresponding sequences of instructions to the n-grams.
	 * <p></p>
	 * Example: 2-grams from (a*b + a/c) will add [(*,+), (/,+)]
	 * @param li
	 * @param currentPath
	 * @param indexes
	 * @param builders
	 */
	private static void addLineagePaths(LineageItem li, ArrayList<Entry<LineageItem, LineageNGramExtension>> currentPath, ArrayList<Integer> indexes, NGramBuilder<String, NGramStats>[] builders) {
		if (li.getType() == LineageItem.LineageItemType.Literal)
			return; // Skip literals as they are no real instruction

		currentPath.add(new AbstractMap.SimpleEntry<>(li, getExtendedLineage(li)));

		int maxSize = 0;
		NGramBuilder<String, NGramStats> matchingBuilder = null;

		for (NGramBuilder<String, NGramStats> builder : builders) {
			if (builder.getSize() == currentPath.size())
				matchingBuilder = builder;
			if (builder.getSize() > maxSize)
				maxSize = builder.getSize();
		}

		if (matchingBuilder != null) {
			// If we have an n-gram builder with n = currentPath.size(), then we want to insert the entry
			// As we cannot incrementally add the instructions (we have a DAG rather than a sequence of instructions)
			// we need to clear the current n-grams
			clearNGramRecording();
			// We then record a new n-gram with all the LineageItems of the current lineage path
			Entry<LineageItem, LineageNGramExtension> currentEntry = currentPath.get(currentPath.size()-1);
			matchingBuilder.append(LineageItemUtils.explainLineageAsInstruction(currentEntry.getKey(), currentEntry.getValue()) + (indexes.size() > 0 ? ("[" + indexes.get(currentPath.size()-2) + "]") : ""), new NGramStats(1, currentEntry.getValue() != null ? currentEntry.getValue().getExecNanos() : 0, 0, currentEntry.getValue() != null ? currentEntry.getValue()._meta : null));
			for (int i = currentPath.size()-2; i >= 0; i--) {
				currentEntry = currentPath.get(i);
				matchingBuilder.append(LineageItemUtils.explainLineageAsInstruction(currentEntry.getKey(), currentEntry.getValue()) + (i > 0 ? ("[" + indexes.get(i-1) + "]") : ""), new NGramStats(1, currentEntry.getValue() != null ? currentEntry.getValue().getExecNanos() : 0, 0, currentEntry.getValue() != null ? currentEntry.getValue()._meta : null));
			}
		}

		if (currentPath.size() < maxSize && li.getInputs() != null) {
			int idx = 0;
			for (LineageItem input : li.getInputs()) {
				indexes.add(idx++);
				addLineagePaths(input, currentPath, indexes, builders);
				indexes.remove(indexes.size()-1);
			}
		}

		currentPath.remove(currentPath.size()-1);
	}

	@SuppressWarnings("unchecked")
	public static void maintainNGrams(String instName, long timeNanos) {
		NGramBuilder<String, NGramStats>[] tmp = _instStatsNGram.computeIfAbsent(Thread.currentThread().getName(), k -> {
			NGramBuilder<String, NGramStats>[] threadEntry = new NGramBuilder[DMLScript.STATISTICS_NGRAM_SIZES.length];
			for (int i = 0; i < threadEntry.length; i++) {
				threadEntry[i] = new NGramBuilder<String, NGramStats>(String.class, NGramStats.class, DMLScript.STATISTICS_NGRAM_SIZES[i], s -> s, NGramStats::merge);
			}
			return threadEntry;
		});

		for (int i = 0; i < tmp.length; i++)
			tmp[i].append(instName, new NGramStats(1, timeNanos, 0, null));
	}

	@SuppressWarnings("unchecked")
	public static NGramBuilder<String, NGramStats>[] mergeNGrams() {
		NGramBuilder<String, NGramStats>[] builders = new NGramBuilder[DMLScript.STATISTICS_NGRAM_SIZES.length];

		for (int i = 0; i < builders.length; i++) {
			builders[i] = new NGramBuilder<String, NGramStats>(String.class, NGramStats.class, DMLScript.STATISTICS_NGRAM_SIZES[i], s -> s, NGramStats::merge);
		}

		for (int i = 0; i < DMLScript.STATISTICS_NGRAM_SIZES.length; i++) {
			for (Map.Entry<String, NGramBuilder<String, NGramStats>[]> entry : _instStatsNGram.entrySet()) {
				NGramBuilder<String, NGramStats> mbuilder = entry.getValue()[i];
				builders[i].merge(mbuilder);
			}
		}

		return builders;
	}

	public static String getNGramStdDevs(NGramStats[] stats, int offset, int prec, boolean displayZero) {
		StringBuilder sb = new StringBuilder();
		sb.append("(");
		boolean containsData = false;
		int actualIndex;
		for (int i = 0; i < stats.length; i++) {
			if (i != 0)
				sb.append(", ");
			actualIndex = (offset + i) % stats.length;
			double var = 1000000000d * stats[actualIndex].n * Math.sqrt(stats[actualIndex].getTimeVariance()) / stats[actualIndex].cumTimeNanos;
			if (displayZero || var >= Math.pow(10, -prec)) {
				sb.append(String.format(Locale.US, "%." + prec + "f", var));
				containsData = true;
			}
		}
		sb.append(")");
		return containsData ? sb.toString() : "-";
	}

	public static String getNGramAvgTimes(NGramStats[] stats, int offset, int prec) {
		StringBuilder sb = new StringBuilder();
		sb.append("(");
		int actualIndex;
		for (int i = 0; i < stats.length; i++) {
			if (i != 0)
				sb.append(", ");
			actualIndex = (offset + i) % stats.length;
			double var = (stats[actualIndex].cumTimeNanos / 1000000000d) / stats[actualIndex].n;
			sb.append(String.format(Locale.US, "%." + prec + "f", var));
		}
		sb.append(")");
		return sb.toString();
	}

	public static void toCSVStream(final NGramBuilder<String, NGramStats> mbuilder, final Consumer<String> lineConsumer) {
		ArrayList<String> colList = new ArrayList<>();
		colList.add("N-Gram");
		colList.add("Time[s]");

		for (int j = 0; j < mbuilder.getSize(); j++)
			colList.add("Col" + (j + 1));
		for (int j = 0; j < mbuilder.getSize(); j++)
			colList.add("Col" + (j + 1) + "::Mean(Time[s])");
		for (int j = 0; j < mbuilder.getSize(); j++)
			colList.add("Col" + (j + 1) + "::StdDev(Time[s])/Col" + (j + 1) + "::Mean(Time[s])");
		for (int j = 0; j < mbuilder.getSize(); j++)
			colList.add("Col" + (j + 1) + "_Meta");

		colList.add("Count");

		NGramBuilder.toCSVStream(colList.toArray(new String[colList.size()]), mbuilder.getTopK(100000, Statistics.NGramStats.getComparator(), true), e -> {
			StringBuilder builder = new StringBuilder();
			builder.append(e.getIdentifier().replace("(", "").replace(")", "").replace(", ", ","));
			builder.append(",");
			builder.append(Statistics.getNGramAvgTimes(e.getStats(), e.getOffset(), 9).replace("-", "").replace("(", "").replace(")", ""));
			builder.append(",");
			String stdDevs = Statistics.getNGramStdDevs(e.getStats(), e.getOffset(), 9, true).replace("-", "").replace("(", "").replace(")", "");
			if (stdDevs.isEmpty()) {
				for (int j = 0; j < mbuilder.getSize()-1; j++)
					builder.append(",");
			} else {
				builder.append(stdDevs);
			}
			//builder.append(",");
			boolean first = true;
			NGramStats[] stats = e.getStats();
			for (int i = 0; i < stats.length; i++) {
				builder.append(",");
				NGramStats stat = stats[i];
				if (stat.getMeta() != null) {
					for (Entry<String, Double> metaData : stat.getMeta().entrySet()) {
						if (first)
							first = false;
						else
							builder.append("&");
						if (metaData.getValue() != null)
							builder.append(metaData.getKey()).append(":").append(metaData.getValue());
					}
				}
			}
			return builder.toString();
		}, lineConsumer);
	}

	public static String nGramToCSV(final NGramBuilder<String, NGramStats> mbuilder) {
		final StringBuilder b = new StringBuilder();
		toCSVStream(mbuilder, b::append);
		return b.toString();
	}

	public static String getCommonNGrams(NGramBuilder<String, NGramStats> builder, int num) {
		if (num <= 0 || _instStatsNGram.size() <= 0)
			return "-";

		//NGramBuilder<String, Long> builder = mergeNGrams();
		@SuppressWarnings("unchecked")
		NGramBuilder.NGramEntry<String, NGramStats>[] topNGrams = builder.getTopK(num, NGramStats.getComparator(), true).toArray(NGramBuilder.NGramEntry[]::new);

		final String numCol = "#";
		final String instCol = "N-Gram";
		final String timeSCol = "Time(s)";
		final String timeSVar = "StdDev(t)/Mean(t)";
		final String countCol = "Count";
		StringBuilder sb = new StringBuilder();
		int len = topNGrams.length;
		int numHittersToDisplay = Math.min(num, len);
		int maxNumLen = String.valueOf(numHittersToDisplay).length();
		int maxInstLen = instCol.length();
		int maxTimeSLen = timeSCol.length();
		int maxTimeSVarLen = timeSVar.length();
		int maxCountLen = countCol.length();
		DecimalFormat sFormat = new DecimalFormat("#,##0.000");

		for (int i = 0; i < numHittersToDisplay; i++) {
			long timeNs = topNGrams[i].getCumStats().cumTimeNanos;
			String instruction = topNGrams[i].getIdentifier();
			double timeS = timeNs / 1000000000d;


			maxInstLen = Math.max(maxInstLen, instruction.length() + 1);

			String timeSString = sFormat.format(timeS);
			String timeSVarString = getNGramStdDevs(topNGrams[i].getStats(), topNGrams[i].getOffset(), 3, false);
			maxTimeSLen = Math.max(maxTimeSLen, timeSString.length());
			maxTimeSVarLen = Math.max(maxTimeSVarLen, timeSVarString.length());

			maxCountLen = Math.max(maxCountLen, String.valueOf(topNGrams[i].getOccurrences()).length());
		}

		maxInstLen = Math.min(maxInstLen, DMLScript.STATISTICS_MAX_WRAP_LEN);
		sb.append(String.format( " %" + maxNumLen + "s  %-" + maxInstLen + "s  %"
				+ maxTimeSLen + "s  %" + maxTimeSVarLen + "s  %" + maxCountLen + "s", numCol, instCol, timeSCol, timeSVar, countCol));
		sb.append("\n");
		for (int i = 0; i < numHittersToDisplay; i++) {
			String instruction = topNGrams[i].getIdentifier();
			String [] wrappedInstruction = wrap(instruction, maxInstLen);

			//long timeNs = tmp[len - 1 - i].getValue().time.longValue();
			double timeS = topNGrams[i].getCumStats().cumTimeNanos / 1000000000d;
			//double timeVar = topNGrams[i].getCumStats().getTimeVariance();
			String timeSString = sFormat.format(timeS);
			String timeVarString = getNGramStdDevs(topNGrams[i].getStats(), topNGrams[i].getOffset(), 3, false);//sFormat.format(timeVar);

			long count = topNGrams[i].getOccurrences();
			int numLines = wrappedInstruction.length;

			for(int wrapIter = 0; wrapIter < numLines; wrapIter++) {
				String instStr = (wrapIter < wrappedInstruction.length) ? wrappedInstruction[wrapIter] : "";
				if(wrapIter == 0) {
					// Display instruction count
					sb.append(String.format(
							" %" + maxNumLen + "d  %-" + maxInstLen + "s  %" + maxTimeSLen + "s %" + maxTimeSVarLen + "s  %" + maxCountLen + "d",
							(i + 1), instStr, timeSString, timeVarString, count));
				}
				else {
					sb.append(String.format(
							" %" + maxNumLen + "s  %-" + maxInstLen + "s  %" + maxTimeSLen + "s %" + maxTimeSVarLen + "s  %" + maxCountLen + "s",
							"", instStr, "", "", ""));
				}
				sb.append("\n");
			}
		}

		return sb.toString();
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
				if (LineageCacheStatistics.ifGpuStats()) {
					sb.append("LinCache GPU (Hit/PF): \t" + LineageCacheStatistics.displayGpuStats() + ".\n");
					sb.append("LinCache GPU (Recyc/Del/Miss): \t" + LineageCacheStatistics.displayGpuPointerStats() + ".\n");
					sb.append("LinCache GPU evict time: \t" + LineageCacheStatistics.displayGpuEvictTime() + " sec.\n");
				}
				if (LineageCacheStatistics.ifSparkStats()) {
					sb.append("LinCache Spark (Col/Loc/Dist): \t" + LineageCacheStatistics.displaySparkHits() + ".\n");
					sb.append("LinCache Spark (Per/Unper/Del):\t" + LineageCacheStatistics.displaySparkPersist() + ".\n");
				}
				sb.append("LinCache writes (Mem/FS/Del): \t" + LineageCacheStatistics.displayWtrites() + ".\n");
				sb.append("LinCache FStimes (Rd/Wr): \t" + LineageCacheStatistics.displayFSTime() + " sec.\n");
				sb.append("LinCache Computetime (S/M/P): \t" + LineageCacheStatistics.displayComputeTime() + " sec.\n");
				sb.append("LinCache Rewrites:    \t\t" + LineageCacheStatistics.displayRewrites() + ".\n");
			}

			if( ConfigurationManager.isCodegenEnabled() )
				sb.append(CodegenStatistics.displayStatistics());

			if( OptimizerUtils.isSparkExecutionMode() )
				sb.append(SparkStatistics.displayStatistics());
			if (SparkStatistics.anyAsyncOp())
				sb.append(SparkStatistics.displayAsyncStats());

			sb.append(ParamServStatistics.displayStatistics());

			sb.append(ParForStatistics.displayStatistics());

			sb.append(FederatedStatistics.displayFedIOExecStatistics());
			sb.append(FederatedStatistics.displayFedWorkerStats());

			sb.append(TransformStatistics.displayStatistics());

			if(ConfigurationManager.isCompressionEnabled() || DMLCompressionStatistics.getDecompressionCount() > 0){
				DMLCompressionStatistics.display(sb);
			}

			sb.append("Total JIT compile time:\t\t" + ((double)getJITCompileTime())/1000 + " sec.\n");
			sb.append("Total JVM GC count:\t\t" + getJVMgcCount() + ".\n");
			sb.append("Total JVM GC time:\t\t" + ((double)getJVMgcTime())/1000 + " sec.\n");
			sb.append("Heavy hitter instructions:\n" + getHeavyHitters(maxHeavyHitters));
		}

		if (DMLScript.STATISTICS_NGRAMS) {
			NGramBuilder<String, NGramStats>[] mergedNGrams = mergeNGrams();
			for (int i = 0; i < DMLScript.STATISTICS_NGRAM_SIZES.length; i++) {
				sb.append("Most common " + DMLScript.STATISTICS_NGRAM_SIZES[i] + "-grams (sorted by absolute time):\n" + getCommonNGrams(mergedNGrams[i], DMLScript.STATISTICS_TOP_K_NGRAMS));
			}
		}

		if(DMLScript.FED_STATISTICS) {
			sb.append("\n");
			sb.append(FederatedStatistics.displayStatistics(DMLScript.FED_STATISTICS_COUNT));
			sb.append("\n");
			sb.append(ParamServStatistics.displayFloStatistics());
		}

		if (DMLScript.OOC_STATISTICS) {
			sb.append('\n');
			sb.append(displayOOCEvictionStats());
		}

		return sb.toString();
	}
}
