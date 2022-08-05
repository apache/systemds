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

package org.apache.sysds.runtime.controlprogram.federated;

import java.io.Serializable;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.ThreadMXBean;
import java.net.InetSocketAddress;
import java.text.DecimalFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.LongAdder;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics.FedStatsCollection.CacheStatsCollection;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics.FedStatsCollection.GCStatsCollection;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics.FedStatsCollection.LineageCacheStatsCollection;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics.FedStatsCollection.MultiTenantStatsCollection;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.lineage.LineageCacheStatistics;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.utils.Statistics;

public class FederatedStatistics {
	// stats of the federated worker on the coordinator site
	private static Set<Pair<String, Integer>> _fedWorkerAddresses = new HashSet<>();
	private static final LongAdder readCount = new LongAdder();
	private static final LongAdder putCount = new LongAdder();
	private static final LongAdder getCount = new LongAdder();
	private static final LongAdder executeInstructionCount = new LongAdder();
	private static final LongAdder executeUDFCount = new LongAdder();
	private static final LongAdder transferredScalarCount = new LongAdder();
	private static final LongAdder transferredListCount = new LongAdder();
	private static final LongAdder transferredMatrixCount = new LongAdder();
	private static final LongAdder transferredFrameCount = new LongAdder();
	private static final LongAdder transferredMatCharCount = new LongAdder();
	private static final LongAdder transferredMatrixBytes = new LongAdder();
	private static final LongAdder transferredFrameBytes = new LongAdder();
	private static final LongAdder asyncPrefetchCount = new LongAdder();
	private static final LongAdder bytesSent = new LongAdder();
	private static final LongAdder bytesReceived = new LongAdder();

	// stats on the federated worker itself
	private static final LongAdder fedLookupTableGetCount = new LongAdder();
	private static final LongAdder fedLookupTableGetTime = new LongAdder(); // in milli sec
	private static final LongAdder fedLookupTableEntryCount = new LongAdder();
	private static final LongAdder fedReuseReadHitCount = new LongAdder();
	private static final LongAdder fedReuseReadBytesCount = new LongAdder();
	private static final LongAdder fedBytesSent = new LongAdder();
	private static final LongAdder fedBytesReceived = new LongAdder();

	private static final LongAdder fedPutLineageCount = new LongAdder();
	private static final LongAdder fedPutLineageItems = new LongAdder();
	private static final LongAdder fedSerializationReuseCount = new LongAdder();
	private static final LongAdder fedSerializationReuseBytes = new LongAdder();
	// Traffic between federated worker and a coordinator site
	// in the form of [{ datetime, coordinatorAddress, transferredBytes }, { ... }] }
	private static CopyOnWriteArrayList<Triple<LocalDateTime, String, Long>> coordinatorsTrafficBytes = new CopyOnWriteArrayList<>();

	public static void logServerTraffic(long read, long written) {
		bytesReceived.add(read);
		bytesSent.add(written);
	}

	public static void logWorkerTraffic(long read, long written) {
		fedBytesReceived.add(read);
		fedBytesSent.add(written);
	}


	public static synchronized void incFederated(RequestType rqt, List<Object> data){
		switch (rqt) {
			case READ_VAR:
				readCount.increment();
				break;
			case PUT_VAR:
				putCount.increment();
				incFedTransfer(data.get(0));
				break;
			case GET_VAR:
				getCount.increment();
				break;
			case EXEC_INST:
				executeInstructionCount.increment();
				break;
			case EXEC_UDF:
				executeUDFCount.increment();
				incFedTransfer(data);
				break;
			default:
				break;
		}
	}

	private static void incFedTransfer(List<Object> data) {
		for(Object dataObj : data)
			incFedTransfer(dataObj);
	}

	private static void incFedTransfer(Object dataObj) {
		incFedTransfer(dataObj, null);
	}

	public static void incFedTransfer(Object dataObj, String host) {
		long byteAmount = 0;
		if(dataObj instanceof MatrixBlock) {
			transferredMatrixCount.increment();
			byteAmount = ((MatrixBlock)dataObj).getInMemorySize();
			transferredMatrixBytes.add(byteAmount);
		}
		else if(dataObj instanceof FrameBlock) {
			transferredFrameCount.increment();
			byteAmount = ((FrameBlock)dataObj).getInMemorySize();
			transferredFrameBytes.add(byteAmount);
		}
		else if(dataObj instanceof ScalarObject)
			transferredScalarCount.increment();
		else if(dataObj instanceof ListObject)
			transferredListCount.increment();
		else if(dataObj instanceof MatrixCharacteristics)
			transferredMatCharCount.increment();

		if (host != null && byteAmount > 0) {
			coordinatorsTrafficBytes.add(new ImmutableTriple<>(LocalDateTime.now(), host, byteAmount));
		}
	}

	public static void incAsyncPrefetchCount(long c) {
		asyncPrefetchCount.add(c);
	}

	public static long getTotalFedTransferCount() {
		return transferredScalarCount.longValue() + transferredListCount.longValue()
			+ transferredMatrixCount.longValue() + transferredFrameCount.longValue()
			+ transferredMatCharCount.longValue();
	}

	public static void reset() {
		readCount.reset();
		putCount.reset();
		getCount.reset();
		executeInstructionCount.reset();
		executeUDFCount.reset();
		transferredScalarCount.reset();
		transferredListCount.reset();
		transferredMatrixCount.reset();
		transferredFrameCount.reset();
		transferredMatCharCount.reset();
		transferredMatrixBytes.reset();
		transferredFrameBytes.reset();
		asyncPrefetchCount.reset();
		fedLookupTableGetCount.reset();
		fedLookupTableGetTime.reset();
		fedLookupTableEntryCount.reset();
		fedReuseReadHitCount.reset();
		fedReuseReadBytesCount.reset();
		fedPutLineageCount.reset();
		fedPutLineageItems.reset();
		fedSerializationReuseCount.reset();
		fedSerializationReuseBytes.reset();
		bytesSent.reset();
		bytesReceived.reset();
		fedBytesSent.reset();
		fedBytesReceived.reset();
		//TODO merge with existing
		coordinatorsTrafficBytes.clear();
	}

	public static String displayFedIOExecStatistics() {
		if( readCount.longValue() > 0){ // only if there happened something on the federated worker
			StringBuilder sb = new StringBuilder();
			sb.append("Federated I/O (Read, Put, Get):\t" +
				readCount.longValue() + "/" +
				putCount.longValue() + "/" +
				getCount.longValue() + ".\n");
			sb.append("Federated Execute (Inst, UDF):\t" +
				executeInstructionCount.longValue() + "/" +
				executeUDFCount.longValue() + ".\n");
			if(getTotalFedTransferCount() > 0)
				sb.append("Fed Put Count (Sc/Li/Ma/Fr/MC):\t" +
					transferredScalarCount.longValue() + "/" +
					transferredListCount.longValue() + "/" +
					transferredMatrixCount.longValue() + "/" +
					transferredFrameCount.longValue() + "/" +
					transferredMatCharCount.longValue() + ".\n");
			if(transferredMatrixBytes.longValue() > 0 || transferredFrameBytes.longValue() > 0)
				sb.append("Fed Put Bytes (Mat/Frame):\t" +
					transferredMatrixBytes.longValue() + "/" +
					transferredFrameBytes.longValue() + " Bytes.\n");
			sb.append("Federated prefetch count:\t" +
				asyncPrefetchCount.longValue() + ".\n");
			return sb.toString();
		}
		return "";
	}

	public static String displayNetworkTrafficStatistics() {
		return "Server I/O bytes (read/written):\t" +
				bytesReceived.longValue() +
				"/" +
				bytesSent.longValue() +
				"\n" +
				"Worker I/O bytes (read/written):\t" +
				fedBytesReceived.longValue() +
				"/" +
				fedBytesSent.longValue() +
				"\n";
	}


	public static void registerFedWorker(String host, int port) {
		_fedWorkerAddresses.add(new ImmutablePair<>(host, Integer.valueOf(port)));
	}

	public static String displayFedWorkers() {
		StringBuilder sb = new StringBuilder();
		sb.append("Federated Worker Addresses:\n");
		for(Pair<String, Integer> fedAddr : _fedWorkerAddresses) {
			sb.append(String.format("  %s:%d", fedAddr.getLeft(), fedAddr.getRight().intValue()));
			sb.append("\n");
		}
		return sb.toString();
	}

	public static String displayFedWorkerStats() {
		StringBuilder sb = new StringBuilder();
		sb.append(displayFedLookupTableStats());
		sb.append(displayFedReuseReadStats());
		sb.append(displayFedPutLineageStats());
		sb.append(displayFedSerializationReuseStats());
		//FIXME: the following statistics need guards to only show
		// results if federated operations where executed, also the CPU
		// and mem usage only probe once at the time of stats printing
		//sb.append(displayFedTransfer());
		//sb.append(displayCPUUsage());
		//sb.append(displayMemoryUsage());
		return sb.toString();
	}

	public static String displayStatistics(int numHeavyHitters) {
		FedStatsCollection fedStats = collectFedStats();
		return displayStatistics(fedStats, numHeavyHitters);
	}

	public static String displayStatistics(FedStatsCollection fedStats, int numHeavyHitters) {
		StringBuilder sb = new StringBuilder();
		sb.append("SystemDS Federated Statistics:\n");
		sb.append(displayCacheStats(fedStats.cacheStats));
		sb.append(String.format("Total JIT compile time:\t\t%.3f sec.\n", fedStats.jitCompileTime));
		sb.append(displayGCStats(fedStats.gcStats));
		sb.append(displayLinCacheStats(fedStats.linCacheStats));
		sb.append(displayMultiTenantStats(fedStats.mtStats));
		sb.append(displayCPUUsage());
		sb.append(displayMemoryUsage());
		sb.append(displayFedTransfer());
		sb.append(displayHeavyHitters(fedStats.heavyHitters, numHeavyHitters));
		sb.append(displayNetworkTrafficStatistics());
		return sb.toString();
	}

	private static String displayCacheStats(CacheStatsCollection csc) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("Cache hits (Mem/Li/WB/FS/HDFS):\t%d/%d/%d/%d/%d.\n",
			csc.memHits, csc.linHits, csc.fsBuffHits, csc.fsHits, csc.hdfsHits));
		sb.append(String.format("Cache writes (Li/WB/FS/HDFS):\t%d/%d/%d/%d.\n",
			csc.linWrites, csc.fsBuffWrites, csc.fsWrites, csc.hdfsWrites));
		sb.append(String.format("Cache times (ACQr/m, RLS, EXP):\t%.3f/%.3f/%.3f/%.3f sec.\n",
			csc.acqRTime, csc.acqMTime, csc.rlsTime, csc.expTime));
		return sb.toString();
	}

	private static String displayGCStats(GCStatsCollection gcsc) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("Total JVM GC count:\t\t%d.\n", gcsc.gcCount));
		sb.append(String.format("Total JVM GC time:\t\t%.3f sec.\n", gcsc.gcTime));
		return sb.toString();
	}

	private static String displayLinCacheStats(LineageCacheStatsCollection lcsc) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("LinCache hits (Mem/FS/Del):\t%d/%d/%d.\n",
			lcsc.numHitsMem, lcsc.numHitsFS, lcsc.numHitsDel));
		sb.append(String.format("LinCache MultiLvl (Ins/SB/Fn):\t%d/%d/%d.\n",
			lcsc.numHitsInst, lcsc.numHitsSB, lcsc.numHitsFunc));
		sb.append(String.format("LinCache writes (Mem/FS/Del):\t%d/%d/%d.\n",
			lcsc.numWritesMem, lcsc.numWritesFS, lcsc.numMemDel));
		return sb.toString();
	}

	private static String displayMultiTenantStats(MultiTenantStatsCollection mtsc) {
		StringBuilder sb = new StringBuilder();
		sb.append(displayFedLookupTableStats(mtsc.fLTGetCount, mtsc.fLTEntryCount, mtsc.fLTGetTime));
		sb.append(displayFedReuseReadStats(mtsc.reuseReadHits, mtsc.reuseReadBytes));
		sb.append(displayFedPutLineageStats(mtsc.putLineageCount, mtsc.putLineageItems));
		sb.append(displayFedSerializationReuseStats(mtsc.serializationReuseCount, mtsc.serializationReuseBytes));
		return sb.toString();
	}

	@SuppressWarnings("unused")
	private static String displayHeavyHitters(HashMap<String, Pair<Long, Double>> heavyHitters) {
		return displayHeavyHitters(heavyHitters, 10);
	}

	private static String displayFedTransfer() {
		StringBuilder sb = new StringBuilder();
		sb.append("Transferred bytes (Host/Datetime/ByteAmount):\n");

		for (var entry: coordinatorsTrafficBytes) {
			sb.append(String.format("%s/%s/%d.\n",
					entry.getLeft().format(DateTimeFormatter.ISO_DATE_TIME), entry.getMiddle(), entry.getRight()));
		}

		return sb.toString();
	}

	private static String displayCPUUsage() {
		StringBuilder sb = new StringBuilder();

		double cpuUsage = getCPUUsage();

		sb.append(String.format("CPU usage %%: %.2f\n", cpuUsage));

		return sb.toString();
	}

	private static String displayMemoryUsage() {
		StringBuilder sb = new StringBuilder();

		double memoryUsage = getMemoryUsage();

		sb.append(String.format("Memory usage %%: %.2f\n", memoryUsage));

		return sb.toString();
	}

	private static String displayHeavyHitters(HashMap<String, Pair<Long, Double>> heavyHitters, int num) {
		StringBuilder sb = new StringBuilder();
		@SuppressWarnings("unchecked")
		Entry<String, Pair<Long, Double>>[] hhArr = heavyHitters.entrySet().toArray(new Entry[0]);
		Arrays.sort(hhArr, new Comparator<Entry<String, Pair<Long, Double>>>() {
			public int compare(Entry<String, Pair<Long, Double>> e1, Entry<String, Pair<Long, Double>> e2) {
				return e1.getValue().getRight().compareTo(e2.getValue().getRight());
			}
		});

		sb.append("Heavy hitter instructions:\n");
		final String numCol = "#";
		final String instCol = "Instruction";
		final String timeSCol = "Time(s)";
		final String countCol = "Count";
		int numHittersToDisplay = Math.min(num, hhArr.length);
		int maxNumLen = String.valueOf(numHittersToDisplay).length();
		int maxInstLen = instCol.length();
		int maxTimeSLen = timeSCol.length();
		int maxCountLen = countCol.length();
		DecimalFormat sFormat = new DecimalFormat("#,##0.000");
		for (int counter = 0; counter < numHittersToDisplay; counter++) {
			Entry<String, Pair<Long, Double>> hh = hhArr[hhArr.length - 1 - counter];
			String instruction = hh.getKey();
			maxInstLen = Math.max(maxInstLen, instruction.length());
			String timeString = sFormat.format(hh.getValue().getRight());
			maxTimeSLen = Math.max(maxTimeSLen, timeString.length());
			maxCountLen = Math.max(maxCountLen, String.valueOf(hh.getValue().getLeft()).length());
		}
		maxInstLen = Math.min(maxInstLen, DMLScript.STATISTICS_MAX_WRAP_LEN);
		sb.append(String.format( " %" + maxNumLen + "s  %-" + maxInstLen + "s  %"
			+ maxTimeSLen + "s  %" + maxCountLen + "s", numCol, instCol, timeSCol, countCol));
		sb.append("\n");

		for (int counter = 0; counter < numHittersToDisplay; counter++) {
			String instruction = hhArr[hhArr.length - 1 - counter].getKey();
			String [] wrappedInstruction = Statistics.wrap(instruction, maxInstLen);

			String timeSString = sFormat.format(hhArr[hhArr.length - 1 - counter].getValue().getRight());

			long count = hhArr[hhArr.length - 1 - counter].getValue().getLeft();
			int numLines = wrappedInstruction.length;
			
			for(int wrapIter = 0; wrapIter < numLines; wrapIter++) {
				String instStr = (wrapIter < wrappedInstruction.length) ? wrappedInstruction[wrapIter] : "";
				if(wrapIter == 0) {
					sb.append(String.format(
						" %" + maxNumLen + "d  %-" + maxInstLen + "s  %" + maxTimeSLen + "s  %" 
						+ maxCountLen + "d", (counter + 1), instStr, timeSString, count));
				}
				else {
					sb.append(String.format(
						" %" + maxNumLen + "s  %-" + maxInstLen + "s  %" + maxTimeSLen + "s  %" 
						+ maxCountLen + "s", "", instStr, "", ""));
				}
				sb.append("\n");
			}
		}

		return sb.toString();
	}

	private static FedStatsCollection collectFedStats() {
		Future<FederatedResponse>[] responses = getFederatedResponses();
		FedStatsCollection aggFedStats = new FedStatsCollection();
		for(Future<FederatedResponse> res : responses) {
			try {
				Object[] tmp = res.get().getData();
				if(tmp[0] instanceof FedStatsCollection)
					aggFedStats.aggregate((FedStatsCollection)tmp[0]);
			} catch(Exception e) {
				throw new DMLRuntimeException("Exception of type " + e.getClass().toString() 
					+ " thrown while " + "getting the federated stats of the federated response: ", e);
			}
		}
		return aggFedStats;
	}

	private static Future<FederatedResponse>[] getFederatedResponses() {
		List<Future<FederatedResponse>> ret = new ArrayList<>();
		for(Pair<String, Integer> fedAddr : _fedWorkerAddresses) {
			InetSocketAddress isa = new InetSocketAddress(fedAddr.getLeft(), fedAddr.getRight());
			FederatedRequest frUDF = new FederatedRequest(RequestType.EXEC_UDF, -1, 
				new FedStatsCollectFunction());
			try {
				ret.add(FederatedData.executeFederatedOperation(isa, frUDF));
			} catch(DMLRuntimeException dre) {
				// silently ignore this exception --> caused by offline federated workers
			} catch (Exception e) {
				System.out.println("Exeption of type " + e.getClass().getName() 
					+ " thrown while getting stats from federated worker: " + e.getMessage());
			}
		}
		@SuppressWarnings("unchecked")
		Future<FederatedResponse>[] retArr = ret.toArray(new Future[0]);
		return retArr;
	}

	public static long getFedLookupTableGetCount() {
		return fedLookupTableGetCount.longValue();
	}

	public static List<Triple<LocalDateTime, String, Long>> getCoordinatorsTrafficBytes() {
		return coordinatorsTrafficBytes;
	}

	public static double getCPUUsage() {
		ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
		double cpuUsage = 0.0f;

		for(Long threadID : threadMXBean.getAllThreadIds()) {
			cpuUsage += threadMXBean.getThreadCpuTime(threadID);
		}

		cpuUsage /= 1000000000; // nanoseconds to seconds

		return cpuUsage;
	}

	public static double getMemoryUsage() {
		MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();

		double maxMemory = (double)memoryMXBean.getHeapMemoryUsage().getMax() / 1073741824;
		double usedMemory = (double)memoryMXBean.getHeapMemoryUsage().getUsed() / 1073741824;

		return (usedMemory / maxMemory) * 100;
	}

	public static long getFedLookupTableGetTime() {
		return fedLookupTableGetTime.longValue();
	}

	public static long getFedLookupTableEntryCount() {
		return fedLookupTableEntryCount.longValue();
	}

	public static long getFedReuseReadHitCount() {
		return fedReuseReadHitCount.longValue();
	}

	public static long getFedReuseReadBytesCount() {
		return fedReuseReadBytesCount.longValue();
	}

	public static long getFedPutLineageCount() {
		return fedPutLineageCount.longValue();
	}

	public static long getFedPutLineageItems() {
		return fedPutLineageItems.longValue();
	}

	public static long getFedSerializationReuseCount() {
		return fedSerializationReuseCount.longValue();
	}

	public static long getFedSerializationReuseBytes() {
		return fedSerializationReuseBytes.longValue();
	}

	public static void incFedLookupTableGetCount() {
		fedLookupTableGetCount.increment();
	}

	public static void incFedLookupTableGetTime(long time) {
		fedLookupTableGetTime.add(time);
	}

	public static void incFedLookupTableEntryCount() {
		fedLookupTableEntryCount.increment();
	}

	public static void incFedReuseReadHitCount() {
		fedReuseReadHitCount.increment();
	}

	public static void incFedReuseReadBytesCount(CacheableData<?> data) {
		fedReuseReadBytesCount.add(data.getDataSize());
	}

	public static void incFedReuseReadBytesCount(CacheBlock cb) {
		fedReuseReadBytesCount.add(cb.getInMemorySize());
	}

	public static void aggFedPutLineage(String serializedLineage) {
		fedPutLineageCount.increment();
		fedPutLineageItems.add(serializedLineage.lines().count());
	}

	public static void aggFedSerializationReuse(long bytes) {
		fedSerializationReuseCount.increment();
		fedSerializationReuseBytes.add(bytes);
	}

	public static String displayFedLookupTableStats() {
		return displayFedLookupTableStats(fedLookupTableGetCount.longValue(),
			fedLookupTableEntryCount.longValue(), fedLookupTableGetTime.doubleValue() / 1000000000);
	}

	public static String displayFedLookupTableStats(long fltGetCount, long fltEntryCount, double fltGetTime) {
		if(fltGetCount > 0) {
			return InstructionUtils.concatStrings(
				"Fed LookupTable (Get, Entries):\t",
				String.valueOf(fltGetCount), "/", String.valueOf(fltEntryCount),".\n");
		}
		return "";
	}

	public static String displayFedReuseReadStats() {
		return displayFedReuseReadStats(
			fedReuseReadHitCount.longValue(),
			fedReuseReadBytesCount.longValue());
	}

	public static String displayFedReuseReadStats(long rrHits, long rrBytes) {
		if(rrHits > 0) {
			return InstructionUtils.concatStrings(
				"Fed ReuseRead (Hits, Bytes):\t",
				String.valueOf(rrHits), "/", String.valueOf(rrBytes), ".\n");
		}
		return "";
	}

	public static String displayFedPutLineageStats() {
		return displayFedPutLineageStats(fedPutLineageCount.longValue(),
			fedPutLineageItems.longValue());
	}

	public static String displayFedPutLineageStats(long plCount, long plItems) {
		if(plCount > 0) {
			return InstructionUtils.concatStrings(
				"Fed PutLineage (Count, Items):\t",
				String.valueOf(plCount), "/", String.valueOf(plItems), ".\n");
		}
		return "";
	}

	public static String displayFedSerializationReuseStats() {
		return displayFedSerializationReuseStats(fedSerializationReuseCount.longValue(),
			fedSerializationReuseBytes.longValue());
	}

	public static String displayFedSerializationReuseStats(long srCount, long srBytes) {
		if(srCount > 0) {
			return InstructionUtils.concatStrings(
				"Fed SerialReuse (Count, Bytes):\t",
				String.valueOf(srCount), "/", String.valueOf(srBytes), ".\n");
		}
		return "";
	}

	public static class FedStatsCollectFunction extends FederatedUDF {
		private static final long serialVersionUID = 1L;

		public FedStatsCollectFunction() {
			super(new long[] { });
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			FedStatsCollection fedStats = new FedStatsCollection();
			fedStats.collectStats();
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, fedStats);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	public static class FedStatsCollection implements Serializable {
		private static final long serialVersionUID = 1L;

		private void collectStats() {
			cacheStats.collectStats();
			jitCompileTime = ((double)Statistics.getJITCompileTime()) / 1000; // in sec
			cpuUsage = getCPUUsage();
			memoryUsage = getMemoryUsage();
			gcStats.collectStats();
			linCacheStats.collectStats();
			mtStats.collectStats();
			heavyHitters = Statistics.getHeavyHittersHashMap();
			coordinatorsTrafficBytes = getCoordinatorsTrafficBytes();
		}
		
		public void aggregate(FedStatsCollection that) {
			cacheStats.aggregate(that.cacheStats);
			jitCompileTime += that.jitCompileTime;
			cpuUsage += that.cpuUsage;
			memoryUsage += that.memoryUsage;
			gcStats.aggregate(that.gcStats);
			linCacheStats.aggregate(that.linCacheStats);
			mtStats.aggregate(that.mtStats);
			that.heavyHitters.forEach(
				(key, value) -> heavyHitters.merge(key, value, (v1, v2) ->
					new ImmutablePair<>(v1.getLeft() + v2.getLeft(), v1.getRight() + v2.getRight()))
			);
			that.coordinatorsTrafficBytes.addAll(coordinatorsTrafficBytes);
		}

		protected static class CacheStatsCollection implements Serializable {
			private static final long serialVersionUID = 1L;

			private void collectStats() {
				memHits = CacheStatistics.getMemHits();
				linHits = CacheStatistics.getLinHits();
				fsBuffHits = CacheStatistics.getFSBuffHits();
				fsHits = CacheStatistics.getFSHits();
				hdfsHits = CacheStatistics.getHDFSHits();
				linWrites = CacheStatistics.getLinWrites();
				fsBuffWrites = CacheStatistics.getFSBuffWrites();
				fsWrites = CacheStatistics.getFSWrites();
				hdfsWrites = CacheStatistics.getHDFSWrites();
				acqRTime = ((double)CacheStatistics.getAcquireRTime()) / 1000000000; // in sec
				acqMTime = ((double)CacheStatistics.getAcquireMTime()) / 1000000000; // in sec
				rlsTime = ((double)CacheStatistics.getReleaseTime()) / 1000000000; // in sec
				expTime = ((double)CacheStatistics.getExportTime()) / 1000000000; // in sec
			}

			private void aggregate(CacheStatsCollection that) {
				memHits += that.memHits;
				linHits += that.linHits;
				fsBuffHits += that.fsBuffHits;
				fsHits += that.fsHits;
				hdfsHits += that.hdfsHits;
				linWrites += that.linWrites;
				fsBuffWrites += that.fsBuffWrites;
				fsWrites += that.fsWrites;
				hdfsWrites += that.hdfsWrites;
				acqRTime += that.acqRTime;
				acqMTime += that.acqMTime;
				rlsTime += that.rlsTime;
				expTime += that.expTime;
			}

			private long memHits = 0;
			private long linHits = 0;
			private long fsBuffHits = 0;
			private long fsHits = 0;
			private long hdfsHits = 0;
			private long linWrites = 0;
			private long fsBuffWrites = 0;
			private long fsWrites = 0;
			private long hdfsWrites = 0;
			private double acqRTime = 0;
			private double acqMTime = 0;
			private double rlsTime = 0;
			private double expTime = 0;
		}

		protected static class GCStatsCollection implements Serializable {
			private static final long serialVersionUID = 1L;

			private void collectStats() {
				gcCount = Statistics.getJVMgcCount();
				gcTime = ((double)Statistics.getJVMgcTime()) / 1000; // in sec
			}

			private void aggregate(GCStatsCollection that) {
				gcCount += that.gcCount;
				gcTime += that.gcTime;
			}

			private long gcCount = 0;
			private double gcTime = 0;
		}

		protected static class LineageCacheStatsCollection implements Serializable {
			private static final long serialVersionUID = 1L;

			private void collectStats() {
				numHitsMem = LineageCacheStatistics.getMemHits();
				numHitsFS = LineageCacheStatistics.getFSHits();
				numHitsDel = LineageCacheStatistics.getDelHits();
				numHitsInst = LineageCacheStatistics.getInstHits();
				numHitsSB = LineageCacheStatistics.getSBHits();
				numHitsFunc = LineageCacheStatistics.getFuncHits();
				numWritesMem = LineageCacheStatistics.getMemWrites();
				numWritesFS = LineageCacheStatistics.getFSWrites();
				numMemDel = LineageCacheStatistics.getMemDeletes();
			}

			private void aggregate(LineageCacheStatsCollection that) {
				numHitsMem += that.numHitsMem;
				numHitsFS += that.numHitsFS;
				numHitsDel += that.numHitsDel;
				numHitsInst += that.numHitsInst;
				numHitsSB += that.numHitsSB;
				numHitsFunc += that.numHitsFunc;
				numWritesMem += that.numWritesMem;
				numWritesFS += that.numWritesFS;
				numMemDel += that.numMemDel;
			}

			private long numHitsMem = 0;
			private long numHitsFS = 0;
			private long numHitsDel = 0;
			private long numHitsInst = 0;
			private long numHitsSB = 0;
			private long numHitsFunc = 0;
			private long numWritesMem = 0;
			private long numWritesFS = 0;
			private long numMemDel = 0;
		}

		protected static class MultiTenantStatsCollection implements Serializable {
			private static final long serialVersionUID = 1L;

			private void collectStats() {
				fLTGetCount = getFedLookupTableGetCount();
				fLTGetTime = ((double)getFedLookupTableGetTime()) / 1000000000; // in sec
				fLTEntryCount = getFedLookupTableEntryCount();
				reuseReadHits = getFedReuseReadHitCount();
				reuseReadBytes = getFedReuseReadBytesCount();
				putLineageCount = getFedPutLineageCount();
				putLineageItems = getFedPutLineageItems();
				serializationReuseCount = getFedSerializationReuseCount();
				serializationReuseBytes = getFedSerializationReuseBytes();
			}

			private void aggregate(MultiTenantStatsCollection that) {
				fLTGetCount += that.fLTGetCount;
				fLTGetTime += that.fLTGetTime;
				fLTEntryCount += that.fLTEntryCount;
				reuseReadHits += that.reuseReadHits;
				reuseReadBytes += that.reuseReadBytes;
				putLineageCount += that.putLineageCount;
				putLineageItems += that.putLineageItems;
				serializationReuseCount += that.serializationReuseCount;
				serializationReuseBytes += that.serializationReuseBytes;
			}

			private long fLTGetCount = 0;
			private double fLTGetTime = 0;
			private long fLTEntryCount = 0;
			private long reuseReadHits = 0;
			private long reuseReadBytes = 0;
			private long putLineageCount = 0;
			private long putLineageItems = 0;
			private long serializationReuseCount = 0;
			private long serializationReuseBytes = 0;
		}

		private CacheStatsCollection cacheStats = new CacheStatsCollection();
		public double jitCompileTime = 0;
		public double cpuUsage = 0;
		public double memoryUsage = 0;
		private GCStatsCollection gcStats = new GCStatsCollection();
		private LineageCacheStatsCollection linCacheStats = new LineageCacheStatsCollection();
		private MultiTenantStatsCollection mtStats = new MultiTenantStatsCollection();
		public HashMap<String, Pair<Long, Double>> heavyHitters = new HashMap<>();
		public List<Triple<LocalDateTime, String, Long>> coordinatorsTrafficBytes = new ArrayList<>();
	}
}
