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
import java.net.InetSocketAddress;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Future;
import javax.net.ssl.SSLException;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics.FedStatsCollection.CacheStatsCollection;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics.FedStatsCollection.GCStatsCollection;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.utils.Statistics;

public class FederatedStatistics {
	private static Set<Pair<String, Integer>> _fedWorkerAddresses = new HashSet<>();

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

	public static String displayFedStatistics(int numHeavyHitters) {
		StringBuilder sb = new StringBuilder();
		FedStatsCollection fedStats = collectFedStats();
		sb.append("SystemDS Federated Statistics:\n");
		sb.append(displayCacheStats(fedStats.cacheStats));
		sb.append(String.format("Total JIT compile time:\t\t%.3f sec.\n", fedStats.jitCompileTime));
		sb.append(displayGCStats(fedStats.gcStats));
		sb.append(displayHeavyHitters(fedStats.heavyHitters, numHeavyHitters));
		return sb.toString();
	}

	public static String displayCacheStats(CacheStatsCollection csc) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("Cache hits (Mem/Li/WB/FS/HDFS):\t%d/%d/%d/%d/%d.\n",
			csc.memHits, csc.linHits, csc.fsBuffHits, csc.fsHits, csc.hdfsHits));
		sb.append(String.format("Cache writes (Li/WB/FS/HDFS):\t%d/%d/%d/%d.\n",
			csc.linWrites, csc.fsBuffWrites, csc.fsWrites, csc.hdfsWrites));
		sb.append(String.format("Cache times (ACQr/m, RLS, EXP):\t%.3f/%.3f/%.3f/%.3f sec.\n",
			csc.acqRTime, csc.acqMTime, csc.rlsTime, csc.expTime));
		return sb.toString();
	}

	public static String displayGCStats(GCStatsCollection gcsc) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("Total JVM GC count:\t\t%d.\n", gcsc.gcCount));
		sb.append(String.format("Total JVM GC time:\t\t%.3f sec.\n", gcsc.gcTime));
		return sb.toString();
	}

	public static String displayHeavyHitters(HashMap<String, Pair<Long, Double>> heavyHitters) {
		return displayHeavyHitters(heavyHitters, 10);
	}

	public static String displayHeavyHitters(HashMap<String, Pair<Long, Double>> heavyHitters, int num) {
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
			} catch(SSLException ssle) {
				System.out.println("SSLException while getting the federated stats from "
					+ isa.toString() + ": " + ssle.getMessage());
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

	private static class FedStatsCollectFunction extends FederatedUDF {
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

	protected static class FedStatsCollection implements Serializable {
		private static final long serialVersionUID = 1L;

		private void collectStats() {
			cacheStats.collectStats();
			jitCompileTime = ((double)Statistics.getJITCompileTime()) / 1000; // in sec
			gcStats.collectStats();
			heavyHitters = Statistics.getHeavyHittersHashMap();
		}
		
		private void aggregate(FedStatsCollection that) {
			cacheStats.aggregate(that.cacheStats);
			jitCompileTime += that.jitCompileTime;
			gcStats.aggregate(that.gcStats);
			that.heavyHitters.forEach(
				(key, value) -> heavyHitters.merge(key, value, (v1, v2) ->
					new ImmutablePair<>(v1.getLeft() + v2.getLeft(), v1.getRight() + v2.getRight()))
			);
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

		private CacheStatsCollection cacheStats = new CacheStatsCollection();
		private double jitCompileTime = 0;
		private GCStatsCollection gcStats = new GCStatsCollection();
		private HashMap<String, Pair<Long, Double>> heavyHitters = new HashMap<>();
	}
}
