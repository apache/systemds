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

/**
 * This singleton provides basic lineage caching statistics in CP.
 * Hit statistics for caching (mem, fs, total)
 */
public class LineageCacheStatistics {
	private static final LongAdder _numHitsMem      = new LongAdder();
	private static final LongAdder _numHitsFS       = new LongAdder();
	private static final LongAdder _numHitsDel      = new LongAdder();
	private static final LongAdder _numHitsInst     = new LongAdder();
	private static final LongAdder _numHitsSB       = new LongAdder();
	private static final LongAdder _numHitsFunc     = new LongAdder();
	private static final LongAdder _numWritesMem    = new LongAdder();
	private static final LongAdder _numWritesFS     = new LongAdder();
	private static final LongAdder _numMemDel       = new LongAdder();
	private static final LongAdder _numRewrites     = new LongAdder();
	// All the time measurements are in nanoseconds
	private static final LongAdder _ctimeFSRead     = new LongAdder();
	private static final LongAdder _ctimeFSWrite    = new LongAdder();
	private static final LongAdder _ctimeSaved      = new LongAdder();
	private static final LongAdder _ctimeMissed     = new LongAdder();
	private static final LongAdder _ctimeProbe      = new LongAdder();
	// Bellow entries are specific to gpu lineage cache
	private static final LongAdder _numHitsGpu      = new LongAdder();
	private static final LongAdder _numAsyncEvictGpu= new LongAdder();
	private static final LongAdder _numSyncEvictGpu = new LongAdder();
	private static final LongAdder _numRecycleGpu   = new LongAdder();
	private static final LongAdder _numDelGpu       = new LongAdder();
	private static final LongAdder _evtimeGpu       = new LongAdder();
	// Below entries are specific to Spark instructions
	private static final LongAdder _numHitsRdd      = new LongAdder();
	private static final LongAdder _numHitsSparkActions = new LongAdder();
	private static final LongAdder _numHitsRddPersist   = new LongAdder();
	private static final LongAdder _numRddPersist   = new LongAdder();
	private static final LongAdder _numRddUnpersist   = new LongAdder();

	public static void reset() {
		_numHitsMem.reset();
		_numHitsFS.reset();
		_numHitsDel.reset();
		_numHitsInst.reset();
		_numHitsSB.reset();
		_numHitsFunc.reset();
		_numWritesMem.reset();
		_numWritesFS.reset();
		_numMemDel.reset();
		_numRewrites.reset();
		_ctimeFSRead.reset();
		_ctimeFSWrite.reset();
		_ctimeSaved.reset();
		_ctimeMissed.reset();
		_ctimeProbe.reset();
		_evtimeGpu.reset();
		_numHitsGpu.reset();
		_numAsyncEvictGpu.reset();
		_numSyncEvictGpu.reset();
		_numRecycleGpu.reset();
		_numDelGpu.reset();
		_numHitsRdd.reset();
		_numHitsSparkActions.reset();
		_numHitsRddPersist.reset();
		_numRddPersist.reset();
		_numRddUnpersist.reset();
	}
	
	public static void incrementMemHits() {
		// Number of times found in cache.
		_numHitsMem.increment();
	}

	public static long getMemHits() {
		return _numHitsMem.longValue();
	}

	public static void incrementFSHits() {
		// Number of times found in local FS.
		_numHitsFS.increment();
	}

	public static long getFSHits() {
		return _numHitsFS.longValue();
	}

	public static void incrementDelHits() {
		// Number of times entry is removed from cache but sought again later.
		_numHitsDel.increment();
	}

	public static long getDelHits() {
		return _numHitsDel.longValue();
	}

	public static void incrementInstHits() {
		// Number of times single instruction results are reused (full and partial).
		_numHitsInst.increment();
	}
	
	public static long getInstHits() {
		return _numHitsInst.longValue();
	}

	public static void incrementSBHits() {
		// Number of times statementblock results are reused.
		_numHitsSB.increment();
	}

	public static long getSBHits() {
		return _numHitsSB.longValue();
	}

	public static void incrementFuncHits() {
		// Number of times function results are reused.
		_numHitsFunc.increment();
	}

	public static long getFuncHits() {
		return _numHitsFunc.longValue();
	}

	public static void incrementMemWrites() {
		// Number of times written in cache.
		_numWritesMem.increment();
	}

	public static long getMemWrites() {
		return _numWritesMem.longValue();
	}

	public static void incrementPRewrites() {
		// Number of partial rewrites.
		_numRewrites.increment();
	}

	public static void incrementFSWrites() {
		// Number of times written in local FS.
		_numWritesFS.increment();
	}

	public static long getFSWrites() {
		return _numWritesFS.longValue();
	}

	public static void incrementMemDeletes() {
		// Number of deletions from cache (including spilling).
		_numMemDel.increment();
	}

	public static long getMemDeletes() {
		return _numMemDel.longValue();
	}

	public static void incrementFSReadTime(long delta) {
		// Total time spent on reading from FS.
		_ctimeFSRead.add(delta);
	}

	public static void incrementFSWriteTime(long delta) {
		// Total time spent writing to FS.
		_ctimeFSWrite.add(delta);
	}

	public static void incrementSavedComputeTime(long delta) {
		// Total time saved by reusing.
		// TODO: Handle overflow
		_ctimeSaved.add(delta);
	}

	public static void incrementMissedComputeTime(long delta) {
		// Total time missed due to eviction.
		// TODO: Handle overflow
		_ctimeMissed.add(delta);
	}

	public static void incrementProbeTime(long delta) {
		_ctimeProbe.add(delta);
	}

	public static long getMultiLevelFnHits() {
		return _numHitsFunc.longValue();
	}
	
	public static long getMultiLevelSBHits() {
		return _numHitsSB.longValue();
	}

	public static void incrementGpuHits() {
		// Number of times single instruction results are reused in the gpu.
		_numHitsGpu.increment();
	}

	public static void incrementGpuAsyncEvicts() {
		// Number of gpu cache entries moved to cpu cache via the background thread
		_numAsyncEvictGpu.increment();
	}

	public static void incrementGpuSyncEvicts() {
		// Number of gpu cache entries moved to cpu cache during malloc 
		_numSyncEvictGpu.increment();
	}

	public static void incrementGpuRecycle() {
		// Number of gpu cached pointers recycled
		_numRecycleGpu.increment();
	}

	public static void incrementGpuDel() {
		// Number of gpu cached pointers deleted to make space
		_numDelGpu.increment();
	}

	public static void incrementEvictTimeGpu(long delta) {
		// Total time spent on evicting from GPU to main memory or deleting from GPU lineage cache
		_evtimeGpu.add(delta);
	}

	public static void incrementRDDHits() {
		// Number of times a locally cached (but not persisted) RDD are reused.
		_numHitsRdd.increment();
	}

	public static void incrementSparkCollectHits() {
		// Spark instructions that bring intermediate back to local.
		// Both synchronous and asynchronous (e.g. tsmm, prefetch)
		_numHitsSparkActions.increment();
	}

	public static void incrementRDDPersistHits() {
		// Number of times a locally cached and persisted RDD are reused.
		_numHitsRddPersist.increment();
	}

	public static void incrementRDDPersists() {
		// Number of RDDs marked for persistence
		_numRddPersist.increment();
	}

	public static void incrementRDDUnpersists() {
		// Number of RDDs unpersisted due the due to memory pressure
		_numRddUnpersist.increment();
	}

	public static String displayHits() {
		StringBuilder sb = new StringBuilder();
		sb.append(_numHitsMem.longValue());
		sb.append("/");
		sb.append(_numHitsFS.longValue());
		sb.append("/");
		sb.append(_numHitsDel.longValue());
		return sb.toString();
	}

	public static String displayMultiLevelHits() {
		StringBuilder sb = new StringBuilder();
		sb.append(_numHitsInst.longValue());
		sb.append("/");
		sb.append(_numHitsSB.longValue());
		sb.append("/");
		sb.append(_numHitsFunc.longValue());
		return sb.toString();
	}

	public static String displayWtrites() {
		StringBuilder sb = new StringBuilder();
		sb.append(_numWritesMem.longValue());
		sb.append("/");
		sb.append(_numWritesFS.longValue());
		sb.append("/");
		sb.append(_numMemDel.longValue());
		return sb.toString();
	}

	public static String displayRewrites() {
		StringBuilder sb = new StringBuilder();
		sb.append(_numRewrites.longValue());
		return sb.toString();
	}
	
	public static String displayFSTime() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%.3f", ((double)_ctimeFSRead.longValue())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeFSWrite.longValue())/1000000000)); //in sec
		return sb.toString();
	}
	public static String displayComputeTime() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%.3f", ((double)_ctimeSaved.longValue())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeMissed.longValue())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeProbe.longValue())/1000000000)); //in sec
		return sb.toString();
	}

	public static String displayGpuStats() {
		StringBuilder sb = new StringBuilder();
		sb.append(_numHitsGpu.longValue());
		sb.append("/");
		sb.append(_numAsyncEvictGpu.longValue());
		sb.append("/");
		sb.append(_numSyncEvictGpu.longValue());
		return sb.toString();
	}

	public static String displayGpuPointerStats() {
		StringBuilder sb = new StringBuilder();
		sb.append(_numRecycleGpu.longValue());
		sb.append("/");
		sb.append(_numDelGpu.longValue());
		return sb.toString();
	}

	public static String displayGpuEvictTime() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%.3f", ((double)_evtimeGpu.longValue())/1000000000)); //in sec
		return sb.toString();
	}

	public static boolean ifGpuStats() {
		return (_numHitsGpu.longValue() + _numAsyncEvictGpu.longValue()
			+ _numSyncEvictGpu.longValue() + _numRecycleGpu.longValue()
			+ _numDelGpu.longValue() + _evtimeGpu.longValue()) != 0;
	}

	public static String displaySparkHits() {
		StringBuilder sb = new StringBuilder();
		sb.append(_numHitsSparkActions.longValue());
		sb.append("/");
		sb.append(_numHitsRdd.longValue());
		sb.append("/");
		sb.append(_numHitsRddPersist.longValue());
		return sb.toString();
	}

	public static String displaySparkPersist() {
		StringBuilder sb = new StringBuilder();
		sb.append(_numRddPersist.longValue());
		sb.append("/");
		sb.append(_numRddUnpersist.longValue());
		return sb.toString();
	}

	public static boolean ifSparkStats() {
		return (_numHitsSparkActions.longValue() + _numHitsRdd.longValue()
		+ _numHitsRddPersist.longValue() + _numRddPersist.longValue()) != 0;
	}
}
