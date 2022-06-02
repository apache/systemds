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
	// Bellow entries are for specific to gpu lineage cache
	private static final LongAdder _numHitsGpu      = new LongAdder();
	private static final LongAdder _numAsyncEvictGpu= new LongAdder();
	private static final LongAdder _numSyncEvictGpu = new LongAdder();

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
		_numHitsGpu.reset();
		_numAsyncEvictGpu.reset();
		_numSyncEvictGpu.reset();
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
}
