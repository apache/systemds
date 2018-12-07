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

package org.apache.sysml.runtime.controlprogram.caching;

import java.util.concurrent.atomic.LongAdder;

/**
 * This singleton provides basic caching statistics in CP.
 * 
 * 1) Hit statistics for caching (mem, fs, hdfs, total)
 * 
 * NOTE: In order to provide accurate statistics in multi-threaded
 * synchronized increments are required. Since those functions are 
 * called potentially very often, we use atomic increments 
 * (compare and swap) instead of heavy-weight 'synchronized' methods. 
 * 
 */
public class CacheStatistics 
{
	//enum used for MR counters
	public enum Stat {
		CACHE_HITS_MEM,
		CACHE_HITS_FSBUFF,
		CACHE_HITS_FS,
		CACHE_HITS_HDFS,
		CACHE_WRITES_FSBUFF,
		CACHE_WRITES_FS,
		CACHE_WRITES_HDFS,
		CACHE_TIME_ACQR, //acquire read
		CACHE_TIME_ACQM, //acquire read
		CACHE_TIME_RLS, //release
		CACHE_TIME_EXP, //export 
	}
	
	//hit statistics (for acquire read)
	private static final LongAdder _numHitsMem      = new LongAdder();
	private static final LongAdder _numHitsFSBuff   = new LongAdder();
	private static final LongAdder _numHitsFS       = new LongAdder();
	private static final LongAdder _numHitsHDFS     = new LongAdder();
	
	//write statistics caching
	private static final LongAdder _numWritesFSBuff = new LongAdder();
	private static final LongAdder _numWritesFS     = new LongAdder();
	private static final LongAdder _numWritesHDFS   = new LongAdder();
	
	//time statistics caching
	private static final LongAdder _ctimeAcquireR   = new LongAdder(); //in nano sec
	private static final LongAdder _ctimeAcquireM   = new LongAdder(); //in nano sec
	private static final LongAdder _ctimeRelease    = new LongAdder(); //in nano sec
	private static final LongAdder _ctimeExport     = new LongAdder(); //in nano sec

	public static void reset() {
		_numHitsMem.reset();
		_numHitsFSBuff.reset();
		_numHitsFS.reset();
		_numHitsHDFS.reset();
		
		_numWritesFSBuff.reset();
		_numWritesFS.reset();
		_numWritesHDFS.reset();
		
		_ctimeAcquireR.reset();
		_ctimeAcquireM.reset();
		_ctimeRelease.reset();
		_ctimeExport.reset();
	}

	public static void incrementMemHits() {
		_numHitsMem.increment();
	}
	
	public static void incrementMemHits(int delta) {
		_numHitsMem.add(delta);
	}
	
	public static long getMemHits() {
		return _numHitsMem.longValue();
	}

	public static void incrementFSBuffHits() {
		_numHitsFSBuff.increment();
	}
	
	public static void incrementFSBuffHits( int delta ) {
		_numHitsFSBuff.add(delta);
	}
	
	public static long getFSBuffHits() {
		return _numHitsFSBuff.longValue();
	}
	
	public static void incrementFSHits() {
		_numHitsFS.increment();
	}
	
	public static void incrementFSHits(int delta) {
		_numHitsFS.add(delta);
	}
	
	public static long getFSHits() {
		return _numHitsFS.longValue();
	}
	
	public static void incrementHDFSHits() {
		_numHitsHDFS.increment();
	}
	
	public static void incrementHDFSHits(int delta) {
		_numHitsHDFS.add(delta);
	}
	
	public static long getHDFSHits() {
		return _numHitsHDFS.longValue();
	}

	public static void incrementFSBuffWrites() {
		_numWritesFSBuff.increment();
	}
	
	public static void incrementFSBuffWrites(int delta) {
		_numWritesFSBuff.add(delta);
	}
	
	public static long getFSBuffWrites() {
		return _numWritesFSBuff.longValue();
	}
	
	public static void incrementFSWrites() {
		_numWritesFS.increment();
	}
	
	public static void incrementFSWrites(int delta) {
		_numWritesFS.add(delta);
	}
	
	public static long getFSWrites() {
		return _numWritesFS.longValue();
	}
	
	public static void incrementHDFSWrites() {
		_numWritesHDFS.increment();
	}
	
	public static void incrementHDFSWrites(int delta) {
		_numWritesHDFS.add(delta);
	}
	
	public static long getHDFSWrites() {
		return _numWritesHDFS.longValue();
	}
	
	public static void incrementAcquireRTime(long delta) {
		_ctimeAcquireR.add(delta);
	}
	
	public static long getAcquireRTime() {
		return _ctimeAcquireR.longValue();
	}
	
	public static void incrementAcquireMTime(long delta) {
		_ctimeAcquireM.add(delta);
	}
	
	public static long getAcquireMTime() {
		return _ctimeAcquireM.longValue();
	}

	public static void incrementReleaseTime(long delta) {
		_ctimeRelease.add(delta);
	}
	
	public static long getReleaseTime() {
		return _ctimeRelease.longValue();
	}

	public static void incrementExportTime(long delta) {
		_ctimeExport.add(delta);
	}
	
	public static long getExportTime() {
		return _ctimeExport.longValue();
	}
	
	public static String displayHits() {	
		StringBuilder sb = new StringBuilder();
		sb.append(_numHitsMem.longValue());
		sb.append("/");
		sb.append(_numHitsFSBuff.longValue());
		sb.append("/");
		sb.append(_numHitsFS.longValue());
		sb.append("/");
		sb.append(_numHitsHDFS.longValue());
		
		return sb.toString();
	}
	
	public static String displayWrites() {	
		StringBuilder sb = new StringBuilder();
		sb.append(_numWritesFSBuff.longValue());
		sb.append("/");
		sb.append(_numWritesFS.longValue());
		sb.append("/");
		sb.append(_numWritesHDFS.longValue());
		
		return sb.toString();
	}
	
	public static String displayTime() {	
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%.3f", ((double)_ctimeAcquireR.longValue())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeAcquireM.longValue())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeRelease.longValue())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeExport.longValue())/1000000000)); //in sec
		
		return sb.toString();
	}
}
