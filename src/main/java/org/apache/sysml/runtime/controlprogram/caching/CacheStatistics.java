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

import java.util.concurrent.atomic.AtomicLong;

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
	private static AtomicLong _numHitsTotal  = null;
	private static AtomicLong _numHitsMem    = null;
	private static AtomicLong _numHitsFSBuff = null;
	private static AtomicLong _numHitsFS     = null;
	private static AtomicLong _numHitsHDFS   = null;
	
	//write statistics caching
	private static AtomicLong _numWritesFSBuff = null;
	private static AtomicLong _numWritesFS     = null;
	private static AtomicLong _numWritesHDFS   = null;
	
	//time statistics caching
	private static AtomicLong _ctimeAcquireR   = null; //in nano sec
	private static AtomicLong _ctimeAcquireM   = null; //in nano sec
	private static AtomicLong _ctimeRelease    = null; //in nano sec
	private static AtomicLong _ctimeExport     = null; //in nano sec

	static
	{
		reset();
	}
	
	public static void reset()
	{
		_numHitsTotal = new AtomicLong(0);
		_numHitsMem = new AtomicLong(0);
		_numHitsFSBuff = new AtomicLong(0);
		_numHitsFS = new AtomicLong(0);
		_numHitsHDFS = new AtomicLong(0);
		
		_numWritesFSBuff = new AtomicLong(0);
		_numWritesFS = new AtomicLong(0);
		_numWritesHDFS = new AtomicLong(0);
		
		_ctimeAcquireR = new AtomicLong(0);
		_ctimeAcquireM = new AtomicLong(0);
		_ctimeRelease = new AtomicLong(0);
		_ctimeExport = new AtomicLong(0);
	}
	
	public static void incrementTotalHits()
	{
		_numHitsTotal.incrementAndGet();
	}
	
	public static void incrementTotalHits(int delta)
	{
		_numHitsTotal.addAndGet(delta);
	}
	
	public static long getTotalHits()
	{
		return _numHitsTotal.get();
	}
	
	public static void incrementMemHits()
	{
		_numHitsMem.incrementAndGet();
	}
	
	public static void incrementMemHits(int delta)
	{
		_numHitsMem.addAndGet(delta);
	}
	
	public static long getMemHits()
	{
		return _numHitsMem.get();
	}

	public static void incrementFSBuffHits()
	{
		_numHitsFSBuff.incrementAndGet();
	}
	
	public static void incrementFSBuffHits( int delta )
	{
		_numHitsFSBuff.addAndGet(delta);
	}
	
	public static long getFSBuffHits()
	{
		return _numHitsFSBuff.get();
	}
	
	public static void incrementFSHits()
	{
		_numHitsFS.incrementAndGet();
	}
	
	public static void incrementFSHits(int delta)
	{
		_numHitsFS.addAndGet(delta);
	}
	
	public static long getFSHits()
	{
		return _numHitsFS.get();
	}
	
	public static void incrementHDFSHits()
	{
		_numHitsHDFS.incrementAndGet();
	}
	
	public static void incrementHDFSHits(int delta)
	{
		_numHitsHDFS.addAndGet(delta);
	}
	
	public static long getHDFSHits()
	{
		return _numHitsHDFS.get();
	}

	public static void incrementFSBuffWrites()
	{
		_numWritesFSBuff.incrementAndGet();
	}
	
	public static void incrementFSBuffWrites(int delta)
	{
		_numWritesFSBuff.addAndGet(delta);
	}
	
	public static long getFSBuffWrites()
	{
		return _numWritesFSBuff.get();
	}
	
	public static void incrementFSWrites()
	{
		_numWritesFS.incrementAndGet();
	}
	
	public static void incrementFSWrites(int delta)
	{
		_numWritesFS.addAndGet(delta);
	}
	
	public static long getFSWrites()
	{
		return _numWritesFS.get();
	}
	
	public static void incrementHDFSWrites()
	{
		_numWritesHDFS.incrementAndGet();
	}
	
	public static void incrementHDFSWrites(int delta)
	{
		_numWritesHDFS.addAndGet(delta);
	}
	
	public static long getHDFSWrites()
	{
		return _numWritesHDFS.get();
	}
	
	public static void incrementAcquireRTime(long delta)
	{
		_ctimeAcquireR.addAndGet(delta);
	}
	
	public static long getAcquireRTime()
	{
		return _ctimeAcquireR.get();
	}
	
	public static void incrementAcquireMTime(long delta)
	{
		_ctimeAcquireM.addAndGet(delta);
	}
	
	public static long getAcquireMTime()
	{
		return _ctimeAcquireM.get();
	}

	public static void incrementReleaseTime(long delta)
	{
		_ctimeRelease.addAndGet(delta);
	}
	
	public static long getReleaseTime()
	{
		return _ctimeRelease.get();
	}

	
	public static void incrementExportTime(long delta)
	{
		_ctimeExport.addAndGet(delta);
	}
	
	public static long getExportTime()
	{
		return _ctimeExport.get();
	}
	

	public static String displayHits()
	{	
		StringBuilder sb = new StringBuilder();
		sb.append(_numHitsMem.get());
		sb.append("/");
		sb.append(_numHitsFSBuff.get());
		sb.append("/");
		sb.append(_numHitsFS.get());
		sb.append("/");
		sb.append(_numHitsHDFS.get());
		
		
		return sb.toString();
	}
	
	public static String displayWrites()
	{	
		StringBuilder sb = new StringBuilder();
		sb.append(_numWritesFSBuff.get());
		sb.append("/");
		sb.append(_numWritesFS.get());
		sb.append("/");
		sb.append(_numWritesHDFS.get());
		
		
		return sb.toString();
	}
	
	public static String displayTime()
	{	
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%.3f", ((double)_ctimeAcquireR.get())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeAcquireM.get())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeRelease.get())/1000000000)); //in sec
		sb.append("/");
		sb.append(String.format("%.3f", ((double)_ctimeExport.get())/1000000000)); //in sec
		
		;
		
		return sb.toString();
	}
	
	
}
