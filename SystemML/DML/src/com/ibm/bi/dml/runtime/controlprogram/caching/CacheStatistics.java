package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.util.concurrent.atomic.AtomicInteger;

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
		CACHE_WRITES_HDFS
	}
	
	//hit statistics (for acquire read)
	private static AtomicInteger _numHitsTotal  = null;
	private static AtomicInteger _numHitsMem    = null;
	private static AtomicInteger _numHitsFSBuff = null;
	private static AtomicInteger _numHitsFS     = null;
	private static AtomicInteger _numHitsHDFS   = null;
	
	//write statistics caching
	private static AtomicInteger _numWritesFSBuff = null;
	private static AtomicInteger _numWritesFS     = null;
	private static AtomicInteger _numWritesHDFS   = null;
	

	static
	{
		reset();
	}
	
	public static void reset()
	{
		_numHitsTotal = new AtomicInteger(0);
		_numHitsMem = new AtomicInteger(0);
		_numHitsFSBuff = new AtomicInteger(0);
		_numHitsFS = new AtomicInteger(0);
		_numHitsHDFS = new AtomicInteger(0);
		
		_numWritesFSBuff = new AtomicInteger(0);
		_numWritesFS = new AtomicInteger(0);
		_numWritesHDFS = new AtomicInteger(0);
	}
	
	public static void incrementTotalHits()
	{
		_numHitsTotal.incrementAndGet();
	}
	
	public static void incrementTotalHits(int delta)
	{
		_numHitsTotal.addAndGet(delta);
	}
	
	public static int getTotalHits()
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
	
	public static int getMemHits()
	{
		return _numHitsMem.get();
	}

	public static void incrementFSBuffHits()
	{
		_numHitsFSBuff.incrementAndGet();
	}
	
	public static void incrementFSHits()
	{
		_numHitsFS.incrementAndGet();
	}
	
	public static void incrementFSHits(int delta)
	{
		_numHitsFS.addAndGet(delta);
	}
	
	public static int getFSHits()
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
	
	public static int getHDFSHits()
	{
		return _numHitsHDFS.get();
	}

	public static void incrementFSBuffWrites()
	{
		_numWritesFSBuff.incrementAndGet();
	}
	
	public static void incrementFSWrites()
	{
		_numWritesFS.incrementAndGet();
	}
	
	public static void incrementFSWrites(int delta)
	{
		_numWritesFS.addAndGet(delta);
	}
	
	public static int getFSWrites()
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
	
	public static int getHDFSWrites()
	{
		return _numWritesHDFS.get();
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
	
	
}
