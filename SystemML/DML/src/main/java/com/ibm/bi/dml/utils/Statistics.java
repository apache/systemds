/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils;

import java.lang.management.CompilationMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;

/**
 * This class captures all statistics.
 */
public class Statistics 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static long lStartTime = 0;
	private static long lEndTime = 0;
	
	//public static long parseTime=0, hopsTime=0, lopsTime=0, piggybackTime=0, execTime=0;
	public static long execTime=0;

	/** number of executed MR jobs */
	private static int iNoOfExecutedMRJobs = 0;

	/** number of compiled MR jobs */
	private static int iNoOfCompiledMRJobs = 0;

	private static long jitCompileTime = 0; //in milli sec
	private static long hopRecompileTime = 0; //in nano sec
	
	private static HashMap<String,Long> _cpInstTime   =  new HashMap<String, Long>();
	private static HashMap<String,Long> _cpInstCounts =  new HashMap<String, Long>();
	
	public static synchronized void setNoOfExecutedMRJobs(int iNoOfExecutedMRJobs) {
		Statistics.iNoOfExecutedMRJobs = iNoOfExecutedMRJobs;
	}

	public static synchronized int getNoOfExecutedMRJobs() {
		return iNoOfExecutedMRJobs;
	}
	
	public static synchronized void incrementNoOfExecutedMRJobs() {
		iNoOfExecutedMRJobs ++;
	}
	
	public static synchronized void decrementNoOfExecutedMRJobs() {
		iNoOfExecutedMRJobs --;
	}

	public static synchronized void setNoOfCompiledMRJobs(int iNoOfCompiledMRJobs) {
		Statistics.iNoOfCompiledMRJobs = iNoOfCompiledMRJobs;
	}

	public static synchronized int getNoOfCompiledMRJobs() {
		return iNoOfCompiledMRJobs;
	}
	
	public static synchronized void incrementNoOfCompiledMRJobs() {
		iNoOfCompiledMRJobs ++;
	}
	
	public static synchronized void incrementJITCompileTime( long time ) {
		jitCompileTime += time;
	}
	
	public static synchronized void incrementHOPRecompileTime( long time ) {
		hopRecompileTime += time;
	}

	/**
	 * Starts the timer, should be invoked immediately before invoking
	 * Program.execute()
	 */
	public static void startRunTimer() {
		lStartTime = System.nanoTime();
	}

	/**
	 * Stops the timer, should be invoked immediately after invoking
	 * Program.execute()
	 */
	public static void stopRunTimer() {
		lEndTime = System.nanoTime();
	}

	/**
	 * Returns the total time of run in nanoseconds.
	 * 
	 * @return
	 */
	public static long getRunTime() {
		return lEndTime - lStartTime;
	}
	
	/**
	 * 
	 */
	public static void resetJITCompileTime(){
		jitCompileTime = -1 * getJITCompileTime();
	}
	
	/**
	 * 
	 */
	public static void resetHOPRecompileTime(){
		hopRecompileTime = 0;
	}
	
	/**
	 * 
	 */
	public static void resetCPHeavyHitters(){
		_cpInstTime.clear();
		_cpInstCounts.clear();
	}
	
	public synchronized static void maintainCPHeavyHitters( String key, long timeNanos )
	{
		Long oldVal = _cpInstTime.get(key);
		Long newVal = timeNanos + ((oldVal!=null) ? oldVal : 0);
		_cpInstTime.put(key, newVal);

		Long oldCnt = _cpInstCounts.get(key);
		Long newCnt = 1 + ((oldCnt!=null) ? oldCnt : 0);
		_cpInstCounts.put(key, newCnt);
	}
	
	/**
	 * 
	 * @param num
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static String getHeavyHitters( int num )
	{
		int len = _cpInstTime.size();
		if( num <= 0 || len <= 0 )
			return "-";
		
		//get top k via sort
		Entry<String,Long>[] tmp = _cpInstTime.entrySet().toArray(new Entry[len]);
		Arrays.sort(tmp, new Comparator<Entry<String, Long>>() {
		    public int compare(Entry<String, Long> e1, Entry<String, Long> e2) {
		        return e1.getValue().compareTo(e2.getValue());
		    }
		});
		
		//prepare output string
		StringBuilder sb = new StringBuilder();
		for( int i=0; i<Math.min(num, len); i++ ){
			String key = tmp[len-1-i].getKey();
			sb.append("-- "+(i+1)+") \t");
			sb.append(key);
			sb.append(" \t");
			sb.append(String.format("%.3f", ((double)tmp[len-1-i].getValue())/1000000000));
			sb.append(" sec \t");
			sb.append(_cpInstCounts.get(key));
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
	/**
	 * Returns the total time of asynchronous JIT compilation in milliseconds.
	 * 
	 * @return
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
	
	public static long getHopRecompileTime(){
		return hopRecompileTime;
	}

	/**
	 * Prints statistics.
	 * 
	 * @return
	 */
	public static String display() 
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append("SystemML Statistics:\n");
		double totalT = getRunTime()*1e-9; // nanoSec --> sec
		sb.append("Total time:\t\t" + totalT + " sec.\n");
		/*sb.append("CompileTime: " + parseTime*1e-9);
		sb.append(" " + hopsTime*1e-9);
		sb.append(" " + lopsTime*1e-9);
		sb.append(" " + piggybackTime*1e-9);
		sb.append(" = " + (parseTime+hopsTime+lopsTime+piggybackTime)*1e-9  + "\n");
		sb.append("RunTime: " + execTime*1e-9 + "\n");*/
		sb.append("Number of compiled MR Jobs:\t" + getNoOfCompiledMRJobs() + ".\n");
		sb.append("Number of executed MR Jobs:\t" + getNoOfExecutedMRJobs() + ".\n");
		
		//show extended caching/compilation statistics
		if( DMLScript.STATISTICS ) 
		{
			sb.append("Cache hits (Mem, WB, FS, HDFS):\t" + CacheStatistics.displayHits() + ".\n");
			sb.append("Cache writes (WB, FS, HDFS):\t" + CacheStatistics.displayWrites() + ".\n");
			sb.append("Cache times (ACQr/m, RLS, EXP):\t" + CacheStatistics.displayTime() + " sec.\n");
			sb.append("Total HOP recompile time:\t" + String.format("%.3f", ((double)getHopRecompileTime())/1000000000) + " sec.\n");
			sb.append("Total JIT compile time:\t\t" + ((double)getJITCompileTime())/1000 + " sec.\n");
			sb.append("Heavy hitter instructions (name, time, count):\n" + getHeavyHitters(10));
		}
		
		return sb.toString();
	}
}
