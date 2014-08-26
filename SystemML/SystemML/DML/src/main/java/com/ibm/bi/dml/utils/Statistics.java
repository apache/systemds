/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils;

import java.lang.management.CompilationMXBean;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;

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

	//JVM stats
	private static long jitCompileTime = 0; //in milli sec
	private static long jvmGCTime = 0; //in milli sec
	private static long jvmGCCount = 0; //count
	
	//HOP DAG recompile stats
	private static long hopRecompileTime = 0; //in nano sec
	private static long hopRecompilePred = 0; //count
	private static long hopRecompileSB = 0;   //count
	
	
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
	
	public static synchronized void incrementJVMgcTime( long time ) {
		jvmGCTime += time;
	}
	
	public static synchronized void incrementJVMgcCount( long delta ) {
		jvmGCCount += delta;
	}
	
	public static synchronized void incrementHOPRecompileTime( long time ) {
		hopRecompileTime += time;
	}
	
	public static synchronized void incrementHOPRecompilePred() {
		hopRecompilePred ++;
	}
	
	public static synchronized void incrementHOPRecompilePred(long delta) {
		hopRecompilePred += delta;
	}
	
	public static synchronized void incrementHOPRecompileSB() {
		hopRecompileSB ++;
	}
	
	public static synchronized void incrementHOPRecompileSB(long delta) {
		hopRecompileSB += delta;
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
	
	public static void reset()
	{
		hopRecompileTime = 0;
		hopRecompilePred = 0;
		hopRecompileSB = 0;
		
		resetJITCompileTime();
		resetJVMgcTime();
		resetJVMgcCount();
		resetCPHeavyHitters();
	}
	
	/**
	 * 
	 */
	public static void resetJITCompileTime(){
		jitCompileTime = -1 * getJITCompileTime();
	}
	
	public static void resetJVMgcTime(){
		jvmGCTime = -1 * getJVMgcTime();
	}
	
	public static void resetJVMgcCount(){
		jvmGCTime = -1 * getJVMgcCount();
	}
	
	/**
	 * 
	 */
	public static void resetCPHeavyHitters(){
		_cpInstTime.clear();
		_cpInstCounts.clear();
	}
	
	public static String getCPHeavyHitterCode( Instruction inst )
	{
		String opcode = null;
		
		if( inst instanceof MRJobInstruction )
		{
			MRJobInstruction mrinst = (MRJobInstruction) inst;
			opcode = "MR-Job_"+mrinst.getJobType();
		}
		else //CPInstructions
		{
			opcode = InstructionUtils.getOpCode(inst.toString());
			if( inst instanceof FunctionCallCPInstruction ) {
				FunctionCallCPInstruction extfunct = (FunctionCallCPInstruction)inst;
				opcode = extfunct.getFunctionName();
				//opcode = extfunct.getNamespace()+Program.KEY_DELIM+extfunct.getFunctionName();
			}		
		}
		
		return opcode;
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
	
	public static Set<String> getCPHeavyHitterOpCodes()
	{
		return _cpInstTime.keySet();
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
		return hopRecompileTime;
	}
	
	public static long getHopRecompiledPredDAGs(){
		return hopRecompilePred;
	}
	
	public static long getHopRecompiledSBDAGs(){
		return hopRecompileSB;
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
		sb.append("Total execution time:\t\t" + String.format("%.3f", totalT) + " sec.\n");
		if( DMLScript.STATISTICS ) //moved into stats on Shiv's request
			sb.append("Number of compiled MR Jobs:\t" + getNoOfCompiledMRJobs() + ".\n");
		sb.append("Number of executed MR Jobs:\t" + getNoOfExecutedMRJobs() + ".\n");
		
		//show extended caching/compilation statistics
		if( DMLScript.STATISTICS ) 
		{
			sb.append("Cache hits (Mem, WB, FS, HDFS):\t" + CacheStatistics.displayHits() + ".\n");
			sb.append("Cache writes (WB, FS, HDFS):\t" + CacheStatistics.displayWrites() + ".\n");
			sb.append("Cache times (ACQr/m, RLS, EXP):\t" + CacheStatistics.displayTime() + " sec.\n");
			sb.append("HOP DAGs recompiled (PRED, SB):\t" + getHopRecompiledPredDAGs() + "/" + getHopRecompiledSBDAGs() + ".\n");
			sb.append("Total HOP DAG recompile time:\t" + String.format("%.3f", ((double)getHopRecompileTime())/1000000000) + " sec.\n");
			sb.append("Total JIT compile time:\t\t" + ((double)getJITCompileTime())/1000 + " sec.\n");
			sb.append("Total JVM GC count:\t\t" + getJVMgcCount() + ".\n");
			sb.append("Total JVM GC time:\t\t" + ((double)getJVMgcTime())/1000 + " sec.\n");
			sb.append("Heavy hitter instructions (name, time, count):\n" + getHeavyHitters(10));
		}
		
		return sb.toString();
	}
}
