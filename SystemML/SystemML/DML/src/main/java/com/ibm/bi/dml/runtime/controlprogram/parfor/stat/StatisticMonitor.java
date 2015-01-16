/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.stat;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map.Entry;

import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.POptMode;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;

/**
 * This singleton statistic monitor is used to consolidate all parfor runtime statistics.
 * Its purpose is mainly for (1) debugging and (2) potential optimization.
 * 
 * 
 *
 */
public class StatisticMonitor 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static HashMap<Long,Long>                              _mapPwPf;       //mapping parfor to parworkers
	private static HashMap<Long, HashMap<Stat,LinkedList<Double>>> _pfstats;       //parfor statistics
	private static HashMap<Long, HashMap<Stat,LinkedList<Double>>> _pwstats;       //parworker statistics
	
	private static boolean _disabled;
	
	static
	{
		_mapPwPf  = new HashMap<Long, Long>();
		_pfstats  = new HashMap<Long, HashMap<Stat,LinkedList<Double>>>();
		_pwstats  = new HashMap<Long, HashMap<Stat,LinkedList<Double>>>();
	}
	
	/**
	 * Register information about parent-child relationships of parworkers and 
	 * parfor program blocks, with a parfor can be related to one or many parworkers.
	 * 
	 * @param pfid
	 * @param pwid
	 */
	public static void putPfPwMapping( long pfid, long pwid )
	{
		if( _disabled )
			return; // do nothing
		
		_mapPwPf.put(pwid, pfid);
	}
	
	/**
	 * Puts a specific parfor statistic for future analysis into the repository.
	 * 
	 * @param id
	 * @param type
	 * @param s
	 */
	public static void putPFStat( long id, Stat type, double s)
	{
		if( _disabled )
			return; // do nothing
		
		//check if parfor exists
		if( !_pfstats.containsKey(id) )
			_pfstats.put(id, new HashMap<Stat,LinkedList<Double>>());
		HashMap<Stat,LinkedList<Double>> allstats = _pfstats.get(id);
		
		//check if stat type exists
		if( !allstats.containsKey(type) )
			allstats.put(type, new LinkedList<Double>());
		LinkedList<Double> stats = allstats.get(type);
		
		//add new stat
		stats.addLast(s);
	}
	
	/**
	 * Puts a specific parworker statistic for future analysis into the repository.
	 * 
	 * @param id
	 * @param type
	 * @param s
	 */
	public static void putPWStat( long id, Stat type, double s)
	{
		if( _disabled )
			return; // do nothing
		
		//check if parworker exists
		if( !_pwstats.containsKey(id) )
			_pwstats.put(id, new HashMap<Stat,LinkedList<Double>>());
		HashMap<Stat,LinkedList<Double>> allstats = _pwstats.get(id);
		
		//check if stat type exists
		if( !allstats.containsKey(type) )
			allstats.put(type, new LinkedList<Double>());
		LinkedList<Double> stats = allstats.get(type);
		
		//add new stat
		stats.addLast(s);
		
	}
	
	/**
	 * Cleans up the whole repository by discarding all collected information.
	 */
	public static void cleanUp()
	{
		_mapPwPf.clear();
		_pfstats.clear();
		_pwstats.clear();
	}
	
	/**
	 * Globally disables statistic monitor for the currently activ JVM.
	 */
	public static void disableStatMonitoring()
	{
		_disabled = true;
	}
	
	/**
	 * Creates a nested statistic report of all parfor and parworker instances.
	 * This should be called after completed execution.
	 * 
	 * NOTE: This report is mainly for analysis and debugging purposes.
	 * 
	 * @return
	 */
	public static String createReport()
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append("############################################## \n");
		sb.append("## ParFOR Runtime Statistics Report         ## \n");
		sb.append("############################################## \n");
		
		//foreach parfor
		for( Long pfid : _pfstats.keySet() )
		{
			sb.append("\n");
			sb.append("##############################################\n");
			sb.append("## ParFOR (ID="+ pfid +") Execution Statistics:\n");
			HashMap<Stat,LinkedList<Double>> stats = _pfstats.get(pfid); 
			
			//foreach parfor execute
			for( int i=0; i<stats.get(Stat.PARFOR_NUMTHREADS).size(); i++ )
			{
				//sb.append(" Optimization = "+stats.get(Stat.STAT_PARSE_T).get(i)+"ms\n");
				//sb.append(" Optimization = "+stats.get(Stat.STAT_OPT_T).get(i)+"ms\n");
				
				sb.append(" Run #"+i+"\n");
				sb.append("  Num Threads      = "+(int)(double)stats.get(Stat.PARFOR_NUMTHREADS).get(i)+"\n");
				sb.append("  TaskSize         = "+(int)(double)stats.get(Stat.PARFOR_TASKSIZE).get(i)+"\n");
				sb.append("  Task Partitioner = "+PTaskPartitioner.values()[(int)(double)stats.get(Stat.PARFOR_TASKPARTITIONER).get(i)]+"\n");
				sb.append("  Data Partitioner = "+PDataPartitioner.values()[(int)(double)stats.get(Stat.PARFOR_DATAPARTITIONER).get(i)]+"\n");
				sb.append("  Exec Mode        = "+PExecMode.values()[(int)(double)stats.get(Stat.PARFOR_EXECMODE).get(i)]+"\n");
				sb.append("  Num Tasks        = "+(int)(double)stats.get(Stat.PARFOR_NUMTASKS).get(i)+"\n");
				sb.append("  Num Iterations   = "+(int)(double)stats.get(Stat.PARFOR_NUMITERS).get(i)+"\n");
				
				if( stats.containsKey(Stat.OPT_OPTIMIZER) )
				{
					sb.append("  Optimizer               = "+POptMode.values()[(int)(double)stats.get(Stat.OPT_OPTIMIZER).get(i)]+"\n");
					sb.append("  Opt Num Total Plans     = "+(int)(double)stats.get(Stat.OPT_NUMTPLANS).get(i)+"\n");
					sb.append("  Opt Num Evaluated Plans = "+(int)(double)stats.get(Stat.OPT_NUMEPLANS).get(i)+"\n");
					sb.append("  Time INIT OPTIM   = "+stats.get(Stat.OPT_T).get(i)+"ms\n");
				}
				
				sb.append("  Time INIT DATA    = "+stats.get(Stat.PARFOR_INIT_DATA_T).get(i)+"ms\n");
				sb.append("  Time INIT PARWRK  = "+stats.get(Stat.PARFOR_INIT_PARWRK_T).get(i)+"ms\n");
				sb.append("  Time INIT TASKS   = "+stats.get(Stat.PARFOR_INIT_TASKS_T).get(i)+"ms\n");
				sb.append("  Time WAIT EXEC    = "+stats.get(Stat.PARFOR_WAIT_EXEC_T).get(i)+"ms\n");
				sb.append("  Time WAIT RESULT  = "+stats.get(Stat.PARFOR_WAIT_RESULTS_T).get(i)+"ms\n");
				
				//foreach parworker of this parfor
				
				int count2=1;
				for( Entry<Long, Long> e : _mapPwPf.entrySet() )
				{	
					if( e.getValue().equals(pfid) )
					{
						long pid = e.getKey();
						HashMap<Stat,LinkedList<Double>> stats2 = _pwstats.get(pid); 
						if(stats2==null)
							continue;
						int ntasks=(int)(double)stats2.get(Stat.PARWRK_NUMTASKS).get(0);
						int niters=(int)(double)stats2.get(Stat.PARWRK_NUMITERS).get(0);
						
						sb.append("   ------------------------\n");
						sb.append("   --- ParWorker #"+count2+" (ID="+ pid +") Execution Statistics:\n");						
						sb.append("       Num Tasks = "+ntasks+"\n");
						sb.append("       Num Iters = "+niters+"\n");
						sb.append("       Time EXEC = "+stats2.get(Stat.PARWRK_EXEC_T).get(0)+"ms\n");
						
						LinkedList<Double> taskexec = stats2.get(Stat.PARWRK_TASK_T);
						LinkedList<Double> tasksize = stats2.get(Stat.PARWRK_TASKSIZE);
						LinkedList<Double> iterexec = stats2.get(Stat.PARWRK_ITER_T);
						
						
						int count3=0;
						for( int k1=0; k1<ntasks; k1++ )
						{
							int ltasksize=(int)(double)tasksize.get(k1);
							
							sb.append("        Task #"+(k1+1)+": \n");
							sb.append("         Task Size = "+ltasksize+"\n");
							sb.append("         Time EXEC = "+taskexec.get(k1)+"ms\n");
							
							for( int k2=0; k2<ltasksize; k2++ )
							{
								sb.append("          Iteration #"+(k2+1)+": Time EXEC = "+iterexec.get(count3)+"ms\n");
								count3++;
							}
						}
						
						count2++;
					}
				}
			}
				
			
		}
		
		sb.append("############################################## \n");
		sb.append("############################################## \n");
		
		return sb.toString();
	}
}
