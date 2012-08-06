package com.ibm.bi.dml.runtime.controlprogram.parfor.stat;

import java.io.IOException;

import org.apache.hadoop.mapred.ClusterStatus;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;

/**
 * Central place for analyzing and obtaining static infrastructure properties
 * such as memory and number of logical processors.
 * 
 */
public class InfrastructureAnalyzer 
{
	//static local master node properties
	public static int  _localPar        = -1;
	public static long _localJVMMaxMem  = -1;
	
	//static hadoop cluster properties
	public static int  _remotePar       = -1;
	public static int  _remoteParMap    = -1;
	public static int  _remoteParReduce = -1;
	public static long _remoteJVMMaxMem = -1;
	public static long _remoteMRSortMem = -1;
	
	
	//static initialization, called for each JVM (on each node)
	static 
	{
		//analyze local node properties
		analyzeLocalMachine();
		
		//analyze remote Hadoop cluster properties
		//analyzeHadoopCluster(); //note: due to overhead - analyze on-demand
	}
	
	
	///////
	//methods for obtaining parallelism properties
	
	/**
	 * Gets the number of logical processors of the current node,
	 * including hyper-threading if enabled.
	 * 
	 * @return
	 */
	public static int getLocalParallelism()
	{
		return _localPar;
	}	
	
	/**
	 * Gets the number of cluster nodes (number of tasktrackers). If multiple tasktracker
	 * are started per node, each tasktracker is viewed as individual node.
	 * 
	 * @return
	 */
	public static int getRemoteParallelNodes() 
	{
		if( _remotePar == -1 )
			analyzeHadoopCluster();
		
		return _remotePar;
	}	
	
	/**
	 * Gets the total number of available map slots.
	 * 
	 * @return
	 */
	public static int getRemoteParallelMapTasks()
	{
		if( _remoteParMap == -1 )
			analyzeHadoopCluster();
		
		return _remoteParMap;
	}
	
	/**
	 * Gets the total number of available reduce slots.
	 * 
	 * @return
	 */
	public static int getRemoteParallelReduceTasks()
	{
		if( _remoteParReduce == -1 )
			analyzeHadoopCluster();
		
		return _remoteParReduce;
	}
	
	/**
	 * Gets the totals number of available map and reduce slots.
	 * 
	 * @return
	 */
	public static int getRemoteParallelTasks()
	{
		if( _remoteParMap == -1 )
			analyzeHadoopCluster();
		
		return _remoteParMap + _remoteParReduce;
	}
	
	
	///////
	//methods for obtaining memory properties
	
	/**
	 * Gets the maximum memory [in bytes] of the current JVM.
	 * 
	 * @return
	 */
	public static long getLocalMaxMemory()
	{
		return _localJVMMaxMem;
	}
	
	/**
	 * Gets the maximum memory [in bytes] of a hadoop task JVM.
	 * 
	 * @return
	 */
	public static long getRemoteMaxMemory()
	{
		if( _remoteJVMMaxMem == -1 )
			analyzeHadoopCluster();
		
		return _remoteJVMMaxMem;
	}
	
	/**
	 * Gets the maximum memory requirement [in bytes] of a given hadoop job.
	 * 
	 * @param conf
	 * @return
	 */
	public static long getRemoteMaxMemory( JobConf job )
	{
		return (1024*1024) * Math.max(
				               job.getMemoryForMapTask(),
				               job.getMemoryForReduceTask() );			
	}
	
	/**
	 * Gets the maximum sort buffer memory requirement [in bytes] of a hadoop task.
	 * 
	 * @return
	 */
	public static long getRemoteMaxMemorySortBuffer( )
	{
		if( _remoteMRSortMem == -1 )
			analyzeHadoopCluster();
		
		return _remoteMRSortMem;		
	}
	
	
	///////
	//methods for obtaining constraints or respective defaults
	
	/**
	 * Gets the maximum local parallelism constraint.
	 * 
	 * @return
	 */
	public static int getCkMaxCP() 
	{
		//default value (if not specified)
		return getLocalParallelism();
	}

	/**
	 * Gets the maximum remote parallelism constraint
	 * 
	 * @return
	 */
	public static int getCkMaxMR() 
	{
		//default value (if not specified)
		return getRemoteParallelMapTasks();
	}

	/**
	 * Gets the maximum memory constraint [in bytes].
	 * 
	 * @return
	 */
	public static long getCmMax() 
	{
		//default value (if not specified)
		return Math.min( getLocalMaxMemory(), getRemoteMaxMemory() );
	}
	
	///////
	//internal methods for analysis
		
	/**
	 * Analyzes properties of local machine and JVM.
	 */
	private static void analyzeLocalMachine()
	{
		_localPar       = Runtime.getRuntime().availableProcessors();
		_localJVMMaxMem = Runtime.getRuntime().maxMemory();
	}
	
	/**
	 * Analyzes properties of hadoop cluster and configuration.
	 */
	private static void analyzeHadoopCluster()
	{
		try 
		{
			JobConf job = new JobConf(InfrastructureAnalyzer.class);
			JobClient client = new JobClient(job);
			ClusterStatus stat = client.getClusterStatus();
			if( stat != null ) //if in cluster mode
			{
				_remotePar = stat.getTaskTrackers();
				_remoteParMap = stat.getMaxMapTasks(); 
				_remoteParReduce = stat.getMaxReduceTasks(); 
				_remoteJVMMaxMem = (1024*1024) * job.getLong("mapred.child.java.opts",1024); //1GB
				_remoteMRSortMem = (1024*1024) * job.getLong("io.sort.mb",100); //1MB
			}		
		} 
		catch (IOException e) 
		{
			throw new RuntimeException("Unable to analyze infrastructure.",e);
		}
	}

}
