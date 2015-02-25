/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.stat;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.mapred.ClusterStatus;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.runtime.matrix.mapred.MRConfigurationNames;

/**
 * Central place for analyzing and obtaining static infrastructure properties
 * such as memory and number of logical processors.
 * 
 * 
 */
public class InfrastructureAnalyzer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final long DEFAULT_JVM_SIZE = 512 * 1024 * 1024;
	
	//static local master node properties
	private static int  _localPar        = -1;
	private static long _localJVMMaxMem  = -1;
	
	//static hadoop cluster properties
	private static int  _remotePar       = -1;
	private static int  _remoteParMap    = -1;
	private static int  _remoteParReduce = -1;
	private static long _remoteJVMMaxMemMap    = -1;
	private static long _remoteJVMMaxMemReduce = -1;
	private static long _remoteMRSortMem = -1;
	private static boolean _localJT      = false;
	private static long _blocksize       = -1;
	private static boolean _yarnEnabled  = false;
	
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
	 * 
	 * @param pmap
	 */
	public static void setRemoteParallelMapTasks(int pmap)
	{
		_remoteParMap = pmap;
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
	 * 
	 * @param preduce
	 */
	public static void setRemoteParallelReduceTasks(int preduce)
	{
		_remoteParReduce = preduce;
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
	 * 
	 * @param localMem
	 */
	public static void setLocalMaxMemory( long localMem )
	{
		_localJVMMaxMem = localMem;
	}
	
	/**
	 * Gets the maximum memory [in bytes] of a hadoop map task JVM.
	 * 
	 * @return
	 */
	public static long getRemoteMaxMemoryMap()
	{
		if( _remoteJVMMaxMemMap == -1 )
			analyzeHadoopCluster();
		
		return _remoteJVMMaxMemMap;
	}
	
	/**
	 * 
	 * @param remoteMem
	 */
	public static void setRemoteMaxMemoryMap( long remoteMem )
	{
		_remoteJVMMaxMemMap = remoteMem;
	}
	
	/**
	 * Gets the maximum memory [in bytes] of a hadoop reduce task JVM.
	 * 
	 * @return
	 */
	public static long getRemoteMaxMemoryReduce()
	{
		if( _remoteJVMMaxMemReduce == -1 )
			analyzeHadoopCluster();
		
		return _remoteJVMMaxMemReduce;
	}
	
	/**
	 * 
	 * @param remoteMem
	 */
	public static void setRemoteMaxMemoryReduce( long remoteMem )
	{
		_remoteJVMMaxMemReduce = remoteMem;
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
	
	public static boolean isLocalMode()
	{
		if( _remoteJVMMaxMemMap == -1 )
			analyzeHadoopCluster();
		
		return _localJT;		
	}
	
	public static boolean isLocalMode(JobConf job)
	{
		if( _remoteJVMMaxMemMap == -1 )
		{
			//analyze if local mode
			String jobTracker = job.get("mapred.job.tracker", "local");
			_localJT = jobTracker.equals("local");
		}
		
		return _localJT;		
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
		return Math.min( getLocalMaxMemory(), getRemoteMaxMemoryMap() );
	}

	/**
	 * Gets the HDFS blocksize of the used cluster in bytes.
	 * 
	 * @return
	 */
	public static long getHDFSBlockSize()
	{
		if( _blocksize == -1 )
			analyzeHadoopCluster();
		
		return _blocksize;		
	}
	

	/**
	 * 
	 * @return
	 */
	public static boolean isYarnEnabled()
	{
		if( _remoteJVMMaxMemMap == -1 )
			analyzeHadoopCluster();
		
		return _yarnEnabled;
	}
	
	
	/**
	 * 
	 * @param javaOpts
	 * @return
	 */
	public static long extractMaxMemoryOpt(String javaOpts)
	{
		long ret = -1; //mem in bytes
		
		try
		{
			StringTokenizer st = new StringTokenizer( javaOpts, " " );
			while( st.hasMoreTokens() )
			{
				String arg = st.nextToken();
				if( !arg.startsWith("-Xmx") ) //search for max mem
					continue;
				
				arg = arg.substring(4); //cut off "-Xmx"
				//parse number and unit
				if ( arg.endsWith("g") || arg.endsWith("G") )
					ret = Long.parseLong(arg.substring(0,arg.length()-1)) * 1024 * 1024 * 1024;
				else if ( arg.endsWith("m") || arg.endsWith("M") )
					ret = Long.parseLong(arg.substring(0,arg.length()-1)) * 1024 * 1024;
				else if( arg.endsWith("k") || arg.endsWith("K") )
					ret = Long.parseLong(arg.substring(0,arg.length()-1)) * 1024;
				else 
					ret = Long.parseLong(arg.substring(0,arg.length()-2));
			}
			
			if( ret < 0 ) // no argument found
			{
				ret = DEFAULT_JVM_SIZE;
			}
		}
		catch(Exception ex)
		{
			//if anything breaks during parsing (e.g., because args not specified correctly)
			ret = DEFAULT_JVM_SIZE;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param job
	 * @param key
	 * @param bytes
	 */
	public static void setMaxMemoryOpt(JobConf job, String key, long bytes)
	{
		String javaOptsOld = job.get( key );
		String javaOptsNew = null;

		//StringTokenizer st = new StringTokenizer( javaOptsOld, " " );
		String[] tokens = javaOptsOld.split(" "); //account also for no ' '
		StringBuilder sb = new StringBuilder();
		for( String arg : tokens )
		{
			if( arg.startsWith("-Xmx") ) //search for max mem
			{
				sb.append("-Xmx");
				sb.append( (bytes/(1024*1024)) );
				sb.append("M");
			}
			else
				sb.append(arg);
			
			sb.append(" ");
		}
		javaOptsNew = sb.toString().trim();		
		job.set(key, javaOptsNew);
	}
	
	/**
	 * Gets the fraction of running map/reduce tasks to existing
	 * map/reduce task slots. 
	 * 
	 * NOTE: on YARN the number of slots is a spurious indicator 
	 * because containers are purely scheduled based on memory. 
	 * 
	 * @return
	 * @throws IOException
	 */
	public static double getClusterUtilization(boolean mapOnly) 
		throws IOException
	{
		//in local mode, the cluster utilization is always 0.0 
		
		JobConf job = ConfigurationManager.getCachedJobConf();
		JobClient client = new JobClient(job);
		ClusterStatus stat = client.getClusterStatus();
		
		double ret = 0.0;
		if( stat != null ) //if in cluster mode
		{	
			if( mapOnly )
			{
				int capacity = stat.getMaxMapTasks();
				int current = stat.getMapTasks();
				ret = ((double)current) / capacity;
			}
			else
			{
				int capacity = stat.getMaxMapTasks() + stat.getMaxReduceTasks();
				int current = stat.getMapTasks() + stat.getReduceTasks();
				ret = ((double)current) / capacity;
			}
		}
		
		return ret;
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
			JobConf job = ConfigurationManager.getCachedJobConf();
			JobClient client = new JobClient(job);
			ClusterStatus stat = client.getClusterStatus();
			if( stat != null ) //if in cluster mode
			{
				_remotePar = stat.getTaskTrackers();
				_remoteParMap = stat.getMaxMapTasks(); 
				_remoteParReduce = stat.getMaxReduceTasks(); 
				_remoteMRSortMem = (1024*1024) * job.getLong("io.sort.mb",100); //1MB
				
				//handle jvm max mem (map mem budget is relevant for map-side distcache and parfor)
				//(for robustness we probe both: child and map configuration parameters)
				String javaOpts1 = job.get("mapred.child.java.opts"); //internally mapred/mapreduce synonym
				String javaOpts2 = job.get("mapreduce.map.java.opts", null); //internally mapred/mapreduce synonym
				String javaOpts3 = job.get("mapreduce.reduce.java.opts", null); //internally mapred/mapreduce synonym
				if( javaOpts2 != null ) //specific value overrides generic
					_remoteJVMMaxMemMap = extractMaxMemoryOpt(javaOpts2); 
				else
					_remoteJVMMaxMemMap = extractMaxMemoryOpt(javaOpts1);
				if( javaOpts3 != null ) //specific value overrides generic
					_remoteJVMMaxMemReduce = extractMaxMemoryOpt(javaOpts3); 
				else
					_remoteJVMMaxMemReduce = extractMaxMemoryOpt(javaOpts1);
				
				//analyze if local mode
				String jobTracker = job.get("mapred.job.tracker", "local");
				_localJT = jobTracker.equals("local");
				
				//HDFS blocksize
				String blocksize = job.get(MRConfigurationNames.DFS_BLOCK_SIZE, "134217728");
				_blocksize = Long.parseLong(blocksize);
				
				//is yarn enabled
				String framework = job.get("mapreduce.framework.name");
				_yarnEnabled = (framework!=null && framework.equals("yarn"));
			}		
		} 
		catch (IOException e) 
		{
			throw new RuntimeException("Unable to analyze infrastructure.",e);
		}
	}
}
