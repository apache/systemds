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

package org.apache.sysds.utils.stats;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.mapred.ClusterStatus;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Central place for analyzing and obtaining static infrastructure properties
 * such as memory and number of logical processors.
 * 
 * 
 */
public class InfrastructureAnalyzer 
{
	public static final long DEFAULT_JVM_SIZE = 512 * 1024 * 1024;
	public static final String MR_FRAMEWORK_NAME = "mapreduce.framework.name"; // mapred-default.xml
	public static final String MR_JOBTRACKER_ADDRESS = "mapreduce.jobtracker.address"; // mapred-default.xml
	
	//static local master node properties
	private static int  _localPar        = -1;
	private static long _localJVMMaxMem  = -1; //mutable
	private static long _fLocalJVMMaxMem = -1; //immutable
	
	//static hadoop cluster properties
	private static int  _remotePar       = -1;
	private static int  _remoteParMap    = -1;
	private static int  _remoteParReduce = -1;
	private static boolean _localJT      = false;
	private static long _blocksize       = -1;

	//static initialization, called for each JVM (on each node)
	static {
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
	 * @return number of local processors of the current node
	 */
	public static int getLocalParallelism() {
		return _localPar;
	}	
	
	/**
	 * Gets the number of cluster nodes (number of tasktrackers). If multiple tasktracker
	 * are started per node, each tasktracker is viewed as individual node.
	 * 
	 * @return number of cluster nodes
	 */
	public static int getRemoteParallelNodes() {
		if( _remotePar == -1 )
			analyzeHadoopCluster();
		return _remotePar;
	}

	/**
	 * Gets the number of remote parallel map slots.
	 * 
	 * @return number of remote parallel map tasks
	 */
	public static int getRemoteParallelMapTasks() {
		if( _remotePar == -1 )
			analyzeHadoopCluster();
		return _remoteParMap;
	}

	public static void setRemoteParallelMapTasks(int pmap) {
		_remoteParMap = pmap;
	}
	
	/**
	 * Gets the total number of remote parallel reduce slots.
	 * 
	 * @return number of remote parallel reduce tasks
	 */
	public static int getRemoteParallelReduceTasks() {
		if( _remotePar == -1 )
			analyzeHadoopCluster();
		return _remoteParReduce;
	}

	public static void setRemoteParallelReduceTasks(int preduce) {
		_remoteParReduce = preduce;
	}
	
	///////
	//methods for obtaining memory properties
	
	/**
	 * Gets the maximum memory [in bytes] of the current JVM.
	 * 
	 * @return maximum memory of the current JVM
	 */
	public static long getLocalMaxMemory() {
		return _localJVMMaxMem;
	}

	public static void setLocalMaxMemory( long localMem ) {
		_localJVMMaxMem = localMem;
	}

	public static void setLocalPar(int localPar) {
		_localPar = localPar;
	}

	public static double getLocalMaxMemoryFraction() {
		//since parfor modifies _localJVMMaxMem, some internal primitives
		//need access to the current fraction of total local memory
		return (double)_localJVMMaxMem / _fLocalJVMMaxMem;
	}

	public static boolean isLocalMode() {
		if( _remotePar == -1 )
			analyzeHadoopConfiguration();
		return _localJT;
	}

	public static boolean isLocalMode(JobConf job) {
		// Due to a bug in HDP related to fetching the "mode" of execution within mappers,
		// we explicitly probe the relevant properties instead of relying on results from 
		// analyzeHadoopCluster().
		String jobTracker = job.get(MR_JOBTRACKER_ADDRESS, "local");
		String framework = job.get(MR_FRAMEWORK_NAME, "local");
		boolean isYarnEnabled = (framework!=null && framework.equals("yarn"));
		
		return ("local".equals(jobTracker) & !isYarnEnabled);
	}
	
	///////
	//methods for obtaining constraints or respective defaults
	
	/**
	 * Gets the maximum local parallelism constraint.
	 * 
	 * @return maximum local parallelism constraint
	 */
	public static int getCkMaxCP() {
		//default value (if not specified)
		return getLocalParallelism();
	}

	/**
	 * Gets the maximum remote parallelism constraint
	 * 
	 * @return maximum remote parallelism constraint
	 */
	public static int getCkMaxMR() {
		//NOTE: we refresh only if there is already a spark context created
		// in order to avoid unnecessary spark context creation in local ops
		boolean refresh = SparkExecutionContext.isSparkContextCreated();
		if( OptimizerUtils.isSparkExecutionMode() )
			return SparkExecutionContext.isLocalMaster() ?
				InfrastructureAnalyzer.getLocalParallelism() :
				SparkExecutionContext.getDefaultParallelism(refresh);
		else
			return getRemoteParallelMapTasks();
	}

	/**
	 * Gets the maximum memory constraint [in bytes].
	 * 
	 * @return maximum memory constraint
	 */
	public static long getCmMax() {
		//default value (if not specified)
		return getLocalMaxMemory();
	}

	public static long getBlockSize(FileSystem fs){
		if(fs instanceof LocalFileSystem)
			// 4 blocks per file at least
			return 4096 * 4; 
		else 
			return getHDFSBlockSize();
	}

	/**
	 * Gets the HDFS blocksize of the used cluster in bytes.
	 * 
	 * @return HDFS block size
	 */
	public static long getHDFSBlockSize() {
		if( _blocksize == -1 )
			analyzeHadoopConfiguration();
		return _blocksize;
	}

	public static long extractMaxMemoryOpt(String javaOpts)
	{
		long ret = -1; //mem in bytes
		
		try
		{
			StringTokenizer st = new StringTokenizer( javaOpts, " " );
			while( st.hasMoreTokens() ) {
				String arg = st.nextToken();
				if( !arg.startsWith("-Xmx") ) //search for max mem
					continue;
				
				//cut off "-Xmx" parameter
				arg = arg.substring(4);
				
				//parse number and unit
				ret = UtilFunctions.parseMemorySize(arg); 
			}
			
			if( ret < 0 ) { // no argument found
				ret = DEFAULT_JVM_SIZE;
			}
		}
		catch(Exception ex) {
			//if anything breaks during parsing (e.g., because args not specified correctly)
			ret = DEFAULT_JVM_SIZE;
		}
		
		return ret;
	}

	public static void setMaxMemoryOpt(JobConf job, String key, long bytes) {
		String javaOptsOld = job.get( key );
		String javaOptsNew = null;
		String[] tokens = javaOptsOld.split(" "); //account also for no ' '
		StringBuilder sb = new StringBuilder();
		for( String arg : tokens ) {
			if( arg.startsWith("-Xmx") ) { //search for max mem
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
	 * @param mapOnly if true, only look at map tasks
	 * @return cluster utilization (current / capacity)
	 * @throws IOException if IOException occurs
	 */
	public static double getClusterUtilization(boolean mapOnly) 
		throws IOException
	{
		//in local mode, the cluster utilization is always 0.0 
		
		JobConf job = ConfigurationManager.getCachedJobConf();
		JobClient client = new JobClient(job);
		ClusterStatus stat = client.getClusterStatus();
		client.close();
		double ret = 0.0;
		if( stat != null ) { //if in cluster mode
			if( mapOnly ) {
				int capacity = stat.getMaxMapTasks();
				int current = stat.getMapTasks();
				ret = ((double)current) / capacity;
			}
			else {
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
	private static void analyzeLocalMachine() {
		//step 1: basic parallelism and memory
		_localPar       = Runtime.getRuntime().availableProcessors();
		_localJVMMaxMem = Runtime.getRuntime().maxMemory();
		_fLocalJVMMaxMem = _localJVMMaxMem;
	}
	
	/**
	 * Analyzes properties of hadoop cluster and configuration.
	 */
	private static void analyzeHadoopCluster() {
		try {
			JobConf job = ConfigurationManager.getCachedJobConf();
			JobClient client = new JobClient(job);
			ClusterStatus stat = client.getClusterStatus();
			client.close();
			if( stat != null ) { //if in cluster mode
				//analyze cluster status
				_remotePar = stat.getTaskTrackers();
				_remoteParMap = stat.getMaxMapTasks(); 
				_remoteParReduce = stat.getMaxReduceTasks(); 
				
				//analyze pure configuration properties
				analyzeHadoopConfiguration();
			}
		} 
		catch (IOException e) {
			throw new RuntimeException("Unable to analyze infrastructure.",e);
		}
	}
	
	/**
	 * Analyzes only properties of hadoop configuration in order to prevent 
	 * expensive call to cluster status .
	 */
	private static void analyzeHadoopConfiguration() {
		JobConf job = ConfigurationManager.getCachedJobConf();
		
		//HDFS blocksize
		String blocksize = job.get(HDFSTool.DFS_BLOCKSIZE, "134217728");
		_blocksize = Long.parseLong(blocksize);
		
		//analyze if local mode (internally requires yarn_enabled)
		_localJT = analyzeLocalMode(job);
	}

	private static boolean analyzeLocalMode(JobConf job) {
		//analyze if local mode (if yarn enabled, we always assume cluster mode
		//in order to workaround configuration issues on >=Hadoop 2.6)
		String jobTracker = job.get("mapreduce.jobtracker.address", "local");
		return "local".equals(jobTracker);
	}
}
