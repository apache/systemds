/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.mapred.MRConfigurationNames;

/**
 * Central place for analyzing and obtaining static infrastructure properties
 * such as memory and number of logical processors.
 * 
 * 
 */
public class YarnClusterAnalyzer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final long DEFAULT_JVM_SIZE = 512 * 1024 * 1024;
	public static final int CPU_HYPER_FACTOR = 1; 
	
	//static local master node properties
	public static int  _localPar        = -1;
	public static long _localJVMMaxMem  = -1;
	
	//default hadoop cluster properties
	public static int  _remotePar       = -1;
	//public static int  _remoteParMap    = -1;
	//public static int  _remoteParReduce = -1;
	public static long _remoteJVMMaxMemMap    = -1;
	public static long _remoteJVMMaxMemReduce = -1;
	public static long _remoteMRSortMem = -1;
	public static boolean _localJT      = false;
	public static long _blocksize       = -1;
	
	// Map from StatementBlock.ID to remoteJVMMaxMem (in bytes)
	// Encodes MR job memory settings in the execution plan, if not found here, use the default MR setting in _remoteJVMMaxMem
	public static HashMap<Long, Long> remoteJVMMaxMemPlan = new HashMap<Long, Long>();
	public static HashSet<Long> probedSb = new HashSet<Long>();
	
	public static List<Long> nodesMaxPhySorted = null;		// Original maximum physical memory per node in Byte, sorted
	public static List<Double> nodesMaxBudgetSorted = null;	// Converted to maximum budget per node in Byte, sorted
	public static int minimumMRContainerPhyMB = -1;			// Suggested minimum mappers physical memory
	public static long mrAMPhy = -1;						// The default physical memory size of MR AM
	
	public static long clusterTotalMem = -1;
	public static int clusterTotalNodes = -1;
	public static int clusterTotalCores = -1;
	public static long minimalPhyAllocate = -1;
	public static long maximumPhyAllocate = -1;
	
	//client for resource utilization updates
	private static YarnClient _client = null;
	
	//static initialization, called for each JVM (on each node)
	static 
	{
		//analyze local node properties
		analyzeLocalMachine();
		
		//analyze remote Hadoop cluster properties
		//analyzeYarnCluster(true); //note: due to overhead - analyze on-demand
	}
	
	public static List<Long> getNodesMaxPhySorted() {
		if (nodesMaxPhySorted == null)
			analyzeYarnCluster(true);
		return nodesMaxPhySorted;
	}
	
	public static List<Double> getNodesMaxBudgetSorted() {
		if (nodesMaxBudgetSorted == null)
			analyzeYarnCluster(true);
		return nodesMaxBudgetSorted;
	}
	
	public static long getMRARPhy() {
		if (mrAMPhy == -1)
			analyzeYarnCluster(true);
		return mrAMPhy;
	}
	
	public static long getClusterTotalMem() {
		if (clusterTotalMem == -1)
			analyzeYarnCluster(true);
		return clusterTotalMem;
	}
	
	public static long getMaxPhyAllocate() {
		if (maximumPhyAllocate == -1)
			analyzeYarnCluster(true);
		return maximumPhyAllocate;
	}

	public static int getMinMRContarinerPhyMB() {
		if (minimumMRContainerPhyMB == -1)
			analyzeYarnCluster(true);
		return minimumMRContainerPhyMB;
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
			analyzeYarnCluster(true);
		
		return _remotePar;
	}
	
	/**
	 * Gets the total number of available map slots.
	 * 
	 * @return
	 */
	public static int getRemoteParallelMapTasks(long jobLookupId)
	{
		if (clusterTotalCores == -1)
			analyzeYarnCluster(true);
		int ret = getRemoteParallelTasksGivenMem(getRemoteMaxMemoryMap(jobLookupId));
		//System.out.print("  jvm size " + OptimizerUtils.toMB(getRemoteMaxMemory(jobLookupId)) + " -> " + ret + " map tasks");
		if (ret >= clusterTotalCores * CPU_HYPER_FACTOR)
			ret = clusterTotalCores * CPU_HYPER_FACTOR;
		
		//System.out.println(jobLookupId + " got " + ret + " jvm = " + OptimizerUtils.toMB(getRemoteMaxMemoryMap(jobLookupId)));
		return ret;
	}
	
	/**
	 * Gets the total number of available reduce slots.
	 * 
	 * @return
	 */
	public static int getRemoteParallelReduceTasks(long jobLookupId)
	{
		if (clusterTotalCores == -1)
			analyzeYarnCluster(true);
		int ret = getRemoteParallelTasksGivenMem(getRemoteMaxMemoryReduce(jobLookupId));
		if (ret >= clusterTotalCores * CPU_HYPER_FACTOR)
			ret = clusterTotalCores * CPU_HYPER_FACTOR;
		return ret;
	}
	
	public static long getYarnPhyAllocate(long requestPhy) {
		if (minimalPhyAllocate == -1)
			analyzeYarnCluster(true);
		if (requestPhy > maximumPhyAllocate)
			throw new RuntimeException("Requested " + OptimizerUtils.toMB(requestPhy) + 
					"MB, while the maximum yarn allocate is " + OptimizerUtils.toMB(maximumPhyAllocate) + "MB");
		
		long ret = (long) Math.ceil((double)requestPhy / minimalPhyAllocate);
		ret = ret * minimalPhyAllocate;
		if (ret > maximumPhyAllocate)
			ret = maximumPhyAllocate;
		return ret;
	}
	
	/**
	 * Gets the totals number of parallel tasks given its max memory size.
	 * 
	 * @return
	 */
	public static int getRemoteParallelTasksGivenMem(long remoteTaskJvmMemory) {
		long taskPhy = getYarnPhyAllocate(ResourceOptimizer.jvmToPhy(remoteTaskJvmMemory, false));
		long cpPhy = getYarnPhyAllocate(ResourceOptimizer.jvmToPhy(getLocalMaxMemory(), false));
		long mrAMPhy = getYarnPhyAllocate(getMRARPhy());
		
		if (nodesMaxPhySorted == null)
			analyzeYarnCluster(true);
		
		if( nodesMaxPhySorted.isEmpty() )
			return -1;
		if (nodesMaxPhySorted.size() == 1) {
			long tmp = nodesMaxPhySorted.get(0) - cpPhy - mrAMPhy;
			if (tmp < 0)
				return -1;
			return (int)(tmp / taskPhy);
		}
		// At least have two nodes
		long first = nodesMaxPhySorted.get(0) - cpPhy;
		long second = nodesMaxPhySorted.get(1);
		
		if (first >= second)
			first -= mrAMPhy;
		else 
			second -= mrAMPhy;
		if (first < 0 || second < 0)
			return -1;
		long taskCount = first / taskPhy + second / taskPhy;
		int counter = 0;
		for (Long node : nodesMaxPhySorted) {
			if (counter++ < 2)
				continue;	// skip first two nodes
			taskCount += node / taskPhy;
		}
		
		//System.out.println(OptimizerUtils.toMB(cpPhy) + " " + OptimizerUtils.toMB(mrAMPhy) + " " + OptimizerUtils.toMB(taskPhy) + " " + OptimizerUtils.toMB(nodesMaxPhySorted.get(1)));
		return (int)taskCount;
	}
	
	public static boolean checkValidMemPlan(boolean hasMRJob) {
		if (nodesMaxPhySorted == null)
			analyzeYarnCluster(true);
		if (!hasMRJob)
			return nodesMaxPhySorted.get(0) >= getYarnPhyAllocate(ResourceOptimizer.jvmToPhy(getLocalMaxMemory(), false));
		return getRemoteParallelTasksGivenMem(getMaximumRemoteMaxMemory()) > 0;
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
	 * Gets the maximum memory [in bytes] of all given hadoop task memory settings.
	 * 
	 * @return
	 */
	public static long getMaximumRemoteMaxMemory()
	{
		if( _remoteJVMMaxMemMap == -1 )
			analyzeYarnCluster(true);
		
		long ret = (_remoteJVMMaxMemMap > _remoteJVMMaxMemReduce) ? _remoteJVMMaxMemMap : _remoteJVMMaxMemReduce;
		for (Map.Entry<Long, Long> entry : remoteJVMMaxMemPlan.entrySet()) {
			if (ret < entry.getValue())
				ret = entry.getValue();
		}
		return ret;
	}
	
	/**
	 * Gets the maximum memory [in bytes] of a hadoop map task JVM.
	 * 
	 * @return
	 */
	public static long getRemoteMaxMemoryMap(long jobLookupId)
	{
		if( _remoteJVMMaxMemMap == -1 )
			analyzeYarnCluster(true);
		
		long ret = getSpecifiedRemoteMaxMemory(jobLookupId);
		if (ret == -1)
			ret = _remoteJVMMaxMemMap;
		return ret;
	}
	
	/**
	 * Gets the maximum memory [in bytes] of a hadoop reduce task JVM.
	 * 
	 * @return
	 */
	public static long getRemoteMaxMemoryReduce(long jobLookupId)
	{
		if( _remoteJVMMaxMemReduce == -1 )
			analyzeYarnCluster(true);
		
		long ret = getSpecifiedRemoteMaxMemory(jobLookupId);
		if (ret == -1)
			ret = _remoteJVMMaxMemReduce;
		return ret;
	}
	
	/**
	 * Gets the maximum memory [in bytes] of a hadoop task JVM.
	 * 
	 * @return
	 */
	public static long getSpecifiedRemoteMaxMemory(long jobLookupId)
	{
		probedSb.add(jobLookupId);
		
		// Look up specified MR job setting
		Long ret = remoteJVMMaxMemPlan.get(jobLookupId);
		if (ret != null)
			return ret;
		
		// Look up specified default MR setting
		ret = remoteJVMMaxMemPlan.get((long)-1);
		if (ret != null)
			return ret;
		
		// No specified setting found
		return -1;
	}
	
	public static void setRemoteMaxMemPlan(HashMap<Long, Double> budgetMRPlan) {
		remoteJVMMaxMemPlan.clear();
		for (Map.Entry<Long, Double> entry : budgetMRPlan.entrySet()) {
			long mapJvm = ResourceOptimizer.budgetToJvm(entry.getValue());
			remoteJVMMaxMemPlan.put(entry.getKey(), mapJvm);
		}
	}
	
	public static void resetSBProbedSet() {
		probedSb.clear();
	}
	
	public static HashSet<Long> getSBProbedSet() {
		return probedSb;
	}
	
	public static void printProbedSet(String message) {
		ArrayList<Long> probed = new ArrayList<Long> (probedSb);
		Collections.sort(probed);
		System.out.print(message);
		for (Long id : probed)
			System.out.print(id + ",");
		System.out.println();
	}
		
	/**
	 * Gets the maximum memory requirement [in bytes] of a given hadoop job.
	 * 
	 * @param conf
	 * @return
	 */
	/*public static long getRemoteMaxMemory( JobConf job )
	{
		return (1024*1024) * Math.max(
				               job.getMemoryForMapTask(),
				               job.getMemoryForReduceTask() );			
	}*/
	
	/**
	 * Gets the maximum sort buffer memory requirement [in bytes] of a hadoop task.
	 * 
	 * @return
	 */
	public static long getRemoteMaxMemorySortBuffer( )
	{
		if( _remoteMRSortMem == -1 )
			analyzeYarnCluster(true);
		
		return _remoteMRSortMem;		
	}
	
	public static boolean isLocalMode()
	{
		if( _remoteJVMMaxMemMap == -1 )
			analyzeYarnCluster(true);
		
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
	public static int getCkMaxMR(long jobLookupId) 
	{
		//default value (if not specified)
		return getRemoteParallelMapTasks(jobLookupId);
	}

	/**
	 * Gets the maximum memory constraint [in bytes].
	 * 
	 * @return
	 */
	public static long getCmMax(long jobLookupId) 
	{
		//default value (if not specified)
		return Math.min(getLocalMaxMemory(), getRemoteMaxMemoryMap(jobLookupId));
	}

	/**
	 * Gets the HDFS blocksize of the used cluster in bytes.
	 * 
	 * @return
	 */
	public static long getHDFSBlockSize()
	{
		if( _blocksize == -1 )
			analyzeYarnCluster(true);
		
		return _blocksize;		
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
	
	public static void analyzeYarnCluster(boolean verbose) {
		YarnConfiguration conf = new YarnConfiguration();
		YarnClient yarnClient = YarnClient.createYarnClient();
		yarnClient.init(conf);
		yarnClient.start();
		analyzeYarnCluster(yarnClient, conf, verbose);
	}
	
	public static long getMinAllocationBytes()
	{
		if( minimalPhyAllocate < 0 )
			analyzeYarnCluster(false);
		return minimalPhyAllocate;
	}
	
	public static long getMaxAllocationBytes()
	{
		if( maximumPhyAllocate < 0 )
			analyzeYarnCluster(false);
		return maximumPhyAllocate;
	}
	
	public static long getNumCores()
	{
		if( clusterTotalCores < 0 )
			analyzeYarnCluster(false);
		return clusterTotalCores;
	}
	
	public static long getNumNodes()
	{
		if( clusterTotalNodes < 0 )
			analyzeYarnCluster(false);
		return clusterTotalNodes;
	}
	
	public static YarnClusterConfig getClusterConfig()
	{
		YarnClusterConfig cc = new YarnClusterConfig();
		cc.setMinAllocationMB( getMinAllocationBytes()/(1024*1024) );
		cc.setMaxAllocationMB( getMaxAllocationBytes()/(1024*1024) );
		cc.setNumNodes( getNumNodes() );
		cc.setNumCores( getNumCores()*CPU_HYPER_FACTOR );
		
		return cc;
	}
	
	/**
	 * 
	 * @return
	 * @throws YarnException
	 * @throws IOException
	 */
	public static double getClusterUtilization() 
		throws IOException
	{
		double util = 0;
		
		try
		{
			if( _client == null )
				_client = createYarnClient();
			List<NodeReport> nodesReport = _client.getNodeReports();
			
			double maxMem = 0;
			double currMem = 0;
			long maxCores = 0;
			long currCores = 0;
			for (NodeReport node : nodesReport) {
				Resource max = node.getCapability();
				Resource used = node.getUsed();
				maxMem += max.getMemory();
				currMem += used.getMemory();
				maxCores += max.getVirtualCores();
				currCores += used.getVirtualCores();
			}
		
			util = Math.max( 
					  Math.min(1, currMem/maxMem),  //memory util
					  Math.min(1, (double)currCores/maxCores) ); //vcore util 	
		}
		catch(Exception ex )
		{
			throw new IOException(ex);
		}
		
		return util;
	}

	/**
	 * Analyzes properties of Yarn cluster and Hadoop configurations.
	 */
	public static void analyzeYarnCluster(YarnClient yarnClient, YarnConfiguration conf, boolean verbose) {
		try {
			List<NodeReport> nodesReport = yarnClient.getNodeReports();
			if (verbose)
				System.out.println("There are " + nodesReport.size() + " nodes in the cluster");
			if( nodesReport.isEmpty() )
				throw new YarnException("There are zero available nodes in the yarn cluster");
			
			nodesMaxPhySorted = new ArrayList<Long> (nodesReport.size());
			clusterTotalMem = 0;
			clusterTotalCores = 0;
			clusterTotalNodes = 0;
			minimumMRContainerPhyMB = -1;
			for (NodeReport node : nodesReport) {
				Resource resource = node.getCapability();
				Resource used = node.getUsed();
				if (used == null)
					used = Resource.newInstance(0, 0);
				int mb = resource.getMemory();
				int cores = resource.getVirtualCores();
				if (mb <= 0)
					throw new YarnException("A node has non-positive memory " + mb);
				
				int myMinMRPhyMB = mb / cores / CPU_HYPER_FACTOR;
				if (minimumMRContainerPhyMB < myMinMRPhyMB)
					minimumMRContainerPhyMB = myMinMRPhyMB;	// minimumMRContainerPhyMB needs to be the largest among the mins
				
				clusterTotalMem += (long)mb * 1024 * 1024;
				nodesMaxPhySorted.add((long)mb * 1024 * 1024);
				clusterTotalCores += cores;
				clusterTotalNodes ++;
				if (verbose)
					System.out.println("\t" + node.getNodeId() + " has " + mb + " MB (" + used.getMemory() + " MB used) memory and " + 
						resource.getVirtualCores() + " (" + used.getVirtualCores() + " used) cores");
				
			}
			Collections.sort(nodesMaxPhySorted, Collections.reverseOrder());
			
			nodesMaxBudgetSorted = new ArrayList<Double> (nodesMaxPhySorted.size());
			for (int i = 0; i < nodesMaxPhySorted.size(); i++)
				nodesMaxBudgetSorted.add(ResourceOptimizer.phyToBudget(nodesMaxPhySorted.get(i)));
			
			_remotePar = nodesReport.size();
			if (_remotePar == 0)
				throw new YarnException("There are no available nodes in the yarn cluster");
			
			// Now get the default cluster settings
			_remoteMRSortMem = (1024*1024) * conf.getLong("io.sort.mb",100); //100MB

			//handle jvm max mem (map mem budget is relevant for map-side distcache and parfor)
			//(for robustness we probe both: child and map configuration parameters)
			String javaOpts1 = conf.get("mapred.child.java.opts"); //internally mapred/mapreduce synonym
			String javaOpts2 = conf.get("mapreduce.map.java.opts", null); //internally mapred/mapreduce synonym
			String javaOpts3 = conf.get("mapreduce.reduce.java.opts", null); //internally mapred/mapreduce synonym
			if( javaOpts2 != null ) //specific value overrides generic
				_remoteJVMMaxMemMap = extractMaxMemoryOpt(javaOpts2); 
			else
				_remoteJVMMaxMemMap = extractMaxMemoryOpt(javaOpts1);
			if( javaOpts3 != null ) //specific value overrides generic
				_remoteJVMMaxMemReduce = extractMaxMemoryOpt(javaOpts3); 
			else
				_remoteJVMMaxMemReduce = extractMaxMemoryOpt(javaOpts1);
			
			//analyze if local mode
			String jobTracker = conf.get("mapred.job.tracker", "local");
			_localJT = jobTracker.equals("local");
			
			//HDFS blocksize
			String blocksize = conf.get(MRConfigurationNames.DFS_BLOCK_SIZE, "134217728");
			_blocksize = Long.parseLong(blocksize);
			
			minimalPhyAllocate = (long) 1024 * 1024 * conf.getInt(YarnConfiguration.RM_SCHEDULER_MINIMUM_ALLOCATION_MB, 
					YarnConfiguration.DEFAULT_RM_SCHEDULER_MINIMUM_ALLOCATION_MB);
			maximumPhyAllocate = (long) 1024 * 1024 * conf.getInt(YarnConfiguration.RM_SCHEDULER_MAXIMUM_ALLOCATION_MB, 
					YarnConfiguration.DEFAULT_RM_SCHEDULER_MAXIMUM_ALLOCATION_MB);
			mrAMPhy = (long)conf.getInt("yarn.app.mapreduce.am.resource.mb", 1536) * 1024 * 1024;
			
		} catch (Exception e) {
			throw new RuntimeException("Unable to analyze yarn cluster ", e);
		}
		
		
		
		/*
		 * This is for AppMaster to query available resource in the cluster during heartbeat 
		 * 
		AMRMClient<ContainerRequest> rmClient = AMRMClient.createAMRMClient();
		rmClient.init(conf);
		rmClient.start();
		AllocateResponse response = rmClient.allocate(0);
		int nodeCount = response.getNumClusterNodes();
		Resource resource = response.getAvailableResources();
		List<NodeReport> nodeUpdate = response.getUpdatedNodes();
		
		LOG.info("This is a " + nodeCount + " node cluster with totally " +
				resource.getMemory() + " memory and " + resource.getVirtualCores() + " cores");
		LOG.info(nodereport.size() + " updatedNode reports received");
		for (NodeReport node : nodeUpdate) {
			resource = node.getCapability();
			LOG.info(node.getNodeId() + " updated with " + resource.getMemory() + " memory and " + resource.getVirtualCores() + " cores");
		}*/
	}
	
	/**
	 * 
	 * @return
	 */
	private static YarnClient createYarnClient()
	{
		YarnConfiguration conf = new YarnConfiguration();
		YarnClient yarnClient = YarnClient.createYarnClient();
		yarnClient.init(conf);
		yarnClient.start();
		return yarnClient;
	}
}

