package com.ibm.bi.dml.runtime.controlprogram.parfor.stat;


//import org.apache.hadoop.mapred.ClusterStatus;
//import org.apache.hadoop.mapred.JobClient;
//import org.apache.hadoop.mapred.JobConf;

/**
 * Central place for analyzing and obtaining infrastructure properties
 * such as memory and number of logical processors.
 * 
 * TODO: system-specific analysis of local physical memory constraints
 * 
 * @author mboehm
 */
public class InfrastructureAnalyzer 
{
	//static local master node properties
	public static int  _localPar        = -1;
	public static long _localMaxMem     = -1;
	public static long _localJVMMaxMem  = -1;
	
	//static hadoop cluster properties
	public static int  _remotePar       = -1;
	public static int  _remoteParMap    = -1;
	public static int  _remoteParReduce = -1;
	public static long _remoteMaxMem    = -1;
	public static long _remoteJVMMaxMem = -1;
	
	
	//static initialization, called for each JVM (on each node)
	static 
	{
		//analyze local node properties
		_localPar       = Runtime.getRuntime().availableProcessors();
		_localMaxMem    = Runtime.getRuntime().maxMemory(); 
		_localJVMMaxMem = Runtime.getRuntime().maxMemory();
		
		//analyze remote Hadoop cluster properties
		/*try 
		{
			JobConf job = new JobConf(InfrastructureAnalyzer.class);
			JobClient client = new JobClient(job);
			ClusterStatus stat = client.getClusterStatus();
			if( stat != null ) //if in cluster mode
			{
				_remotePar = stat.getTaskTrackers();
				_remoteParMap = stat.getMaxMapTasks(); 
				_remoteParReduce = stat.getMaxReduceTasks(); 
				_remoteMaxMem = job.getMemoryForMapTask();//job.getMaxPhysicalMemoryForTask(); //job.getMaxVirtualMemoryForTask();
				_remoteJVMMaxMem = job.getLong("mapred.child.java.opts",1024*1024*1024); //1GB
			
				//stat.getMaxMemory();//max memory (job tracker)
			}		
		} 
		catch (IOException e) 
		{
			throw new RuntimeException("Unable to analyze infrastructure.",e);
		}*/
		
	}
	
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
	
	
	public static int getCkMaxCP() {
		// TODO Auto-generated method stub
		return 0;
	}

	public static int getCkMaxMR() {
		// TODO Auto-generated method stub
		return 0;
	}

	public static double getCmMax() {
		// TODO Auto-generated method stub
		return 0;
	}
}
