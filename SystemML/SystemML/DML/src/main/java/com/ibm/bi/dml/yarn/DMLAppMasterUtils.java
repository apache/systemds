/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn;

import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;

public class DMLAppMasterUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * 
	 * @param conf
	 */
	public static void setupConfigRemoteMaxMemory(DMLConfig conf)
	{
		//set remote max memory (if in yarn appmaster context)
		if( DMLScript.isActiveAM() && conf.getIntValue(DMLConfig.YARN_MAPREDUCEMEM)>0 ){
			//ensure cluster has been analyzed
			InfrastructureAnalyzer.getRemoteMaxMemoryMap();
			
			//set max map and reduce memory (to be used by the compiler)
			//see GMR and parfor EMR and DPEMR for runtime configuration
			long mem = ((long)conf.getIntValue(DMLConfig.YARN_MAPREDUCEMEM)) * 1024 * 1024;
			InfrastructureAnalyzer.setRemoteMaxMemoryMap(mem);
			InfrastructureAnalyzer.setRemoteMaxMemoryReduce(mem);
		}
	}
	
	/**
	 * 
	 * @param job
	 * @param conf
	 */
	public static void setupMRJobRemoteMaxMemory(JobConf job, DMLConfig conf)
	{
		if( DMLScript.isActiveAM() && conf.getBooleanValue(DMLConfig.YARN_APPMASTER) )
		{
			int memMB = conf.getIntValue(DMLConfig.YARN_MAPREDUCEMEM);
			
			if( memMB > 0 ){ //ignored if negative
				String memOpts = "-Xmx"+memMB+"m -Xms"+memMB+"m -Xmn"+(int)(memMB/10)+"m";
						
				//set mapper heapsizes
				job.set( "mapreduce.map.java.opts", memOpts );
				job.set( "mapreduce.map.memory.mb", String.valueOf((int)(memMB*1.5)) );
				
				//set reducer heapsizes
				job.set( "mapreduce.reduce.java.opts", memOpts );
				job.set( "mapreduce.reduce.memory.mb", String.valueOf((int)(memMB*1.5)) );
			}
		}
	}
	
}
