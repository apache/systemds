/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.yarn.api.records.ApplicationId;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.yarn.ropt.ResourceConfig;
import com.ibm.bi.dml.yarn.ropt.ResourceOptimizer;
import com.ibm.bi.dml.yarn.ropt.YarnClusterConfig;

public class DMLAppMasterUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static ResourceConfig _rc = null;
	private static HashMap<ProgramBlock, Long> _rcMap = null;
	

	/**
	 * 
	 * @param conf
	 * @param appId
	 * @return
	 */
	public static String constructHDFSWorkingDir(DMLConfig conf, ApplicationId appId)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( conf.getTextValue(DMLConfig.SCRATCH_SPACE) );
		sb.append( Lop.FILE_SEPARATOR );
		sb.append( appId );
		sb.append( Lop.FILE_SEPARATOR );
		return sb.toString();	
	}
	
	/**
	 * 
	 * @param conf
	 */
	public static void setupConfigRemoteMaxMemory(DMLConfig conf)
	{
		//set remote max memory (if in yarn appmaster context)
		if( DMLScript.isActiveAM() ){
			
			if( DMLYarnClientProxy.RESOURCE_OPTIMIZER )
			{
				//handle optimized memory (mr memory budget per program block)
				//ensure cluster has been analyzed
				InfrastructureAnalyzer.getRemoteMaxMemoryMap();
				
				String memStr = conf.getTextValue(DMLConfig.YARN_MAPREDUCEMEM);
				ResourceConfig rc = ResourceConfig.deserialize(memStr);
				_rc = rc; //keep resource config for later program mapping
			}
			else
			{
				//handle user configuration
				if( conf.getIntValue(DMLConfig.YARN_MAPREDUCEMEM)>0 )
				{
					//ensure cluster has been analyzed
					InfrastructureAnalyzer.getRemoteMaxMemoryMap();
					
					//set max map and reduce memory (to be used by the compiler)
					//see GMR and parfor EMR and DPEMR for runtime configuration
					long mem = ((long)conf.getIntValue(DMLConfig.YARN_MAPREDUCEMEM)) * 1024 * 1024;
					InfrastructureAnalyzer.setRemoteMaxMemoryMap(mem);
					InfrastructureAnalyzer.setRemoteMaxMemoryReduce(mem);		
				}
			}
		}
	}
	
	/**
	 * 
	 * @param prog
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	public static void setupProgramMappingRemoteMaxMemory(Program prog) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if( DMLScript.isActiveAM() && DMLYarnClientProxy.RESOURCE_OPTIMIZER )
		{
			ArrayList<ProgramBlock> pbProg = getRuntimeProgramBlocks( prog ); 
			ArrayList<ProgramBlock> B = ResourceOptimizer.compileProgram( pbProg, _rc );
			
			_rcMap = new HashMap<ProgramBlock, Long>();
			for( int i=0; i<B.size(); i++ ){
				_rcMap.put(B.get(i), _rc.getMRResources(i));
			}
		}
	}
	
	/**
	 * 
	 * @param sb
	 */
	public static void setupProgramBlockRemoteMaxMemory(ProgramBlock pb)
	{
		if( DMLScript.isActiveAM() && DMLYarnClientProxy.RESOURCE_OPTIMIZER )
		{
			if( _rcMap != null && _rcMap.containsKey(pb) ){ 
				//set max map and reduce memory (to be used by the compiler)
				long mem = _rcMap.get(pb);
				InfrastructureAnalyzer.setRemoteMaxMemoryMap(mem);
				InfrastructureAnalyzer.setRemoteMaxMemoryReduce(mem);			
				OptimizerUtils.setDefaultSize();
			}
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
			int memMB = -1;
			
			//obtain the current configuation (optimized or user-specified)
			if( DMLYarnClientProxy.RESOURCE_OPTIMIZER )
				memMB = (int)(InfrastructureAnalyzer.getRemoteMaxMemoryMap() / (1024*1024));
			else
				memMB = conf.getIntValue(DMLConfig.YARN_MAPREDUCEMEM);
			
			//set the memory configuration into the job conf
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
	

	/**
	 * 
	 * @param args
	 * @return
	 * @throws DMLException
	 */
	protected static ArrayList<ProgramBlock> getRuntimeProgramBlocks(Program prog) 
		throws DMLRuntimeException
	{			
		//construct single list of all program blocks including functions
		ArrayList<ProgramBlock> ret = new ArrayList<ProgramBlock>();
		ret.addAll(prog.getProgramBlocks());
		ret.addAll(prog.getFunctionProgramBlocks().values());
		
		return ret;
	}
	
	/**
	 * 
	 * @param cc
	 */
	protected static void setupRemoteParallelTasks( YarnClusterConfig cc )
	{
		int pmap = (int) cc.getNumCores();
		int preduce = (int) cc.getNumCores()/2;
		InfrastructureAnalyzer.setRemoteParallelMapTasks(pmap);
		InfrastructureAnalyzer.setRemoteParallelReduceTasks(preduce);
	}
	
	
}
