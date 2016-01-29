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

package org.apache.sysml.yarn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.OptimizerUtils.OptimizationLevel;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.yarn.ropt.ResourceConfig;
import org.apache.sysml.yarn.ropt.ResourceOptimizer;
import org.apache.sysml.yarn.ropt.YarnClusterConfig;

public class DMLAppMasterUtils 
{
	
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
	 * @throws DMLRuntimeException 
	 */
	public static void setupConfigRemoteMaxMemory(DMLConfig conf) 
		throws DMLRuntimeException
	{
		//set remote max memory (if in yarn appmaster context)
		if( DMLScript.isActiveAM() ){
			
			//set optimization level (for awareness of resource optimization)
			OptimizerUtils.setOptimizationLevel( conf.getIntValue(DMLConfig.OPTIMIZATION_LEVEL) );
	 		
			if( isResourceOptimizerEnabled() )
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
		if( DMLScript.isActiveAM() && isResourceOptimizerEnabled() )
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
		if( DMLScript.isActiveAM() && isResourceOptimizerEnabled() )
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
			if( isResourceOptimizerEnabled() )
				memMB = (int)(InfrastructureAnalyzer.getRemoteMaxMemoryMap() / (1024*1024));
			else
				memMB = conf.getIntValue(DMLConfig.YARN_MAPREDUCEMEM);
			
			//set the memory configuration into the job conf
			if( memMB > 0 ){ //ignored if negative
				String memOpts = "-Xmx"+memMB+"m -Xms"+memMB+"m -Xmn"+(int)(memMB/10)+"m";
						
				//set mapper heapsizes
				job.set( MRConfigurationNames.MR_MAP_JAVA_OPTS, memOpts );
				job.set( MRConfigurationNames.MR_MAP_MEMORY_MB, String.valueOf(DMLYarnClient.computeMemoryAllocation(memMB)) );
				
				//set reducer heapsizes
				job.set( "mapreduce.reduce.java.opts", memOpts );
				job.set( "mapreduce.reduce.memory.mb", String.valueOf(DMLYarnClient.computeMemoryAllocation(memMB)) );
			}
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public static boolean isResourceOptimizerEnabled()
	{
		return ( DMLYarnClientProxy.RESOURCE_OPTIMIZER
				|| OptimizerUtils.isOptLevel(OptimizationLevel.O3_LOCAL_RESOURCE_TIME_MEMORY) );
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
