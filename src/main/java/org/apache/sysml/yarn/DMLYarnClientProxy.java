/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.yarn;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils.OptimizationLevel;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.yarn.ropt.ResourceConfig;
import org.apache.sysml.yarn.ropt.ResourceOptimizer;
import org.apache.sysml.yarn.ropt.YarnClusterAnalyzer;
import org.apache.sysml.yarn.ropt.YarnClusterConfig;
import org.apache.sysml.yarn.ropt.YarnOptimizerUtils;
import org.apache.sysml.yarn.ropt.YarnOptimizerUtils.GridEnumType;

/**
 * The sole purpose of this class is to serve as a proxy to
 * DMLYarnClient to handle class not found exceptions or any
 * other issues of spawning the DML App Master.
 * 
 */
public class DMLYarnClientProxy 
{	
	
	private static final Log LOG = LogFactory.getLog(DMLYarnClientProxy.class);

	//flags to enabled resource optimizer / debugging (this does not disable external configurations)
	protected static boolean RESOURCE_OPTIMIZER = false;
	protected static boolean LDEBUG = false;
	
	static
	{
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("org.apache.sysml.yarn")
			      .setLevel((Level) Level.DEBUG);
		}
	}
	
	/**
	 * 
	 * @param dmlScriptStr
	 * @param conf
	 * @param allArgs
	 * @return
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 */
	public static boolean launchDMLYarnAppmaster(String dmlScriptStr, DMLConfig conf, String[] allArgs, Program rtprog) 
		throws IOException, DMLRuntimeException
	{
		boolean ret = false;
		
		try
		{
			//check for need for resource optimization
			//if( conf.getIntValue(DMLConfig.YARN_APPMASTERMEM) < 0 )
			//	RESOURCE_OPTIMIZER = true;
			
			//optimize resources (and update configuration)
			if( DMLAppMasterUtils.isResourceOptimizerEnabled() )
			{
				LOG.warn("Optimization level '" + OptimizationLevel.O3_LOCAL_RESOURCE_TIME_MEMORY + "' " +
						"is still in experimental state and not intended for production use.");
				
				YarnClusterConfig cc = YarnClusterAnalyzer.getClusterConfig();
				DMLAppMasterUtils.setupRemoteParallelTasks( cc );
				ArrayList<ProgramBlock> pb = DMLAppMasterUtils.getRuntimeProgramBlocks(rtprog);
				ResourceConfig rc = ResourceOptimizer.optimizeResourceConfig( pb, cc, 
						                 GridEnumType.HYBRID_MEM_EXP_GRID, GridEnumType.HYBRID_MEM_EXP_GRID );
				conf.updateYarnMemorySettings(String.valueOf(YarnOptimizerUtils.toMB(rc.getCPResource())), rc.serialize());
				//alternative: only use the max mr memory for all statement blocks
				//conf.updateYarnMemorySettings(String.valueOf(rc.getCPResource()), String.valueOf(rc.getMaxMRResource()));
			}
			
			//launch dml yarn app master
			DMLYarnClient yclient = new DMLYarnClient(dmlScriptStr, conf, allArgs);
			ret = yclient.launchDMLYarnAppmaster();
		}
		catch(NoClassDefFoundError ex)
		{
			LOG.warn("Failed to instantiate DML yarn client " +
					 "(NoClassDefFoundError: "+ex.getMessage()+"). " +
					 "Resume with default client processing.");
			ret = false;
		}
		
		return ret;
	}
}
