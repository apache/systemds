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

package org.apache.sysml.hops.globalopt;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.globalopt.gdfgraph.GDFGraph;
import org.apache.sysml.hops.globalopt.gdfgraph.GraphBuilder;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.utils.Explain;

/**
 * Main entry point for Global Data Flow Optimization. It is intended to be invoked after 
 * the initial runtime program was created (and thus with constructed hops and lops).
 * 
 * 
 */
public class GlobalOptimizerWrapper 
{	
	
	private static final Log LOG = LogFactory.getLog(GlobalOptimizerWrapper.class);
	private static final boolean LDEBUG = true; //local debug flag
	
	//supported optimizers
	public enum GlobalOptimizerType{
		ENUMERATE_DP,
		TRANSFORM,
	}
	
	//internal parameters
	private static final GlobalOptimizerType OPTIM = GlobalOptimizerType.ENUMERATE_DP; 
	
	static
	{
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("org.apache.sysml.hops.globalopt")
			      .setLevel((Level) Level.DEBUG);
		}
	}
	
	/**
	 * 
	 * @param prog
	 * @param rtprog
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws HopsException
	 * @throws LopsException  
	 */
	public static Program optimizeProgram(DMLProgram prog, Program rtprog) 
		throws DMLRuntimeException, HopsException, LopsException
	{
		LOG.debug("Starting global data flow optimization.");
		Timing time = new Timing(true);
		
		//create optimizer instance
		GlobalOptimizer optimizer = createGlobalOptimizer( OPTIM );
		
		//create global data flow graph
		Summary summary = new Summary();
		GDFGraph graph = GraphBuilder.constructGlobalDataFlowGraph(rtprog, summary);
		if( LOG.isDebugEnabled() ) {
			LOG.debug("EXPLAIN GDFGraph:\n" + Explain.explainGDFNodes(graph.getGraphRootNodes(),1));
		}
		
		//core global data flow optimization 
		graph = optimizer.optimize(graph, summary);
		
		//get the final runtime program
		rtprog = graph.getRuntimeProgram();
		
		//print global optimizer summary
		LOG.info( summary );
		
		LOG.debug("Finished global data flow optimization in " + time.stop() + " ms.");
		return rtprog;
	}
	
	/**
	 * 
	 * @param type
	 * @param graphCreator
	 * @return
	 * @throws HopsException 
	 * @throws DMLRuntimeException 
	 */
	private static GlobalOptimizer createGlobalOptimizer( GlobalOptimizerType type ) 
		throws HopsException, DMLRuntimeException
	{
		GlobalOptimizer optimizer = null;
		
		switch( type )
		{
			case ENUMERATE_DP: 
				optimizer = new GDFEnumOptimizer();
				break;
				
			//case TRANSFORM: 
			//	optimizer = new GlobalTransformationOptimizer(Strategy.CANONICAL);
			//	((GlobalTransformationOptimizer)optimizer).addRule(new BlockSizeRule());
			//	break;
			
			default:
				throw new HopsException("Unsupported global optimizer type: "+type+".");
		}
		
		return optimizer;
	}
}
