/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.enumerate.GlobalEnumerationOptimizer;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFNode;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GraphBuilder;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;

/**
 * Main entry point for Global Data Flow Optimization. It is intended to be invoked after 
 * the initial runtime program was created (and thus with constructed hops and lops).
 * 
 * 
 */
public class GlobalOptimizerWrapper 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());
	
	//supported optimizers
	public enum GlobalOptimizerType{
		ENUMERATE_DP,
		TRANSFORM,
	}
	
	//internal parameters
	private static final GlobalOptimizerType OPTIM = GlobalOptimizerType.ENUMERATE_DP; 
	
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
		LOG.info("Starting global data flow optimization.");
		Timing time = new Timing(true);
		
		//create optimizer instance
		GlobalOptimizer optimizer = createGlobalOptimizer( OPTIM );
		
		//create global data flow graph
		ArrayList<GDFNode> roots = GraphBuilder.constructGlobalDataFlowGraph(rtprog);
		for( GDFNode node : roots )
			System.out.println(node+":\n"+node.explain(1));
		
		//core optimization FIXME work on new global data flow graph
		//rtprog = optimizer.optimize(prog, rtprog);
		
		LOG.info("Finished optimization in " + time.stop() + " ms.");
		return rtprog;
	}
	
	/**
	 * 
	 * @param type
	 * @param graphCreator
	 * @return
	 * @throws HopsException 
	 */
	private static GlobalOptimizer createGlobalOptimizer( GlobalOptimizerType type ) 
		throws HopsException
	{
		GlobalOptimizer optimizer = null;
		
		switch( type )
		{
			case ENUMERATE_DP: 
				optimizer = new GlobalEnumerationOptimizer();
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
