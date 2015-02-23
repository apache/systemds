/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.cost.CostEstimationWrapper;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFGraph;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFNode;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFNode.NodeType;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;

/**
 * Global data flow optimization via enumeration-based optimizer (dynamic programming). 
 * 
 */
public class GDFEnumOptimizer extends GlobalOptimizer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final Log LOG = LogFactory.getLog(GDFEnumOptimizer.class);
	
	//internal configuration parameters //TODO remove -1 
	public static final int[] BLOCK_SIZES         = new int[]{-1,1000,2000,4000};
	public static final int[] REPLICATION_FACTORS = new int[]{1,3,5};
		
	//TODO cache for interesting properties
	
	private MemoStructure _memo = null; //plan memoization table
	private static long _enumeratedPlans = 0;
	private static long _prunedPlans = 0;
	
	
	public GDFEnumOptimizer( ) 
	{
		//init internal memo structure
		_memo = new MemoStructure();
	}

	@Override
	public GDFGraph optimize(GDFGraph gdfgraph) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, HopsException, LopsException 
	{
		Timing time = new Timing(true);
		Program prog = gdfgraph.getRuntimeProgram();
		ArrayList<GDFNode> roots = gdfgraph.getGraphRootNodes();
		
		//Step 1: baseline costing for branch and bound costs
		double initCosts = CostEstimationWrapper.getTimeEstimate(prog, new ExecutionContext());
	
		//Step 2: dynamic programming plan generation
		//(finally, pick optimal root plans over all interesting property sets)
		ArrayList<Plan> rootPlans = new ArrayList<Plan>(); 
		for( GDFNode node : roots ) {
			PlanSet ps = enumOpt(node, _memo, initCosts);
			Plan optPlan = ps.getPlanWithMinCosts();
			rootPlans.add( optPlan );
		}
		
		//check for final containment of independent roots and pick optimal
		//Plan = 
		
		//generate final runtime plan (w/ optimal config)
		//TODO apply final configurations to runtime plan, recompile, and cost
		
		double optCosts = CostEstimationWrapper.getTimeEstimate(prog, new ExecutionContext());
		
		
		//print optimization summary
		LOG.info("Optimization summary:");
		LOG.info("-- costs of intial plan: "+initCosts);
		LOG.info("-- costs of optimal plan: "+optCosts);
		LOG.info("-- # enumerated plans:    "+_enumeratedPlans);
		LOG.info("-- # pruned plans:        "+_prunedPlans);
		LOG.info("-- # of block compiles:   "+0);
		LOG.info("-- # of block costings:   "+0);
		LOG.info("-- optimization time:     "+String.format("%.3f", (double)time.stop()/1000)+" sec.");
		
		return gdfgraph;
	}
	
	/**
	 * Core dynamic programming enumeration algorithm
	 * for global data flow optimization.
	 * 
	 * @param node
	 * @param maxCosts
	 * @return
	 */
	public static PlanSet enumOpt( GDFNode node, MemoStructure memo, double maxCosts )
	{
		//memoization of already enumerated subgraphs
		if( memo.constainsEntry(node) )
			return memo.getEntry(node);
		
		//enumerate node plans
		PlanSet P = enumNodePlans( node );
		//System.out.println("Plans after enumNodePlan:\n"+P.toString());
		
		//combine local node plan with optimal child plans
		for( GDFNode c : node.getInputs() )
		{
			//recursive optimization
			PlanSet Pc = enumOpt( c, memo, maxCosts );
			P = P.crossProductChild(Pc);
			_enumeratedPlans += P.size();			
			
			//System.out.println("Plans after crossProduct:\n"+P.toString());
			
			//prune invalid plans
			pruneInvalidPlans( node, P );
		}
		
		//prune suboptimal plans
		pruneSuboptimalPlans( P, maxCosts );
		
		//memoization of created entries
		memo.putEntry(node, P);
		
		return P;
	}
	
	/**
	 * 
	 * @param node
	 * @return
	 */
	private static PlanSet enumNodePlans( GDFNode node )
	{
		ArrayList<Plan> plans = new ArrayList<Plan>();
		
		//ENUMERATE HOP PLANS
		//do nothing for scalars (leads to pass-through for something like sum)
		if(    node.getNodeType() == NodeType.HOP_NODE
			&& node.getDataType() == DataType.MATRIX ) 
		{
			//create cp plan (most interesting proeprties are irrelevant for CP)
			RewriteConfig rccp = new RewriteConfig(ExecType.CP, -1);
			InterestingProperties ipscp = rccp.deriveInterestingProperties();
			Plan cpplan = new Plan(ipscp, rccp, null);
			plans.add( cpplan );
			
			//create mr plans
			for( Integer bs : BLOCK_SIZES )
			{
				RewriteConfig rcmr = new RewriteConfig(ExecType.MR, bs);
				InterestingProperties ipsmr = rcmr.deriveInterestingProperties();
				Plan mrplan = new Plan(ipsmr, rcmr, null);
				plans.add( mrplan );
					
			}
		}
		//ENUMERATE LOOP PLANS
		else if( node.getNodeType() == NodeType.LOOP_NODE )
		{
			//TODO
		}
		//CREATE DUMMY CROSSBLOCK PLAN
		else if( node.getNodeType() == NodeType.CROSS_BLOCK_NODE )
		{
			//do nothing (leads to pass-through on crossProductChild)
		}
		
		return new PlanSet(plans);
	}
	
	/**
	 * 
	 * @param plans
	 */
	private static void pruneInvalidPlans( GDFNode node, PlanSet plans )
	{
		ArrayList<Plan> valid = new ArrayList<Plan>();
		
		//check each plan in planset for validity
		for( Plan plan : plans.getPlans() )
		{
			//a) check matching blocksizes if operation in MR
			if( !plan.checkValidBlocksizesInMR() )
				continue;
			
			if( !plan.checkValidFormatInMR(node) )
				continue;
				
			valid.add( plan );
		}
		
		//debug output
		int sizeBefore = plans.size();
		int sizeAfter = valid.size();
		_prunedPlans += (sizeBefore-sizeAfter);
		LOG.debug("Pruned invalid plans: "+sizeBefore+" --> "+sizeAfter);
		
		plans.setPlans( valid );
	}
	
	/**
	 * 
	 * @param plans
	 * @param maxCosts 
	 */
	private static void pruneSuboptimalPlans( PlanSet plans, double maxCosts )
	{
		//TODO costing of all plans incl containment check
		
		//build and probe for optimal plans (hash-groupby on IPC, min costs) 
		HashMap<InterestingProperties, Plan> probeMap = new HashMap<InterestingProperties, Plan>();
		for( Plan p : plans.getPlans() )
		{
			//max cost pruning filter (branch-and-bound)
			if( p.getCosts() > maxCosts )
				continue;
			
			//best plan per IPS pruning filter
			Plan best = probeMap.get(p.getInterestingProperties());
			if( best!=null && p.getCosts() > best.getCosts() )
				continue;
			
			//add plan as best per IPS
			probeMap.put(p.getInterestingProperties(), p);
		}
		
		//copy over plans per IPC into one plan set
		ArrayList<Plan> optimal = new ArrayList<Plan>(probeMap.values());
		
		int sizeBefore = plans.size();
		int sizeAfter = optimal.size();
		_prunedPlans += (sizeBefore-sizeAfter);
		LOG.debug("Pruned suboptimal plans: "+sizeBefore+" --> "+sizeAfter);
		
		plans.setPlans(optimal);
	}
	
	
	
}
