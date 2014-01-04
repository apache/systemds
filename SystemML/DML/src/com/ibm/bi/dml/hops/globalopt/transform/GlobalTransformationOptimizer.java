/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.transform;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.cost.CostEstimationWrapper;
import com.ibm.bi.dml.hops.globalopt.GlobalOptimizer;
import com.ibm.bi.dml.hops.globalopt.HopsDag;
import com.ibm.bi.dml.hops.globalopt.MaximalGlobalGraphCreator;
import com.ibm.bi.dml.hops.globalopt.PrintVisitor;
import com.ibm.bi.dml.hops.globalopt.RewriteRule;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.Program;

/**
 * Global DML data flow optimizer. Takes a {@link DMLProgram} as input on which local optimizations 
 * have already been executed. The optimization process is executed in a transformation based manner. 
 * That means the optimizer maintains a list of {@link RewriteRule}s sorted by impact (TODO: how to define the order?).
 * This list is applied one by one only optimizations that reduce the execution cost are applied to the actual program.
 * TODO: guarantee?
 */
public class GlobalTransformationOptimizer extends GlobalOptimizer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(GlobalTransformationOptimizer.class);
	
	private MaximalGlobalGraphCreator globalGraphCreator;
	private List<RewriteRule> rules;
	private Strategy strategy;
	public enum Strategy {SIMPLE, CANONICAL};
	
	public GlobalTransformationOptimizer(Strategy strategy) {
		this.globalGraphCreator = new MaximalGlobalGraphCreator(); //TODO consolidate
		this.rules = new LinkedList<RewriteRule>();
		this.strategy = strategy;
	}
	
	/**
	 * (1) Creates a map of maximal global graphs(mggs) (represented by {@link HopsDag})
	 * (2) Iterates over these mggs in order to optimize them independently
	 * (3) compute the initial cost
	 * (4) for each mgg apply all the rewrites (the heaviest impact first)
	 * (5) cost the rewritten candidate
	 * (6) compare to opt cost and replace the opt candidate in case the candidateCost < optCost.
	 * @param program
	 * @return
	 * @throws HopsException
	 */
	public Program optimize(DMLProgram prog, Program rtprog) 
		throws HopsException 
	{
		Map<String, HopsDag> globalGraphSet = this.globalGraphCreator.createGraph(prog);
		
		for(Entry<String, HopsDag> e : globalGraphSet.entrySet()) {
			HopsDag optPlan = e.getValue();
			
			switch(this.strategy){ 
			
			case CANONICAL: 
				optimizeCanonical(optPlan);
				break;
			case SIMPLE:
			default:
				optimizeSimple(optPlan);
			}
			
			PrintVisitor visitor = new PrintVisitor(null);
			if(e.getValue() != null 
					&& e.getValue().getDagOutputs() != null 
					&& e.getValue().getDagOutputs().values().iterator().hasNext()) {
				e.getValue().getDagOutputs().values().iterator().next().accept(visitor);
			}
		}
		
		return rtprog;
	}

	/**
	 * Very expensive yet easy strategy to optimize.
	 * 
	(1) create a global graph g
	(2) take a set of rewrites and iterate over it
	(3) apply each rewrite r_i to g and get g'_ri
	(4) pick the g'_rj with min(cost(g'_rj) 
	(5) remove r_j from rewrite set 
	(6) repeat with (2)
	 * 
	 * @param optPlan
	 * @throws HopsException 
	 */
	private void optimizeCanonical(HopsDag optPlan) {
		List<Double> candidateCosts = new ArrayList<Double>();
		Map<Double, HopsDag> candidateMap = new HashMap<Double, HopsDag>();
		Map<HopsDag, RewriteRule> candidateToRule = new HashMap<HopsDag, RewriteRule>();
		Set<RewriteRule> ruleSet = new HashSet<RewriteRule>();
		ruleSet.addAll(this.rules);
		
		double optCost = this.getPlanCost(optPlan);
		if(LOG.isInfoEnabled()) {
			LOG.info("original plan cost: " + optCost);
		}
		while(!ruleSet.isEmpty())
		{
			//apply each rule in the rule set
			for(RewriteRule r : ruleSet) {
				HopsDag candidate = r.rewrite(optPlan);
				double candCost = this.getPlanCost(candidate);
				candidateMap.put(candCost, candidate);
				candidateCosts.add(candCost);
				candidateToRule.put(candidate, r);
			}
			//TODO: add a check to skip optPlan assignment if none of the rules gave an improvement
			//determine the optimal plan and remove that rule from the ruleSet
			if(candidateCosts != null && candidateCosts.size() > 0){
				Collections.sort(candidateCosts);
				double optimalCost = candidateCosts.get(0);
				if(LOG.isInfoEnabled()) {
					LOG.info("original rewrite cost: " + optimalCost);
				}
				
				HopsDag optCandidate = candidateMap.get(optimalCost);
				if( optimalCost <= optCost ) {
					optPlan = optCandidate;
					optCost = optimalCost;
				}
				RewriteRule toApply = candidateToRule.get(optCandidate);
//				toApply.applyChanges();
				ruleSet.remove(toApply);
				candidateCosts.clear();
				candidateMap.clear();
				candidateToRule.clear();
			}
		}
	}
	
	
	/**
	 * Iterate over all the source hops in optPlan and call cost estimator.
	 * Overall cost is the sum of all source hops cost.
	 * @param optPlan
	 * @return
	 */
	private double getPlanCost(HopsDag optPlan) 
	{
		double intermediateCost = 0.0;

		ArrayList<Hop> sourceList = new ArrayList<Hop>();
		sourceList.addAll(optPlan.getOriginalRootHops());
		try {
			double estimate = CostEstimationWrapper.getTimeEstimate(sourceList, new ExecutionContext());
			intermediateCost += estimate;
		} 
		catch (DMLException e) {
			LOG.error(e);
		}
		catch (IOException e) {
			LOG.error(e);
		}
		
		return intermediateCost;
	}

	private void optimizeSimple(HopsDag optPlan) {
		double optCost = this.getPlanCost(optPlan);
		for(RewriteRule rule : this.rules) {
			HopsDag candidate = rule.rewrite(optPlan);
			double candidateCost = this.getPlanCost(candidate);
			if( optCost >= candidateCost ) {
				optCost = candidateCost;
				optPlan = candidate;
			}
		}
	}
	
	public void addRule(RewriteRule toAdd) {
		this.rules.add(toAdd);
	}
}
