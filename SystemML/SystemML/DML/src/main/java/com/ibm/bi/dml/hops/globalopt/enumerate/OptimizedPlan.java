/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.globalopt.MergeOp;
import com.ibm.bi.dml.hops.globalopt.enumerate.RewriteConfig.RewriteConfigType;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.runtime.controlprogram.Program;

/**
 * Internal representation of a (partial) {@link DMLProgram}. Consists of a reference to the root operator 
 * for this part of the program, a configuration for this node and a list of input plans.
 * TODO: naming
 * TODO: add Lop as subplans, DO NOT make this structure recursive
 */
public class OptimizedPlan 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Hop operator;
	private Lop generatedLop;
	private double cost;
	private double cumulatedCost;
	private RewriteConfigSet config;
	private RewriteConfigSet extractedConfig;
	private Program runtimeProgram;
	
	private List<Long> hopsOnPath = new ArrayList<Long>();
	
	private Map<RewriteConfigType, Rewrite> rewriteMap = new HashMap<RewriteConfigType, Rewrite>();
	
	
	private List<MemoEntry> inputPlans = new ArrayList<MemoEntry>();
	
	public Hop getOperator() {
		return operator;
	}
	
	public void setOperator(Hop operator) {
		this.operator = operator;
	}
	
	public RewriteConfigSet getConfig() {
		return config;
	}
	
	public void setConfig(RewriteConfigSet config) {
		this.config = config;
	}

	public List<MemoEntry> getInputPlans() {
		return inputPlans;
	}

	public void setInputPlans(List<MemoEntry> inputLops) {
		this.inputPlans = inputLops;
	}
	
	public void addInputLop(MemoEntry toAdd) {
		this.inputPlans.add(toAdd);
	}

	public Lop getGeneratedLop() {
		return generatedLop;
	}

	public void setGeneratedLop(Lop generatedLop) {
		this.generatedLop = generatedLop;
	}

	public void setCost(double cost) {
		this.cost = cost;
	}
	
	public double getCost() {
		return this.cost;
	}

	public double getCumulatedCost() {
		return cumulatedCost;
	}

	public void setCumulatedCost(double cumulatedCost) {
		this.cumulatedCost = cumulatedCost;
	}

	public void computeCumulatedCosts() {
		this.cumulatedCost = 0.0;
		this.cumulatedCost += this.cost;
		
		if(this.operator instanceof MergeOp) {
			for(MemoEntry p : inputPlans) {
				this.cumulatedCost += p.getCost();
			}
			this.cumulatedCost = this.cumulatedCost / this.inputPlans.size();
		}else {
			for(MemoEntry p : inputPlans) {
				this.cumulatedCost += p.getCost();
			}
		}
	}

	public Rewrite getRewrite(RewriteConfigType type) {
		return rewriteMap.get(type);
	}

	public void addRewrite(RewriteConfigType type, Rewrite rewrite) {
		Rewrite oldRewrite = this.rewriteMap.get(type);
		if(oldRewrite != null) {
//			throw new IllegalStateException("++++++++++++++++++++++++++++OVERWRITING OLD REWRITE: " + oldRewrite);
		} else {
			this.rewriteMap.put(type, rewrite);
		}
	}
	

	public Map<RewriteConfigType, Rewrite> getRewriteMap() {
		return rewriteMap;
	}

	public void setRewriteMap(Map<RewriteConfigType, Rewrite> rewriteMap) {
		this.rewriteMap = rewriteMap;
	}

	/**
	 * Apply all the rewrites and take the current configuration into account.
	 */
	public void applyRewrites() {
		for(Rewrite r : this.rewriteMap.values()) {
			r.apply(this);
		}
		
	}

	public RewriteConfigSet getExtractedConfig() {
		return extractedConfig;
	}

	public void setExtractedConfig(RewriteConfigSet extractedConfig) {
		this.extractedConfig = extractedConfig;
	}

	public Program getRuntimeProgram() {
		return runtimeProgram;
	}

	public void setRuntimeProgram(Program program) {
		this.runtimeProgram = program;
	}
	
	public void addHopsOnPath(Hop h) {
		this.hopsOnPath.add(h.getHopID());
	}
	
	public boolean sharesHopOnPath(Long toTest) {
		return this.hopsOnPath.contains(toTest);
	}
	
	public long getClosestSharedHop(List<Long> path) {
		  ListIterator<Long> iter = path.listIterator(path.size());
		  while(iter.hasPrevious()) {
			 long current = iter.previous();
			 if(this.sharesHopOnPath(current)) {
				 return current;
			 }
		  }
		return -1L;
	}
}
