/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.lops.Lop;

public class MemoEntry 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//redundant 
	private OptimizedPlan optPlan;
	private Hop rootHop;
	private Lop rootLop;
	private Long lopId;
	private double cost;
	private Configuration config;
	private InterestingPropertyCombination interestingProperties;
	
	public Hop getRootHop() {
		return rootHop;
	}
	
	public void setRootHop(Hop rootHop) {
		this.rootHop = rootHop;
	}
	
	public Lop getRootLop() {
		return rootLop;
	}
	
	public void setRootLop(Lop rootLop) {
		this.rootLop = rootLop;
	}
	
	public double getCost() {
		return cost;
	}
	
	public void setCost(double cost) {
		this.cost = cost;
	}
	
	public Configuration getConfig() {
		return config;
	}
	
	public void setConfig(Configuration config) {
		this.config = config;
	}
	
	public InterestingPropertyCombination getInterestingProperties() {
		return interestingProperties;
	}
	
	public void setInterestingProperties(
			InterestingPropertyCombination interestingProperties) {
		this.interestingProperties = interestingProperties;
	}
	
	public Long getLopId() {
		return lopId;
	}
	
	public void setLopId(Long lopId) {
		this.lopId = lopId;
	}
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		
		buffer.append("[");
		buffer.append("cost: " + cost + ", " + this.getConfig());
		buffer.append("\n");
		buffer.append("lop " + lopId);
		buffer.append("]");
		return buffer.toString();
	}

	public OptimizedPlan getOptPlan() {
		return optPlan;
	}

	public void setOptPlan(OptimizedPlan optPlan) {
		this.optPlan = optPlan;
	}
}
