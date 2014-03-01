/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.LinkedList;
import java.util.List;

import com.ibm.bi.dml.hops.Hop;

/**
 * Captures optimization. 
 */
public class Summary 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private List<SummaryEntry> entries = new LinkedList<SummaryEntry>();
	private long costFunctionCalls;
	private long prunedInvalid;
	private long prunedSuboptimal;
	private long generatedPlans;
	private long descends;
	private int entryLevel = 0;
	private int numberOfConfigs = 0;
	private int numberOfInterestingProperties;
	
	public SummaryEntry startRecording(Hop hop) {
		SummaryEntry entry = new SummaryEntry(this.entryLevel, hop);
		return entry;
	}
	
	public void stopRecording(SummaryEntry entry) {
		this.entries.add(entry);
	}

	public long getCostFunctionCalls() {
		return costFunctionCalls;
	}
	
	public void setCostFunctionCalls(long costFunctionCalls) {
		this.costFunctionCalls = costFunctionCalls;
	}
	
	public void incrementCostFunctionCalls() {
		this.costFunctionCalls++;
	}
	
	public long getPrunedInvalid() {
		return prunedInvalid;
	}
	
	public void setPrunedInvalid(long prunedInvalid) {
		this.prunedInvalid = prunedInvalid;
	}
	
	public void incrementPrunedInvalid() {
		this.prunedInvalid++;
	}
	
	public long getGeneratedPlans() {
		return generatedPlans;
	}
	
	public void setGeneratedPlans(long generatedPlans) {
		this.generatedPlans = generatedPlans;
	}
	
	public void incrementGeneratedPlan() {
		this.generatedPlans++;
	}
	
	public long getPrunedSuboptimal() {
		return prunedSuboptimal;
	}

	public void setPrunedSuboptimal(long prunedSuboptimal) {
		this.prunedSuboptimal = prunedSuboptimal;
	}
	
	public void incrementPrunedSuboptimal() {
		this.prunedSuboptimal++;
	}
	
	public void addPrunedCounter(int pruned) {
		this.prunedSuboptimal += pruned;
	}
	
	public long getDescends() {
		return descends;
	}

	public void setDescends(long descends) {
		this.descends = descends;
	}
	
	public void incrementDescents() {
		this.descends++;
	}
	
	public void increaseLevel() {
		this.entryLevel++;
	}
	
	public void decreaseLevel() {
		this.entryLevel--;
	}
	
	public int getNumberOfConfigs() {
		return numberOfConfigs;
	}

	public void setNumberOfConfigs(int numberOfConfigs) {
		this.numberOfConfigs = numberOfConfigs;
	}

	public int getNumberOfInterestingProperties() {
		return numberOfInterestingProperties;
	}

	public void setNumberOfInterestingProperties(int numberOfInterestingProperties) {
		this.numberOfInterestingProperties = numberOfInterestingProperties;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("number of configs: " + this.numberOfConfigs);
		builder.append("\n");
		builder.append("number of interesting property combinations: " + this.numberOfInterestingProperties);
		builder.append("\n");
		builder.append("generated subplans plans: " + this.generatedPlans);
		builder.append("\n");
		builder.append("pruned invalid subplans: " + this.prunedInvalid);
		builder.append("\n");
		builder.append("pruned suboptimal subplans: " + this.prunedSuboptimal);
		builder.append("\n");
		builder.append("number of cost function calls: " + this.costFunctionCalls);
		builder.append("\n");
		builder.append("memo entry history: [\n");
		for(int i = 0; i < this.entries.size(); i++) {
			SummaryEntry e = this.entries.get(i);
			builder.append(e);
		}
		builder.append("]\n");
		builder.append("number of descends: " + this.descends);
		builder.append("\n");
	
			
		return builder.toString();	
	}
	
}
