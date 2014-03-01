/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.ArrayList;
import java.util.List;

import com.ibm.bi.dml.hops.Hop;

public class SummaryEntry 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private List<String> action = new ArrayList<String>();
	private List<Long> planCount = new ArrayList<Long>();
	private Hop operator;
	private int level = 0;
	
	public SummaryEntry(int lvl, Hop hop) {
		this.level = lvl;
		this.operator = hop;
	}
	
	public List<Long> getMemEntryHistory() {
		return planCount;
	}

	public void addEntryVal(Long value) {
		this.planCount.add(value);
	}

	public void addAction(String action) {
		this.action.add(action);
	}
	
	public List<String> getAction() {
		return action;
	}

	public Hop getOperator() {
		return operator;
	}

	public void setOperator(Hop operator) {
		this.operator = operator;
	}

	public int getLevel() {
		return level;
	}

	public void increaseLevel() {
		this.level++;
	}
	
	public void decreaseLevel() {
		this.level--;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for(int i = 0; i <= this.level; i++) {
			builder.append("-");
		}
		builder.append(this.operator.get_name() + ", " + this.operator.getClass().getSimpleName() + ", ID: " + this.operator.getHopID() + ", ");
		builder.append("[");
		for (int i = 0; i < this.planCount.size(); i++) {
			Long l = this.planCount.get(i);
			String action = this.action.get(i);
			String appendString = "after " + action + ": " + l;
			builder.append(appendString);
			if (i < (this.planCount.size() - 1)) {
				 builder.append(", ");
			} 
		}
		builder.append("]\n");
		return builder.toString();
	}

}
