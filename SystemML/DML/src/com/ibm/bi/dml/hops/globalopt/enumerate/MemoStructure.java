/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.lops.Lop;

/**
 * This MemoStructure is the central location for storing enumerated plans and severes basically
 * two purposes:
 * 
 * 1) Plan Memoization: Due to the DAG structure (where a single node is reachable over alternative 
 * paths), our top-down, recursive optimization procedure might visit an operator multiple times. 
 * The memo structure memoizes and reuses already generated plans.
 * 2) Config Combination Cache:  
 * 
 * TODO
 * 
 * The internal structure is as follows:
 * | LONG HopID | InterestingPropertyCombination IPC | 
 * 
 */
public class MemoStructure 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	private Map<Hop, Map<InterestingPropertySet, MemoEntry>> entries = new HashMap<Hop, Map<InterestingPropertySet, MemoEntry>>();
	private Long currentId = new Long(1);
	private Map<Long, Lop> planDirectory = new HashMap<Long, Lop>();
	
	public Map<InterestingPropertySet, MemoEntry> getEntry(Hop root) {
		return this.entries.get(root);
	}

	public MemoEntry getMemoEntry(Hop root, InterestingPropertySet combin) {
		Map<InterestingPropertySet, MemoEntry> map = this.entries.get(root);
		MemoEntry retVal = null;
		retVal = map.get(combin);
		return retVal;
	}
	
	public void add(Map<InterestingPropertySet, MemoEntry> nodePlans, Hop root) {
		this.entries.put(root, nodePlans);
	}

	public synchronized Long addPlan(Lop toAdd){
		Long retVal = new Long(currentId);
		this.planDirectory.put(currentId++, toAdd);
		return retVal;
	}
	
	public Lop getPlan(Long planId) {
		return this.planDirectory.get(planId);
	}
	
	public void setPlan(Long planId, Lop toOverwrite) {
		this.planDirectory.put(planId, toOverwrite);
	}
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		buffer.append("Memo: \n");
		buffer.append("number of entries: " + this.entries.size() + "\n");
		for(Entry<Hop, Map<InterestingPropertySet, MemoEntry>> e : this.entries.entrySet()) {
			Hop root = e.getKey();
			Map<InterestingPropertySet, MemoEntry> value = e.getValue();
			
			buffer.append("root hop: " + root.get_name() + ", ID: " + root.getHopID() + ", class: " + root.getClass().getSimpleName());
			buffer.append("\n");
			buffer.append("number of interesting property combinations: " + value.size() +"\n");
			buffer.append("-------------------------\n");
			for(Entry<InterestingPropertySet, MemoEntry> valEntry : value.entrySet()) {
				buffer.append(valEntry.getKey());
				buffer.append(" - ");
				buffer.append(valEntry.getValue());
				buffer.append("\n");
			}
			buffer.append("-------------------------\n");
		}
		
		return buffer.toString();
	}

	public int size() {
		return entries.size();
	}
}
