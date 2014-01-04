/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class InterestingPropertyCombination 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Map<String, InterestingProperty> properties = new HashMap<String, InterestingProperty>();

	public Set<InterestingProperty> getProperties() {
		return new HashSet<InterestingProperty> (properties.values());
	}

	public void setProperties(Set<InterestingProperty> properties) {
		for(InterestingProperty p : properties) {
			this.properties.put(p.getName(), p);
		}
	}
	
	public InterestingProperty getPropertyByName(String name) {
		return this.properties.get(name);
	}
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		
		buffer.append("IPC [");
		for(InterestingProperty p : this.getProperties()) 
			buffer.append(p.getValueString() + " ");
		buffer.append("]");
		return buffer.toString();
	}
}
