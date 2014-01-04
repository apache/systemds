/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashSet;
import java.util.Set;

import com.ibm.bi.dml.hops.Hop;


public abstract class ConfigParam 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected String name;
	protected Integer value;
	protected Set<Integer> definedValues = new HashSet<Integer>();
	
	public ConfigParam(){
	}
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public Integer getValue() {
		return value;
	}
	public void setValue(Integer value) {
		this.value = value;
	}
	public Set<Integer> getDefinedValues() {
		return definedValues;
	}
	public void setDefinedValues(Integer... definedValues) {
		for(Integer l : definedValues) {
			this.definedValues.add(l);
		}
	}
	
	/**
	 * Creates possibly a set of interesting properties.
	 * @return
	 */
	public abstract Set<InterestingProperty> createsInterestingProperties();
	
	/** 
	 * In case an interesting property has to be enforced, what rewrites are necessary???	
	 * @param toCreate
	 * @return
	 */
	public abstract Rewrite requiresRewrite(InterestingProperty toCreate);
	
	/**
	 * Create an instance of a given class of a config param
	 * @param value
	 */
	public abstract ConfigParam createInstance(Integer value);
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		buffer.append("[");
		buffer.append(this.getName());
		buffer.append(", ");
		buffer.append(this.getValue());
		buffer.append(", def: [");
		for(Integer dv : this.getDefinedValues()) 
		{
			buffer.append(dv);
			buffer.append(", ");
		}
		buffer.append("]]");
		
		return buffer.toString();
	}

	public abstract void applyToHop(Hop hop);

	public abstract boolean isValidForOperator(Hop operator);

	public abstract String getValueString();

	public abstract ConfigParam extractParamFromHop(Hop hop);
}
