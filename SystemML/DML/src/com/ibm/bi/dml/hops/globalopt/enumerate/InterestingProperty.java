/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.Set;

public interface InterestingProperty 
{
	public String getName(); 
	public Integer getValue();
	public void setValue(Integer toSet);
	public Set<Integer> getDefinedValues();
	
	public String getValueString();
	
}
