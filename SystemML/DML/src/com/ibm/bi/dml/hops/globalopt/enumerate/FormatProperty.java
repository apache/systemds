/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.Set;

public class FormatProperty implements InterestingProperty
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String NAME = "format";
	private Integer value;
	
	//imposes ugly code duplication with {@see FormatParam} constants
	//TODO: refactor this to one location
	public static final Integer TEXT = 0;
	public static final Integer BINARY_BLOCK = 1;
	public static final Integer BINARY_CELL = 2;
	
	
	@Override
	public Set<Integer> getDefinedValues() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public Integer getValue() {
		return this.value;
	}

	@Override
	public void setValue(Integer toSet) {
		this.value = toSet;
	}

	public String getValueString() {
		String valString = "BINARY_BLOCK";
		if(value.equals(TEXT) ) {
			valString = "TEXT"; 
		} 
		if(value.equals(BINARY_CELL) ) {
			valString = "BINARY_CELL"; 
		}
		return valString;
	}
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		String valString = this.getValueString();
		buffer.append("[" + NAME + ", " + valString + "]");
		return buffer.toString();
	}
}
