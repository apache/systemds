/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashSet;
import java.util.Set;



public class BlockSizeProperty implements InterestingProperty 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String NAME = "blockSize";
	private Set<Integer> definedValues = new HashSet<Integer>();
	private Integer value;
	
	//TODO: synchronize with configuration param
	public BlockSizeProperty(){
		this.definedValues.add(1000);
		this.definedValues.add(2000);
		this.definedValues.add(-1);
	}
	
	@Override
	public Set<Integer> getDefinedValues() {
		return this.definedValues;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public Integer getValue() {
		return value;
	}

	public void setValue(Integer value) {
		this.value = value;
	}

	public String getValueString() {
		return String.valueOf(this.getValue());
	}
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		buffer.append("[" + NAME + ", " + value + "]");
		return buffer.toString();
	}
}
