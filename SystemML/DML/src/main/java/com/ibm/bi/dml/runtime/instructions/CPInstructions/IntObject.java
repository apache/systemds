/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;

public class IntObject extends ScalarObject  
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int _value;

	public IntObject(int val){
		this(null,val);
	}

	public IntObject(String name,int val){
		super(name, ValueType.INT);
		_value = val;
	}

	public int getIntValue(){
		return _value;
	}

	public double getDoubleValue(){
		return (double) _value;
	}

	public long getLongValue(){
		return (long) _value;
	}
	
	public Object getValue(){
		return _value;
	}

	public boolean getBooleanValue(){
		throw new UnsupportedOperationException();
	}

	public String getStringValue(){
		return Integer.toString(_value);
	}

	public String toString() { 
		return getStringValue();
	}

	@Override
	public String getDebugName() {
		// TODO Auto-generated method stub
		return null;
	}

}
