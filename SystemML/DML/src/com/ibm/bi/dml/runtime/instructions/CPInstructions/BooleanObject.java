/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;


public class BooleanObject extends ScalarObject  
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean _value;

	public BooleanObject(boolean val){
		this(null,val);
	}

	public BooleanObject(String name,boolean val){
		super(name, ValueType.BOOLEAN);
		_value = val;
	}

	public int getIntValue(){
		throw new UnsupportedOperationException();
	}

	public double getDoubleValue(){
		throw new UnsupportedOperationException();
	}

	public boolean getBooleanValue(){
		return _value;
	}

	public Object getValue(){
		return _value;
	}

	public String getStringValue(){
		return Boolean.toString(_value);
	}
	
	public String toString() { 
		return getStringValue();
	}

	@Override
	public long getLongValue() {
		throw new UnsupportedOperationException();
	}

	@Override
	public String getDebugName() {
		// TODO Auto-generated method stub
		return null;
	}
	
}
