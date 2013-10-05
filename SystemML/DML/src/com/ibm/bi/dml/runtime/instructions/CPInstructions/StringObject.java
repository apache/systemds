/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;

public class StringObject extends ScalarObject 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String _value;

	public StringObject (String val){
		this(null,val);
	}

	public StringObject(String name, String val){
		super(name, ValueType.STRING);
		_value = val;
	}

	public double getDoubleValue(){
		throw new UnsupportedOperationException();
	}

	public int getIntValue(){
		throw new UnsupportedOperationException();
	}

	public long getLongValue(){
		throw new UnsupportedOperationException();
	}
	
	public Object getValue(){
		return _value;
	}
	
	public String getStringValue(){
		return _value;
	}

	public String toString() { 
		return getStringValue();
	}

	public boolean getBooleanValue(){
		throw new UnsupportedOperationException();
	}

	@Override
	public String getDebugName() {
		return _value;
	}


}
