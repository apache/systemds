/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class DoubleObject extends ScalarObject 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private double _value;

	public DoubleObject(double val){
		this(null,val);
	}

	public DoubleObject(String name, double val){
		super(name, ValueType.DOUBLE);
		_value = val;
	}

	public double getDoubleValue(){
		return _value;
	}

	public int getIntValue(){
		return UtilFunctions.toInt(_value);
	}
	
	public long getLongValue() {
		return UtilFunctions.toLong(_value);
	}
	
	public Object getValue(){
		return _value;
	}
	
	public String getStringValue(){
		return Double.toString(_value);
	}

	public String toString() { 
		return getStringValue();
	}

	public boolean getBooleanValue(){
		return (_value != 0);
	}

	@Override
	public String getDebugName() {
		return null;
	}
}
