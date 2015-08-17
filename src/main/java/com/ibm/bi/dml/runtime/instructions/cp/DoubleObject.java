/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class DoubleObject extends ScalarObject 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -8525290101679236360L;

	private double _value;

	public DoubleObject(double val){
		this(null,val);
	}

	public DoubleObject(String name, double val){
		super(name, ValueType.DOUBLE);
		_value = val;
	}

	@Override
	public boolean getBooleanValue(){
		return (_value != 0);
	}

	@Override
	public long getLongValue() {
		return UtilFunctions.toLong(_value);
	}
	
	@Override
	public double getDoubleValue(){
		return _value;
	}
	
	@Override
	public String getStringValue(){
		return Double.toString(_value);
	}
	
	@Override
	public Object getValue(){
		return _value;
	}
	

	public String toString() { 
		return getStringValue();
	}

	@Override
	public String getDebugName() {
		return null;
	}
}
