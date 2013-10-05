/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;



public class BooleanIdentifier extends ConstIdentifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean _val;
	
	
	public BooleanIdentifier(boolean val){
		super();
		 _val = val;
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.BOOLEAN);
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	
	public boolean getValue(){
		return _val;
	}
	
	public String toString(){
		return Boolean.toString(_val);
	}
	
	@Override
	public VariableSet variablesRead() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
		// TODO Auto-generated method stub
		return null;
	}
}
