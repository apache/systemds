/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;



public class StringIdentifier extends ConstIdentifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String _val;
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	
	public StringIdentifier(String val, String filename, int blp, int bcp, int elp, int ecp){
		super();
		 _val = val;
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.STRING);
        this.setAllPositions(filename, blp, bcp, elp, ecp);
		
	}
	
	public StringIdentifier(StringIdentifier s){
		super();
		 _val = s.getValue();
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.STRING);
	}
	
	public String getValue(){
		return _val;
	}
	
	public String toString(){
		return _val;
	}
	
	@Override
	public VariableSet variablesRead() {
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
		return null;
	}
}
