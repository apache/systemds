/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;



public class IntIdentifier extends ConstIdentifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _val;
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	
	public IntIdentifier(long val, String filename, int blp, int bcp, int elp, int ecp){
		super();
		 _val = val;
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.INT);
        this.setAllPositions(filename, blp, bcp, elp, ecp);
	}
	
	public IntIdentifier(IntIdentifier i, String filename, int blp, int bcp, int elp, int ecp){
		super();
		 _val = i.getValue();
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.INT);
        this.setAllPositions(filename, blp, bcp, elp, ecp);
	}
	
	// Used only by the parser for unary operation
	public void multiplyByMinusOne() {
		_val = -1 * _val;
	}
	
	public long getValue(){
		return _val;
	}
	
	public String toString(){
		return Long.toString(_val);
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
