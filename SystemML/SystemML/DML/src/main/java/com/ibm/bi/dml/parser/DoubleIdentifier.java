/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;



public class DoubleIdentifier extends ConstIdentifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private double _val;
	
	
	public DoubleIdentifier(double val, String filename, int blp, int bcp, int elp, int ecp){
		super();
		 _val = val;
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.DOUBLE);
        this.setAllPositions(filename, blp, bcp, elp, ecp);
	}
	
	public DoubleIdentifier(DoubleIdentifier d, String filename, int blp, int bcp, int elp, int ecp){
		super();
		 _val = d.getValue();
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.DOUBLE);
        this.setAllPositions(filename, blp, bcp, elp, ecp);
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	
	public double getValue(){
		return _val;
	}
	
	public void setValue(double v) {
		_val = v;
	}
	
	public String toString(){
		return Double.toString(_val);
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
