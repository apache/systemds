package com.ibm.bi.dml.parser;

import com.ibm.bi.dml.utils.LanguageException;


public class DoubleIdentifier extends ConstIdentifier {

	private double _val;
	
	
	public DoubleIdentifier(double val){
		super();
		 _val = val;
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.DOUBLE);
	}
	
	public DoubleIdentifier(DoubleIdentifier d){
		super();
		 _val = d.getValue();
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.DOUBLE);
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
