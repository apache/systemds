package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.parser.Expression.Kind;
import com.ibm.bi.dml.utils.LanguageException;


public class IntIdentifier extends ConstIdentifier {

	private long _val;
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	
	public IntIdentifier(long val){
		super();
		 _val = val;
		_kind = Kind.Data;
	}
	
	public IntIdentifier(IntIdentifier i){
		super();
		 _val = i.getValue();
		_kind = Kind.Data;
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
