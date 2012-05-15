package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.parser.Expression.Kind;
import com.ibm.bi.dml.utils.LanguageException;


public class StringIdentifier extends ConstIdentifier {

	private String _val;
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	public StringIdentifier(String val){
		super();
		 _val = val;
		_kind = Kind.Data;
	}
	
	public StringIdentifier(StringIdentifier s){
		super();
		 _val = s.getValue();
		_kind = Kind.Data;
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
