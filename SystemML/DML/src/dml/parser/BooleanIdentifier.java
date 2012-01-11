package dml.parser;

import java.util.HashMap;

import dml.utils.LanguageException;

public class BooleanIdentifier extends ConstIdentifier {

	private boolean _val;
	
	public BooleanIdentifier(boolean val){
		super();
		 _val = val;
		_kind = Kind.Data;
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	
	public BooleanIdentifier(BooleanIdentifier b){
		super();
		 _val = b.getValue();
		_kind = Kind.Data;
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
