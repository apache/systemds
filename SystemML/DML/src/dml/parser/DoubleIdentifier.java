package dml.parser;

import java.util.HashMap;

import dml.parser.Expression.Kind;
import dml.utils.LanguageException;

public class DoubleIdentifier extends ConstIdentifier {

	private double _val;
	
	public DoubleIdentifier(double val){
		super();
		 _val = val;
		_kind = Kind.Data;
	}
	
	public DoubleIdentifier(DoubleIdentifier d){
		super();
		 _val = d.getValue();
		_kind = Kind.Data;
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	
	public double getValue(){
		return _val;
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
