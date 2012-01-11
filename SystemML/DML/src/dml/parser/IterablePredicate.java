package dml.parser;

public class IterablePredicate {
	Expression _expr;
	Identifier _iterVar;
	
	public IterablePredicate(Expression expr){
		_expr = expr;
	}
	
	public IterablePredicate(Identifier var, Expression expr){
		_expr = expr;
		_iterVar = var;
	}
	
	public Expression getPredicate(){
		return _expr;
	}
	
	public String toString(){
		return _expr.toString();
	}
	
	 
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_expr.variablesRead());
	 	return result;
	}

	 
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariables(_expr.variablesUpdated());
	 	return result;
	}
		
}
