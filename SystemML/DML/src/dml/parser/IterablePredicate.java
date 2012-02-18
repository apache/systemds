package dml.parser;

import java.util.HashMap;

import dml.utils.LanguageException;

public class IterablePredicate extends Expression {
	private DataIdentifier _iterVar;	// variable being iterated over
	private Expression _fromExpr;
	private Expression _toExpr;
	private Expression _incrementExpr;
	private HashMap<String,String> _parforParams;
		
	public IterablePredicate(DataIdentifier iterVar, Expression fromExpr, Expression toExpr, Expression incrementExpr, HashMap<String,String> parForParamValues)
	{
		_iterVar = iterVar;
		_fromExpr = fromExpr;
		_toExpr = toExpr;
		_incrementExpr = incrementExpr;
		
	}
		
	public String toString(){
		
		String retVal = "(" + _iterVar + " in seq(" + _fromExpr.toString() + "," + _toExpr.toString() + "," + _incrementExpr.toString();
		if (_parforParams != null && _parforParams.size() > 0){
			for (String key : _parforParams.keySet()){
				retVal += "," + key + "=" + _parforParams.get(key).toString();
			}
		}
		retVal = retVal + ")";
		return retVal;
	}
	
	 
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		return result;
	}

	 
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariable(_iterVar.getName(), _iterVar);
	 	return result;
	}

	@Override
	public Expression rewriteExpression(String prefix) throws LanguageException {
		//DataIdentifier newIterVar = (DataIdentifier)_iterVar.rewriteExpression(prefix);
		//return new IterablePredicate(newIterVar, _from, _to, _increment);
		throw new LanguageException("rewriteExpression not supported for IterablePredicate");
		
	}

	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids) throws LanguageException {
		
		// check the variable has either 1) not been defined already OR 2) defined as integer scalar   
		if (ids.containsKey(_iterVar.getName())){
			DataIdentifier otherIdentifier = ids.get(_iterVar.getName());
			if (!otherIdentifier.getDataType().equals(DataType.SCALAR) || !otherIdentifier.getDataType().equals(ValueType.INT)){
				throw new LanguageException("iterable predicate in for loop " + _iterVar.getName() + " must be a scalar integer");
			}	
		}
		
		// set the values for DataIdentifer iterable variable
		_iterVar.setIntProperties();
			
		// add the iterVar to the variable set
		ids.put(_iterVar.getName(), _iterVar);
	}
		
	public DataIdentifier getIterVar() {
		return _iterVar;
	}

	public void setIterVar(DataIdentifier iterVar) {
		_iterVar = iterVar;
	}

	public Expression getFromExpr() {
		return _fromExpr;
	}

	public void setFromExpr(Expression from) {
		_fromExpr = from;
	}

	public Expression getToExpr() {
		return _toExpr;
	}

	public void setToExpr(Expression to) {
		_toExpr = to;
	}

	public Expression getIncrementExpr() {
		return _incrementExpr;
	}

	public void setIncrementExpr(Expression increment) {
		_incrementExpr = increment;
	}
	
	public HashMap<String,String> getParForParams(){
		return _parforParams;
	}

} // end class
