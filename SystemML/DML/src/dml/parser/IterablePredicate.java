package dml.parser;

import java.util.HashMap;

import dml.utils.LanguageException;

public class IterablePredicate extends Expression {
	private DataIdentifier _iterVar;	// variable being iterated over
	private int _from;
	private int _to;
	private int _increment;
		
	public IterablePredicate(DataIdentifier iterVar, int from, int to, int increment){
		_iterVar = iterVar;
		_from = from;
		_to = to;
		_increment = (_from < _to) ? increment : (increment * -1);
		
		// create the expression to initialize the variable 
		
		// create the expression to increment the variable
		
	}
			
	public String toString(){
		return "(" + _iterVar + " in seq(" + _from + "," + _to + "," + _increment + ")";
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
		DataIdentifier newIterVar = (DataIdentifier)_iterVar.rewriteExpression(prefix);
		return new IterablePredicate(newIterVar, _from, _to, _increment);
		
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

	public int getFrom() {
		return _from;
	}

	public void setFrom(int from) {
		_from = from;
	}

	public int getTo() {
		return _to;
	}

	public void setTo(int to) {
		_to = to;
	}

	public int getIncrement() {
		return _increment;
	}

	public void setIncrement(int increment) {
		_increment = increment;
	}

} // end class
