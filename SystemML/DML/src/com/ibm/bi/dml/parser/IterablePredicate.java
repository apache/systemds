package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.utils.LanguageException;


public class IterablePredicate extends Expression 
{
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
		
		_parforParams = parForParamValues;
	}
		
	public String toString()
	{ 
		StringBuffer sb = new StringBuffer();
		sb.append( "(" );
		sb.append( _iterVar.getName() );
		sb.append(" in seq(");
		sb.append(_fromExpr.toString());
		sb.append(",");
		sb.append(_toExpr.toString());
		sb.append(",");
		sb.append(_incrementExpr.toString());
		sb.append( ")" );
		if (_parforParams != null && _parforParams.size() > 0){
			for (String key : _parforParams.keySet())
			{
				sb.append( "," );
				sb.append( key );
				sb.append( "=" );
				sb.append( _parforParams.get(key).toString() );
			}
		}
		sb.append( ")" );
		return sb.toString();
	}
	
	 
	public VariableSet variablesRead() 
	{
		VariableSet result = new VariableSet();
		result.addVariables( _fromExpr.variablesRead()      );
		result.addVariables( _toExpr.variablesRead()        );
		result.addVariables( _incrementExpr.variablesRead() );

		return result;
	}

	 
	public VariableSet variablesUpdated() 
	{
		VariableSet result = new VariableSet();
		result.addVariable(_iterVar.getName(), _iterVar);
		
	 	return result;
	}

	@Override
	public Expression rewriteExpression(String prefix) throws LanguageException {
		//DataIdentifier newIterVar = (DataIdentifier)_iterVar.rewriteExpression(prefix);
		//return new IterablePredicate(newIterVar, _from, _to, _increment);
		LOG.error(this.printErrorLocation() + "rewriteExpression not supported for IterablePredicate");
		throw new LanguageException(this.printErrorLocation() + "rewriteExpression not supported for IterablePredicate");
		
	}

	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException 
	{		
		//1) VALIDATE ITERATION VARIABLE (index)
		// check the variable has either 1) not been defined already OR 2) defined as integer scalar   
		if (ids.containsKey(_iterVar.getName())){
			DataIdentifier otherDI = ids.get(_iterVar.getName());
			if( otherDI.getDataType() != DataType.SCALAR || otherDI.getValueType() != ValueType.INT ){
				LOG.error(this.printErrorLocation() + "iterable predicate in for loop '" + _iterVar.getName() + "' must be a scalar integer");
				throw new LanguageException(this.printErrorLocation() + "iterable predicate in for loop '" + _iterVar.getName() + "' must be a scalar integer");
			}	
		}
		
		// set the values for DataIdentifer iterable variable
		_iterVar.setIntProperties();
			
		// add the iterVar to the variable set
		ids.put(_iterVar.getName(), _iterVar);
		
		
		//2) VALIDATE FOR PREDICATE in (from, to, increment)		
		//recursively validate the individual expression
		_fromExpr.validateExpression(ids, constVars);
		_toExpr.validateExpression(ids, constVars);
		_incrementExpr.validateExpression(ids, constVars);
		
		//check for scalar expression output
		checkNumericScalarOutput( _fromExpr );
		checkNumericScalarOutput( _toExpr );
		checkNumericScalarOutput( _incrementExpr );
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
	
	public void setParForParams(HashMap<String,String> params){
		_parforParams = params;
	}
	
	public static String[] createIterablePredicateVariables( String varName, Lops from, Lops to, Lops incr )
	{
		String[] ret = new String[4]; //varname, from, to, incr
		
		ret[0] = varName;
		
		if( from.getType()==Lops.Type.Data )
			ret[1] = from.getOutputParameters().getLabel();
		if( to.getType()==Lops.Type.Data )
			ret[2] = to.getOutputParameters().getLabel();
		if( incr.getType()==Lops.Type.Data )
			ret[3] = incr.getOutputParameters().getLabel();
		
		return ret;
	}

	private void checkNumericScalarOutput( Expression expr )
		throws LanguageException
	{
		if( expr == null || expr.getOutput() == null )
			return;
		
		Identifier ident = expr.getOutput();
		if( ident.getDataType() == DataType.MATRIX || ident.getDataType() == DataType.OBJECT ||
			(ident.getDataType() == DataType.SCALAR && (ident.getValueType() == ValueType.BOOLEAN || 
					                                    ident.getValueType() == ValueType.STRING || 
					                                    ident.getValueType() == ValueType.OBJECT)) )
		{
			LOG.error(this.printErrorLocation() + "expression in iterable predicate in for loop '" + expr.toString() + "' must return a numeric scalar");
			throw new LanguageException(this.printErrorLocation() + "expression in iterable predicate in for loop '" + expr.toString() + "' must return a numeric scalar");
		}
	}

} // end class
