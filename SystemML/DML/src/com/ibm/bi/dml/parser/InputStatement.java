package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.utils.LanguageException;

 
public class InputStatement extends IOStatement{
	
	// rewrites statement to support function inlining (creates deep copy)
	public Statement rewriteStatement(String prefix) throws LanguageException {
		
		InputStatement newStatement = new InputStatement();
		
		// rewrite target variable name (creates deep copy)
		newStatement._id = (DataIdentifier)this._id.rewriteExpression(prefix);
		
		// rewrite Input filename expression (creates deep copy)
		Expression newFilenameExpr = _filenameExpr.rewriteExpression(prefix);
		newStatement.setFilenameExpr(newFilenameExpr);
		
		// rewrite InputStatement expr parameters (creates deep copies)
		HashMap<String,Expression> newExprParams = new HashMap<String,Expression>();
		for (String key : _exprParams.keySet()){
			Expression newExpr = _exprParams.get(key).rewriteExpression(prefix);
			newExprParams.put(key, newExpr);
		}	
		newStatement.setExprParams(newExprParams);
		return newStatement;
	}

	public InputStatement(){
		super();
	}
		
	public InputStatement(DataIdentifier t, Expression fname){
		super(t,fname);
	}

	public void initializeforwardLV(VariableSet activeIn){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		 sb.append(_id.toString() + " = " + Statement.INPUTSTATEMENT + " ( " );
		 sb.append(_filenameExpr.toString());
		 for (String key : _exprParams.keySet()){
			 sb.append(", " + key + "=" + _exprParams.get(key).toString());
		 }
		 sb.append(" );"); 
		 return sb.toString(); 
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		// add variables read by filename expression
		result.addVariables(_filenameExpr.variablesRead());
		
		// add variables read by parameter expressions
		for (String key : _exprParams.keySet())	
			result.addVariables(_exprParams.get(key).variablesRead()) ;
		
		// for LHS IndexedIdentifier, add variables for indexing expressions
		if (_id instanceof IndexedIdentifier) {
			IndexedIdentifier target = (IndexedIdentifier) _id;
			result.addVariables(target.variablesRead());
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		
		// add variable being populated by InputStatement
		result.addVariable(_id.getName(),_id);
	 	return result;
	}
}
