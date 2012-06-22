package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.parser.Expression.DataOp;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.utils.LanguageException;


public class InputStatement extends IOStatement{
	
	public static final String[] READ_VALID_PARAM_NAMES = 
		{ IO_FILENAME, READROWPARAM, READCOLPARAM, READNUMNONZEROPARAM, FORMAT_TYPE,
			ROWBLOCKCOUNTPARAM, COLUMNBLOCKCOUNTPARAM, DATATYPEPARAM, VALUETYPEPARAM, DESCRIPTIONPARAM }; 
	
	public InputStatement(){
		super();
	}
	
	public InputStatement(DataOp op){
		super (op);
	}

	public static boolean isValidParamName(String key){
		for (String paramName : READ_VALID_PARAM_NAMES)
			if (paramName.equals(key)){
				return true;
			}
	
		return false;
	}
	
	// rewrites statement to support function inlining (creates deep copy)
	public Statement rewriteStatement(String prefix) throws LanguageException {
		
		InputStatement newStatement = new InputStatement();
		
		// rewrite target variable name (creates deep copy)
		newStatement._id = (DataIdentifier)this._id.rewriteExpression(prefix);
	
		// rewrite InputStatement expr parameters (creates deep copies)
		DataOp op = _paramsExpr.getOpCode();
		HashMap<String,Expression> newExprParams = new HashMap<String,Expression>();
		for (String key : _paramsExpr.getVarParams().keySet()){
			Expression newExpr = _paramsExpr.getVarParam(key).rewriteExpression(prefix);
			newExprParams.put(key, newExpr);
		}	

		DataExpression newParamerizedExpr = new DataExpression(op, newExprParams);
		newStatement.setExprParams(newParamerizedExpr);
		return newStatement;
			
	}

	
	public void initializeforwardLV(VariableSet activeIn){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		 sb.append(_id.toString() + " = " + Statement.INPUTSTATEMENT + " ( " );
		 sb.append(_paramsExpr.getVarParam(IO_FILENAME));
		 for (String key : _paramsExpr.getVarParams().keySet()){
			 if (key.equals(IO_FILENAME))
				 sb.append(", " + key + "=" + _paramsExpr.getVarParam(key).toString());
		 }
		 sb.append(" );"); 
		 return sb.toString(); 
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		// add variables read by parameter expressions
		for (String key : _paramsExpr.getVarParams().keySet())	
			result.addVariables(_paramsExpr.getVarParam(key).variablesRead()) ;
		
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
