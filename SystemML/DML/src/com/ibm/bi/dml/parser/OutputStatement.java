package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.utils.LanguageException;

 
public class OutputStatement extends IOStatement{
	
	public OutputStatement(DataIdentifier t, Expression fname){
		super(t,null);
	}
	
	// rewrites statement to support function inlining (create deep copy)
	public Statement rewriteStatement(String prefix) throws LanguageException{
		
		OutputStatement newStatement = new OutputStatement();

		// rewrite outputStatement variable name (creates deep copy)
		newStatement._id = (DataIdentifier)this._id.rewriteExpression(prefix);
		
		// rewrite output filename expression (creates deep copy)
		Expression newFilenameExpr = _filenameExpr.rewriteExpression(prefix);
		newStatement.setFilenameExpr(newFilenameExpr);
		
		// rewrite parameter expressions (creates deep copy)
		HashMap<String,Expression> newExprParams = new HashMap<String,Expression>();
		for (String key : _exprParams.keySet()){
			Expression newExpr = _exprParams.get(key).rewriteExpression(prefix);
			newExprParams.put(key, newExpr);
		}
		
		newStatement.setExprParams(newExprParams);
		return newStatement;
	}
	
	
	public OutputStatement(){
		super();
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	public String toString(){
		 StringBuffer sb = new StringBuffer();
		 sb.append(Statement.OUTPUTSTATEMENT + " ( " );
		 sb.append( _id.toString() + ", " +  _filenameExpr.toString());
		 for (String key : _exprParams.keySet()){
			 sb.append(", " + key + "=" + _exprParams.get(key));
		 }
		 sb.append(" );");
		 return sb.toString(); 
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		// handle variable that is being written out
		result.addVariables(_id.variablesRead());
		
		// handle variables for output filename expression
		result.addVariables(_filenameExpr.variablesRead());
		
		// add variables for parameter expressions 
		for (String key : _exprParams.keySet())
			result.addVariables(_exprParams.get(key).variablesRead()) ;
		
		
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return null;
	}
}
