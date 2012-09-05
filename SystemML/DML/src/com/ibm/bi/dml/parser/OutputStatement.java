package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.parser.Expression.DataOp;
import com.ibm.bi.dml.utils.LanguageException;

 
public class OutputStatement extends IOStatement{
	
	public static final String[] WRITE_VALID_PARAM_NAMES = { IO_FILENAME, FORMAT_TYPE};

	public static boolean isValidParamName(String key){
	for (String paramName : WRITE_VALID_PARAM_NAMES)
		if (paramName.equals(key)){
			return true;
		}

		return false;
	}
	
	public OutputStatement(){
		super();
	}
	public OutputStatement(DataIdentifier t, DataOp op){
		super(t, op);
	}
		
	// rewrites statement to support function inlining (create deep copy)
	public Statement rewriteStatement(String prefix) throws LanguageException{
		
		OutputStatement newStatement = new OutputStatement();

		// rewrite outputStatement variable name (creates deep copy)
		newStatement._id = (DataIdentifier)this._id.rewriteExpression(prefix);
		
		// rewrite output filename expression (creates deep copy)
		//Expression newFilenameExpr = _filenameExpr.rewriteExpression(prefix);
		//newStatement.setFilenameExpr(newFilenameExpr);
		
		// rewrite parameter expressions (creates deep copy)
		DataOp op = _paramsExpr.getOpCode();
		HashMap<String,Expression> newExprParams = new HashMap<String,Expression>();
		for (String key : _paramsExpr.getVarParams().keySet()){
			Expression newExpr = _paramsExpr.getVarParam(key).rewriteExpression(prefix);
			newExprParams.put(key, newExpr);
		}
		DataExpression newParamerizedExpr = new DataExpression(op, newExprParams);
		newParamerizedExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		newStatement.setExprParams(newParamerizedExpr);
		return newStatement;
	}
	
	
	public void initializeforwardLV(VariableSet activeIn){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		 sb.append(Statement.OUTPUTSTATEMENT + " ( " );
		 sb.append( _id.toString() + ", " +  _paramsExpr.getVarParam(IO_FILENAME).toString());
		 for (String key : _paramsExpr.getVarParams().keySet()){
			 if (!key.equals(IO_FILENAME))
				 sb.append(", " + key + "=" + _paramsExpr.getVarParam(key));
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
		//result.addVariables(_filenameExpr.variablesRead());
		
		// add variables for parameter expressions 
		for (String key : _paramsExpr.getVarParams().keySet())
			result.addVariables(_paramsExpr.getVarParam(key).variablesRead()) ;
		
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return null;
	}
}
