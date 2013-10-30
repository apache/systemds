/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.parser.Expression.DataOp;

 
public class OutputStatement extends IOStatement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public static final String[] WRITE_VALID_PARAM_NAMES = { IO_FILENAME, FORMAT_TYPE, DELIM_DELIMITER, DELIM_HAS_HEADER_ROW, DELIM_SPARSE};

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
