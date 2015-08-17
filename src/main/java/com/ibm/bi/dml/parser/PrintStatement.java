/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.debug.DMLBreakpointManager;

 
public class PrintStatement extends Statement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	public enum PRINTTYPE {PRINT, STOP};
	
	protected PRINTTYPE _type; // print or stop
	protected Expression _expr;

	private static PRINTTYPE getPrintType(String type) throws LanguageException {
		if(type.equalsIgnoreCase("print")) {
			return PRINTTYPE.PRINT;
		}
		else if (type.equalsIgnoreCase("stop")) {
			return PRINTTYPE.STOP;
		}
		else
			throw new LanguageException("Unknown statement type: " + type);
	}
	
	public PrintStatement(String type, Expression expr, int beginLine, int beginCol, int endLine, int endCol) 
		throws LanguageException
	{
		this(getPrintType(type), expr);
		
		setBeginLine(beginLine);
		setBeginColumn(beginCol);
		setEndLine(endLine);
		setEndColumn(endCol);
	}
	 
	public PrintStatement(PRINTTYPE type, Expression expr) throws LanguageException{
		_type = type;
		_expr = expr; 
	}
	 
	public Statement rewriteStatement(String prefix) throws LanguageException{
		Expression newExpr = _expr.rewriteExpression(prefix);
		PrintStatement retVal = new PrintStatement(_type, newExpr);
		retVal.setBeginLine(this.getBeginLine());
		retVal.setBeginColumn(this.getBeginColumn());
		retVal.setEndLine(this.getEndLine());
		retVal.setEndColumn(this.getEndColumn());
		
		return retVal;
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	
	public String toString(){
		 StringBuilder sb = new StringBuilder();
		 sb.append(_type + " (" );
		 if (_expr != null){
			 sb.append(_expr.toString());
		 }
		 sb.append(");");
		 return sb.toString(); 
	}
	
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result =  _expr.variablesRead();
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return new VariableSet();
	}

	@Override
	public boolean controlStatement() {	 
		// ensure that breakpoints end up in own statement block
		if (DMLScript.ENABLE_DEBUG_MODE) {
			DMLBreakpointManager.insertBreakpoint(_expr.getBeginLine());
			return true;
		}
		
		// Keep stop() statement in a separate statement block
		if(getType() == PRINTTYPE.STOP)
			return true;
		
		return false;
	}

	public Expression getExpression(){
		return _expr;
	}
	
	public PRINTTYPE getType() {
		return _type;
	}
	 
}
