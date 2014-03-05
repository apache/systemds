/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.HashMap;


public class BooleanExpression extends Expression
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private Expression _left;
	private Expression _right;
	private BooleanOp _opcode;
	
	public BooleanExpression(BooleanOp bop){
		_kind = Kind.BooleanOp;
		_opcode = bop;
		
		setFilename("MAIN SCRIPT");
		setBeginLine(0);
		setBeginColumn(0);
		setEndLine(0);
		setEndColumn(0);
	}
	
	public BooleanExpression(BooleanOp bop, String filename, int beginLine, int beginColumn, int endLine, int endColumn){
		_kind = Kind.BooleanOp;
		_opcode = bop;
		
		setFilename(filename);
		setBeginLine(beginLine);
		setBeginColumn(beginColumn);
		setEndLine(endLine);
		setEndColumn(endColumn);
	}
	
	public BooleanOp getOpCode(){
		return _opcode;
	}
	
	public void setLeft(Expression l){
		_left = l;
		
		// update script location information --> left expression is BEFORE in script
		if (_left != null){
			this.setFilename(_left.getFilename());
			this.setBeginLine(_left.getBeginLine());
			this.setBeginColumn(_left.getBeginColumn());
		}
	}
	
	public void setRight(Expression r){
		_right = r;
		
		// update script location information --> right expression is AFTER in script
		if (_right != null){
			this.setFilename(_right.getFilename());
			this.setBeginLine(_right.getBeginLine());
			this.setBeginColumn(_right.getBeginColumn());
		}
	}
	
	public Expression getLeft(){
		return _left;
	}
	
	public Expression getRight(){
		return _right;
	}

	public Expression rewriteExpression(String prefix) throws LanguageException{
		
		
		BooleanExpression newExpr = new BooleanExpression(this._opcode, this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		newExpr.setLeft(_left.rewriteExpression(prefix));
		newExpr.setRight(_right.rewriteExpression(prefix));
		return newExpr;
	}
	
	/**
	 * Validate parse tree : Process Boolean Expression  
	 */
	public void validateExpression(HashMap<String,DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException{
		 	 
		this.getLeft().validateExpression(ids, constVars);
		if (this.getRight() != null) {
			this.getRight().validateExpression(ids, constVars);
		}
			
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		output.setBooleanProperties();
		this.setOutput(output);
		if ((_opcode == Expression.BooleanOp.CONDITIONALAND) ||
				(_opcode == Expression.BooleanOp.CONDITIONALOR)) {
			throw new LanguageException(this.printErrorLocation() + "Unsupported boolean operation " + _opcode.toString(), LanguageException.LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}		
	
	public String toString(){
		if (_opcode == BooleanOp.NOT) {
			return "(" + _opcode.toString() + " " + _left.toString() + ")";
		} else {
			return "(" + _left.toString() + " " + _opcode.toString() + " " + _right.toString() + ")";
		}
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_left.variablesRead());
		if (_right != null){
			result.addVariables(_right.variablesRead());
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariables(_left.variablesUpdated());
		if (_right != null){
			result.addVariables(_right.variablesUpdated());
		}
		return result;
	}
}
