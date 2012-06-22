package com.ibm.bi.dml.parser;

import java.util.HashMap;
import com.ibm.bi.dml.utils.LanguageException;


public class BooleanExpression extends Expression{
	
	private Expression _left;
	private Expression _right;
	private BooleanOp _opcode;
	
	public BooleanExpression(BooleanOp bop){
		_kind = Kind.BooleanOp;
		_opcode = bop;
	}
	
	public BooleanOp getOpCode(){
		return _opcode;
	}
	
	public void setLeft(Expression l){
		_left = l;
	}
	
	public void setRight(Expression r){
		_right = r;
	}
	
	public Expression getLeft(){
		return _left;
	}
	
	public Expression getRight(){
		return _right;
	}

	public Expression rewriteExpression(String prefix) throws LanguageException{
		
		
		BooleanExpression newExpr = new BooleanExpression(this._opcode);
		newExpr.setLeft(_left.rewriteExpression(prefix));
		newExpr.setRight(_right.rewriteExpression(prefix));
		return newExpr;
	}
	
	/**
	 * Validate parse tree : Process Boolean Expression  
	 */
	public void validateExpression(HashMap<String,DataIdentifier> ids) throws LanguageException{
		 	 
		this.getLeft().validateExpression(ids);
		if (this.getRight() != null) {
			this.getRight().validateExpression(ids);
		}
			
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setBooleanProperties();
		this.setOutput(output);
		if ((_opcode == Expression.BooleanOp.CONDITIONALAND) ||
				(_opcode == Expression.BooleanOp.CONDITIONALOR)) {
			throw new LanguageException("Unsupported boolean operation " + _opcode.toString(),
					LanguageException.LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
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
