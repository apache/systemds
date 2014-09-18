/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.HashMap;


public class BinaryExpression extends Expression 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Expression _left;
	private Expression _right;
	private BinaryOp _opcode;

	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		
		
		BinaryExpression newExpr = new BinaryExpression(this._opcode,
				this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		newExpr.setLeft(_left.rewriteExpression(prefix));
		newExpr.setRight(_right.rewriteExpression(prefix));
		return newExpr;
	}
	
	public BinaryExpression(BinaryOp bop) {
		_kind = Kind.BinaryOp;
		_opcode = bop;
		
		setFilename("MAIN SCRIPT");
		setBeginLine(0);
		setBeginColumn(0);
		setEndLine(0);
		setEndColumn(0);
	}
	
	public BinaryExpression(BinaryOp bop, String filename, int beginLine, int beginColumn, int endLine, int endColumn) {
		_kind = Kind.BinaryOp;
		_opcode = bop;
		
		setFilename(filename);
		setBeginLine(beginLine);
		setBeginColumn(beginColumn);
		setEndLine(endLine);
		setEndColumn(endColumn);
	}
	

	public BinaryOp getOpCode() {
		return _opcode;
	}

	public void setLeft(Expression l) {
		_left = l;
		
		// update script location information --> left expression is BEFORE in script
		if (_left != null){
			setFilename(_left.getFilename());
			setBeginLine(_left.getBeginLine());
			setBeginColumn(_left.getBeginColumn());
		}
		
	}

	public void setRight(Expression r) {
		_right = r;
		
		// update script location information --> right expression is AFTER in script
		if (_right != null){
			setFilename(_right.getFilename());
			setBeginLine(_right.getEndLine());
			setBeginColumn(_right.getEndColumn());
		}
	}

	public Expression getLeft() {
		return _left;
	}

	public Expression getRight() {
		return _right;
	}

	/**
	 * Validate parse tree : Process Binary Expression in an assignment
	 * statement
	 * 
	 * @throws LanguageException
	 */
	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional)
			throws LanguageException 
	{	
		//recursive validate
		if (_left instanceof FunctionCallIdentifier || _right instanceof FunctionCallIdentifier){
			raiseValidateError("user-defined function calls not supported in binary expressions", 
		            false, LanguageException.LanguageErrorCodes.UNSUPPORTED_EXPRESSION);
		}
			
		_left.validateExpression(ids, constVars, conditional);
		_right.validateExpression(ids, constVars, conditional);
		
		//constant propagation (precondition for more complex constant folding rewrite)
		if( _left instanceof DataIdentifier && constVars.containsKey(((DataIdentifier) _left).getName()) )
			_left = constVars.get(((DataIdentifier) _left).getName());
		if( _right instanceof DataIdentifier && constVars.containsKey(((DataIdentifier) _right).getName()) )
			_right = constVars.get(((DataIdentifier) _right).getName());
		
		
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());		
		output.setDataType(computeDataType(this.getLeft(), this.getRight(), true));
		ValueType resultVT = computeValueType(this.getLeft(), this.getRight(), true);

		// Override the computed value type, if needed
		if (this.getOpCode() == Expression.BinaryOp.POW
				|| this.getOpCode() == Expression.BinaryOp.DIV) {
			resultVT = ValueType.DOUBLE;
		}

		output.setValueType(resultVT);

		checkAndSetDimensions(output);
		if (this.getOpCode() == Expression.BinaryOp.MATMULT) {
			if ((this.getLeft().getOutput().getDataType() != DataType.MATRIX) || (this.getRight().getOutput().getDataType() != DataType.MATRIX)) {
		// remove exception for now
		//		throw new LanguageException(
		//				"Matrix multiplication not supported for scalars",
		//				LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			if (this.getLeft().getOutput().getDim2() != -1
					&& this.getRight().getOutput().getDim1() != -1
					&& this.getLeft().getOutput().getDim2() != this.getRight()
							.getOutput().getDim1()) 
			{
				raiseValidateError("invalid dimensions for matrix multiplication (k1="+this.getLeft().getOutput().getDim2()+", k2="+this.getRight().getOutput().getDim1()+")", 
						            conditional, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			output.setDimensions(this.getLeft().getOutput().getDim1(), this
					.getRight().getOutput().getDim2());
		}

		this.setOutput(output);
	}

	private void checkAndSetDimensions(DataIdentifier output)
			throws LanguageException {
		Identifier left = this.getLeft().getOutput();
		Identifier right = this.getRight().getOutput();
		Identifier pivot = null;
		Identifier aux = null;

		if (left.getDataType() == DataType.MATRIX) {
			pivot = left;
			if (right.getDataType() == DataType.MATRIX) {
				aux = right;
			}
		} else if (right.getDataType() == DataType.MATRIX) {
			pivot = right;
		}

		if ((pivot != null) && (aux != null)) {
			if (isSameDimensionBinaryOp(this.getOpCode())) {
		//		if ((pivot.getDim1() != aux.getDim1())
		//				|| (pivot.getDim2() != aux.getDim2())) {
		//			throw new LanguageException(
		//					"Mismatch in dimensions for operation "
		//							+ this.toString(),
		//					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		//		}
			}
		}

		if (pivot != null) {
			output.setDimensions(pivot.getDim1(), pivot.getDim2());
		}
		return;
	}
	
	public String toString() {

		return "(" + _left.toString() + " " + _opcode.toString() + " "
				+ _right.toString() + ")";

	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_left.variablesRead());
		result.addVariables(_right.variablesRead());
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariables(_left.variablesUpdated());
		result.addVariables(_right.variablesUpdated());
		return result;
	}

	public static boolean isSameDimensionBinaryOp(BinaryOp op) {
		return (op == BinaryOp.PLUS) || (op == BinaryOp.MINUS)
				|| (op == BinaryOp.MULT) || (op == BinaryOp.DIV);
	}
}
