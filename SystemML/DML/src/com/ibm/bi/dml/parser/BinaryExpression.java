package com.ibm.bi.dml.parser;

import java.util.HashMap;
import com.ibm.bi.dml.utils.LanguageException;


public class BinaryExpression extends Expression {

	private Expression _left;
	private Expression _right;
	private BinaryOp _opcode;

	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		
		
		BinaryExpression newExpr = new BinaryExpression(this._opcode);
		newExpr._beginLine 		 = this._beginLine; 
		newExpr._beginColumn	 = this._beginColumn;
		newExpr._endLine		 = this._endLine;
		newExpr._endColumn 		 = this._endColumn;	
		
		newExpr.setLeft(_left.rewriteExpression(prefix));
		newExpr.setRight(_right.rewriteExpression(prefix));
		return newExpr;
	}
	
	public BinaryExpression(BinaryOp bop) {
		_kind = Kind.BinaryOp;
		_opcode = bop;
		
		_beginLine 	 = 0;
		_beginColumn = 0;
		_endLine	 = 0;
		_endColumn 	 = 0;
	}
	
	public BinaryExpression(BinaryOp bop, int beginLine, int beginColumn, int endLine, int endColumn) {
		_kind = Kind.BinaryOp;
		_opcode = bop;
		
		_beginLine 	 = beginLine;
		_beginColumn = beginColumn;
		_endLine	 = endLine;
		_endColumn 	 = endColumn;
	}
	

	public BinaryOp getOpCode() {
		return _opcode;
	}

	public void setLeft(Expression l) {
		_left = l;
		
		// update script location information --> left expression is BEFORE in script
		if (_left != null){
			this._beginLine   = _left.getBeginLine();
			this._beginColumn = _left.getBeginColumn();
		}
		
	}

	public void setRight(Expression r) {
		_right = r;
		
		// update script location information --> right expression is AFTER in script
		if (_right != null){
			this._beginLine = _right.getEndLine();
			this._beginColumn = _right.getEndColumn();
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
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars)
			throws LanguageException {
		
		this.getLeft().validateExpression(ids, constVars);
		this.getRight().validateExpression(ids, constVars);
		
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		output.setDataType(computeDataType(this.getLeft(), this.getRight(), true));

		ValueType resultVT = computeValueType(this.getLeft(), this.getRight(),
				true);

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
							.getOutput().getDim1()) {
				throw new LanguageException(this.printErrorLocation() + "invalid dimensions for matrix multiplication (k1="+this.getLeft().getOutput().getDim2()+", k2="+this.getRight().getOutput().getDim1()+")", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			output.setDimensions(this.getLeft().getOutput().getDim1(), this
					.getRight().getOutput().getDim2());
		}

		if (this.getOpCode() == Expression.BinaryOp.POW) {
			if (this.getRight().getOutput().getDataType() != DataType.SCALAR) {
				throw new LanguageException(this.printErrorLocation() + "Second operand to ^ should be a scalar in " + this.toString(), LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
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
