package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.utils.LanguageException;


public class BuiltinFunctionExpression extends Expression {

	private Expression _first;
	private Expression _second;
	private Expression _third;
	private BuiltinFunctionOp _opcode;

	public BuiltinFunctionExpression(BuiltinFunctionOp bifop, Expression first,
			Expression second, Expression third) {
		_kind = Kind.BuiltinFunctionOp;
		_opcode = bifop;
		_first = first;
		_second = second;
		_third = third;
	}

	public Expression rewriteExpression(String prefix) throws LanguageException {

		Expression newFirst = (this._first == null) ? null : this._first
				.rewriteExpression(prefix);
		Expression newSecond = (this._second == null) ? null : this._second
				.rewriteExpression(prefix);
		Expression newThird = (this._third == null) ? null : this._third
				.rewriteExpression(prefix);
		return new BuiltinFunctionExpression(this._opcode, newFirst, newSecond,
				newThird);
	}

	public BuiltinFunctionOp getOpCode() {
		return _opcode;
	}

	public void setFirstExpr(Expression e) {
		_first = e;
	}

	public void setSecondExpr(Expression e) {
		_second = e;
	}

	public void setThirdExpr(Expression e) {
		_third = e;
	}

	public Expression getFirstExpr() {
		return _first;
	}

	public Expression getSecondExpr() {
		return _second;
	}

	public Expression getThirdExpr() {
		return _third;
	}

	/**
	 * Validate parse tree : Process BuiltinFunction Expression in an assignment
	 * statement
	 * 
	 * @throws LanguageException
	 */
	public void validateExpression(HashMap<String, DataIdentifier> ids)
			throws LanguageException {
		this.getFirstExpr().validateExpression(ids);
		if (_second != null)
			_second.validateExpression(ids);
		if (_third != null)
			_third.validateExpression(ids);

		// checkIdentifierParams();
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		Identifier id = this.getFirstExpr().getOutput();
		output.setProperties(this.getFirstExpr().getOutput());
		this.setOutput(output);

		switch (this.getOpCode()) {
		case COLSUM:
		case COLMAX:
		case COLMIN:
		case COLMEAN:
			// colSums(X);
			checkNumParameters(1);
			checkMatrixParam(_first);
			output.setDataType(DataType.MATRIX);
			output.setDimensions(1, id.getDim2());
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			break;
		case ROWSUM:
		case ROWMAX:
		case ROWINDEXMAX:
		case ROWMIN:
		case ROWMEAN:
			//rowSums(X);
			checkNumParameters(1);
			checkMatrixParam(_first);
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), 1);
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			break;
		case SUM:
		case PROD:
		case TRACE:
			// sum(X);
			checkNumParameters(1);
			checkMatrixParam(_first);
			
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(id.getValueType());
			break;
		
		case MEAN:
			//checkNumParameters(2, false); // mean(Y) or mean(Y,W)
            if (_second != null) {
            	checkNumParameters (2);
            }
            else {
            	checkNumParameters (1);
            }
			
			checkMatrixParam(_first);
			if ( _second != null ) {
				// x = mean(Y,W);
				checkMatchingDimensions(_first, _second);
			}
			
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(id.getValueType());
			break;
			
		case MIN:
		case MAX:
			if (_first.getOutput().getDataType() == DataType.SCALAR) {
				// Example: x = min(2,5)
				checkNumParameters(2);
			} else {
				// Example: x = min(A)
				checkNumParameters(1);
				checkMatrixParam(_first);
			}
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(id.getValueType());
			break;
		case CAST_AS_SCALAR:
			checkNumParameters(1);
			checkMatrixParam(_first);
			if (( _first.getOutput().getDim1() != -1 && _first.getOutput().getDim1() !=1) || 
				( _first.getOutput().getDim2() != -1 && _first.getOutput().getDim2() !=1)) {
				throw new LanguageException(
						"dimension mismatch while casting matrix to scalar: dim1: " + 
						_first.getOutput().getDim1() +  " dim2 " + _first.getOutput().getDim2(),
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);

			}
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(id.getValueType());
			break;
		case APPEND:
			checkNumParameters(2);
			checkMatrixParam(_first);
			checkMatrixParam(_second);
			//checkMatchingDimensions();
			output.setDataType(id.getDataType());
			output.setValueType(id.getValueType());
			output.setDimensions(-1, -1); // unknown output dimensions at compile time
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			break;
		case PMIN:
		case PMAX:
			// pmin (X, Y) or pmin(X, y)
			checkNumParameters(2); 
			checkMatrixParam(_first);
			
			if (_second.getOutput().getDataType() == DataType.MATRIX) {
			 checkMatrixParam(_second);
			 checkMatchingDimensions(_first, _second);
			}
	
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(),id.getDim2());
			output.setBlockDimensions(id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			break;
		case PPRED:
			// ppred (X,Y, "<"); or ppred (X,y, "<");
			checkNumParameters(3);
			checkMatrixParam(_first);
			
			if (_second.getOutput().getDataType() == DataType.MATRIX) {
			      checkMatrixParam(_second);
			      checkMatchingDimensions(_first, _second);
			}
			
			if (_third.getOutput().getDataType() != DataType.SCALAR || 
				_third.getOutput().getValueType() != ValueType.STRING) {
					throw new LanguageException(
							"Third argument in ppred() is not an operator ",
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), id.getDim2());
			output.setBlockDimensions(id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			break;

		case TRANS:
			checkNumParameters(1);
			checkMatrixParam(_first);
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim2(), id.getDim1());
			output.setBlockDimensions (id.getColumnsInBlock(), id.getRowsInBlock());
			output.setValueType(id.getValueType());
			break;
		case DIAG:
			checkNumParameters(1);
			checkMatrixParam(_first);
			output.setDataType(DataType.MATRIX);
			if ((id.getDim1() == 1) || (id.getDim2() == 1)) {
				long dim = (id.getDim1() == 1) ? id.getDim2() : id.getDim1();
				output.setDimensions(dim, dim);
			} else {
				if (id.getDim1() != id.getDim2()) {
					throw new LanguageException(
							"Invoking diag on matrix with dimensions ("
									+ id.getDim1() + "," + id.getDim2()
									+ ") in " + this.toString(),
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
				output.setDimensions(id.getDim2(), 1);
			}
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			break;
		case NROW:
		case NCOL:
		case LENGTH:
			checkNumParameters(1);
			checkMatrixParam(_first);
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(ValueType.INT);
			break;

		// Contingency tables
		case CTABLE:
			if (_third != null) {
			   checkNumParameters(3);
			}
			else {
			   checkNumParameters(2);
			}
			checkMatrixParam(_first);
			// second and third parameters can either be a scalar or a matrix
			// example: F = ctable(A,1); and F = ctable(A,B,1)
			
			// check for matching dimensions appropriately
			if ( _second.getOutput().getDataType() == DataType.MATRIX)
				checkMatchingDimensions(_first,_second);
			if (_third != null) {
				if ( _third.getOutput().getDataType() == DataType.MATRIX )
					checkMatchingDimensions(_first,_third);
			}
			
			// The dimensions for the output matrix will be known only at the
			// run time
			output.setDimensions(-1, -1);
			output.setBlockDimensions (-1, -1);
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			break;

		case SPEARMAN:
			if (_third != null) {
			   checkNumParameters(3);
			}
			else {
			   checkNumParameters(2);
			}
			checkMatrixParam(_first);
			checkMatrixParam(_second);
			checkMatchingDimensions(_first,_second);
			if (_third != null) {
				checkMatrixParam(_third);
			    checkMatchingDimensions(_first, _third);
			}

			// the output is a scalar
			output.setDimensions(0, 0);
			output.setBlockDimensions(0, 0);
			output.setDataType(DataType.SCALAR);
			output.setValueType(ValueType.DOUBLE);
			break;

		case ROUND:
			checkNumParameters(1);
			checkMatrixParam(_first);

			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), id.getDim2());
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());

			break;

		case CENTRALMOMENT:
			/*
			 * x = centralMoment(V,order) or xw = centralMoment(V,W,order)
			 */
			checkMatrixParam(_first);
			if (_third != null) {
			   checkNumParameters(3);
			   checkMatrixParam(_second);
			   checkMatchingDimensions(_first,_second);
			   checkScalarParam(_third);
			}
			else {
			   checkNumParameters(2);
			   checkScalarParam(_second);
			}

			// output is a scalar
			output.setDataType(DataType.SCALAR);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(0, 0);
			output.setBlockDimensions(0,0);
			break;

		case COVARIANCE:
			/*
			 * x = cov(V1,V2) or xw = cov(V1,V2,W)
			 */
			if (_third != null) {
				checkNumParameters(3);
			}
			else {
				checkNumParameters(2);
			}
			checkMatrixParam(_first);
			checkMatrixParam(_second);
			checkMatchingDimensions(_first,_second);
			
			if (_third != null) {
				checkMatrixParam(_third);
			 checkMatchingDimensions(_first, _third);
			}

			// output is a scalar
			output.setDataType(DataType.SCALAR);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(0, 0);
			output.setBlockDimensions(0,0);
			break;

		case QUANTILE:
			/*
			 * q = quantile(V1,0.5) computes median in V1 
			 * or Q = quantile(V1,P) computes the vector of quantiles as specified by P
			 * or qw = quantile(V1,W,0.5) computes median when weights (W) are given
			 * or QW = quantile(V1,W,P) computes the vector of quantiles as specified by P, when weights (W) are given
			 */
			if(_third != null) {
			    checkNumParameters(3);
			}
			else {
				checkNumParameters(2);
			}
			
			// first parameter must always be a 1D matrix 
			check1DMatrixParam(getFirstExpr());
			
			// check for matching dimensions for other matrix parameters
			if (_third != null) {
			    checkMatrixParam(_second);
				checkMatchingDimensions(_first, _second);
			}
			
			// set the properties for _output expression
			// output dimensions = dimensions of second, if third is null
			//                   = dimensions of the third, otherwise.

			if (_third != null) {
				output.setDimensions(_third._output.getDim1(), _third._output
						.getDim2());
				output.setBlockDimensions(_third._output.getRowsInBlock(), 
						                  _third._output.getColumnsInBlock());
				output.setDataType(_third._output.getDataType());
			} else {
				output.setDimensions(_second._output.getDim1(), _second._output
						.getDim2());
				output.setBlockDimensions(_second._output.getRowsInBlock(), 
		                  _second._output.getColumnsInBlock());
				output.setDataType(_second._output.getDataType());
			}
			break;

		case INTERQUANTILE:
			if (_third != null) {
			    checkNumParameters(3);
			}
			else {
				checkNumParameters(2);
			}
			checkMatrixParam(_first);
			if (_third != null) {
				// i.e., second input is weight vector
				checkMatrixParam(_second);
				checkMatchingDimensionsQuantile();
			}

			if ((_third == null && _second._output.getDataType() != DataType.SCALAR)
					&& (_third != null && _third._output.getDataType() != DataType.SCALAR)) {
				throw new LanguageException("Invalid parameters to "
						+ this.getOpCode(),
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}

			output.setValueType(id.getValueType());
			// output dimensions are unknown
			output.setDimensions(-1, -1);
			output.setBlockDimensions(-1,-1);
			output.setDataType(DataType.MATRIX);
			break;

		case IQM:
			/*
			 * Usage: iqm = InterQuartileMean(A,W); iqm = InterQuartileMean(A);
			 */
			if (_second != null){
			    checkNumParameters(2);
		    }
			else {
				checkNumParameters(1);
			}
			checkMatrixParam(_first);

			if (_second != null) {
				// i.e., second input is weight vector
				checkMatrixParam(_second);
				checkMatchingDimensions(_first, _second);
			}

			// Output is a scalar
			output.setValueType(id.getValueType());
			output.setDimensions(0, 0);
			output.setBlockDimensions(0,0);
			output.setDataType(DataType.SCALAR);

			break;

		default:
			if (this.isMathFunction()) {
				// datatype and dimensions are same as this.getExpr()
				if (this.getOpCode() == BuiltinFunctionOp.ABS) {
					output.setValueType(_first.getOutput().getValueType());
				} else {
					output.setValueType(ValueType.DOUBLE);
				}
				checkMathFunctionParam();
				output.setDataType(id.getDataType());
				output.setDimensions(id.getDim1(), id.getDim2());
				output.setBlockDimensions(id.getRowsInBlock(), id.getColumnsInBlock()); 
			} else
				throw new LanguageException("Unsupported function "
						+ this.getOpCode(),
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
		return;
	}

	private boolean isMathFunction() {
		switch (this.getOpCode()) {
		case COS:
		case SIN:
		case TAN:
		case SQRT:
		case ABS:
		case LOG:
		case EXP:
		case ROUND:
			return true;
		default:
			return false;
		}
	}

	private void checkMathFunctionParam() throws LanguageException {
		switch (this.getOpCode()) {
		case COS:
		case SIN:
		case TAN:
		case SQRT:
		case ABS:
		case EXP:
		case ROUND:
			checkNumParameters(1);
			break;
		case LOG:
			if (_second != null) {
			  checkNumParameters(2);
			}
			else {
			  checkNumParameters(1);
			}
			break;
		default:
			throw new LanguageException("Unknown math function "
					+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.UNSUPPORTED_EXPRESSION);
		}
	}

	public String toString() {
		StringBuffer sb = new StringBuffer(_opcode.toString() + " ( "
				+ _first.toString());

		if (_second != null) {
			sb.append("," + _second.toString());
		}
		if (_third != null) {
			sb.append("," + _third.toString());
		}
		sb.append(" )");
		return sb.toString();
	}

	@Override
	// third part of expression IS NOT a variable -- it is the OP to be applied
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_first.variablesRead());
		if (_second != null) {
			result.addVariables(_second.variablesRead());
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		// result.addVariables(_first.variablesUpdated());
		return result;
	}

	private void checkNumParameters(int count)
			throws LanguageException {
		if (_first == null)
			throw new LanguageException("Missing parameter for function "
					+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
            	
       	if (((count == 1) && (_second!= null || _third != null)) || 
        		((count == 2) && (_third != null))) 
			    throw new LanguageException("Invalid number of parameters for function "
					  + this.getOpCode(),
					  LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
	
       	else if (((count == 2) && (_second == null)) || 
		             ((count == 3) && (_second == null || _third == null)))
			throw new LanguageException("Missing parameter for function "
					+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
	}

	private void checkMatrixParam(Expression e) throws LanguageException {
		if (e.getOutput().getDataType() != DataType.MATRIX) {
			throw new LanguageException(
					"Expecting matrix parameter for function "
							+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	private void checkScalarParam(Expression e) throws LanguageException {
		if (e.getOutput().getDataType() != DataType.SCALAR) {
			throw new LanguageException(
					"Expecting scalar parameter for function "
							+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	private boolean is1DMatrix(Expression e) {
		return (e.getOutput().getDim1() == 1 || e.getOutput().getDim2() == 1 );
	}
	
	private boolean dimsKnown(Expression e) {
		return (e.getOutput().getDim1() != -1 && e.getOutput().getDim2() != -1);
	}
	
	private void check1DMatrixParam(Expression e) throws LanguageException {
		
		checkMatrixParam(e);
		
		// throw an exception, when e's output is NOT a one-dimensional matrix 
		// the check must be performed only when the dimensions are known at compilation time
		if ( dimsKnown(e) && !is1DMatrix(e)) {
			throw new LanguageException(
					"Expecting one-dimensional matrix parameter for function "
							+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}

	private void checkMatchingDimensions(Expression expr1, Expression expr2) throws LanguageException {
		if (expr1 != null && expr2 != null) {
			
			// if any matrix has unknown dimensions, simply return
			if (expr1.getOutput().getDim1() == -1 || 
			 expr2.getOutput().getDim1() == -1) {

				return;
			}
			else if (expr1.getOutput().getDim1() != expr2.getOutput().getDim1() 
				|| expr1.getOutput().getDim2() != expr2.getOutput().getDim2() ) {
				throw new LanguageException(
						"Mismatch in matrix dimensions of parameters for function "
								+ this.getOpCode(),
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		   }
		}
	}
	

	private void checkMatchingDimensionsQuantile() throws LanguageException {
		if (_first.getOutput().getDim1() != _second.getOutput().getDim1()) {
			throw new LanguageException("Mismatch in matrix dimensions for "
					+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);

		}
	}

}
