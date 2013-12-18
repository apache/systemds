/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;


public class BuiltinFunctionExpression extends DataIdentifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected Expression  	  _first;
	protected Expression  	  _second;
	protected Expression 	  _third;
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

		Expression newFirst = (this._first == null) ? null : this._first.rewriteExpression(prefix);
		Expression newSecond = (this._second == null) ? null : this._second.rewriteExpression(prefix);
		Expression newThird = (this._third == null) ? null : this._third.rewriteExpression(prefix);
		BuiltinFunctionExpression retVal = new BuiltinFunctionExpression(this._opcode, newFirst, newSecond, newThird);
	
		retVal._beginLine 	= this._beginLine;
		retVal._beginColumn = this._beginColumn;
		retVal._endLine		= this._endLine;
		retVal._endColumn	= this._endColumn;
		
		return retVal;
	
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

	public void validateExpression(MultiAssignmentStatement stmt, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars)
			throws LanguageException {
		this.getFirstExpr().validateExpression(ids, constVars);
		if (_second != null)
			_second.validateExpression(ids, constVars);
		if (_third != null)
			_third.validateExpression(ids, constVars);

		_outputs = new Identifier[stmt.getTargetList().size()];
		int count = 0;
		for (DataIdentifier outParam: stmt.getTargetList()){
			DataIdentifier tmp = new DataIdentifier(outParam);
			tmp.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			_outputs[count++] = tmp;
		}
		
		switch (_opcode) {
		case QR:
			checkNumParameters(1);
			checkMatrixParam(_first);
			
			// setup output properties
			DataIdentifier qrOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier qrOut2 = (DataIdentifier) getOutputs()[1];
			
			long rows = _first.getOutput().getDim1();
			long cols = _first.getOutput().getDim2();
			
			// Output1 - Q
			qrOut1.setDataType(DataType.MATRIX);
			qrOut1.setValueType(ValueType.DOUBLE);
			qrOut1.setDimensions(rows, rows);
			qrOut1.setBlockDimensions(_first.getOutput().getRowsInBlock(), _first.getOutput().getColumnsInBlock());
			
			// Output2 - R
			qrOut2.setDataType(DataType.MATRIX);
			qrOut2.setValueType(ValueType.DOUBLE);
			qrOut2.setDimensions(rows, cols);
			qrOut2.setBlockDimensions(_first.getOutput().getRowsInBlock(), _first.getOutput().getColumnsInBlock());
			
			break;

		case LU:
			checkNumParameters(1);
			checkMatrixParam(_first);
			
			// setup output properties
			DataIdentifier luOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier luOut2 = (DataIdentifier) getOutputs()[1];
			DataIdentifier luOut3 = (DataIdentifier) getOutputs()[2];
			
			long inrows = _first.getOutput().getDim1();
			long incols = _first.getOutput().getDim2();
			
			if ( inrows != incols ) {
				throw new LanguageException("LU Decomposition can only be done on a square matrix. Input matrix is rectangular (rows=" + inrows + ", cols="+incols+")");
			}
			
			// Output1 - P
			luOut1.setDataType(DataType.MATRIX);
			luOut1.setValueType(ValueType.DOUBLE);
			luOut1.setDimensions(inrows, inrows);
			luOut1.setBlockDimensions(_first.getOutput().getRowsInBlock(), _first.getOutput().getColumnsInBlock());
			
			// Output2 - L
			luOut2.setDataType(DataType.MATRIX);
			luOut2.setValueType(ValueType.DOUBLE);
			luOut2.setDimensions(inrows, inrows);
			luOut2.setBlockDimensions(_first.getOutput().getRowsInBlock(), _first.getOutput().getColumnsInBlock());
			
			// Output3 - U
			luOut3.setDataType(DataType.MATRIX);
			luOut3.setValueType(ValueType.DOUBLE);
			luOut3.setDimensions(inrows, inrows);
			luOut3.setBlockDimensions(_first.getOutput().getRowsInBlock(), _first.getOutput().getColumnsInBlock());
			
			break;

		default:
			throw new LanguageException("Unknown Builtin Function opcode: " + _opcode);
		}
	}

	/**
	 * Validate parse tree : Process BuiltinFunction Expression in an assignment
	 * statement
	 * 
	 * @throws LanguageException
	 */
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars)
			throws LanguageException {
		this.getFirstExpr().validateExpression(ids, constVars);
		if (_second != null)
			_second.validateExpression(ids, constVars);
		if (_third != null)
			_third.validateExpression(ids, constVars);

		// checkIdentifierParams();
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
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
			if (( _first.getOutput().getDim1() != -1 && _first.getOutput().getDim1() !=1) || ( _first.getOutput().getDim2() != -1 && _first.getOutput().getDim2() !=1)) {
				throw new LanguageException(this.printErrorLocation() + "dimension mismatch while casting matrix to scalar: dim1: " + _first.getOutput().getDim1() +  " dim2 " + _first.getOutput().getDim2(), LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
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
			
			// set output dimensions
			long appendDim1 = -1, appendDim2 = -1;
			if (_first.getOutput().getDim1() > 0 && _second.getOutput().getDim1() > 0){
				if (_first.getOutput().getDim1() != _second.getOutput().getDim1()){
					
					throw new LanguageException(this.printErrorLocation() +
							"inputs to append must have same number of rows: input 1 rows: " + 
							_first.getOutput().getDim1() +  ", input 2 rows " + _second.getOutput().getDim1(),
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
				appendDim1 = _first.getOutput().getDim1();
			}
			else if (_first.getOutput().getDim1() > 0)	
				appendDim1 = _first.getOutput().getDim1(); 
			else if (_second.getOutput().getDim1() > 0 )
				appendDim1 = _second.getOutput().getDim1(); 
				
			if (_first.getOutput().getDim2() > 0 && _second.getOutput().getDim2() > 0){
				appendDim2 = _first.getOutput().getDim2() + _second.getOutput().getDim2();
			}
			
			output.setDimensions(appendDim1, appendDim2); 
			
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
				
					throw new LanguageException(this.printErrorLocation() +
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
					
					throw new LanguageException(this.printErrorLocation() +
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
				output.setDimensions(_third.getOutput().getDim1(), _third.getOutput()
						.getDim2());
				output.setBlockDimensions(_third.getOutput().getRowsInBlock(), 
						                  _third.getOutput().getColumnsInBlock());
				output.setDataType(_third.getOutput().getDataType());
			} else {
				output.setDimensions(_second.getOutput().getDim1(), _second.getOutput()
						.getDim2());
				output.setBlockDimensions(_second.getOutput().getRowsInBlock(), 
		                  _second.getOutput().getColumnsInBlock());
				output.setDataType(_second.getOutput().getDataType());
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

			if ((_third == null && _second.getOutput().getDataType() != DataType.SCALAR)
					&& (_third != null && _third.getOutput().getDataType() != DataType.SCALAR)) {
				
				throw new LanguageException(this.printErrorLocation() + "Invalid parameters to "
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

		case SEQ:
			
			checkScalarParam(_first);
			checkScalarParam(_second);
			if ( _third != null ) {
				checkNumParameters(3);
				checkScalarParam(_third);
			}
			else
				checkNumParameters(2);
			
			// check if dimensions can be inferred
			long dim1=-1, dim2=1;
			if ( isConstant(_first) && isConstant(_second) && (_third != null ? isConstant(_third) : true) ) {
				double from, to, incr;
				boolean neg;
				try {
					from = getDoubleValue(_first);
					to = getDoubleValue(_second);
					
					// Setup the value of increment
					// default value: 1 if from <= to; -1 if from > to
					neg = (from > to);
					if(_third == null) {
						_third = new DoubleIdentifier((neg? -1.0 : 1.0));
					}
					incr = getDoubleValue(_third); // (_third != null ? getDoubleValue(_third) : (neg? -1:1) );
					
				}
				catch (LanguageException e) {
					throw new LanguageException("Arguments for seq() must be numeric.");
				}

				if (neg != (incr < 0))
					throw new LanguageException("Wrong sign for the increment in a call to seq()");
				
				// Both end points of the range must included i.e., [from,to] both inclusive.
				// Note that, "to" is included only if (to-from) is perfectly divisible by incr
				// For example, seq(0,1,0.5) produces (0.0 0.5 1.0) whereas seq(0,1,0.6) produces only (0.0 0.6) but not (0.0 0.6 1.0) 
				dim1 = 1 + (long)Math.floor((to-from)/incr); 
				//System.out.println("seq("+from+","+to+","+incr+") -> dims("+dim1+","+dim2+")");
			}
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(dim1, dim2);
			output.setBlockDimensions(0, 0);
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
			} else{
				
				throw new LanguageException(this.printErrorLocation() + "Unsupported function "
						+ this.getOpCode(),
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
		return;
	}
	
	@Override
	public boolean multipleReturns() {
		switch(_opcode) {
		case QR:
		case LU:
			return true;
		default:
			return false;
		}
	}

	private boolean isConstant(Expression expr) {
		return ( expr instanceof ConstIdentifier );
	}
	
	private double getDoubleValue(Expression expr) throws LanguageException {
		if ( expr instanceof DoubleIdentifier )
			return ((DoubleIdentifier)expr).getValue();
		else if ( expr instanceof IntIdentifier)
			return ((IntIdentifier)expr).getValue();
		else
			throw new LanguageException("Expecting a numeric value.");
	}
	private boolean isMathFunction() {
		switch (this.getOpCode()) {
		case COS:
		case SIN:
		case TAN:
		case ACOS:
		case ASIN:
		case ATAN:
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
		case ACOS:
		case ASIN:
		case ATAN:
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
			
			throw new LanguageException(this.printErrorLocation() + "Unknown math function "
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
		if( _third != null ) {
			result.addVariables(_third.variablesRead());
		}
		
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		// result.addVariables(_first.variablesUpdated());
		return result;
	}

	protected void checkNumParameters(int count)
			throws LanguageException {
		if (_first == null){
			
			throw new LanguageException(this.printErrorLocation()  + "Missing parameter for function "
					+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
       	if (((count == 1) && (_second!= null || _third != null)) || 
        		((count == 2) && (_third != null))){ 
       		
       			throw new LanguageException(this.printErrorLocation() + "Invalid number of parameters for function "
					  + this.getOpCode(),
					  LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
       	}
       	else if (((count == 2) && (_second == null)) || 
		             ((count == 3) && (_second == null || _third == null))){
       		
       		throw new LanguageException(this.printErrorLocation()  + "Missing parameter for function "
					+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
       	}
	}

	protected void checkMatrixParam(Expression e) throws LanguageException {
		if (e.getOutput().getDataType() != DataType.MATRIX) {
			
			throw new LanguageException(this.printErrorLocation() +
					"Expecting matrix parameter for function "
							+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	private void checkScalarParam(Expression e) throws LanguageException {
		if (e.getOutput().getDataType() != DataType.SCALAR) {
			
			throw new LanguageException(this.printErrorLocation() +
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
			
			throw new LanguageException(this.printErrorLocation() +
					"Expecting one-dimensional matrix parameter for function "
							+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}

	private void checkMatchingDimensions(Expression expr1, Expression expr2) throws LanguageException {
		if (expr1 != null && expr2 != null) {
			
			// if any matrix has unknown dimensions, simply return
			if(  expr1.getOutput().getDim1() == -1 || expr2.getOutput().getDim1() == -1 
			   ||expr1.getOutput().getDim2() == -1 || expr2.getOutput().getDim2() == -1 ) 
			{
				return;
			}
			else if (expr1.getOutput().getDim1() != expr2.getOutput().getDim1() 
				|| expr1.getOutput().getDim2() != expr2.getOutput().getDim2() ) {
				
				throw new LanguageException(this.printErrorLocation() +
						"Mismatch in matrix dimensions of parameters for function "
								+ this.getOpCode(),
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		   }
		}
	}
	

	private void checkMatchingDimensionsQuantile() throws LanguageException {
		if (_first.getOutput().getDim1() != _second.getOutput().getDim1()) {
			
			throw new LanguageException(this.printErrorLocation() + "Mismatch in matrix dimensions for "
					+ this.getOpCode(),
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);

		}
	}

	public static BuiltinFunctionExpression getBuiltinFunctionExpression(String functionName, ArrayList<Expression> exprs) {
		
		// check if the function name is built-in function
		//	(assign built-in function op if function is built-in
		Expression.BuiltinFunctionOp bifop = null;
		
		if (functionName.equals("avg"))
			bifop = Expression.BuiltinFunctionOp.MEAN;
		else if (functionName.equals("cos"))
			bifop = Expression.BuiltinFunctionOp.COS;
		else if (functionName.equals("sin"))
			bifop = Expression.BuiltinFunctionOp.SIN;
		else if (functionName.equals("tan"))
			bifop = Expression.BuiltinFunctionOp.TAN;
		else if (functionName.equals("acos"))
			bifop = Expression.BuiltinFunctionOp.ACOS;
		else if (functionName.equals("asin"))
			bifop = Expression.BuiltinFunctionOp.ASIN;
		else if (functionName.equals("atan"))
			bifop = Expression.BuiltinFunctionOp.ATAN;
		else if (functionName.equals("diag"))
			bifop = Expression.BuiltinFunctionOp.DIAG;
		else if (functionName.equals("exp"))
			 bifop = Expression.BuiltinFunctionOp.EXP;
		else if (functionName.equals("abs"))
			bifop = Expression.BuiltinFunctionOp.ABS;
		else if (functionName.equals("min"))
			bifop = Expression.BuiltinFunctionOp.MIN;
		else if (functionName.equals("max"))
			 bifop = Expression.BuiltinFunctionOp.MAX;
		else if (functionName.equals("pmin"))
			 bifop = Expression.BuiltinFunctionOp.PMIN;
		else if (functionName.equals("pmax"))
			 bifop = Expression.BuiltinFunctionOp.PMAX;
		else if (functionName.equals("ppred"))
			bifop = Expression.BuiltinFunctionOp.PPRED;
		else if (functionName.equals("log"))
			bifop = Expression.BuiltinFunctionOp.LOG;
		else if (functionName.equals("length"))
			bifop = Expression.BuiltinFunctionOp.LENGTH;
		else if (functionName.equals("ncol"))
			 bifop = Expression.BuiltinFunctionOp.NCOL;
		else if (functionName.equals("nrow"))
			bifop = Expression.BuiltinFunctionOp.NROW;
		else if (functionName.equals("sqrt"))
			 bifop = Expression.BuiltinFunctionOp.SQRT;
		else if (functionName.equals("sum"))
			bifop = Expression.BuiltinFunctionOp.SUM;
		else if (functionName.equals("mean"))
			bifop = Expression.BuiltinFunctionOp.MEAN;
		else if (functionName.equals("trace"))
			bifop = Expression.BuiltinFunctionOp.TRACE;
		else if (functionName.equals("t"))
			 bifop = Expression.BuiltinFunctionOp.TRANS;
		else if (functionName.equals("append"))
			bifop = Expression.BuiltinFunctionOp.APPEND;
		else if (functionName.equals("range"))
			bifop = Expression.BuiltinFunctionOp.RANGE;
		else if (functionName.equals("prod"))
			bifop = Expression.BuiltinFunctionOp.PROD;
		else if (functionName.equals("rowSums"))
			bifop = Expression.BuiltinFunctionOp.ROWSUM;
		else if (functionName.equals("colSums"))
			bifop = Expression.BuiltinFunctionOp.COLSUM;
		else if (functionName.equals("rowMins"))
			bifop = Expression.BuiltinFunctionOp.ROWMIN;
		else if (functionName.equals("colMins"))
			bifop = Expression.BuiltinFunctionOp.COLMIN;
		else if (functionName.equals("rowMaxs"))
			bifop = Expression.BuiltinFunctionOp.ROWMAX;
		else if (functionName.equals("rowIndexMax"))
			bifop = Expression.BuiltinFunctionOp.ROWINDEXMAX;
		else if (functionName.equals("colMaxs"))
			bifop = Expression.BuiltinFunctionOp.COLMAX;
		else if (functionName.equals("rowMeans"))
			bifop = Expression.BuiltinFunctionOp.ROWMEAN;
		else if (functionName.equals("colMeans"))
			 bifop = Expression.BuiltinFunctionOp.COLMEAN;
		else if (functionName.equals("castAsScalar"))
			bifop = Expression.BuiltinFunctionOp.CAST_AS_SCALAR;
		else if (functionName.equals("quantile"))
			bifop= Expression.BuiltinFunctionOp.QUANTILE;
		else if (functionName.equals("interQuantile"))
			bifop= Expression.BuiltinFunctionOp.INTERQUANTILE;
		else if (functionName.equals("interQuartileMean"))
			bifop= Expression.BuiltinFunctionOp.IQM;
		else if (functionName.equals("ctable"))
			bifop = Expression.BuiltinFunctionOp.CTABLE;
		else if (functionName.equals("round"))
			bifop = Expression.BuiltinFunctionOp.ROUND;
		else if (functionName.equals("centralMoment"))
			 bifop = Expression.BuiltinFunctionOp.CENTRALMOMENT;
		else if (functionName.equals("cov"))
			bifop = Expression.BuiltinFunctionOp.COVARIANCE;
		else if (functionName.equals("seq"))
			bifop = Expression.BuiltinFunctionOp.SEQ;
		else if (functionName.equals("qr"))
			bifop = Expression.BuiltinFunctionOp.QR;
		else if (functionName.equals("lu"))
			bifop = Expression.BuiltinFunctionOp.LU;
		else
			return null;
		
		Expression expr1 = exprs.size() >= 1 ? expr1 = exprs.get(0) : null;
		Expression expr2 = exprs.size() >= 2 ? expr2 = exprs.get(1) : null;
		Expression expr3 = exprs.size() >= 3 ? expr3 = exprs.get(2) : null;
		BuiltinFunctionExpression retVal = new BuiltinFunctionExpression(bifop,expr1, expr2, expr3);
	
		return retVal;
	} // end method getBuiltinFunctionExpression

}