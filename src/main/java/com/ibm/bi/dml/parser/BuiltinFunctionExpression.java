/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.parser.LanguageException.LanguageErrorCodes;

public class BuiltinFunctionExpression extends DataIdentifier 
{
	
	protected Expression[] 	  _args = null;
	private BuiltinFunctionOp _opcode;

	public BuiltinFunctionExpression(BuiltinFunctionOp bifop, ArrayList<ParameterExpression> args, String fname, int blp, int bcp, int elp, int ecp) {
		_kind = Kind.BuiltinFunctionOp;
		_opcode = bifop;
		_args = new Expression[args.size()];
		for(int i=0; i < args.size(); i++) {
			_args[i] = args.get(i).getExpr();
		}
		this.setAllPositions(fname, blp, bcp, elp, ecp);
	}

	public BuiltinFunctionExpression(BuiltinFunctionOp bifop, Expression[] args, String fname, int blp, int bcp, int elp, int ecp) {
		_kind = Kind.BuiltinFunctionOp;
		_opcode = bifop;
		_args = new Expression[args.length];
		for(int i=0; i < args.length; i++) {
			_args[i] = args[i];
		}
		this.setAllPositions(fname, blp, bcp, elp, ecp);
	}

	public Expression rewriteExpression(String prefix) throws LanguageException {

		Expression[] newArgs = new Expression[_args.length];
		for(int i=0; i < _args.length; i++) {
			newArgs[i] = _args[i].rewriteExpression(prefix);
		}
		BuiltinFunctionExpression retVal = new BuiltinFunctionExpression(this._opcode, newArgs, 
				this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		return retVal;
	
	}

	public BuiltinFunctionOp getOpCode() {
		return _opcode;
	}

	public Expression getFirstExpr() {
		return (_args.length >= 1 ? _args[0] : null);
	}

	public Expression getSecondExpr() {
		return (_args.length >= 2 ? _args[1] : null);
	}

	public Expression getThirdExpr() {
		return (_args.length >= 3 ? _args[2] : null);
	}

	public Expression[] getAllExpr(){
		return _args;
	}
	
	@Override
	public void validateExpression(MultiAssignmentStatement stmt, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional)
			throws LanguageException 
	{
		if (this.getFirstExpr() instanceof FunctionCallIdentifier){
			raiseValidateError("UDF function call not supported as parameter to built-in function call", false);
		}
		
		this.getFirstExpr().validateExpression(ids, constVars, conditional);
		if (getSecondExpr() != null){
			if (this.getSecondExpr() instanceof FunctionCallIdentifier){
				raiseValidateError("UDF function call not supported as parameter to built-in function call", false);
			}
			getSecondExpr().validateExpression(ids, constVars, conditional);
		}
		if (getThirdExpr() != null) {
			if (this.getThirdExpr() instanceof FunctionCallIdentifier){
				raiseValidateError("UDF function call not supported as parameter to built-in function call", false);
			}
			getThirdExpr().validateExpression(ids, constVars, conditional);
		}
		_outputs = new Identifier[stmt.getTargetList().size()];
		int count = 0;
		for (DataIdentifier outParam: stmt.getTargetList()){
			DataIdentifier tmp = new DataIdentifier(outParam);
			tmp.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			_outputs[count++] = tmp;
		}
		
		switch (_opcode) {
		case QR:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			
			// setup output properties
			DataIdentifier qrOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier qrOut2 = (DataIdentifier) getOutputs()[1];
			
			long rows = getFirstExpr().getOutput().getDim1();
			long cols = getFirstExpr().getOutput().getDim2();
			
			// Output1 - Q
			qrOut1.setDataType(DataType.MATRIX);
			qrOut1.setValueType(ValueType.DOUBLE);
			qrOut1.setDimensions(rows, cols);
			qrOut1.setBlockDimensions(getFirstExpr().getOutput().getRowsInBlock(), getFirstExpr().getOutput().getColumnsInBlock());
			
			// Output2 - R
			qrOut2.setDataType(DataType.MATRIX);
			qrOut2.setValueType(ValueType.DOUBLE);
			qrOut2.setDimensions(rows, cols);
			qrOut2.setBlockDimensions(getFirstExpr().getOutput().getRowsInBlock(), getFirstExpr().getOutput().getColumnsInBlock());
			
			break;

		case LU:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			
			// setup output properties
			DataIdentifier luOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier luOut2 = (DataIdentifier) getOutputs()[1];
			DataIdentifier luOut3 = (DataIdentifier) getOutputs()[2];
			
			long inrows = getFirstExpr().getOutput().getDim1();
			long incols = getFirstExpr().getOutput().getDim2();
			
			if ( inrows != incols ) {
				raiseValidateError("LU Decomposition can only be done on a square matrix. Input matrix is rectangular (rows=" + inrows + ", cols="+incols+")", conditional);
			}
			
			// Output1 - P
			luOut1.setDataType(DataType.MATRIX);
			luOut1.setValueType(ValueType.DOUBLE);
			luOut1.setDimensions(inrows, inrows);
			luOut1.setBlockDimensions(getFirstExpr().getOutput().getRowsInBlock(), getFirstExpr().getOutput().getColumnsInBlock());
			
			// Output2 - L
			luOut2.setDataType(DataType.MATRIX);
			luOut2.setValueType(ValueType.DOUBLE);
			luOut2.setDimensions(inrows, inrows);
			luOut2.setBlockDimensions(getFirstExpr().getOutput().getRowsInBlock(), getFirstExpr().getOutput().getColumnsInBlock());
			
			// Output3 - U
			luOut3.setDataType(DataType.MATRIX);
			luOut3.setValueType(ValueType.DOUBLE);
			luOut3.setDimensions(inrows, inrows);
			luOut3.setBlockDimensions(getFirstExpr().getOutput().getRowsInBlock(), getFirstExpr().getOutput().getColumnsInBlock());
			
			break;

		case EIGEN:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			
			// setup output properties
			DataIdentifier eigenOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier eigenOut2 = (DataIdentifier) getOutputs()[1];
			
			if ( getFirstExpr().getOutput().getDim1() != getFirstExpr().getOutput().getDim2() ) {
				raiseValidateError("Eigen Decomposition can only be done on a square matrix. Input matrix is rectangular (rows=" + getFirstExpr().getOutput().getDim1() + ", cols="+ getFirstExpr().getOutput().getDim2() +")", conditional);
			}
			
			// Output1 - Eigen Values
			eigenOut1.setDataType(DataType.MATRIX);
			eigenOut1.setValueType(ValueType.DOUBLE);
			eigenOut1.setDimensions(getFirstExpr().getOutput().getDim1(), 1);
			eigenOut1.setBlockDimensions(getFirstExpr().getOutput().getRowsInBlock(), getFirstExpr().getOutput().getColumnsInBlock());
			
			// Output2 - Eigen Vectors
			eigenOut2.setDataType(DataType.MATRIX);
			eigenOut2.setValueType(ValueType.DOUBLE);
			eigenOut2.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			eigenOut2.setBlockDimensions(getFirstExpr().getOutput().getRowsInBlock(), getFirstExpr().getOutput().getColumnsInBlock());
			
			break;
		
		default: //always unconditional
			raiseValidateError("Unknown Builtin Function opcode: " + _opcode, false);
		}
	}

	/**
	 * Validate parse tree : Process BuiltinFunction Expression in an assignment
	 * statement
	 * 
	 * @throws LanguageException
	 */
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional)
			throws LanguageException {
		
		for(int i=0; i < _args.length; i++ ) {
			
			if (_args[i] instanceof FunctionCallIdentifier){
				raiseValidateError("UDF function call not supported as parameter to built-in function call", false);
			}
			
			_args[i].validateExpression(ids, constVars, conditional);
		}
		
		// checkIdentifierParams();
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		Identifier id = this.getFirstExpr().getOutput();
		output.setProperties(this.getFirstExpr().getOutput());
		output.setNnz(-1); //conservatively, cannot use input nnz!
		this.setOutput(output);
		
		switch (this.getOpCode()) {
		case COLSUM:
		case COLMAX:
		case COLMIN:
		case COLMEAN:
			// colSums(X);
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(1, id.getDim2());
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			break;
		case ROWSUM:
		case ROWMAX:
		case ROWINDEXMAX:
		case ROWMIN:
		case ROWINDEXMIN:
		case ROWMEAN:
			//rowSums(X);
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
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
			checkMatrixParam(getFirstExpr());
			
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(id.getValueType());
			
			break;
		
		case MEAN:
			//checkNumParameters(2, false); // mean(Y) or mean(Y,W)
            if (getSecondExpr() != null) {
            	checkNumParameters (2);
            }
            else {
            	checkNumParameters (1);
            }
			
			checkMatrixParam(getFirstExpr());
			if ( getSecondExpr() != null ) {
				// x = mean(Y,W);
				checkMatchingDimensions(getFirstExpr(), getSecondExpr());
			}
			
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(id.getValueType());
			break;
			
		case MIN:
		case MAX:
			//min(X), min(X,s), min(s,X), min(s,r), min(X,Y)
			
			//unary aggregate
			if (getSecondExpr() == null) 
			{
				checkNumParameters(1);
				checkMatrixParam(getFirstExpr());
				output.setDataType( DataType.SCALAR );
				output.setDimensions(0, 0);
				output.setBlockDimensions (0, 0);
			}
			//binary operation
			else
			{
				checkNumParameters(2);
				DataType dt1 = getFirstExpr().getOutput().getDataType();
				DataType dt2 = getSecondExpr().getOutput().getDataType();
				DataType dtOut = (dt1==DataType.MATRIX || dt2==DataType.MATRIX)?
				                   DataType.MATRIX : DataType.SCALAR;				
				if( dt1==DataType.MATRIX && dt2==DataType.MATRIX )
					checkMatchingDimensions(getFirstExpr(), getSecondExpr(), true);
				//determine output dimensions
				long[] dims = getBinaryMatrixCharacteristics(getFirstExpr(), getSecondExpr());
				output.setDataType( dtOut );
				output.setDimensions(dims[0], dims[1]);
				output.setBlockDimensions (dims[2], dims[3]);
			}
			output.setValueType(id.getValueType());
			
			break;
		
		case CUMSUM:
		case CUMPROD:
		case CUMMIN:
		case CUMMAX:
			// cumsum(X);
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), id.getDim2());
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			
			break;
			
		case CAST_AS_SCALAR:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			if (( getFirstExpr().getOutput().getDim1() != -1 && getFirstExpr().getOutput().getDim1() !=1) || ( getFirstExpr().getOutput().getDim2() != -1 && getFirstExpr().getOutput().getDim2() !=1)) {
				raiseValidateError("dimension mismatch while casting matrix to scalar: dim1: " + getFirstExpr().getOutput().getDim1() +  " dim2 " + getFirstExpr().getOutput().getDim2(), 
				          conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(id.getValueType());
			break;
		case CAST_AS_MATRIX:
			checkNumParameters(1);
			checkScalarParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(1, 1);
			output.setBlockDimensions(id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			break;
		case CAST_AS_DOUBLE:
			checkNumParameters(1);
			checkScalarParam(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			//output.setDataType(id.getDataType()); //TODO whenever we support multiple matrix value types, currently noop.
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(ValueType.DOUBLE);
			break;
		case CAST_AS_INT:
			checkNumParameters(1);
			checkScalarParam(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			//output.setDataType(id.getDataType()); //TODO whenever we support multiple matrix value types, currently noop.
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(ValueType.INT);
			break;
		case CAST_AS_BOOLEAN:
			checkNumParameters(1);
			checkScalarParam(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			//output.setDataType(id.getDataType()); //TODO whenever we support multiple matrix value types, currently noop.
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(ValueType.BOOLEAN);
			break;	
		case APPEND:
			checkNumParameters(2);
			
			//scalar string append (string concatenation with \n)
			if( getFirstExpr().getOutput().getDataType()==DataType.SCALAR )
			{
				checkScalarParam(getFirstExpr());
				checkScalarParam(getSecondExpr());
				checkValueTypeParam(getFirstExpr(), ValueType.STRING);
				checkValueTypeParam(getSecondExpr(), ValueType.STRING);
			}
			//matrix append (cbind)
			else
			{				
				checkMatrixParam(getFirstExpr());
				checkMatrixParam(getSecondExpr());
			}
			
			output.setDataType(id.getDataType());
			output.setValueType(id.getValueType());
			
			// set output dimensions
			long appendDim1 = -1, appendDim2 = -1;
			if (getFirstExpr().getOutput().getDim1() > 0 && getSecondExpr().getOutput().getDim1() > 0){
				if (getFirstExpr().getOutput().getDim1() != getSecondExpr().getOutput().getDim1()){
					raiseValidateError("inputs to append must have same number of rows: input 1 rows: " + 
							getFirstExpr().getOutput().getDim1() +  ", input 2 rows " + getSecondExpr().getOutput().getDim1(), 
							 conditional, LanguageErrorCodes.INVALID_PARAMETERS);
				}
				appendDim1 = getFirstExpr().getOutput().getDim1();
			}
			else if (getFirstExpr().getOutput().getDim1() > 0)	
				appendDim1 = getFirstExpr().getOutput().getDim1(); 
			else if (getSecondExpr().getOutput().getDim1() > 0 )
				appendDim1 = getSecondExpr().getOutput().getDim1(); 
				
			if (getFirstExpr().getOutput().getDim2() > 0 && getSecondExpr().getOutput().getDim2() > 0){
				appendDim2 = getFirstExpr().getOutput().getDim2() + getSecondExpr().getOutput().getDim2();
			}
			
			output.setDimensions(appendDim1, appendDim2); 
			
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			break;
		case PPRED:
			// ppred (X,Y, "<"); ppred (X,y, "<"); ppred (y,X, "<");
			checkNumParameters(3);
			
			DataType dt1 = getFirstExpr().getOutput().getDataType();
			DataType dt2 = getSecondExpr().getOutput().getDataType();
			
			//check input data types
			if( dt1 == DataType.SCALAR && dt2 == DataType.SCALAR ) {
				raiseValidateError("ppred() requires at least one matrix input.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}			
			if( dt1 == DataType.MATRIX )
				checkMatrixParam(getFirstExpr());
			if( dt2 == DataType.MATRIX )
				checkMatrixParam(getSecondExpr());
			if( dt1==DataType.MATRIX && dt2==DataType.MATRIX ) //dt1==dt2
			      checkMatchingDimensions(getFirstExpr(), getSecondExpr(), true);
			
			//check operator
			if (getThirdExpr().getOutput().getDataType() != DataType.SCALAR || 
				getThirdExpr().getOutput().getValueType() != ValueType.STRING) 
			{	
				raiseValidateError("Third argument in ppred() is not an operator ", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			//determine output dimensions
			long[] dims = getBinaryMatrixCharacteristics(getFirstExpr(), getSecondExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(dims[0], dims[1]);
			output.setBlockDimensions(dims[2], dims[3]);
			output.setValueType(id.getValueType());
			break;

		case TRANS:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim2(), id.getDim1());
			output.setBlockDimensions (id.getColumnsInBlock(), id.getRowsInBlock());
			output.setValueType(id.getValueType());
			break;
		case DIAG:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			if( id.getDim2() != -1 ) { //type known
				if ( id.getDim2() == 1 ) 
				{
					//diag V2M
					output.setDimensions(id.getDim1(), id.getDim1());
				} 
				else 
				{
					if (id.getDim1() != id.getDim2()) {
						raiseValidateError("Invoking diag on matrix with dimensions ("
								+ id.getDim1() + "," + id.getDim2()
								+ ") in " + this.toString(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
					//diag M2V
					output.setDimensions(id.getDim1(), 1);
				}
			}
			output.setBlockDimensions (id.getRowsInBlock(), id.getColumnsInBlock());
			output.setValueType(id.getValueType());
			break;
		case NROW:
		case NCOL:
		case LENGTH:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlockDimensions (0, 0);
			output.setValueType(ValueType.INT);
			break;

		// Contingency tables
		case TABLE:
			
			/*
			 * Allowed #of arguments: 2,3,4,5
			 * table(A,B)
			 * table(A,B,W)
			 * table(A,B,1)
			 * table(A,B,dim1,dim2)
			 * table(A,B,W,dim1,dim2)
			 * table(A,B,1,dim1,dim2)
			 */
			
			// Check for validity of input arguments, and setup output dimensions
			
			// First input: is always of type MATRIX
			checkMatrixParam(getFirstExpr());
			
			if ( getSecondExpr() == null )
				raiseValidateError("Invalid number of arguments to table(): " 
						+ this.toString(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			
			// Second input: can be MATRIX or SCALAR
			// cases: table(A,B) or table(A,1)
			if ( getSecondExpr().getOutput().getDataType() == DataType.MATRIX)
				checkMatchingDimensions(getFirstExpr(),getSecondExpr());
			
			long outputDim1=-1, outputDim2=-1;
			
			switch(_args.length) {
			case 2:
				// nothing to do
				break;
				
			case 3:
				// case - table w/ weights
				//        - weights specified as a matrix: table(A,B,W) or table(A,1,W)
				//        - weights specified as a scalar: table(A,B,1) or table(A,1,1)
				if ( getThirdExpr().getOutput().getDataType() == DataType.MATRIX)
					checkMatchingDimensions(getFirstExpr(),getThirdExpr());
				break;
				
			case 4:
				// case - table w/ output dimensions: table(A,B,dim1,dim2) or table(A,1,dim1,dim2)
				// third and fourth arguments must be scalars
				if ( getThirdExpr().getOutput().getDataType() != DataType.SCALAR || _args[3].getOutput().getDataType() != DataType.SCALAR ) {
					raiseValidateError("Invalid argument types to table(): output dimensions must be of type scalar: " 
							+ this.toString(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
				}
				else {
					// constant propagation
					if( getThirdExpr() instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)getThirdExpr()).getName()) )
						_args[2] = constVars.get(((DataIdentifier)getThirdExpr()).getName());
					if( _args[3] instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)_args[3]).getName()) )
						_args[3] = constVars.get(((DataIdentifier)_args[3]).getName());
					
					if ( getThirdExpr().getOutput() instanceof ConstIdentifier ) 
						outputDim1 = ((ConstIdentifier) getThirdExpr().getOutput()).getLongValue();
					if ( _args[3].getOutput() instanceof ConstIdentifier ) 
						outputDim2 = ((ConstIdentifier) _args[3].getOutput()).getLongValue();
				}
				break;
				
			case 5:
				// case - table w/ weights and output dimensions: 
				//        - table(A,B,W,dim1,dim2) or table(A,1,W,dim1,dim2)
				//        - table(A,B,1,dim1,dim2) or table(A,1,1,dim1,dim2)
				
				if ( getThirdExpr().getOutput().getDataType() == DataType.MATRIX)
					checkMatchingDimensions(getFirstExpr(),getThirdExpr());
				
				// fourth and fifth arguments must be scalars
				if ( _args[3].getOutput().getDataType() != DataType.SCALAR || _args[4].getOutput().getDataType() != DataType.SCALAR ) {
					raiseValidateError("Invalid argument types to table(): output dimensions must be of type scalar: " 
							+ this.toString(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
				}
				else {
					// constant propagation
					if( _args[3] instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)_args[3]).getName()) )
						_args[3] = constVars.get(((DataIdentifier)_args[3]).getName());
					if( _args[4] instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)_args[4]).getName()) )
						_args[4] = constVars.get(((DataIdentifier)_args[4]).getName());
					
					if ( _args[3].getOutput() instanceof ConstIdentifier ) 
						outputDim1 = ((ConstIdentifier) _args[3].getOutput()).getLongValue();
					if ( _args[4].getOutput() instanceof ConstIdentifier ) 
						outputDim2 = ((ConstIdentifier) _args[4].getOutput()).getLongValue();
				}
				break;

			default:
				raiseValidateError("Invalid number of arguments to table(): " 
						+ this.toString(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			// The dimensions for the output matrix will be known only at the
			// run time
			output.setDimensions(outputDim1, outputDim2);
			output.setBlockDimensions (-1, -1);
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			break;

		case MOMENT:
			/*
			 * x = centralMoment(V,order) or xw = centralMoment(V,W,order)
			 */
			checkMatrixParam(getFirstExpr());
			if (getThirdExpr() != null) {
			   checkNumParameters(3);
			   checkMatrixParam(getSecondExpr());
			   checkMatchingDimensions(getFirstExpr(),getSecondExpr());
			   checkScalarParam(getThirdExpr());
			}
			else {
			   checkNumParameters(2);
			   checkScalarParam(getSecondExpr());
			}

			// output is a scalar
			output.setDataType(DataType.SCALAR);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(0, 0);
			output.setBlockDimensions(0,0);
			break;

		case COV:
			/*
			 * x = cov(V1,V2) or xw = cov(V1,V2,W)
			 */
			if (getThirdExpr() != null) {
				checkNumParameters(3);
			}
			else {
				checkNumParameters(2);
			}
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			checkMatchingDimensions(getFirstExpr(),getSecondExpr());
			
			if (getThirdExpr() != null) {
				checkMatrixParam(getThirdExpr());
			 checkMatchingDimensions(getFirstExpr(), getThirdExpr());
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
			if(getThirdExpr() != null) {
			    checkNumParameters(3);
			}
			else {
				checkNumParameters(2);
			}
			
			// first parameter must always be a 1D matrix 
			check1DMatrixParam(getFirstExpr());
			
			// check for matching dimensions for other matrix parameters
			if (getThirdExpr() != null) {
			    checkMatrixParam(getSecondExpr());
				checkMatchingDimensions(getFirstExpr(), getSecondExpr());
			}
			
			// set the properties for _output expression
			// output dimensions = dimensions of second, if third is null
			//                   = dimensions of the third, otherwise.

			if (getThirdExpr() != null) {
				output.setDimensions(getThirdExpr().getOutput().getDim1(), getThirdExpr().getOutput()
						.getDim2());
				output.setBlockDimensions(getThirdExpr().getOutput().getRowsInBlock(), 
						                  getThirdExpr().getOutput().getColumnsInBlock());
				output.setDataType(getThirdExpr().getOutput().getDataType());
			} else {
				output.setDimensions(getSecondExpr().getOutput().getDim1(), getSecondExpr().getOutput()
						.getDim2());
				output.setBlockDimensions(getSecondExpr().getOutput().getRowsInBlock(), 
		                  getSecondExpr().getOutput().getColumnsInBlock());
				output.setDataType(getSecondExpr().getOutput().getDataType());
			}
			break;

		case INTERQUANTILE:
			if (getThirdExpr() != null) {
			    checkNumParameters(3);
			}
			else {
				checkNumParameters(2);
			}
			checkMatrixParam(getFirstExpr());
			if (getThirdExpr() != null) {
				// i.e., second input is weight vector
				checkMatrixParam(getSecondExpr());
				checkMatchingDimensionsQuantile();
			}

			if ((getThirdExpr() == null && getSecondExpr().getOutput().getDataType() != DataType.SCALAR)
					&& (getThirdExpr() != null && getThirdExpr().getOutput().getDataType() != DataType.SCALAR)) {
				
				raiseValidateError("Invalid parameters to "+ this.getOpCode(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
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
			if (getSecondExpr() != null){
			    checkNumParameters(2);
		    }
			else {
				checkNumParameters(1);
			}
			checkMatrixParam(getFirstExpr());

			if (getSecondExpr() != null) {
				// i.e., second input is weight vector
				checkMatrixParam(getSecondExpr());
				checkMatchingDimensions(getFirstExpr(), getSecondExpr());
			}

			// Output is a scalar
			output.setValueType(id.getValueType());
			output.setDimensions(0, 0);
			output.setBlockDimensions(0,0);
			output.setDataType(DataType.SCALAR);

			break;
		
		case MEDIAN:
			if (getSecondExpr() != null){
			    checkNumParameters(2);
		    }
			else {
				checkNumParameters(1);
			}
			checkMatrixParam(getFirstExpr());

			if (getSecondExpr() != null) {
				// i.e., second input is weight vector
				checkMatrixParam(getSecondExpr());
				checkMatchingDimensions(getFirstExpr(), getSecondExpr());
			}

			// Output is a scalar
			output.setValueType(id.getValueType());
			output.setDimensions(0, 0);
			output.setBlockDimensions(0,0);
			output.setDataType(DataType.SCALAR);

			break;
			
		case SAMPLE:
		{
			Expression[] in = getAllExpr(); 
			
			for(Expression e : in)
				checkScalarParam(e);
			
			if (in[0].getOutput().getValueType() != ValueType.DOUBLE && in[0].getOutput().getValueType() != ValueType.INT) 
				throw new LanguageException("First argument to sample() must be a number.");
			if (in[1].getOutput().getValueType() != ValueType.DOUBLE && in[1].getOutput().getValueType() != ValueType.INT) 
				throw new LanguageException("Second argument to sample() must be a number.");
			
			boolean check = false;
			if ( isConstant(in[0]) && isConstant(in[1]) )
			{
				long range = ((ConstIdentifier)in[0]).getLongValue();
				long size = ((ConstIdentifier)in[1]).getLongValue();
				if ( range < size )
					check = true;
			}
			
			if(in.length == 4 )
			{
				checkNumParameters(4);
				if (in[3].getOutput().getValueType() != ValueType.INT) 
					throw new LanguageException("Fourth arugment, seed, to sample() must be an integer value.");
				if (in[2].getOutput().getValueType() != ValueType.BOOLEAN ) 
					throw new LanguageException("Third arugment to sample() must either denote replacement policy (boolean) or seed (integer).");
			}
			else if(in.length == 3) 
			{
				checkNumParameters(3);
				if (in[2].getOutput().getValueType() != ValueType.BOOLEAN 
						&& in[2].getOutput().getValueType() != ValueType.INT ) 
					throw new LanguageException("Third arugment to sample() must either denote replacement policy (boolean) or seed (integer).");
			}
			
			if ( check && in.length >= 3 
					&& isConstant(in[2]) 
					&& in[2].getOutput().getValueType() == ValueType.BOOLEAN  
					&& !((BooleanIdentifier)in[2]).getValue() )
				throw new LanguageException("Sample (size=" + ((ConstIdentifier)in[0]).getLongValue() 
						+ ") larger than population (size=" + ((ConstIdentifier)in[1]).getLongValue() 
						+ ") can only be generated with replacement.");
			
			// Output is a column vector
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			
			if ( isConstant(in[1]) )
	 			output.setDimensions(((ConstIdentifier)in[1]).getLongValue(), 1);
			else
				output.setDimensions(-1, 1);
 			setBlockDimensions(id.getRowsInBlock(), id.getColumnsInBlock());
 			
			break;
		}
		case SEQ:
			
			//basic parameter validation
			checkScalarParam(getFirstExpr());
			checkScalarParam(getSecondExpr());
			if ( getThirdExpr() != null ) {
				checkNumParameters(3);
				checkScalarParam(getThirdExpr());
			}
			else
				checkNumParameters(2);
			
			// constant propagation (from, to, incr)
			if( getFirstExpr() instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)getFirstExpr()).getName()) )
				_args[0] = constVars.get(((DataIdentifier)getFirstExpr()).getName());
			if( getSecondExpr() instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)getSecondExpr()).getName()) )
				_args[1] = constVars.get(((DataIdentifier)getSecondExpr()).getName());
			if( getThirdExpr()!=null && getThirdExpr() instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)getThirdExpr()).getName()) )
				_args[2] = constVars.get(((DataIdentifier)getThirdExpr()).getName());
			
			// check if dimensions can be inferred
			long dim1=-1, dim2=1;
			if ( isConstant(getFirstExpr()) && isConstant(getSecondExpr()) && (getThirdExpr() != null ? isConstant(getThirdExpr()) : true) ) {
				double from, to, incr;
				boolean neg;
				try {
					from = getDoubleValue(getFirstExpr());
					to = getDoubleValue(getSecondExpr());
					
					// Setup the value of increment
					// default value: 1 if from <= to; -1 if from > to
					neg = (from > to);
					if(getThirdExpr() == null) {
						expandArguments();
						_args[2] = new DoubleIdentifier((neg? -1.0 : 1.0),
								this.getFilename(), this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
					}
					incr = getDoubleValue(getThirdExpr()); 
					
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

		case SOLVE:
			checkNumParameters(2);
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			
			if ( getSecondExpr().getOutput().dimsKnown() && !is1DMatrix(getSecondExpr()) )
				raiseValidateError("Second input to solve() must be a vector", conditional);
			
			if ( getFirstExpr().getOutput().dimsKnown() && getSecondExpr().getOutput().dimsKnown() && 
					getFirstExpr().getOutput().getDim1() != getSecondExpr().getOutput().getDim1() )
				raiseValidateError("Dimension mismatch in a call to solve()", conditional);
			
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(getFirstExpr().getOutput().getDim2(), 1);
			output.setBlockDimensions(0, 0);
			break;
		
		case INVERSE:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			
			Identifier in = getFirstExpr().getOutput();
			if(in.dimsKnown() && in.getDim1() != in.getDim2()) 
				raiseValidateError("Input to inv() must be square matrix -- given: a " + in.getDim1() + "x" + in.getDim2() + " matrix.", conditional);
			
			output.setDimensions(in.getDim1(), in.getDim2());
			output.setBlockDimensions(in.getRowsInBlock(), in.getColumnsInBlock());
			break;
			
		case OUTER:
			Identifier id2 = this.getSecondExpr().getOutput();
			
			//check input types and characteristics
			checkNumParameters(3);
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			checkScalarParam(getThirdExpr());
			checkValueTypeParam(getThirdExpr(), ValueType.STRING);
			if( id.getDim2() > 1 || id2.getDim1()>1 ) {
				raiseValidateError("Outer vector operations require a common dimension of one: " +
			                       id.getDim1()+"x"+id.getDim2()+" o "+id2.getDim1()+"x"+id2.getDim2()+".", false);
			}
			
			//set output characteristics
			output.setDataType(id.getDataType());
			output.setDimensions(id.getDim1(), id2.getDim2());
			output.setBlockDimensions(id.getRowsInBlock(), id.getColumnsInBlock()); 
			break;
			
		default:
			if (this.isMathFunction()) {
				// datatype and dimensions are same as this.getExpr()
				if (this.getOpCode() == BuiltinFunctionOp.ABS) {
					output.setValueType(getFirstExpr().getOutput().getValueType());
				} else {
					output.setValueType(ValueType.DOUBLE);
				}
				checkMathFunctionParam();
				output.setDataType(id.getDataType());
				output.setDimensions(id.getDim1(), id.getDim2());
				output.setBlockDimensions(id.getRowsInBlock(), id.getColumnsInBlock()); 
			} else{
				// always unconditional (because unsupported operation)
				raiseValidateError("Unsupported function "+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
		return;
	}
	
	private void expandArguments() {
	
		if ( _args == null ) {
			_args = new Expression[1];
			return;
		}
		Expression [] temp = _args.clone();
		_args = new Expression[_args.length + 1];
	    System.arraycopy(temp, 0, _args, 0, temp.length);
	}
	
	@Override
	public boolean multipleReturns() {
		switch(_opcode) {
		case QR:
		case LU:
		case EIGEN:
			return true;
		default:
			return false;
		}
	}

	/**
	 * 
	 * @param expr
	 * @return
	 */
	private boolean isConstant(Expression expr) {
		return ( expr != null && expr instanceof ConstIdentifier );
	}
	
	/**
	 * 
	 * @param expr
	 * @return
	 * @throws LanguageException
	 */
	private double getDoubleValue(Expression expr) 
		throws LanguageException 
	{
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
		case CEIL:
		case FLOOR:
		case MEDIAN:
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
		case CEIL:
		case FLOOR:
		case MEDIAN:
			checkNumParameters(1);
			break;
		case LOG:
			if (getSecondExpr() != null) {
			  checkNumParameters(2);
			}
			else {
			  checkNumParameters(1);
			}
			break;
		default:
			//always unconditional
			raiseValidateError("Unknown math function "+ this.getOpCode(), false);
		}
	}

	public String toString() {
		StringBuilder sb = new StringBuilder(_opcode.toString() + "(" + _args[0].toString());
		for(int i=1; i < _args.length; i++) {
			sb.append(",");
			sb.append(_args[i].toString());
		}
		sb.append(")");
		return sb.toString();
	}

	@Override
	// third part of expression IS NOT a variable -- it is the OP to be applied
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		for(int i=0; i<_args.length; i++) {
			result.addVariables(_args[i].variablesRead());
		}
		
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		// result.addVariables(_first.variablesUpdated());
		return result;
	}

	/**
	 * 
	 * @param count
	 * @throws LanguageException
	 */
	protected void checkNumParameters(int count) //always unconditional
		throws LanguageException 
	{
		if (getFirstExpr() == null){
			raiseValidateError("Missing parameter for function "+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		
       	if (((count == 1) && (getSecondExpr()!= null || getThirdExpr() != null)) || 
        		((count == 2) && (getThirdExpr() != null))){ 
       		raiseValidateError("Invalid number of parameters for function "+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
       	}
       	else if (((count == 2) && (getSecondExpr() == null)) || 
		             ((count == 3) && (getSecondExpr() == null || getThirdExpr() == null))){
       		raiseValidateError( "Missing parameter for function "+this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
       	}
	}

	/**
	 * 
	 * @param e
	 * @throws LanguageException
	 */
	protected void checkMatrixParam(Expression e) //always unconditional
		throws LanguageException 
	{
		if (e.getOutput().getDataType() != DataType.MATRIX) {
			raiseValidateError("Expecting matrix parameter for function "+ this.getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	/**
	 * 
	 * @param e
	 * @throws LanguageException
	 */
	private void checkScalarParam(Expression e) //always unconditional
		throws LanguageException 
	{
		if (e.getOutput().getDataType() != DataType.SCALAR) 
		{
			raiseValidateError("Expecting scalar parameter for function " + this.getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	/**
	 * 
	 * @param e
	 * @param vt
	 * @throws LanguageException
	 */
	private void checkValueTypeParam(Expression e, ValueType vt) //always unconditional
		throws LanguageException 
	{
		if (e.getOutput().getValueType() != vt) {
			raiseValidateError("Expecting parameter of different value type " + this.getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	private boolean is1DMatrix(Expression e) {
		return (e.getOutput().getDim1() == 1 || e.getOutput().getDim2() == 1 );
	}
	
	private boolean dimsKnown(Expression e) {
		return (e.getOutput().getDim1() != -1 && e.getOutput().getDim2() != -1);
	}
	
	/**
	 * 
	 * @param e
	 * @throws LanguageException
	 */
	private void check1DMatrixParam(Expression e) //always unconditional
		throws LanguageException 
	{	
		checkMatrixParam(e);
		
		// throw an exception, when e's output is NOT a one-dimensional matrix 
		// the check must be performed only when the dimensions are known at compilation time
		if ( dimsKnown(e) && !is1DMatrix(e)) {
			raiseValidateError("Expecting one-dimensional matrix parameter for function "
					          + this.getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}

	/**
	 * 
	 * @param expr1
	 * @param expr2
	 * @throws LanguageException
	 */
	private void checkMatchingDimensions(Expression expr1, Expression expr2) 
		throws LanguageException 
	{
		checkMatchingDimensions(expr1, expr2, false);
	}
	
	/**
	 * 
	 * @param expr1
	 * @param expr2
	 * @throws LanguageException
	 */
	private void checkMatchingDimensions(Expression expr1, Expression expr2, boolean allowsMV) 
		throws LanguageException 
	{
		if (expr1 != null && expr2 != null) {
			
			// if any matrix has unknown dimensions, simply return
			if(  expr1.getOutput().getDim1() == -1 || expr2.getOutput().getDim1() == -1 
			   ||expr1.getOutput().getDim2() == -1 || expr2.getOutput().getDim2() == -1 ) 
			{
				return;
			}
			else if( (!allowsMV && expr1.getOutput().getDim1() != expr2.getOutput().getDim1())
				  || (allowsMV && expr1.getOutput().getDim1() != expr2.getOutput().getDim1() && expr2.getOutput().getDim1() != 1)
				  || (!allowsMV && expr1.getOutput().getDim2() != expr2.getOutput().getDim2()) 
				  || (allowsMV && expr1.getOutput().getDim2() != expr2.getOutput().getDim2() && expr2.getOutput().getDim2() != 1) ) 
			{
				raiseValidateError("Mismatch in matrix dimensions of parameters for function "
						+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
	}
	
	/**
	 * 
	 * @throws LanguageException
	 */
	private void checkMatchingDimensionsQuantile() 
		throws LanguageException 
	{
		if (getFirstExpr().getOutput().getDim1() != getSecondExpr().getOutput().getDim1()) {
			raiseValidateError("Mismatch in matrix dimensions for "
					+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
		}
	}

	public static BuiltinFunctionExpression getBuiltinFunctionExpression(
			String functionName, ArrayList<ParameterExpression> paramExprsPassed,
			String filename, int blp, int bcp, int elp, int ecp) {
		
		if (functionName == null || paramExprsPassed == null)
			return null;
		
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
		//NOTE: pmin and pmax are just kept for compatibility to R
		// min and max is capable of handling all unary and binary
		// operations (in contrast to R)
		else if (functionName.equals("pmin"))
			bifop = Expression.BuiltinFunctionOp.MIN;
		else if (functionName.equals("pmax"))
			 bifop = Expression.BuiltinFunctionOp.MAX;
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
		else if (functionName.equals("rowIndexMin"))
			bifop = Expression.BuiltinFunctionOp.ROWINDEXMIN;
		else if (functionName.equals("colMaxs"))
			bifop = Expression.BuiltinFunctionOp.COLMAX;
		else if (functionName.equals("rowMeans"))
			bifop = Expression.BuiltinFunctionOp.ROWMEAN;
		else if (functionName.equals("colMeans"))
			 bifop = Expression.BuiltinFunctionOp.COLMEAN;
		else if (functionName.equals("cummax"))
			 bifop = Expression.BuiltinFunctionOp.CUMMAX;
		else if (functionName.equals("cummin"))
			 bifop = Expression.BuiltinFunctionOp.CUMMIN;
		else if (functionName.equals("cumprod"))
			 bifop = Expression.BuiltinFunctionOp.CUMPROD;
		else if (functionName.equals("cumsum"))
			 bifop = Expression.BuiltinFunctionOp.CUMSUM;
		//'castAsScalar' for backwards compatibility
		else if (functionName.equals("as.scalar") || functionName.equals("castAsScalar")) 
			bifop = Expression.BuiltinFunctionOp.CAST_AS_SCALAR;
		else if (functionName.equals("as.matrix"))
			bifop = Expression.BuiltinFunctionOp.CAST_AS_MATRIX;
		else if (functionName.equals("as.double"))
			bifop = Expression.BuiltinFunctionOp.CAST_AS_DOUBLE;
		else if (functionName.equals("as.integer"))
			bifop = Expression.BuiltinFunctionOp.CAST_AS_INT;
		else if (functionName.equals("as.logical")) //alternative: as.boolean
			bifop = Expression.BuiltinFunctionOp.CAST_AS_BOOLEAN;
		else if (functionName.equals("quantile"))
			bifop= Expression.BuiltinFunctionOp.QUANTILE;
		else if (functionName.equals("interQuantile"))
			bifop= Expression.BuiltinFunctionOp.INTERQUANTILE;
		else if (functionName.equals("interQuartileMean"))
			bifop= Expression.BuiltinFunctionOp.IQM;
		//'ctable' for backwards compatibility 
		else if (functionName.equals("table") || functionName.equals("ctable"))
			bifop = Expression.BuiltinFunctionOp.TABLE;
		else if (functionName.equals("round"))
			bifop = Expression.BuiltinFunctionOp.ROUND;
		//'centralMoment' for backwards compatibility 
		else if (functionName.equals("moment") || functionName.equals("centralMoment"))
			 bifop = Expression.BuiltinFunctionOp.MOMENT;
		else if (functionName.equals("cov"))
			bifop = Expression.BuiltinFunctionOp.COV;
		else if (functionName.equals("seq"))
			bifop = Expression.BuiltinFunctionOp.SEQ;
		else if (functionName.equals("qr"))
			bifop = Expression.BuiltinFunctionOp.QR;
		else if (functionName.equals("lu"))
			bifop = Expression.BuiltinFunctionOp.LU;
		else if (functionName.equals("eigen"))
			bifop = Expression.BuiltinFunctionOp.EIGEN;
		else if (functionName.equals("solve"))
			bifop = Expression.BuiltinFunctionOp.SOLVE;
		else if (functionName.equals("ceil"))
			bifop = Expression.BuiltinFunctionOp.CEIL;
		else if (functionName.equals("floor"))
			bifop = Expression.BuiltinFunctionOp.FLOOR;
		else if (functionName.equals("median"))
			bifop = Expression.BuiltinFunctionOp.MEDIAN;
		else if (functionName.equals("inv"))
			bifop = Expression.BuiltinFunctionOp.INVERSE;
		else if (functionName.equals("sample"))
			bifop = Expression.BuiltinFunctionOp.SAMPLE;
		else if ( functionName.equals("outer") )
			bifop = Expression.BuiltinFunctionOp.OUTER;
		else
			return null;
		
		BuiltinFunctionExpression retVal = new BuiltinFunctionExpression(bifop, paramExprsPassed,
				filename, blp, bcp, elp, ecp);
	
		return retVal;
	} // end method getBuiltinFunctionExpression

	/**
	 * 
	 * @param vt
	 * @return
	 * @throws LanguageException
	 */
	public static BuiltinFunctionOp getValueTypeCastOperator( ValueType vt ) 
		throws LanguageException
	{
		switch( vt )
		{
			case DOUBLE:
				return BuiltinFunctionOp.CAST_AS_DOUBLE;
			case INT:
				return BuiltinFunctionOp.CAST_AS_INT;
			case BOOLEAN:
				return BuiltinFunctionOp.CAST_AS_BOOLEAN;
			default:
				throw new LanguageException("No cast for value type "+vt);
		}
	}
}