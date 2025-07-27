/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import org.antlr.v4.runtime.ParserRuleContext;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.LanguageException.LanguageErrorCodes;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DnnUtils;
import org.apache.sysds.runtime.util.UtilFunctions;

public class BuiltinFunctionExpression extends DataIdentifier {
	protected static final Log LOG = LogFactory.getLog(BuiltinFunctionExpression.class.getName());
	protected Expression[] _args = null;
	private Builtins _opcode;

	public BuiltinFunctionExpression(ParserRuleContext ctx, Builtins bifop, ArrayList<ParameterExpression> args, String fname) {
		_opcode = bifop;
		setCtxValuesAndFilename(ctx, fname);
		args = expandDnnArguments(args);
		_args = new Expression[args.size()];
		for(int i=0; i < args.size(); i++) {
			_args[i] = args.get(i).getExpr();
		}
	}

	public BuiltinFunctionExpression(Builtins bifop, Expression[] args, ParseInfo parseInfo) {
		_opcode = bifop;
		_args = new Expression[args.length];
		for (int i = 0; i < args.length; i++) {
			_args[i] = args[i];
		}
		setParseInfo(parseInfo);
	}

	public BuiltinFunctionExpression(ParserRuleContext ctx, Builtins bifop, Expression[] args, String fname) {
		_opcode = bifop;
		_args = new Expression[args.length];
		for(int i=0; i < args.length; i++) {
			_args[i] = args[i];
		}
		setCtxValuesAndFilename(ctx, fname);
	}

	@Override
	public Expression rewriteExpression(String prefix) {
		Expression[] newArgs = new Expression[_args.length];
		for (int i = 0; i < _args.length; i++) {
			newArgs[i] = _args[i].rewriteExpression(prefix);
		}
		BuiltinFunctionExpression retVal = new BuiltinFunctionExpression(this._opcode, newArgs, this);
		return retVal;
	}

	public Builtins getOpCode() {
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
	
	public Expression getFourthExpr() {
		return (_args.length >= 4 ? _args[3] : null);
	}
	
	public Expression getFifthExpr() {
		return (_args.length >= 5 ? _args[4] : null);
	}
	
	public Expression getSixthExpr() {
		return (_args.length >= 6 ? _args[5] : null);
	}
	
	public Expression getSeventhExpr() {
		return (_args.length >= 7 ? _args[6] : null);
	}

	public Expression getEighthExpr() {
		return (_args.length >= 8 ? _args[7] : null);
	}


	public Expression[] getAllExpr(){
		return _args;
	}
	
	public Expression getExpr(int i) {
		return (_args.length > i ? _args[i] : null);
	}
	
	@Override
	public void validateExpression(MultiAssignmentStatement stmt, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional)
	{
		if (this.getFirstExpr() instanceof FunctionCallIdentifier){
			raiseValidateError("UDF function call not supported as parameter to built-in function call", false);
		}
		
		this.getFirstExpr().validateExpression(ids, constVars, conditional);
		Expression [] expr = getAllExpr();
		if(expr != null && expr.length > 1) {
			for(int i = 1; i < expr.length; i++) {
				if (expr[i] instanceof FunctionCallIdentifier){
					raiseValidateError("UDF function call not supported as parameter to built-in function call", false);
				}
				expr[i].validateExpression(ids, constVars, conditional);
			}
		}
		_outputs = new Identifier[stmt.getTargetList().size()];
		int count = 0;
		for (DataIdentifier outParam: stmt.getTargetList()){
			DataIdentifier tmp = new DataIdentifier(outParam);
			tmp.setParseInfo(this);
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
			qrOut1.setValueType(ValueType.FP64);
			qrOut1.setDimensions(rows, cols);
			qrOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			// Output2 - R
			qrOut2.setDataType(DataType.MATRIX);
			qrOut2.setValueType(ValueType.FP64);
			qrOut2.setDimensions(rows, cols);
			qrOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
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

			if (inrows != incols) {
				raiseValidateError("LU Decomposition requires a square matrix. Matrix " + getFirstExpr() + " is "
						+ inrows + "x" + incols + ".", conditional);
			}

			// Output1 - P
			luOut1.setDataType(DataType.MATRIX);
			luOut1.setValueType(ValueType.FP64);
			luOut1.setDimensions(inrows, inrows);
			luOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			// Output2 - L
			luOut2.setDataType(DataType.MATRIX);
			luOut2.setValueType(ValueType.FP64);
			luOut2.setDimensions(inrows, inrows);
			luOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			// Output3 - U
			luOut3.setDataType(DataType.MATRIX);
			luOut3.setValueType(ValueType.FP64);
			luOut3.setDimensions(inrows, inrows);
			luOut3.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			break;

		case LSTM:
		{
			//TODO: LSTM on GPU has different INPUT/OUTPUT than LSTM on CPU

			// X,  W, bias, out0, c0, return_sequences
			checkNumParameters(6);
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			checkMatrixParam(getThirdExpr());
			checkMatrixParam(getFourthExpr());
			checkMatrixParam(getFifthExpr());
			
			// setup output properties, on CPU there are 3 more additionally outputs (cache_out, cache_c, cache_ifog)
			if(getOutputs() == null || (getOutputs().length != 2 && getOutputs().length != 5)) {
				int numOutputs = getOutputs() == null ? 0 : getOutputs().length;
				raiseValidateError("The builtin function lstm has two outputs, but instead found: " + numOutputs, conditional);
			}
			DataIdentifier out = (DataIdentifier) getOutputs()[0];
			DataIdentifier cy = (DataIdentifier) getOutputs()[1];
			
			// Output1 - out: If `return_sequences` is True, outputs for all timesteps, else outputs for the final timestep.
			out.setDataType(DataType.MATRIX);
			out.setValueType(ValueType.FP64);
			out.setDimensions(-1, -1);
			out.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			// Output2 - Cell state for final timestep.
			cy.setDataType(DataType.MATRIX);
			cy.setValueType(ValueType.FP64);
			cy.setDimensions(getExpr(4).getOutput().getDim1(), getExpr(4).getOutput().getDim2());
			cy.setBlocksize(getExpr(4).getOutput().getBlocksize());

			if(getOutputs().length == 5){
				DataIdentifier cache_out = (DataIdentifier) getOutputs()[2];
				DataIdentifier cache_c = (DataIdentifier) getOutputs()[3];
				DataIdentifier cache_ifog = (DataIdentifier) getOutputs()[4];

				// Output3 - cache_out: (T,N*M) T is unknown upfront
				cache_out.setDataType(DataType.MATRIX);
				cache_out.setValueType(ValueType.FP64);
				cache_out.setDimensions(-1, -1);
				cache_out.setBlocksize(getFirstExpr().getOutput().getBlocksize());

				// Output4 - cache_c: (T,N*M)
				cache_c.setDataType(DataType.MATRIX);
				cache_c.setValueType(ValueType.FP64);
				cache_out.setDimensions(-1, -1);
				cache_out.setBlocksize(getFirstExpr().getOutput().getBlocksize());

				// Output5 - cache_ifog: (T,N*M)
				cache_ifog.setDataType(DataType.MATRIX);
				cache_ifog.setValueType(ValueType.FP64);
				cache_ifog.setDimensions(-1, -1);
				cache_ifog.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			}
			break;
		}
		case LSTM_BACKWARD:
		{
			// Input: X, W, b, out0, c0, return_sequences, dout, cy
			checkNumParameters(8);
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			checkMatrixParam(getThirdExpr());
			checkMatrixParam(getFourthExpr());
			checkMatrixParam(getFifthExpr());
			checkMatrixParam(getSeventhExpr());
			checkMatrixParam(getEighthExpr());
			
			// Output: dx, dw, db, dout0, dc0
			// setup output properties
			if(getOutputs().length != 5)
				raiseValidateError("lstm_backward has 5 outputs", false);
			 
			DataIdentifier dx = (DataIdentifier) getOutputs()[0];
			DataIdentifier dw = (DataIdentifier) getOutputs()[1];
			DataIdentifier db = (DataIdentifier) getOutputs()[2];
			DataIdentifier dout0 = (DataIdentifier) getOutputs()[3];
			DataIdentifier dc0 = (DataIdentifier) getOutputs()[4];
			
			setDimensions(dx, getFirstExpr());
			setDimensions(dw, getSecondExpr());
			setDimensions(db, getThirdExpr());
			setDimensions(dout0, getFourthExpr());
			setDimensions(dc0, getFifthExpr());
			break;
		}
		case BATCH_NORM2D:
		{
			// Input: image, scale, bias, runningMean, runningVar, mode, epsilon, exponentialAverageFactor
			checkNumParameters(8);
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			checkMatrixParam(getThirdExpr());
			checkMatrixParam(getFourthExpr());
			checkMatrixParam(getFifthExpr());
			
			// Output: ret, retRunningMean, retRunningVar, resultSaveMean, resultSaveInvVariance
			// setup output properties
			if(getOutputs().length != 5)
				raiseValidateError("batch_norm2d has 5 outputs", false);
			 
			DataIdentifier ret = (DataIdentifier) getOutputs()[0];
			DataIdentifier retRunningMean = (DataIdentifier) getOutputs()[1];
			DataIdentifier retRunningVar = (DataIdentifier) getOutputs()[2];
			DataIdentifier resultSaveMean = (DataIdentifier) getOutputs()[3];
			DataIdentifier resultSaveInvVariance = (DataIdentifier) getOutputs()[4];
			
			setDimensions(ret, getFirstExpr());
			setDimensions(retRunningMean, getFourthExpr());
			setDimensions(retRunningVar, getFourthExpr());
			setDimensions(resultSaveMean, getFourthExpr());
			setDimensions(resultSaveInvVariance, getFourthExpr());
			break;
		}
		case BATCH_NORM2D_BACKWARD:
		{
			// Input: image, dout, scale, epsilon, savedMean, savedInvVariance
			checkNumParameters(6);
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			checkMatrixParam(getThirdExpr());
			checkMatrixParam(getFifthExpr());
			checkMatrixParam(getSixthExpr());
			
			// Output: dX, dScale, dBias 
			// setup output properties
			if(getOutputs().length != 3)
				raiseValidateError("batch_norm2d_backward has 3 outputs", false);
			
			DataIdentifier dX = (DataIdentifier) getOutputs()[0];
			DataIdentifier dScale = (DataIdentifier) getOutputs()[1];
			DataIdentifier dBias = (DataIdentifier) getOutputs()[2];
			
			setDimensions(dX, getFirstExpr());
			setDimensions(dScale, getThirdExpr());
			setDimensions(dBias, getThirdExpr());
			break;
		}
		case EIGEN: {
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
			eigenOut1.setValueType(ValueType.FP64);
			eigenOut1.setDimensions(getFirstExpr().getOutput().getDim1(), 1);
			eigenOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			// Output2 - Eigen Vectors
			eigenOut2.setDataType(DataType.MATRIX);
			eigenOut2.setValueType(ValueType.FP64);
			eigenOut2.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			eigenOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			break;
		}
		case RCM: {
			checkNumParameters(2);
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			long nr = Math.max(getFirstExpr().getOutput().getDim1(),
				getSecondExpr().getOutput().getDim1());
			long nc = Math.max(getFirstExpr().getOutput().getDim2(),
				getSecondExpr().getOutput().getDim2());
			for(int i=0; i<2; i++) {
				DataIdentifier out = (DataIdentifier) getOutputs()[i];
				out.setDataType(DataType.MATRIX);
				out.setValueType(ValueType.FP64);
				out.setDimensions(nr, nc);
				out.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			}
			break;
		}
		case FFT: {

			Expression expressionOne = getFirstExpr();
			Expression expressionTwo = getSecondExpr();

			if(expressionOne == null) {
				raiseValidateError("The first argument to " + _opcode + " cannot be null.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(expressionOne.getOutput() == null || expressionOne.getOutput().getDim1() == 0 ||
				expressionOne.getOutput().getDim2() == 0) {
				raiseValidateError("The first argument to " + _opcode + " cannot be an empty matrix.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(expressionTwo != null) {
				raiseValidateError("Too many arguments. This FFT implementation is only defined for real inputs.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(!isPowerOfTwo(expressionOne.getOutput().getDim1()) ||
				!isPowerOfTwo(expressionOne.getOutput().getDim2())) {
				raiseValidateError(
					"This FFT implementation is only defined for matrices with dimensions that are powers of 2.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}

			checkNumParameters(1);
			checkMatrixParam(expressionOne);

			DataIdentifier fftOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier fftOut2 = (DataIdentifier) getOutputs()[1];

			fftOut1.setDataType(DataType.MATRIX);
			fftOut1.setValueType(ValueType.FP64);
			fftOut1.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			fftOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			fftOut2.setDataType(DataType.MATRIX);
			fftOut2.setValueType(ValueType.FP64);
			fftOut2.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			fftOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			break;

		}
		case IFFT: {
			Expression expressionTwo = getSecondExpr();
			Expression expressionOne = getFirstExpr();

			if(expressionOne == null) {
				raiseValidateError("The first argument to " + _opcode + " cannot be null.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(expressionOne.getOutput() == null || expressionOne.getOutput().getDim1() == 0 ||
				expressionOne.getOutput().getDim2() == 0) {
				raiseValidateError("The first argument to " + _opcode + " cannot be an empty matrix.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(expressionTwo != null) {
				if(expressionTwo.getOutput() == null || expressionTwo.getOutput().getDim1() == 0 ||
					expressionTwo.getOutput().getDim2() == 0) {
					raiseValidateError("The second argument to " + _opcode
						+ " cannot be an empty matrix. Provide either only a real matrix or a filled real and imaginary one.",
						false, LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}

			checkNumParameters(expressionTwo != null ? 2 : 1);
			checkMatrixParam(expressionOne);
			if(expressionTwo != null && expressionOne != null) {
				checkMatrixParam(expressionTwo);
				if(expressionOne.getOutput().getDim1() != expressionTwo.getOutput().getDim1() ||
					expressionOne.getOutput().getDim2() != expressionTwo.getOutput().getDim2())
					raiseValidateError("The real and imaginary part of the provided matrix are of different dimensions.",
						false);
				else if(!isPowerOfTwo(expressionTwo.getOutput().getDim1()) ||
					!isPowerOfTwo(expressionTwo.getOutput().getDim2())) {
					raiseValidateError(
						"This IFFT implementation is only defined for matrices with dimensions that are powers of 2.", false,
						LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}
			else if(expressionOne != null) {
				if(!isPowerOfTwo(expressionOne.getOutput().getDim1()) ||
					!isPowerOfTwo(expressionOne.getOutput().getDim2())) {
					raiseValidateError(
						"This IFFT implementation is only defined for matrices with dimensions that are powers of 2.", false,
						LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}

			DataIdentifier ifftOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier ifftOut2 = (DataIdentifier) getOutputs()[1];

			ifftOut1.setDataType(DataType.MATRIX);
			ifftOut1.setValueType(ValueType.FP64);
			ifftOut1.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			ifftOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			// Output2 - ifft Vectors
			ifftOut2.setDataType(DataType.MATRIX);
			ifftOut2.setValueType(ValueType.FP64);
			ifftOut2.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			ifftOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			break;
		}
		case FFT_LINEARIZED: {

			Expression expressionOne = getFirstExpr();
			Expression expressionTwo = getSecondExpr();

			if(expressionOne == null) {
				raiseValidateError("The first argument to " + _opcode + " cannot be null.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(expressionOne.getOutput() == null || expressionOne.getOutput().getDim1() == 0 ||
				expressionOne.getOutput().getDim2() == 0) {
				raiseValidateError("The first argument to " + _opcode + " cannot be an empty matrix.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(expressionTwo != null) {
				raiseValidateError(
					"Too many arguments. This FFT_LINEARIZED implementation is only defined for real inputs.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(!isPowerOfTwo(expressionOne.getOutput().getDim2())) {
				raiseValidateError(
					"This FFT_LINEARIZED implementation is only defined for matrices with columns that are powers of 2.",
					false, LanguageErrorCodes.INVALID_PARAMETERS);
			}

			checkNumParameters(1);
			checkMatrixParam(expressionOne);

			DataIdentifier fftOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier fftOut2 = (DataIdentifier) getOutputs()[1];

			fftOut1.setDataType(DataType.MATRIX);
			fftOut1.setValueType(ValueType.FP64);
			fftOut1.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			fftOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			fftOut2.setDataType(DataType.MATRIX);
			fftOut2.setValueType(ValueType.FP64);
			fftOut2.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			fftOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			break;

		}
		case IFFT_LINEARIZED: {
			Expression expressionTwo = getSecondExpr();
			Expression expressionOne = getFirstExpr();

			if(expressionOne == null) {
				raiseValidateError("The first argument to " + _opcode + " cannot be null.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(expressionOne.getOutput() == null || expressionOne.getOutput().getDim1() == 0 ||
				expressionOne.getOutput().getDim2() == 0) {
				raiseValidateError("The first argument to " + _opcode + " cannot be an empty matrix.", false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(expressionTwo != null) {
				if(expressionTwo.getOutput() == null || expressionTwo.getOutput().getDim1() == 0 ||
					expressionTwo.getOutput().getDim2() == 0) {
					raiseValidateError("The second argument to " + _opcode
						+ " cannot be an empty matrix. Provide either only a real matrix or a filled real and imaginary one.",
						false, LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}

			checkNumParameters(expressionTwo != null ? 2 : 1);
			checkMatrixParam(expressionOne);
			if(expressionTwo != null && expressionOne != null) {
				checkMatrixParam(expressionTwo);
				if(expressionOne.getOutput().getDim1() != expressionTwo.getOutput().getDim1() ||
					expressionOne.getOutput().getDim2() != expressionTwo.getOutput().getDim2())
					raiseValidateError("The real and imaginary part of the provided matrix are of different dimensions.",
						false);
				else if(!isPowerOfTwo(expressionTwo.getOutput().getDim2())) {
					raiseValidateError(
						"This IFFT_LINEARIZED implementation is only defined for matrices with columns that are powers of 2.",
						false, LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}
			else if(expressionOne != null) {
				if(!isPowerOfTwo(expressionOne.getOutput().getDim2())) {
					raiseValidateError(
						"This IFFT_LINEARIZED implementation is only defined for matrices with columns that are powers of 2.",
						false, LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}

			DataIdentifier ifftOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier ifftOut2 = (DataIdentifier) getOutputs()[1];

			ifftOut1.setDataType(DataType.MATRIX);
			ifftOut1.setValueType(ValueType.FP64);
			ifftOut1.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			ifftOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			ifftOut2.setDataType(DataType.MATRIX);
			ifftOut2.setValueType(ValueType.FP64);
			ifftOut2.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			ifftOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			break;
		}
		case STFT: {
			checkMatrixParam(getFirstExpr());

			if((getFirstExpr() == null || getSecondExpr() == null || getThirdExpr() == null) && _args.length > 0) {
				raiseValidateError("Missing argument for function " + this.getOpCode(), false,
					LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if(getFifthExpr() != null) {
				raiseValidateError("Invalid number of arguments for function " + this.getOpCode().toString().toLowerCase()
					+ "(). This function only takes 3 or 4 arguments.", false);
			}
			else if(_args.length == 3) {
				checkScalarParam(getSecondExpr());
				checkScalarParam(getThirdExpr());
				if(!isPowerOfTwo(((ConstIdentifier) getSecondExpr().getOutput()).getLongValue())) {
					raiseValidateError(
						"This FFT implementation is only defined for matrices with dimensions that are powers of 2."
							+ "The window size (2nd argument) is not a power of two",
						false, LanguageErrorCodes.INVALID_PARAMETERS);
				}
				else if(((ConstIdentifier) getSecondExpr().getOutput())
					.getLongValue() <= ((ConstIdentifier) getThirdExpr().getOutput()).getLongValue()) {
					raiseValidateError("Overlap can't be larger than or equal to the window size.", false,
						LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}
			else if(_args.length == 4) {
				checkMatrixParam(getSecondExpr());
				checkScalarParam(getThirdExpr());
				checkScalarParam(getFourthExpr());
				if(!isPowerOfTwo(((ConstIdentifier) getThirdExpr().getOutput()).getLongValue())) {
					raiseValidateError(
						"This FFT implementation is only defined for matrices with dimensions that are powers of 2."
							+ "The window size (3rd argument) is not a power of two",
						false, LanguageErrorCodes.INVALID_PARAMETERS);
				}
				else if(getFirstExpr().getOutput().getDim1() != getSecondExpr().getOutput().getDim1() ||
					getFirstExpr().getOutput().getDim2() != getSecondExpr().getOutput().getDim2()) {
					raiseValidateError("The real and imaginary part of the provided matrix are of different dimensions.",
						false);
				}
				else if(((ConstIdentifier) getThirdExpr().getOutput())
					.getLongValue() <= ((ConstIdentifier) getFourthExpr().getOutput()).getLongValue()) {
					raiseValidateError("Overlap can't be larger than or equal to the window size.", false,
						LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}

			// setup output properties
			DataIdentifier stftOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier stftOut2 = (DataIdentifier) getOutputs()[1];

			// Output1 - stft Values
			stftOut1.setDataType(DataType.MATRIX);
			stftOut1.setValueType(ValueType.FP64);
			stftOut1.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			stftOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			// Output2 - stft Vectors
			stftOut2.setDataType(DataType.MATRIX);
			stftOut2.setValueType(ValueType.FP64);
			stftOut2.setDimensions(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());
			stftOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			break;
		}
		case REMOVE: {
			checkNumParameters(2);
			checkListParam(getFirstExpr());
			
			// setup output properties
			DataIdentifier out1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier out2 = (DataIdentifier) getOutputs()[1];
			
			// Output1 - list after removal
			long nrow = getFirstExpr().getOutput().getDim1() > 0 ? 
				getFirstExpr().getOutput().getDim1() + 1 : -1;
			out1.setDataType(DataType.LIST);
			out1.setValueType(getFirstExpr().getOutput().getValueType());
			out1.setDimensions(nrow, 1);
			out1.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			// Output2 - list of removed element
			out2.setDataType(DataType.LIST);
			out2.setValueType(getFirstExpr().getOutput().getValueType());
			out2.setDimensions(1, 1);
			out2.setBlocksize(getFirstExpr().getOutput().getBlocksize());
			
			break;
		}
		case SVD:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			
			long minMN = Math.min(getFirstExpr().getOutput().getDim1(), getFirstExpr().getOutput().getDim2());

			// setup output properties
			DataIdentifier svdOut1 = (DataIdentifier) getOutputs()[0];
			DataIdentifier svdOut2 = (DataIdentifier) getOutputs()[1];
			DataIdentifier svdOut3 = (DataIdentifier) getOutputs()[2];

			// Output 1
			svdOut1.setDataType(DataType.MATRIX);
			svdOut1.setValueType(ValueType.FP64);
			svdOut1.setDimensions(getFirstExpr().getOutput().getDim1(), minMN);
			svdOut1.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			// Output 2
			svdOut2.setDataType(DataType.MATRIX);
			svdOut2.setValueType(ValueType.FP64);
			svdOut2.setDimensions(minMN, minMN);
			svdOut2.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			// Output 3
			svdOut3.setDataType(DataType.MATRIX);
			svdOut3.setValueType(ValueType.FP64);
			svdOut3.setDimensions(getFirstExpr().getOutput().getDim2(), minMN);
			svdOut3.setBlocksize(getFirstExpr().getOutput().getBlocksize());

			break;

		case COMPRESS:
			if(OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND) {
				Expression expressionTwo = getSecondExpr();
				checkNumParameters(getSecondExpr() != null ? 2 : 1);
				checkMatrixFrameParam(getFirstExpr());
				if(expressionTwo != null)
					checkMatrixParam(getSecondExpr());

				Identifier compressInput1 = getFirstExpr().getOutput();
				// Identifier compressInput2 = getSecondExpr().getOutput();

				DataIdentifier compressOutput = (DataIdentifier) getOutputs()[0];
				compressOutput.setDataType(DataType.MATRIX);
				compressOutput.setDimensions(compressInput1.getDim1(), compressInput1.getDim2());
				compressOutput.setBlocksize(compressInput1.getBlocksize());
				compressOutput.setValueType(compressInput1.getValueType());

				DataIdentifier metaOutput = (DataIdentifier) getOutputs()[1];
				metaOutput.setDataType(DataType.FRAME);
				metaOutput.setDimensions(compressInput1.getDim1(), -1);
			}
			else
				raiseValidateError("Compress/DeCompress instruction not allowed in dml script");
			break;
							
		default: //always unconditional
			raiseValidateError("Unknown Builtin Function opcode: " + _opcode, false);
		}
	}

	private static boolean isPowerOfTwo(long n) {
		return (n > 0) && ((n & (n - 1)) == 0);
	}
	
	private static void setDimensions(DataIdentifier out, Expression exp) {
		out.setDataType(DataType.MATRIX);
		out.setValueType(ValueType.FP64);
		out.setDimensions(exp.getOutput().getDim1(), exp.getOutput().getDim2());
		out.setBlocksize(exp.getOutput().getBlocksize());
	}
	
	private static ArrayList<ParameterExpression> orderDnnParams(ArrayList<ParameterExpression> paramExpression, int skip) {
		ArrayList<ParameterExpression> newParams = new ArrayList<>();

		for(int i = 0; i < skip; i++)
			newParams.add(paramExpression.get(i));

		String [] orderedParams = {
				"stride1", "stride2", "padding1", "padding2",  
				"input_shape1", "input_shape2", "input_shape3", "input_shape4", 
				"filter_shape1", "filter_shape2", "filter_shape3", "filter_shape4"	
		};
		for(int i = 0; i < orderedParams.length; i++) {
			boolean found = false;
			for(ParameterExpression param : paramExpression) {
				if(param.getName() != null &&  param.getName().equals(orderedParams[i])) {
					found = true;
					newParams.add(param);
				}
			}
			if(!found) {
				throw new LanguageException("Incorrect parameters. Expected " + orderedParams[i] + " to be expanded.");
			}
		}

		return newParams;
	}

	private static ArrayList<ParameterExpression> replaceListParams(ArrayList<ParameterExpression> paramExpression,
			String inputVarName, String outputVarName, int startIndex) {
		ArrayList<ParameterExpression> newParamExpression = new ArrayList<>();
		int i = startIndex;
		int j = 1; // Assumption: sequential ordering pool_size1, pool_size2 
		for (ParameterExpression expr : paramExpression) {
			if(expr.getName() != null && expr.getName().equals(inputVarName + j)) {
				newParamExpression.add(new ParameterExpression(outputVarName + i, expr.getExpr()));
				i++; j++;
			}
			else {
				newParamExpression.add(expr);
			}
		}
		return newParamExpression;
	}

	private static ArrayList<ParameterExpression> expandListParams(ArrayList<ParameterExpression> paramExpression, 
			HashSet<String> paramsToExpand) {
		ArrayList<ParameterExpression> newParamExpressions = new ArrayList<>();
		for(ParameterExpression expr : paramExpression) {
			if(paramsToExpand.contains(expr.getName())) {
				if(expr.getExpr() instanceof ExpressionList) {
					int i = 1;
					for(Expression e : ((ExpressionList)expr.getExpr()).getValue()) {
						newParamExpressions.add(new ParameterExpression(expr.getName() + i, e));
						i++;
					}
				}
			}
			else if(expr.getExpr() instanceof ExpressionList) {
				throw new LanguageException("The parameter " + expr.getName() + " cannot be list or is not supported for the given function");
			}
			else {
				newParamExpressions.add(expr);
			}
		}
		return newParamExpressions;
	}
	
	private ArrayList<ParameterExpression> expandDnnArguments(ArrayList<ParameterExpression> paramExpression) {
		try {
			if(_opcode == Builtins.CONV2D || _opcode == Builtins.CONV2D_BACKWARD_FILTER 
					|| _opcode == Builtins.CONV2D_BACKWARD_DATA) {
				HashSet<String> expand = new HashSet<>();
				expand.add("input_shape"); expand.add("filter_shape"); expand.add("stride"); expand.add("padding");
				paramExpression = expandListParams(paramExpression, expand);
				paramExpression = orderDnnParams(paramExpression, 2);
			}
			else if(_opcode == Builtins.MAX_POOL || _opcode == Builtins.AVG_POOL ||  
					_opcode == Builtins.MAX_POOL_BACKWARD || _opcode == Builtins.AVG_POOL_BACKWARD) {
				HashSet<String> expand = new HashSet<>();
				expand.add("input_shape"); expand.add("pool_size"); expand.add("stride"); expand.add("padding");
				paramExpression = expandListParams(paramExpression, expand);
				paramExpression.add(new ParameterExpression("filter_shape1", new IntIdentifier(1, this)));
				paramExpression.add(new ParameterExpression("filter_shape2", new IntIdentifier(1, this)));
				paramExpression = replaceListParams(paramExpression, "pool_size", "filter_shape", 3);
				if(_opcode == Builtins.MAX_POOL_BACKWARD || _opcode == Builtins.AVG_POOL_BACKWARD)
					paramExpression = orderDnnParams(paramExpression, 2);
				else
					paramExpression = orderDnnParams(paramExpression, 1);
			}
		}
		catch(LanguageException e) {
			throw new RuntimeException(e);
		}
		return paramExpression;
	}
	
	private boolean isValidNoArgumentFunction() {
		return getOpCode() == Builtins.TIME
			|| getOpCode() == Builtins.LIST;
	}

	/**
	 * Validate parse tree : Process BuiltinFunction Expression in an assignment
	 * statement
	 */
	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional) 
	{
		for(int i=0; i < _args.length; i++ ) {
			if (_args[i] instanceof FunctionCallIdentifier){
				raiseValidateError("UDF function call not supported as parameter to built-in function call", false);
			}
			_args[i].validateExpression(ids, constVars, conditional);
		}
		
		// checkIdentifierParams();
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setParseInfo(this);
		
		if (getFirstExpr() == null && !isValidNoArgumentFunction()) { // time has no arguments 
			raiseValidateError("Function " + this + " has no arguments.", false);
		}
		Identifier id = (_args.length != 0) ?
			getFirstExpr().getOutput() : null;
		if (_args.length != 0)
			output.setProperties(this.getFirstExpr().getOutput());
		output.setNnz(-1); //conservatively, cannot use input nnz!
		setOutput(output);
		
		switch (getOpCode()) {
		case EVAL:
		case EVALLIST:
			if (_args.length == 0)
				raiseValidateError("Function eval should provide at least one argument, i.e., the function name.", false);
			checkValueTypeParam(_args[0], ValueType.STRING);
			boolean listReturn = (getOpCode()==Builtins.EVALLIST);
			output.setDataType(listReturn ? DataType.LIST : DataType.MATRIX);
			output.setValueType(listReturn ? ValueType.UNKNOWN : ValueType.FP64);
			output.setDimensions(-1, -1);
			output.setBlocksize(ConfigurationManager.getBlocksize());
			break;
		case COLSUM:
		case COLMAX:
		case COLMIN:
		case COLMEAN:
		case COLPROD:
		case COLSD:
		case COLVAR:
			// colSums(X);
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(1, id.getDim2());
			output.setBlocksize (id.getBlocksize());
			output.setValueType(id.getValueType());
			break;
		case ROWSUM:
		case ROWMAX:
		case ROWINDEXMAX:
		case ROWMIN:
		case ROWINDEXMIN:
		case ROWMEAN:
		case ROWPROD:
		case ROWSD:
		case ROWVAR:
			//rowSums(X);
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), 1);
			output.setBlocksize (id.getBlocksize());
			output.setValueType(id.getValueType());
			break;
		case TRACE:
			if(getFirstExpr().getOutput().dimsKnown() 
				&& getFirstExpr().getOutput().getDim1() != getFirstExpr().getOutput().getDim2()) 
			{
				raiseValidateError("Trace is only defined on squared matrices but found ["
					+getFirstExpr().getOutput().getDim1()+"x"+getFirstExpr().getOutput().getDim2()+"].", conditional);
			}
		case SUM:
		case PROD:
		case SD:
		case VAR:
			// sum(X);
			checkNumParameters(1);
			checkMatrixTensorParam(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			switch (id.getValueType()) {
				case INT64:
				case INT32:
				case UINT8:
				case UINT4:
				case BOOLEAN:
					output.setValueType(ValueType.INT64);
					break;
				case STRING:
				case CHARACTER:
				case FP64:
				case FP32:
				case HASH32:
				case HASH64: //default
					output.setValueType(ValueType.FP64);
					break;
				case UNKNOWN:
					throw new NotImplementedException();
			}
			break;
		
		case MEAN:
			//checkNumParameters(2, false); // mean(Y) or mean(Y,W)
			if (getSecondExpr() != null) {
				checkNumParameters(2);
			}
			else {
				checkNumParameters(1);
			}
			
			checkMatrixParam(getFirstExpr());
			if ( getSecondExpr() != null ) {
				// x = mean(Y,W);
				checkMatchingDimensions(getFirstExpr(), getSecondExpr());
			}
			
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType(id.getValueType());
			break;
		
		case XOR:
		case BITWAND:
		case BITWOR:
		case BITWXOR:
		case BITWSHIFTL:
		case BITWSHIFTR:
			checkNumParameters(2);
			setBinaryOutputProperties(output);
			break;
		
		case MIN:
		case MAX:
			//min(X), min(X,s), min(s,X), min(s,r), min(X,Y)
			if (getSecondExpr() == null) { //unary
				checkNumParameters(1);
				checkMatrixParam(getFirstExpr());
				output.setDataType(DataType.SCALAR);
				output.setValueType(id.getValueType());
				output.setDimensions(0, 0);
				output.setBlocksize(0);
			}
			else if( getAllExpr().length == 2 ) { //binary
				checkNumParameters(2);
				setBinaryOutputProperties(output);
			}
			else { //nary
				for( Expression e : getAllExpr() )
					checkMatrixScalarParam(e);
				setNaryOutputProperties(output);
			}
			break;
		
		case CUMSUM:
		case ROWCUMSUM:
		case CUMPROD:
		case CUMSUMPROD:
		case CUMMIN:
		case CUMMAX:
			// cumsum(X);
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			boolean cumSP = getOpCode() == Builtins.CUMSUMPROD;
			if( cumSP && id.getDim2() > 2 )
				raiseValidateError("Cumsumprod only supported over two-column matrices", conditional);
			
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), cumSP ? 1 : id.getDim2());
			output.setBlocksize (id.getBlocksize());
			output.setValueType(id.getValueType());
			
			break;
		case CAST_AS_SCALAR:
			checkNumParameters(1);
			checkDataTypeParam(getFirstExpr(),
				DataType.MATRIX, DataType.FRAME, DataType.LIST);
			if (( getFirstExpr().getOutput().getDim1() != -1 && getFirstExpr().getOutput().getDim1() !=1)
				|| ( getFirstExpr().getOutput().getDim2() != -1 && getFirstExpr().getOutput().getDim2() !=1)) {
				raiseValidateError("dimension mismatch while casting matrix to scalar: dim1: " + getFirstExpr().getOutput().getDim1() 
					+  " dim2 " + getFirstExpr().getOutput().getDim2(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType((id.getValueType()!=ValueType.UNKNOWN 
				|| id.getDataType()==DataType.LIST) ? id.getValueType() : ValueType.FP64);
			break;
		case CAST_AS_MATRIX:
			checkNumParameters(1);
			checkDataTypeParam(getFirstExpr(),
				DataType.SCALAR, DataType.FRAME, DataType.LIST);
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), id.getDim2());
			if( getFirstExpr().getOutput().getDataType()==DataType.SCALAR )
				output.setDimensions(1, 1); //correction scalars
			if( getFirstExpr().getOutput().getDataType()==DataType.LIST )
				output.setDimensions(-1, -1); //correction list: arbitrary object
			output.setBlocksize(id.getBlocksize());
			output.setValueType(ValueType.FP64); //matrices always in double
			break;
		case CAST_AS_LIST: //list unnesting
			checkNumParameters(1);
			checkDataTypeParam(getFirstExpr(), DataType.LIST);
			output.setDataType(DataType.LIST);
			output.setDimensions(-1, 1);
			output.setBlocksize(id.getBlocksize());
			output.setValueType(ValueType.UNKNOWN);
			break;
		case TYPEOF:
		case DETECTSCHEMA:
		case COLNAMES:
			checkNumParameters(1);
			checkMatrixFrameParam(getFirstExpr());
			output.setDataType(DataType.FRAME);
			output.setDimensions(1, id.getDim2());
			output.setBlocksize (id.getBlocksize());
			output.setValueType(ValueType.STRING);
			break;
		case CAST_AS_FRAME:
			// operation as.frame
			// overloaded to take either one argument or 2 where second is column names
			if( getSecondExpr() == null) {// there is no column names
				checkNumParameters(1);
			}
			else{ // there is column names
				checkNumParameters(2);
				checkDataTypeParam(getSecondExpr(), DataType.LIST);
			}

			checkDataTypeParam(getFirstExpr(), DataType.SCALAR, DataType.MATRIX, DataType.LIST);
			output.setDataType(DataType.FRAME);
			output.setDimensions(id.getDim1(), id.getDim2());
			if(getFirstExpr().getOutput().getDataType() == DataType.SCALAR)
				output.setDimensions(1, 1); // correction scalars
			if(getFirstExpr().getOutput().getDataType() == DataType.LIST)
				output.setDimensions(-1, -1); // correction list: arbitrary object
			output.setBlocksize(id.getBlocksize());
			output.setValueType(id.getValueType());
			break;
		case CAST_AS_DOUBLE:
			checkNumParameters(1);
			checkScalarParam(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			//output.setDataType(id.getDataType()); //TODO whenever we support multiple matrix value types, currently noop.
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType(ValueType.FP64);
			break;
		case CAST_AS_INT:
			checkNumParameters(1);
			checkScalarParam(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			//output.setDataType(id.getDataType()); //TODO whenever we support multiple matrix value types, currently noop.
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType(ValueType.INT64);
			break;
		case CAST_AS_BOOLEAN:
			checkNumParameters(1);
			checkScalarParam(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			//output.setDataType(id.getDataType()); //TODO whenever we support multiple matrix value types, currently noop.
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType(ValueType.BOOLEAN);
			break;
			
		case IFELSE:
			checkNumParameters(3);
			setTernaryOutputProperties(output, conditional);
			break;
			
		case CBIND:
		case RBIND:
			//scalar string append (string concatenation with \n)
			if( getFirstExpr().getOutput().getDataType()==DataType.SCALAR ) {
				checkNumParameters(2);
				checkScalarParam(getFirstExpr());
				checkScalarParam(getSecondExpr());
				checkValueTypeParam(getFirstExpr(), ValueType.STRING);
				checkValueTypeParam(getSecondExpr(), ValueType.STRING);
			}
			// append (rbind/cbind) all the elements of a list
			else if( getAllExpr().length == 1 ) {
				checkDataTypeParam(getFirstExpr(), DataType.LIST);
			}
			else {
				if( getAllExpr().length < 2 )
					raiseValidateError("Invalid number of arguments for "+getOpCode(), conditional);
				//list append
				if(getFirstExpr().getOutput().getDataType().isList() )
					for(int i=1; i<getAllExpr().length; i++)
						checkDataTypeParam(getExpr(i), DataType.SCALAR, DataType.MATRIX, DataType.FRAME, DataType.LIST);
				//matrix append (rbind/cbind)
				else
					for(int i=0; i<getAllExpr().length; i++)
						checkMatrixFrameParam(getExpr(i));
			}
			
			output.setDataType(id.getDataType());
			output.setValueType(id.getValueType());
			
			//special handling of concatenating all list elements
			if( id.getDataType() == DataType.LIST && getAllExpr().length == 1) {
				output.setDataType(DataType.MATRIX);
				output.setValueType(ValueType.FP64);
			}
			
			// set output dimensions and validate consistency
			long m1rlen = getFirstExpr().getOutput().getDim1();
			long m1clen = getFirstExpr().getOutput().getDim2();
			long appendDim1 = m1rlen, appendDim2 = m1clen;
			
			// best-effort dimension propagation and validation
			if( id.getDataType() == DataType.LIST ) {
				appendDim1 = -1;
				appendDim2 = -1;
			}
			else {
				for(int i=1; i<getAllExpr().length; i++) {
					long m2rlen = getExpr(i).getOutput().getDim1();
					long m2clen = getExpr(i).getOutput().getDim2();
					
					if( getOpCode() == Builtins.CBIND ) {
						if (m1rlen >= 0 && m2rlen >= 0 && m1rlen!=m2rlen) {
							raiseValidateError("inputs to cbind must have same number of rows: input 1 rows: " + 
								m1rlen+", input 2 rows: "+m2rlen, conditional, LanguageErrorCodes.INVALID_PARAMETERS);
						}
						appendDim1 = (m2rlen>=0) ? m2rlen : appendDim1;
						appendDim2 = (appendDim2>=0 && m2clen>=0) ? appendDim2 + m2clen : -1;
					}
					else if( getOpCode() == Builtins.RBIND ) {
						if (m1clen >= 0 && m2clen >= 0 && m1clen!=m2clen) {
							raiseValidateError("inputs to rbind must have same number of columns: input 1 columns: " + 
								m1clen+", input 2 columns: "+m2clen, conditional, LanguageErrorCodes.INVALID_PARAMETERS);
						}
						appendDim1 = (appendDim1>=0 && m2rlen>=0)? appendDim1 + m2rlen : -1;
						appendDim2 = (m2clen>=0) ? m2clen : appendDim2;
					}
				}
			}
			
			output.setDimensions(appendDim1, appendDim2);
			output.setBlocksize (id.getBlocksize());
			
			break;
			
		case PPRED:
			// TODO: remove this when ppred has been removed from DML
			raiseValidateError("ppred() has been deprecated. Please use the operator directly.", true);

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
			
			//check operator
			if (getThirdExpr().getOutput().getDataType() != DataType.SCALAR || 
				getThirdExpr().getOutput().getValueType() != ValueType.STRING) 
			{	
				raiseValidateError("Third argument in ppred() is not an operator ", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			setBinaryOutputProperties(output);
			break;

		case TRANS:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim2(), id.getDim1());
			output.setBlocksize (id.getBlocksize());
			output.setValueType(id.getValueType());
			break;
		
		case REV:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), id.getDim2());
			output.setBlocksize (id.getBlocksize());
			output.setValueType(id.getValueType());
			break;

		case ROLL:
			checkNumParameters(2);
			checkMatrixParam(getFirstExpr());
			checkScalarParam(getSecondExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), id.getDim2());
			output.setBlocksize(id.getBlocksize());
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
						raiseValidateError("diag can either: (1) create diagonal matrix from (n x 1) matrix, or (2) take diagonal from a square matrix. "
								+ "Error invoking diag on matrix with dimensions ("
								+ id.getDim1() + "," + id.getDim2()
								+ ") in " + this.toString(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
					//diag M2V
					output.setDimensions(id.getDim1(), 1);
				}
			}
			output.setBlocksize(id.getBlocksize());
			output.setValueType(id.getValueType());
			break;
		case DET:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			if ( id.getDim2() == -1 || id.getDim1() != id.getDim2() ) {
				raiseValidateError("det requires a square matrix as first argument.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType(ValueType.FP64);
			break;
		case NROW:
		case NCOL:
		case LENGTH:
			checkNumParameters(1);
			checkDataTypeParam(getFirstExpr(), 
				DataType.FRAME, DataType.LIST, DataType.MATRIX);
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType(ValueType.INT64);
			break;
		case LINEAGE:
			checkNumParameters(1);
			checkDataTypeParam(getFirstExpr(),
				DataType.MATRIX, DataType.FRAME, DataType.LIST);
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType(ValueType.STRING);
			break;
		case LIST:
			output.setDataType(DataType.LIST);
			output.setValueType(ValueType.UNKNOWN);
			output.setDimensions(getAllExpr().length, 1);
			output.setBlocksize(-1);
			break;
		case EXISTS:
			checkNumParameters(1);
			checkStringOrDataIdentifier(getFirstExpr());
			output.setDataType(DataType.SCALAR);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setValueType(ValueType.BOOLEAN);
			break;
		
		// Contingency tables
		case TABLE:
			
			/*
			 * Allowed #of arguments: 2,3,4,5,6
			 * table(A,B)
			 * table(A,B,W)
			 * table(A,B,1)
			 * table(A,B,dim1,dim2)
			 * table(A,B,W,dim1,dim2)
			 * table(A,B,1,dim1,dim2)
			 * table(A,B,1,dim1,dim2,TRUE)
			 */
			
			// Check for validity of input arguments, and setup output dimensions
			
			// First input: is always of type MATRIX
			checkMatrixParam(getFirstExpr());

			if (getSecondExpr() == null)
				raiseValidateError("Invalid number of arguments to table(). "
					+ "The table() function requires 2, 3, 4, 5, or 6 arguments.", conditional);

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
					if( getThirdExpr() instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)getThirdExpr()).getName()) && !conditional )
						_args[2] = constVars.get(((DataIdentifier)getThirdExpr()).getName());
					if( _args[3] instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)_args[3]).getName()) && !conditional )
						_args[3] = constVars.get(((DataIdentifier)_args[3]).getName());
					
					if ( getThirdExpr().getOutput() instanceof ConstIdentifier ) 
						outputDim1 = ((ConstIdentifier) getThirdExpr().getOutput()).getLongValue();
					if ( _args[3].getOutput() instanceof ConstIdentifier ) 
						outputDim2 = ((ConstIdentifier) _args[3].getOutput()).getLongValue();
				}
				break;
				
			case 5:
			case 6:
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
					if( _args[3] instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)_args[3]).getName())  && !conditional )
						_args[3] = constVars.get(((DataIdentifier)_args[3]).getName());
					if( _args[4] instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)_args[4]).getName())  && !conditional )
						_args[4] = constVars.get(((DataIdentifier)_args[4]).getName());
					
					if ( _args[3].getOutput() instanceof ConstIdentifier ) 
						outputDim1 = ((ConstIdentifier) _args[3].getOutput()).getLongValue();
					if ( _args[4].getOutput() instanceof ConstIdentifier ) 
						outputDim2 = ((ConstIdentifier) _args[4].getOutput()).getLongValue();
				}
				if( _args.length == 6 ) {
					if( !_args[5].getOutput().isScalarBoolean() )
						raiseValidateError("The 6th ctable parameter (outputEmptyBlocks) must be a boolean literal.", conditional);
				}
				break;

			default:
				raiseValidateError("Invalid number of arguments to table(): " 
						+ this.toString(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			// The dimensions for the output matrix will be known only at the
			// run time
			output.setDimensions(outputDim1, outputDim2);
			output.setBlocksize (-1);
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.FP64);
			break;

		case MOMENT:
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
			output.setValueType(ValueType.FP64);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
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
			output.setValueType(ValueType.FP64);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
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
				output.setDimensions(getThirdExpr().getOutput().getDim1(), getThirdExpr().getOutput().getDim2());
				output.setBlocksize(getThirdExpr().getOutput().getBlocksize());
				output.setDataType(getThirdExpr().getOutput().getDataType());
			} else {
				output.setDimensions(getSecondExpr().getOutput().getDim1(), getSecondExpr().getOutput().getDim2());
				output.setBlocksize(getSecondExpr().getOutput().getBlocksize());
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
			output.setBlocksize(-1);
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
			output.setBlocksize(0);
			output.setDataType(DataType.SCALAR);

			break;
		
		case ISNA:
		case ISNAN:
		case ISINF:
			checkNumParameters(1);
			checkMatrixScalarParam(getFirstExpr());
			output.setDataType(id.getDataType());
			output.setDimensions(id.getDim1(), id.getDim2());
			output.setBlocksize (id.getBlocksize());
			//TODO set output type to boolean when supported
			output.setValueType(id.getValueType());
			break;
			
		case MEDIAN:
			checkNumParameters((getSecondExpr()!=null) ? 2 : 1);
			checkMatrixParam(getFirstExpr());

			if (getSecondExpr() != null) {
				// i.e., second input is weight vector
				checkMatrixParam(getSecondExpr());
				checkMatchingDimensions(getFirstExpr(), getSecondExpr());
			}

			// Output is a scalar
			output.setValueType(id.getValueType());
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			output.setDataType(DataType.SCALAR);

			break;
			
		case SAMPLE:
		{
			Expression[] in = getAllExpr(); 
			
			for(Expression e : in)
				checkScalarParam(e);
			
			if (in[0].getOutput().getValueType() != ValueType.FP64 && in[0].getOutput().getValueType() != ValueType.INT64) 
				throw new LanguageException("First argument to sample() must be a number.");
			if (in[1].getOutput().getValueType() != ValueType.FP64 && in[1].getOutput().getValueType() != ValueType.INT64) 
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
				if (in[3].getOutput().getValueType() != ValueType.INT64) 
					throw new LanguageException("Fourth argument, seed, to sample() must be an integer value.");
				if (in[2].getOutput().getValueType() != ValueType.BOOLEAN ) 
					throw new LanguageException("Third argument to sample() must either denote replacement policy (boolean) or seed (integer).");
			}
			else if(in.length == 3) 
			{
				checkNumParameters(3);
				if (in[2].getOutput().getValueType() != ValueType.BOOLEAN 
						&& in[2].getOutput().getValueType() != ValueType.INT64 ) 
					throw new LanguageException("Third argument to sample() must either denote replacement policy (boolean) or seed (integer).");
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
			output.setValueType(ValueType.FP64);
			
			if ( isConstant(in[1]) )
	 			output.setDimensions(((ConstIdentifier)in[1]).getLongValue(), 1);
			else
				output.setDimensions(-1, 1);
 			setBlocksize(id.getBlocksize());
 			
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
			if( !conditional ) {
				if( getFirstExpr() instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)getFirstExpr()).getName()) )
					_args[0] = constVars.get(((DataIdentifier)getFirstExpr()).getName());
				if( getSecondExpr() instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)getSecondExpr()).getName()) )
					_args[1] = constVars.get(((DataIdentifier)getSecondExpr()).getName());
				if( getThirdExpr()!=null && getThirdExpr() instanceof DataIdentifier && constVars.containsKey(((DataIdentifier)getThirdExpr()).getName()) )
					_args[2] = constVars.get(((DataIdentifier)getThirdExpr()).getName());
			}
			
			// check if dimensions can be inferred
			long dim1=-1, dim2=1;
			if ( isConstant(getFirstExpr()) && isConstant(getSecondExpr()) && (getThirdExpr() != null ? isConstant(getThirdExpr()) : true) ) {
				double from, to, incr;
				try {
					from = getDoubleValue(getFirstExpr());
					to = getDoubleValue(getSecondExpr());
					
					// Setup the value of increment
					// default value: 1 if from <= to; -1 if from > to
					if(getThirdExpr() == null) {
						expandArguments();
						_args[2] = new DoubleIdentifier(((from > to) ? -1.0 : 1.0), this);
					}
					incr = getDoubleValue(getThirdExpr()); 
					
				}
				catch (LanguageException e) {
					throw new LanguageException("Arguments for seq() must be numeric.");
				}

				if( (from > to) && (incr >= 0) )
					throw new LanguageException("Wrong sign for the increment in a call to seq()");
				
				// Both end points of the range must included i.e., [from,to] both inclusive.
				// Note that, "to" is included only if (to-from) is perfectly divisible by incr
				// For example, seq(0,1,0.5) produces (0.0 0.5 1.0) whereas seq(0,1,0.6) produces only (0.0 0.6) but not (0.0 0.6 1.0)
				dim1 = UtilFunctions.getSeqLength(from, to, incr);
			}
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.FP64);
			output.setDimensions(dim1, dim2);
			output.setBlocksize(0);
			break;

		case SOLVE:
			checkNumParameters(2);
			checkMatrixParam(getFirstExpr());
			checkMatrixParam(getSecondExpr());
			
			if ( getSecondExpr().getOutput().dimsKnown() && !is1DMatrix(getSecondExpr()) )
				raiseValidateError("Second input to solve() must be a vector", conditional);
			
			if ( getFirstExpr().getOutput().dimsKnown() && getSecondExpr().getOutput().dimsKnown() && 
					getFirstExpr().getOutput().getDim1() != getSecondExpr().getOutput().getDim1() &&
					getFirstExpr().getOutput().getDim1() != getFirstExpr().getOutput().getDim2())
				raiseValidateError("Dimension mismatch in a call to solve()", conditional);
			
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.FP64);
			output.setDimensions(getFirstExpr().getOutput().getDim2(), 1);
			output.setBlocksize(0);
			break;
		
		case INVERSE:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.FP64);
			
			Identifier in = getFirstExpr().getOutput();
			if(in.dimsKnown() && in.getDim1() != in.getDim2()) 
				raiseValidateError("Input to inv() must be square matrix -- given: a " + in.getDim1() + "x" + in.getDim2() + " matrix.", conditional);
			
			output.setDimensions(in.getDim1(), in.getDim2());
			output.setBlocksize(in.getBlocksize());
			break;

		case SQRT_MATRIX_JAVA:

			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.FP64);
			Identifier sqrt = getFirstExpr().getOutput();
			if(sqrt.dimsKnown() && sqrt.getDim1() != sqrt.getDim2())
				raiseValidateError("Input to sqrtMatrix() must be square matrix -- given: a " + sqrt.getDim1() + "x" + sqrt.getDim2() + " matrix.", conditional);
			output.setDimensions( sqrt.getDim1(),  sqrt.getDim2());
			output.setBlocksize( sqrt.getBlocksize());
			break;
		
		case CHOLESKY:
		{
			// A = L%*%t(L) where L is the lower triangular matrix
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());

			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.FP64);
			
			Identifier inA = getFirstExpr().getOutput();
			if(inA.dimsKnown() && inA.getDim1() != inA.getDim2()) 
				raiseValidateError("Input to cholesky() must be square matrix -- given: a " + inA.getDim1() + "x" + inA.getDim2() + " matrix.", conditional);
			
			output.setDimensions(inA.getDim1(), inA.getDim2());
			output.setBlocksize(inA.getBlocksize());
			break;
		}

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
			output.setBlocksize(id.getBlocksize());
			break;
		
		case BIASADD:
		case BIASMULT:
		{
			Expression input = _args[0];
			Expression bias = _args[1];
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.FP64);
			output.setDimensions(input.getOutput().getDim1(), input.getOutput().getDim2());
			output.setBlocksize(input.getOutput().getBlocksize());
			checkMatrixParam(input);
			checkMatrixParam(bias);
			break;
		}
		case CONV2D:
		case CONV2D_BACKWARD_FILTER:
		case CONV2D_BACKWARD_DATA:
		case MAX_POOL:
		case AVG_POOL:
		case MAX_POOL_BACKWARD:
		case AVG_POOL_BACKWARD:
		{
			// At DML level:
			// output = conv2d(input, filter, input_shape=[1, 3, 2, 2], filter_shape=[1, 3, 2, 2], 
			// strides=[1, 1], padding=[1,1])
			// 
			// Converted to following in constructor (only supported NCHW):
			// output = conv2d(input, filter, stride1, stride2, padding1,padding2,  
			// input_shape1, input_shape2, input_shape3, input_shape4, 
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4)
			//
			// Similarly,
			// conv2d_backward_filter and conv2d_backward_data
			Expression input = _args[0]; // For conv2d_backward_filter, this is input and for conv2d_backward_data, this is filter
			
			Expression input2 = null;
			if(!(this.getOpCode() == Builtins.MAX_POOL || this.getOpCode() == Builtins.AVG_POOL)) {
				input2 = _args[1];			// For conv2d_backward functions, this is dout
				checkMatrixParam(input2);
			}
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.FP64);
			output.setBlocksize(input.getOutput().getBlocksize());
			
			if(this.getOpCode() == Builtins.MAX_POOL_BACKWARD || this.getOpCode() == Builtins.AVG_POOL_BACKWARD) {
				output.setDimensions(input.getOutput().getDim1(), input.getOutput().getDim2());
			}
			else {
				// stride1, stride2, padding1, padding2, numImg, numChannels, imgSize, imgSize, 
	 			// filter_shape1=1, filter_shape2=1, filterSize/poolSize1, filterSize/poolSize1
				try {
					int start = 2;
					if(!(this.getOpCode() == Builtins.MAX_POOL || this.getOpCode() == Builtins.AVG_POOL)) {
						start = 1;
					}
					long stride_h = (long) getDoubleValue(_args[start++]);
					long stride_w = (long) getDoubleValue(_args[start++]);
					long pad_h = (long) getDoubleValue(_args[start++]);
					long pad_w = (long) getDoubleValue(_args[start++]); 
					long N = (long) getDoubleValue(_args[start++]);
					long C = (long) getDoubleValue(_args[start++]);
					long H = (long) getDoubleValue(_args[start++]);
					long W = (long) getDoubleValue(_args[start++]);
					long K = -1;
					if(!(this.getOpCode() == Builtins.MAX_POOL || this.getOpCode() == Builtins.AVG_POOL)) {
						K = (long) getDoubleValue(_args[start]);
					}
					start++; start++; // Increment index for K and C
					long R = (long) getDoubleValue(_args[start++]);
					long S = (long) getDoubleValue(_args[start++]);
					
					if(this.getOpCode() == Builtins.CONV2D_BACKWARD_FILTER) {
						output.setDimensions(K, C*R*S);
					}
					else if(this.getOpCode() == Builtins.CONV2D_BACKWARD_DATA) {
						output.setDimensions(N, C*H*W);
					}
					else if(H > 0 && W > 0 && stride_h > 0 && stride_w > 0 && pad_h >= 0 && pad_w >= 0 && R > 0 && S > 0) {
						long P = DnnUtils.getP(H, R, stride_h, pad_h);
						long Q = DnnUtils.getQ(W, S, stride_w, pad_w);
						
						// Try to set both rows and columns
						if(this.getOpCode() == Builtins.CONV2D) 
							output.setDimensions(N, K*P*Q);
						else if(this.getOpCode() == Builtins.MAX_POOL || this.getOpCode() == Builtins.AVG_POOL)
							output.setDimensions(N, C*P*Q);
						else
							throw new LanguageException("");
					}
					else {
						// Since columns cannot be computed, set only rows
						if(this.getOpCode() == Builtins.CONV2D) 
							output.setDimensions(input.getOutput().getDim1(), -1);
						else if(this.getOpCode() == Builtins.MAX_POOL || this.getOpCode() == Builtins.AVG_POOL)
							output.setDimensions(input.getOutput().getDim1(), -1);
						else
							throw new LanguageException("");
					}
				}
				catch(Exception e) {
					output.setDimensions(-1, -1); // To make sure that output dimensions are not incorrect even if getDoubleValue doesnot return value
				}
			}
			checkMatrixParam(input);
			if(input2 != null)
				checkMatrixParam(input2);
			break;
		}
		case TIME: 
			checkNumParameters(0);
			// Output of TIME() is scalar and long
			output.setDataType(DataType.SCALAR);
			output.setValueType(ValueType.INT64);
			output.setDimensions(0, 0);
			output.setBlocksize(0);
			break;

		case DROP_INVALID_TYPE:
		case VALUE_SWAP:
		case FRAME_ROW_REPLICATE:
			checkNumParameters(2);
			checkMatrixFrameParam(getFirstExpr());
			checkMatrixFrameParam(getSecondExpr());
			output.setDataType(DataType.FRAME);
			output.setDimensions(id.getDim1(), id.getDim2());
			output.setBlocksize (id.getBlocksize());
			output.setValueType(ValueType.STRING);
			break;

		case DROP_INVALID_LENGTH:
			checkNumParameters(2);
			checkMatrixFrameParam(getFirstExpr());
			checkMatrixFrameParam(getSecondExpr());
			output.setDataType(DataType.FRAME);
			output.setDimensions(id.getDim1(), id.getDim2());
			output.setBlocksize (id.getBlocksize());
			output.setValueType(id.getValueType());
			break;
		case APPLY_SCHEMA:
				checkNumParameters(2);
				checkMatrixFrameParam(getFirstExpr());
				checkMatrixFrameParam(getSecondExpr());
				output.setDataType(DataType.FRAME);
				output.setDimensions(id.getDim1(), id.getDim2());
				output.setBlocksize (id.getBlocksize());
				break;
		case MAP:
			checkNumParameters(getThirdExpr() != null ? 3 : 2);
			checkMatrixFrameParam(getFirstExpr());
			checkScalarParam(getSecondExpr());
			if(getThirdExpr() != null)
				checkScalarParam(getThirdExpr()); // margin
			output.setDataType(DataType.FRAME);
			if(_args[1].getText().contains("jaccardSim")) {
				output.setDimensions(id.getDim1(), id.getDim1());
				output.setValueType(ValueType.FP64);
			}
			else {
				output.setDimensions(id.getDim1(), id.getDim2());
				output.setValueType(ValueType.STRING);
			}
			break;
		case LOCAL:
			if(OptimizerUtils.ALLOW_SCRIPT_LEVEL_LOCAL_COMMAND){
				checkNumParameters(1);
				checkMatrixParam(getFirstExpr());
				output.setDataType(DataType.MATRIX);
				output.setDimensions(id.getDim1(), id.getDim2());
				output.setBlocksize (id.getBlocksize());
				output.setValueType(id.getValueType());
			}
			else 
				raiseValidateError("Local instruction not allowed in dml script");
		case COMPRESS:
		case DECOMPRESS:
			if(OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND){
				checkNumParameters(1);
				checkMatrixFrameParam(getFirstExpr());
				output.setDataType(getFirstExpr().getOutput().getDataType());
				output.setDimensions(id.getDim1(), id.getDim2());
				output.setBlocksize (id.getBlocksize());
				output.setValueType(id.getValueType());
			}
			else
				raiseValidateError("The compress or decompress instruction is not allowed in dml scripts");
			break;
		case QUANTIZE_COMPRESS:
			if(OptimizerUtils.ALLOW_SCRIPT_LEVEL_QUANTIZE_COMPRESS_COMMAND) {
				checkNumParameters(2);
				Expression firstExpr = getFirstExpr();
				Expression secondExpr = getSecondExpr();

				checkMatrixParam(getFirstExpr());

				if(secondExpr != null) {
					// check if scale factor is a scalar, vector or matrix
					checkMatrixScalarParam(secondExpr);
					// if scale factor is a vector or matrix, make sure it has an appropriate shape
					if(secondExpr.getOutput().getDataType() != DataType.SCALAR) {
						if(is1DMatrix(secondExpr)) {
							long vectorLength = secondExpr.getOutput().getDim1();
							if(vectorLength != firstExpr.getOutput().getDim1()) {
								raiseValidateError(
									"The length of the row-wise scale factor vector must match the number of rows in the matrix.");
							}
						}
						else {
							checkMatchingDimensions(firstExpr, secondExpr);
						}
					}
				}
			}
			else 
				raiseValidateError("The quantize_compress instruction not allowed in dml scripts");
			break;

		case ROW_COUNT_DISTINCT:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(id.getDim1(), 1);
			output.setBlocksize (id.getBlocksize());
			output.setValueType(ValueType.INT64);
			output.setNnz(id.getDim1());
			break;

		case COL_COUNT_DISTINCT:
			checkNumParameters(1);
			checkMatrixParam(getFirstExpr());
			output.setDataType(DataType.MATRIX);
			output.setDimensions(1, id.getDim2());
			output.setBlocksize (id.getBlocksize());
			output.setValueType(ValueType.INT64);
			output.setNnz(id.getDim2());
			break;

		default:
			if( isMathFunction() ) {
				checkMathFunctionParam();
				//unary operations
				if( getSecondExpr() == null ) {
					output.setDataType(id.getDataType());
					output.setValueType((output.getDataType()==DataType.SCALAR
						&& getOpCode()==Builtins.ABS)?id.getValueType():ValueType.FP64 );
					output.setDimensions(id.getDim1(), id.getDim2());
					output.setBlocksize(id.getBlocksize()); 
				}
				//binary operations
				else {
					setBinaryOutputProperties(output);
					// override computed value type for special cases
					if( getOpCode() == Builtins.LOG )
						output.setValueType(ValueType.FP64);
				}
			} 
			else {
				// always unconditional (because unsupported operation)
				Builtins op = getOpCode();
				if( op==Builtins.EIGEN || op==Builtins.LU || op==Builtins.QR || op==Builtins.SVD 
						|| op==Builtins.LSTM || op==Builtins.LSTM_BACKWARD
						|| op==Builtins.BATCH_NORM2D || op==Builtins.BATCH_NORM2D_BACKWARD)
					raiseValidateError("Function "+op+" needs to be called with multi-return assignment.", false, LanguageErrorCodes.INVALID_PARAMETERS);
				else
					raiseValidateError("Unsupported function "+op, false, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
	}

	private void setBinaryOutputProperties(DataIdentifier output) {
		DataType dt1 = getFirstExpr().getOutput().getDataType();
		DataType dt2 = getSecondExpr().getOutput().getDataType();
		DataType dtOut = (dt1==DataType.MATRIX || dt2==DataType.MATRIX) ?
			DataType.MATRIX : DataType.SCALAR;
		if( dt1==DataType.MATRIX && dt2==DataType.MATRIX )
			checkMatchingDimensions(getFirstExpr(), getSecondExpr(), true);
		MatrixCharacteristics dims = getBinaryMatrixCharacteristics(getFirstExpr(), getSecondExpr());
		output.setDataType(dtOut);
		output.setValueType(dtOut==DataType.MATRIX ? ValueType.FP64 :
			computeValueType(getFirstExpr(), getSecondExpr(), true));
		output.setDimensions(dims.getRows(), dims.getCols());
		output.setBlocksize (dims.getBlocksize());
	}
	
	private void setTernaryOutputProperties(DataIdentifier output, boolean conditional) {
		DataType dt1 = getFirstExpr().getOutput().getDataType();
		DataType dt2 = getSecondExpr().getOutput().getDataType();
		DataType dt3 = getThirdExpr().getOutput().getDataType();
		DataType dtOut = (dt1.isMatrix() || dt2.isMatrix() || dt3.isMatrix()) ?
			DataType.MATRIX : DataType.SCALAR;
		if( dt1==DataType.MATRIX && dt2==DataType.MATRIX )
			checkMatchingDimensions(getFirstExpr(), getSecondExpr(), false, conditional);
		if( dt1==DataType.MATRIX && dt3==DataType.MATRIX )
			checkMatchingDimensions(getFirstExpr(), getThirdExpr(), false, conditional);
		if( dt2==DataType.MATRIX && dt3==DataType.MATRIX )
			checkMatchingDimensions(getSecondExpr(), getThirdExpr(), false, conditional);
		MatrixCharacteristics dims1 = getBinaryMatrixCharacteristics(getFirstExpr(), getSecondExpr());
		MatrixCharacteristics dims2 = getBinaryMatrixCharacteristics(getSecondExpr(), getThirdExpr());
		output.setDataType(dtOut);
		output.setValueType(dtOut==DataType.MATRIX ? ValueType.FP64 :
			computeValueType(getSecondExpr(), getThirdExpr(), true));
		output.setDimensions(Math.max(dims1.getRows(), dims2.getRows()), Math.max(dims1.getCols(), dims2.getCols()));
		output.setBlocksize(Math.max(dims1.getBlocksize(), dims2.getBlocksize()));
	}

	private void setNaryOutputProperties(DataIdentifier output) {
		DataType dt = Arrays.stream(getAllExpr()).allMatch(
			e -> e.getOutput().getDataType().isScalar()) ? DataType.SCALAR : DataType.MATRIX;
		Expression firstM = dt.isMatrix() ? Arrays.stream(getAllExpr()).filter(
			e -> e.getOutput().getDataType().isMatrix()).findFirst().get() : null;
		ValueType vt = dt.isMatrix() ? ValueType.FP64 : ValueType.INT64;
		for( Expression e : getAllExpr() ) {
			vt = computeValueType(e, e.getOutput().getValueType(), vt, true);
			if( e.getOutput().getDataType().isMatrix() )
				checkMatchingDimensions(firstM, e, true);
		}
		output.setDataType(dt);
		output.setValueType(vt);
		output.setDimensions(dt.isMatrix() ? firstM.getOutput().getDim1() : 0,
			dt.isMatrix() ? firstM.getOutput().getDim2() : 0);
		output.setBlocksize (dt.isMatrix() ? firstM.getOutput().getBlocksize() : 0);
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
		return _opcode.isMultiReturn();
	}

	private static boolean isConstant(Expression expr) {
		return ( expr != null && expr instanceof ConstIdentifier );
	}
	
	private static double getDoubleValue(Expression expr) {
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
		case COSH:
		case SINH:
		case TANH:
		case SIGN:
		case SQRT:
		case ABS:
		case LOG:
		case EXP:
		case ROUND:
		case CEIL:
		case FLOOR:
		case MEDIAN:
		case XOR:
		case BITWAND:
		case BITWOR:
		case BITWXOR:
		case BITWSHIFTL:
		case BITWSHIFTR:
			return true;
		default:
			return false;
		}
	}

	private void checkMathFunctionParam() {
		switch (this.getOpCode()) {
		case COS:
		case SIN:
		case TAN:
		case ACOS:
		case ASIN:
		case ATAN:
		case COSH:
		case SINH:
		case TANH:
		case SIGN:	
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

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_opcode.toString() + "(");
		if (_args != null) {
			for (int i = 0; i < _args.length; i++) {
				if (i > 0) {
					sb.append(",");
				}
				sb.append(_args[i].toString());
			}
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

	protected void checkNumParameters(int count) { //always unconditional
		if (getFirstExpr() == null && _args.length > 0) {
			raiseValidateError("Missing argument for function " + this.getOpCode(), false,
					LanguageErrorCodes.INVALID_PARAMETERS);
		}

		// Not sure the rationale for the first two if loops, but will keep them for backward compatibility
		if (((count == 1) && (getSecondExpr() != null || getThirdExpr() != null))
				|| ((count == 2) && (getThirdExpr() != null))) {
			raiseValidateError("Invalid number of arguments for function " + this.getOpCode().toString().toLowerCase()
					+ "(). This function only takes 1 or 2 arguments.", false);
		} else if (((count == 2) && (getSecondExpr() == null))
				|| ((count == 3) && (getSecondExpr() == null || getThirdExpr() == null))) {
			raiseValidateError("Missing argument for function " + this.getOpCode(), false,
					LanguageErrorCodes.INVALID_PARAMETERS);
		} else if(count > 0 && (_args == null || _args.length < count)) {
			raiseValidateError("Missing argument for function " + this.getOpCode(), false,
					LanguageErrorCodes.INVALID_PARAMETERS);
		} else if (count == 0 && (_args.length > 0
				|| getSecondExpr() != null || getThirdExpr() != null)) {
			raiseValidateError("Missing argument for function " + this.getOpCode()
					+ "(). This function doesn't take any arguments.", false);
		}
	}

	protected void checkMatrixParam(Expression e) {
		if (e.getOutput().getDataType() != DataType.MATRIX) {
			raiseValidateError("Expected " + e.getText() + " to be a matrix argument for function " +
					this.getOpCode().toString().toLowerCase() + "().", false);
		}
	}
	
	protected void checkMatrixTensorParam(Expression e) {
		if (e.getOutput().getDataType() != DataType.MATRIX) {
			// Param is not a matrix
			// TODO get supported Operations form builtins
			if (e.getOutput().getDataType() != DataType.TENSOR || getOpCode() != Builtins.SUM) {
				// Param is also not a tensor, or the operation is not supported on tensor
				raiseValidateError("Expected " + e.getText() + " to be a matrix or tensor argument for function "
						+ this.getOpCode().toString().toLowerCase() + "().", false);
			}
		}
	}

	protected void checkDataTypeParam(Expression e, DataType... dt) { //always unconditional
		if( !ArrayUtils.contains(dt, e.getOutput().getDataType()) )
			raiseValidateError("Non-matching expected data type for function "+ getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
	}

	protected void checkMatrixFrameParam(Expression e) { //always unconditional
		if (e.getOutput().getDataType() != DataType.MATRIX && e.getOutput().getDataType() != DataType.FRAME) {
			raiseValidateError("Expecting matrix or frame parameter for function "+ getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	protected void checkMatrixScalarParam(Expression e) { //always unconditional
		if (e.getOutput().getDataType() != DataType.MATRIX && e.getOutput().getDataType() != DataType.SCALAR) {
			raiseValidateError("Expecting matrix or scalar parameter for function "+ getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	private void checkScalarParam(Expression e) { //always unconditional
		if (e.getOutput().getDataType() != DataType.SCALAR) {
			raiseValidateError("Expecting scalar parameter for function " + getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	private void checkListParam(Expression e) { //always unconditional
		if (e.getOutput().getDataType() != DataType.LIST) {
			raiseValidateError("Expecting scalar parameter for function " + getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	@SuppressWarnings("unused")
	private void checkScalarFrameParam(Expression e) { //always unconditional
		if (e.getOutput().getDataType() != DataType.SCALAR && e.getOutput().getDataType() != DataType.FRAME) {
			raiseValidateError("Expecting scalar parameter for function " + getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}

	private void checkValueTypeParam(Expression e, ValueType vt) { //always unconditional
		if (e.getOutput().getValueType() != vt) {
			raiseValidateError("Expecting parameter of different value type " + this.getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	protected void checkStringOrDataIdentifier(Expression e) { //always unconditional
		if( !(e.getOutput().getDataType().isScalar() && e.getOutput().getValueType()==ValueType.STRING)
			&& !(e instanceof DataIdentifier && !(e instanceof IndexedIdentifier)) ) {
			raiseValidateError("Expecting variable name or data identifier "+ getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}
	
	private static boolean is1DMatrix(Expression e) {
		return (e.getOutput().getDim1() == 1 || e.getOutput().getDim2() == 1 );
	}
	
	private static boolean dimsKnown(Expression e) {
		return (e.getOutput().getDim1() != -1 && e.getOutput().getDim2() != -1);
	}
	
	private void check1DMatrixParam(Expression e) { //always unconditional
		checkMatrixParam(e);
		
		// throw an exception, when e's output is NOT a one-dimensional matrix 
		// the check must be performed only when the dimensions are known at compilation time
		if ( dimsKnown(e) && !is1DMatrix(e)) {
			raiseValidateError("Expecting one-dimensional matrix parameter for function "
					          + this.getOpCode(), false, LanguageErrorCodes.UNSUPPORTED_PARAMETERS);
		}
	}

	private void checkMatchingDimensions(Expression expr1, Expression expr2) {
		checkMatchingDimensions(expr1, expr2, false);
	}
	
	private void checkMatchingDimensions(Expression expr1, Expression expr2, boolean allowsMV) {
		checkMatchingDimensions(expr1, expr2, allowsMV, false);
	}
	
	private void checkMatchingDimensions(Expression expr1, Expression expr2, boolean allowsMV, boolean conditional) 
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
						+ this.getOpCode(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
	}
	
	private void checkMatchingDimensionsQuantile() 
	{
		if (getFirstExpr().getOutput().getDim1() != getSecondExpr().getOutput().getDim1()) {
			raiseValidateError("Mismatch in matrix dimensions for "
					+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
		}
	}

	public static BuiltinFunctionExpression getBuiltinFunctionExpression(ParserRuleContext ctx, 
		String functionName, ArrayList<ParameterExpression> paramExprsPassed, String filename) {
		
		if (functionName == null || paramExprsPassed == null)
			return null;
		
		// check if the function name is built-in function
		//	(assign built-in function op if function is built-in
		
				
		return (Builtins.contains(functionName, false, false) 
			&& (paramExprsPassed.stream().anyMatch(p -> p.getName()==null) //at least one unnamed
				|| paramExprsPassed.size() == 0)) ? 
			new BuiltinFunctionExpression(ctx, Builtins.get(functionName), paramExprsPassed, filename) : null;
	}
	
	/**
	 * Convert a value type (double, int, or boolean) to a built-in function operator.
	 * 
	 * @param vt Value type ({@code ValueType.DOUBLE}, {@code ValueType.INT}, or {@code ValueType.BOOLEAN}).
	 * @return Built-in function operator ({@code Builtins.AS_DOUBLE},
	 * {@code Builtins.AS_INT}, or {@code Builtins.AS_BOOLEAN}).
	 */
	public static Builtins getValueTypeCastOperator( ValueType vt ) {
		switch( vt )
		{
			case FP64:
				return Builtins.CAST_AS_DOUBLE;
			case INT64:
				return Builtins.CAST_AS_INT;
			case BOOLEAN:
				return Builtins.CAST_AS_BOOLEAN;
			default:
				throw new LanguageException("No cast for value type "+vt);
		}
	}
}
