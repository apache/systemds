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

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.misc.Interval;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import java.util.ArrayList;
import java.util.HashMap;


public abstract class Expression implements ParseInfo
{
	/**
	 * Binary operators.
	 */
	public enum BinaryOp {
		PLUS, MINUS, MULT, DIV, MODULUS, INTDIV, MATMULT, POW, INVALID
	}

	/**
	 * Relational operators.
	 */
	public enum RelationalOp {
		LESSEQUAL, LESS, GREATEREQUAL, GREATER, EQUAL, NOTEQUAL, INVALID
	}

	/**
	 * Boolean operators.
	 */
	public enum BooleanOp {
		CONDITIONALAND, CONDITIONALOR, LOGICALAND, LOGICALOR, NOT, INVALID
	}
	
	/**
	 * Data operators.
	 */
	public enum DataOp {
		READ, WRITE, RAND, MATRIX, FRAME, TENSOR, SQL, FEDERATED
	}

	/**
	 * Function call operators.
	 */
	public enum FunctCallOp {
		INTERNAL, EXTERNAL
	}
	
	private static final Log LOG = LogFactory.getLog(Expression.class.getName());
	
	private static final IDSequence _tempId = new IDSequence();
	protected Identifier[] _outputs;

	public Expression() {
		_outputs = null;
	}

	public abstract Expression rewriteExpression(String prefix);
	
	public void setOutput(Identifier output) {
		if ( _outputs == null) {
			_outputs = new Identifier[1];
		}
		_outputs[0] = output;
	}

	/**
	 * Obtain identifier.
	 * 
	 * @return Identifier
	 */
	public Identifier getOutput() {
		if (_outputs != null && _outputs.length > 0)
			return _outputs[0];
		else
			return null;
	}
	
	/** Obtain identifiers.
	 * 
	 * @return Identifiers
	 */
	public Identifier[] getOutputs() {
		return _outputs;
	}
	
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars, boolean conditional) {
		raiseValidateError("Should never be invoked in Baseclass 'Expression'", false);
	}
	
	public void validateExpression(MultiAssignmentStatement mas, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars, boolean conditional) {
		raiseValidateError("Should never be invoked in Baseclass 'Expression'", false);
	}

	/**
	 * Convert string value to binary operator.
	 * 
	 * @param val String value ('+', '-', '*', '/', '%%', '%/%', '^', %*%')
	 * @return Binary operator ({@code BinaryOp.PLUS}, {@code BinaryOp.MINUS}, 
	 * {@code BinaryOp.MULT}, {@code BinaryOp.DIV}, {@code BinaryOp.MODULUS}, 
	 * {@code BinaryOp.INTDIV}, {@code BinaryOp.POW}, {@code BinaryOp.MATMULT}).
	 * Returns {@code BinaryOp.INVALID} if string value not recognized.
	 */
	public static BinaryOp getBinaryOp(String val) {
		if (val.equalsIgnoreCase(Opcodes.PLUS.toString()))
			return BinaryOp.PLUS;
		else if (val.equalsIgnoreCase(Opcodes.MINUS.toString()))
			return BinaryOp.MINUS;
		else if (val.equalsIgnoreCase(Opcodes.MULT.toString()))
			return BinaryOp.MULT;
		else if (val.equalsIgnoreCase(Opcodes.DIV.toString()))
			return BinaryOp.DIV;
		else if (val.equalsIgnoreCase(Opcodes.MODULUS.toString()))
			return BinaryOp.MODULUS;
		else if (val.equalsIgnoreCase(Opcodes.INTDIV.toString()))
			return BinaryOp.INTDIV;
		else if (val.equalsIgnoreCase(Opcodes.POW.toString()))
			return BinaryOp.POW;
		else if (val.equalsIgnoreCase("%*%"))
			return BinaryOp.MATMULT;
		return BinaryOp.INVALID;
	}

	/**
	 * Convert string value to relational operator.
	 * 
	 * @param val String value ('&lt;', '&lt;=', '&gt;', '&gt;=', '==', '!=')
	 * @return Relational operator ({@code RelationalOp.LESS}, {@code RelationalOp.LESSEQUAL}, 
	 * {@code RelationalOp.GREATER}, {@code RelationalOp.GREATEREQUAL}, {@code RelationalOp.EQUAL}, 
	 * {@code RelationalOp.NOTEQUAL}).
	 * Returns {@code RelationalOp.INVALID} if string value not recognized.
	 */
	public static RelationalOp getRelationalOp(String val) {
		if (val == null) 
			return null;
		else if (val.equalsIgnoreCase(Opcodes.LESS.toString()))
			return RelationalOp.LESS;
		else if (val.equalsIgnoreCase(Opcodes.LESSEQUAL.toString()))
			return RelationalOp.LESSEQUAL;
		else if (val.equalsIgnoreCase(Opcodes.GREATER.toString()))
			return RelationalOp.GREATER;
		else if (val.equalsIgnoreCase(Opcodes.GREATEREQUAL.toString()))
			return RelationalOp.GREATEREQUAL;
		else if (val.equalsIgnoreCase(Opcodes.EQUAL.toString()))
			return RelationalOp.EQUAL;
		else if (val.equalsIgnoreCase(Opcodes.NOTEQUAL.toString()))
			return RelationalOp.NOTEQUAL;
		return RelationalOp.INVALID;
	}

	/**
	 * Convert string value to boolean operator.
	 * 
	 * @param val String value ('&amp;&amp;', '&amp;', '||', '|', '!')
	 * @return Boolean operator ({@code BooleanOp.CONDITIONALAND}, {@code BooleanOp.LOGICALAND}, 
	 * {@code BooleanOp.CONDITIONALOR}, {@code BooleanOp.LOGICALOR}, {@code BooleanOp.NOT}).
	 * Returns {@code BooleanOp.INVALID} if string value not recognized.
	 */
	public static BooleanOp getBooleanOp(String val) {
		if (val.equalsIgnoreCase(Opcodes.AND.toString()))
			return BooleanOp.CONDITIONALAND;
		else if (val.equalsIgnoreCase("&"))
			return BooleanOp.LOGICALAND;
		else if (val.equalsIgnoreCase(Opcodes.OR.toString()))
			return BooleanOp.CONDITIONALOR;
		else if (val.equalsIgnoreCase("|"))
			return BooleanOp.LOGICALOR;
		else if (val.equalsIgnoreCase(Opcodes.NOT.toString()))
			return BooleanOp.NOT;
		return BooleanOp.INVALID;
	}

	/**
	 * Convert string format type to {@code Hop.FileFormatTypes}.
	 * 
	 * @param format String format type ("text", "binary", "mm", "csv")
	 * @return Format as {@code Hop.FileFormatTypes}. Can be
	 * {@code FileFormatTypes.TEXT}, {@code FileFormatTypes.BINARY}, 
	 * {@code FileFormatTypes.MM}, or {@code FileFormatTypes.CSV}. Unrecognized
	 * type is set to {@code FileFormatTypes.TEXT}.
	 */
	public static FileFormat convertFormatType(String format) {
		if( format == null )
			return FileFormat.defaultFormat();
		return FileFormat.safeValueOf(format);
	}

	/**
	 * Obtain temporary name ("parsertemp" + _tempId) for expression. Used to construct Hops from
	 * parse tree.
	 * 
	 * @return Temporary name of expression.
	 */
	public static String getTempName() {
		return "parsertemp" + _tempId.getNextID();
	}

	public abstract VariableSet variablesRead();

	public abstract VariableSet variablesUpdated();

	/**
	 * Compute data type based on expressions. The identifier for each expression is obtained and passed to
	 * {@link #computeDataType(Identifier, Identifier, boolean)}. If the identifiers have the same data type, the shared data type is
	 * returned. Otherwise, if {@code cast} is {@code true} and one of the identifiers is a matrix and the other
	 * identifier is a scalar, return {@code DataType.MATRIX}. Otherwise, throw a LanguageException.
	 * 
	 * @param expression1 First expression
	 * @param expression2 Second expression
	 * @param cast Whether a cast should potentially be performed
	 * @return The data type ({@link DataType})
	 */
	public static DataType computeDataType(Expression expression1, Expression expression2, boolean cast) {
		return computeDataType(expression1.getOutput(), expression2.getOutput(), cast);
	}

	/**
	 * Compute data type based on identifiers. If the identifiers have the same data type, the shared data type is
	 * returned. Otherwise, if {@code cast} is {@code true} and one of the identifiers is a matrix and the other
	 * identifier is a scalar, return {@code DataType.MATRIX}. Otherwise, throw a LanguageException.
	 * 
	 * @param identifier1 First identifier
	 * @param identifier2 Second identifier
	 * @param cast Whether a cast should potentially be performed
	 * @return The data type ({@link DataType})
	 */
	public static DataType computeDataType(Identifier identifier1, Identifier identifier2, boolean cast) {
		DataType d1 = identifier1.getDataType();
		DataType d2 = identifier2.getDataType();

		if (d1 == d2)
			return d1;

		if (cast) {
			if (d1 == DataType.MATRIX && d2 == DataType.SCALAR)
				return DataType.MATRIX;
			if (d1 == DataType.SCALAR && d2 == DataType.MATRIX)
				return DataType.MATRIX;
		}

		//raise error with id1 location
		identifier1.raiseValidateError("Invalid Datatypes for operation "+d1+" "+d2, false, 
				LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		return null; //never reached because unconditional
	}

	/**
	 * Compute value type based on expressions. The identifier for each expression is obtained and passed to
	 * {@link #computeValueType(Identifier, Identifier, boolean)}. If the identifiers have the same value type, the shared value type is
	 * returned. Otherwise, if {@code cast} is {@code true} and one value type is a double and the other is an int,
	 * return {@code ValueType.DOUBLE}. If {@code cast} is {@code true} and one value type is a string or the other value type is a string, return
	 * {@code ValueType.STRING}. Otherwise, throw a LanguageException.
	 * 
	 * @param expression1 First expression
	 * @param expression2 Second expression
	 * @param cast Whether a cast should potentially be performed
	 * @return The value type ({@link ValueType})
	 */
	public static ValueType computeValueType(Expression expression1, Expression expression2, boolean cast) {
		return computeValueType(expression1.getOutput(), expression2.getOutput(), cast);
	}
	
	/**
	 * Compute value type based on identifiers. If the identifiers have the same value type, the shared value type is
	 * returned. Otherwise, if {@code cast} is {@code true} and one value type is a double and the other is an int,
	 * return {@code ValueType.DOUBLE}. If {@code cast} is {@code true} and one value type is a string or the other value type is a string, return
	 * {@code ValueType.STRING}. Otherwise, throw a LanguageException.
	 * 
	 * @param identifier1 First identifier
	 * @param identifier2 Second identifier
	 * @param cast Whether a cast should potentially be performed
	 * @return The value type ({@link ValueType})
	 */
	public static ValueType computeValueType(Identifier identifier1, Identifier identifier2, boolean cast) {
		return computeValueType(identifier1, identifier1.getValueType(), identifier2.getValueType(), cast);
	}
	
	public static ValueType computeValueType(Expression expr1, ValueType v1, ValueType v2, boolean cast) {
		if (v1 == v2)
			return v1;
		if (v1 == ValueType.UNKNOWN && v2 != ValueType.UNKNOWN)
			return v2;
		if (v1 != ValueType.UNKNOWN && v2 == ValueType.UNKNOWN)
			return v1;

		if (cast) {
			if (v1 == ValueType.FP64 && v2 == ValueType.INT64)
				return ValueType.FP64;
			if (v2 == ValueType.FP64 && v1 == ValueType.INT64)
				return ValueType.FP64;
			
			// String value type will override others
			// Primary operation involving strings is concatenation (+)
			if ( v1 == ValueType.STRING || v2 == ValueType.STRING )
				return ValueType.STRING;
		}

		//raise error with id1 location
		expr1.raiseValidateError("Invalid Valuetypes for operation "+v1+" "+v2, false,
			LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		return null; //never reached because unconditional
	}

	@Override
	public boolean equals(Object that)
	{
		//empty check for robustness
		if( that == null || !(that instanceof Expression) )
			return false;
		
		Expression thatExpr = (Expression) that;
		
		//approach is to compare string representation of expression
		String thisStr = this.toString();
		String thatStr = thatExpr.toString();
		
		return thisStr.equalsIgnoreCase(thatStr);
	}
	
	@Override
	public int hashCode()
	{
		//use identity hash code
		return super.hashCode();
	}
	
	///////////////////////////////////////////////////////////////
	// validate error handling (consistent for all expressions)
	
	
	/**
	 * Throw a LanguageException with the message.
	 * 
	 * @param message the error message
	 */
	public void raiseValidateError( String message ) {
		raiseValidateError(message, false, null);
	}
	
	/**
	 * Throw a LanguageException with the message if conditional is {@code false};
	 * otherwise log the message as a warning.
	 * 
	 * @param message the error (or warning) message
	 * @param conditional if {@code true}, display log warning message. Otherwise, the message
	 * will be thrown as a LanguageException
	 */
	public void raiseValidateError( String message, boolean conditional ) {
		raiseValidateError(message, conditional, null);
	}
	
	/**
	 * Throw a LanguageException with the message (and optional error code) if conditional is {@code false};
	 * otherwise log the message as a warning.
	 * 
	 * @param msg the error (or warning) message
	 * @param conditional if {@code true}, display log warning message. Otherwise, the message (and optional
	 * error code) will be thrown as a LanguageException
	 * @param errorCode optional error code
	 */
	public void raiseValidateError(String msg, boolean conditional, String errorCode) {
		if (conditional) {// warning if conditional
			String fullMsg = this.printWarningLocation() + msg;
			LOG.warn(fullMsg);
		} else {// error and exception if unconditional
			String fullMsg = this.printErrorLocation() + msg;
			if (errorCode != null)
				throw new LanguageException(fullMsg, errorCode);
			else
				throw new LanguageException(fullMsg);
		}
	}

	/**
	 * Returns the matrix characteristics for scalar-scalar, scalar-matrix, matrix-scalar, matrix-matrix
	 * operations. This method is aware of potentially unknowns and matrix-vector (col/row) operations.
	 * 
	 * @param expression1 The first expression
	 * @param expression2 The second expression
	 * @return matrix characteristics
	 * [1] is the number of columns (clen), [2] is the number of rows in a block (blen),
	 * and [3] is the number of columns in a block (blen). Default (unknown) values are
	 * -1. Scalar values are all 0.
	 */
	public static MatrixCharacteristics getBinaryMatrixCharacteristics(Expression expression1, Expression expression2) {
		Identifier idleft = expression1.getOutput();
		Identifier idright = expression2.getOutput();
		if( idleft.getDataType()==DataType.SCALAR && idright.getDataType()==DataType.SCALAR ) {
			return new MatrixCharacteristics(0, 0, 0, 0);
		}
		else if( idleft.getDataType()==DataType.SCALAR && idright.getDataType()==DataType.MATRIX ) {
			return new MatrixCharacteristics(idright.getDim1(), idright.getDim2(), idright.getBlocksize());
		}
		else if( idleft.getDataType()==DataType.MATRIX && idright.getDataType()==DataType.SCALAR ) {
			return new MatrixCharacteristics(idleft.getDim1(), idleft.getDim2(), idleft.getBlocksize());
		}
		else if( idleft.getDataType()==DataType.MATRIX && idright.getDataType()==DataType.MATRIX ) {
			MatrixCharacteristics mc = new MatrixCharacteristics(
				idleft.getDim1(), idleft.getDim2(), idleft.getBlocksize());
			if( mc.getRows() < 0 && idright.getDim1() > 1 ) //robustness for row vectors
				mc.setRows(idright.getDim1());
			if( mc.getCols() < 0 && idright.getDim2() > 1 ) //robustness for row vectors
				mc.setCols(idright.getDim2());
			return mc;
		}
		return new MatrixCharacteristics(-1, -1, -1, -1);
	}
	
	///////////////////////////////////////////////////////////////////////////
	// store exception info + position information for expressions
	///////////////////////////////////////////////////////////////////////////
	private String _filename;
	private int _beginLine, _beginColumn;
	private int _endLine, _endColumn;
	private String _text;
	private ArrayList<String> _parseExceptionList = new ArrayList<>();
	
	@Override
	public void setFilename(String passed)  { _filename = passed;   }
	@Override
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	@Override
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	@Override
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	@Override
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	@Override
	public void setText(String text) { _text = text; }
	
	public void setParseExceptionList(ArrayList<String> passed) { _parseExceptionList = passed;}

	/**
	 * Set parse information.
	 *
	 * @param parseInfo
	 *            parse information, such as beginning line position, beginning
	 *            column position, ending line position, ending column position,
	 *            text, and filename
	 */
	public void setParseInfo(ParseInfo parseInfo) {
		_beginLine = parseInfo.getBeginLine();
		_beginColumn = parseInfo.getBeginColumn();
		_endLine = parseInfo.getEndLine();
		_endColumn = parseInfo.getEndColumn();
		_text = parseInfo.getText();
		_filename = parseInfo.getFilename();
	}

	/**
	 * Set ParserRuleContext values (begin line, begin column, end line, end
	 * column, and text).
	 *
	 * @param ctx
	 *            the antlr ParserRuleContext
	 */
	public void setCtxValues(ParserRuleContext ctx) {
		setBeginLine(ctx.start.getLine());
		setBeginColumn(ctx.start.getCharPositionInLine());
		setEndLine(ctx.stop.getLine());
		setEndColumn(ctx.stop.getCharPositionInLine());
		// preserve whitespace if possible
		if ((ctx.start != null) && (ctx.stop != null) && (ctx.start.getStartIndex() != -1)
				&& (ctx.stop.getStopIndex() != -1) && (ctx.start.getStartIndex() <= ctx.stop.getStopIndex())
				&& (ctx.start.getInputStream() != null)) {
			String text = ctx.start.getInputStream()
					.getText(Interval.of(ctx.start.getStartIndex(), ctx.stop.getStopIndex()));
			if (text != null) {
				text = text.trim();
			}
			setText(text);
		} else {
			String text = ctx.getText();
			if (text != null) {
				text = text.trim();
			}
			setText(text);
		}
	}

	/**
	 * Set ParserRuleContext values (begin line, begin column, end line, end
	 * column, and text) and file name.
	 *
	 * @param ctx
	 *            the antlr ParserRuleContext
	 * @param filename
	 *            the filename (if it exists)
	 */
	public void setCtxValuesAndFilename(ParserRuleContext ctx, String filename) {
		setCtxValues(ctx);
		setFilename(filename);
	}

	@Override
	public String getFilename()	{ return _filename;   }
	@Override
	public int getBeginLine()	{ return _beginLine;   }
	@Override
	public int getBeginColumn() { return _beginColumn; }
	@Override
	public int getEndLine() 	{ return _endLine;   }
	@Override
	public int getEndColumn()	{ return _endColumn; }
	@Override
	public String getText() { return _text; }
	
	public ArrayList<String> getParseExceptionList() { return _parseExceptionList; }

	public String printErrorLocation() {
		String file = _filename;
		if (file == null) {
			file = "";
		} else {
			file = file + " ";
		}
		if (getText() != null) {
			return "ERROR: " + file + "[line " + _beginLine + ":" + _beginColumn + "] -> " + getText() + " -- ";
		} else {
			return "ERROR: " + file + "[line " + _beginLine + ":" + _beginColumn + "] -- ";
		}
	}

	public String printWarningLocation() {
		String file = _filename;
		if (file == null) {
			file = "";
		} else {
			file = file + " ";
		}
		if (getText() != null) {
			return "WARNING: " + file + "[line " + _beginLine + ":" + _beginColumn + "] -> " + getText() + " -- ";
		} else {
			return "WARNING: " + file + "[line " + _beginLine + ":" + _beginColumn + "] -- ";
		}
	}

	/**
	 * Return info message containing the filename, the beginning line position, and the beginning column position.
	 * 
	 * @return the info message
	 */
	public String printInfoLocation(){
		return "INFO: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
}
