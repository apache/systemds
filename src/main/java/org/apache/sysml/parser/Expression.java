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

package org.apache.sysml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.hops.Hop.FileFormatTypes;


public abstract class Expression 
{
	/**
	 * The kind of expression. Can be an operator (unary operator, binary operator, boolean operator, built-in function operator,
	 * parameterized built-in function operator, data operator, relational operator, external built-in function operator, function call operator), data, or literal.
	 */
	public enum Kind {
		UnaryOp, BinaryOp, BooleanOp, BuiltinFunctionOp, ParameterizedBuiltinFunctionOp, DataOp, Data, Literal, RelationalOp, ExtBuiltinFunctionOp, FunctionCallOp
	};

	/**
	 * Binary operators.
	 */
	public enum BinaryOp {
		PLUS, MINUS, MULT, DIV, MODULUS, INTDIV, MATMULT, POW, INVALID
	};

	/**
	 * Relational operators.
	 */
	public enum RelationalOp {
		LESSEQUAL, LESS, GREATEREQUAL, GREATER, EQUAL, NOTEQUAL, INVALID
	};

	/**
	 * Boolean operators.
	 */
	public enum BooleanOp {
		CONDITIONALAND, CONDITIONALOR, LOGICALAND, LOGICALOR, NOT, INVALID
	};

	/**
	 * Built-in function operators.
	 */
	public enum BuiltinFunctionOp { 
		ABS,
		ACOS,
		ASIN,
		ATAN,
		AVG,
		CAST_AS_BOOLEAN,
		CAST_AS_DOUBLE,
		CAST_AS_FRAME,
		CAST_AS_INT,
		CAST_AS_MATRIX,
		CAST_AS_SCALAR,
		CBIND, //previously APPEND
		CEIL,
		CHOLESKY,
		COLMAX,
		COLMEAN,
		COLMIN,
		COLSD,
		COLSUM,
		COLVAR,
		COS,
		COV,
		CUMMAX,
		CUMMIN,
		CUMPROD,
		CUMSUM,
		DIAG,
		EIGEN,
		EXP,
		FLOOR,
		INTERQUANTILE,
		INVERSE,
		IQM,
		LENGTH, 
		LOG,
		LU,
		MAX,
		MEAN,
		MEDIAN,
		MIN,
		MOMENT, 
		NCOL, 
		NROW,
		OUTER,
		PPRED, 
		PROD,
		QR,
		QUANTILE,
		RANGE,
		RBIND,
		REV,
		ROUND,
		ROWINDEXMAX,
		ROWINDEXMIN,
		ROWMAX,
		ROWMEAN, 
		ROWMIN,
		ROWSD,
		ROWSUM,
		ROWVAR,
		SAMPLE,
		SD,
		SEQ,
		SIN,
		SIGN,
		SOLVE,
		SQRT,
		SUM,
		TABLE,
		TAN,
		TRACE, 
		TRANS,
		VAR
	};

	/**
	 * Parameterized built-in function operators.
	 */
	public enum ParameterizedBuiltinFunctionOp {
		GROUPEDAGG, RMEMPTY, REPLACE, ORDER, 
		// Distribution Functions
		CDF, INVCDF, PNORM, QNORM, PT, QT, PF, QF, PCHISQ, QCHISQ, PEXP, QEXP,
		TRANSFORM, TRANSFORMAPPLY, TRANSFORMDECODE, TRANSFORMENCODE, TRANSFORMMETA,
		CAST_AS_STRING,	// The "toString" method for DML; named arguments accepted to format output
		INVALID
	};
	
	/**
	 * Data operators.
	 */
	public enum DataOp {
		READ, WRITE, RAND, MATRIX, INVALID	
	}

	/**
	 * Function call operators.
	 */
	public enum FunctCallOp {
		INTERNAL, EXTERNAL
	};
	
	/**
	 * External built-in function operators.
	 */
	public enum ExtBuiltinFunctionOp {
		EIGEN, CHOLESKY
	};

	/**
	 * Data types (matrix, scalar, frame, object, unknown).
	 */
	public enum DataType {
		MATRIX, SCALAR, FRAME, OBJECT, UNKNOWN;
		
		public boolean isMatrix() {
			return (this == MATRIX);
		}
		public boolean isScalar() {
			return (this == SCALAR);
		}
	};

	/**
	 * Value types (int, double, string, boolean, object, unknown).
	 */
	public enum ValueType {
		INT, DOUBLE, STRING, BOOLEAN, OBJECT, UNKNOWN
	};

	/**
	 * Format types (text, binary, matrix market, csv, unknown).
	 */
	public enum FormatType {
		TEXT, BINARY, MM, CSV, UNKNOWN
	};
	
	protected static final Log LOG = LogFactory.getLog(Expression.class.getName());

	public abstract Expression rewriteExpression(String prefix) throws LanguageException;
		
	
	protected Kind _kind;
	protected Identifier[] _outputs;

	private static int _tempId;

	public Expression() {
		_outputs = null;
	}

	public void setOutput(Identifier output) {
		if ( _outputs == null) {
			_outputs = new Identifier[1];
		}
		_outputs[0] = output;
	}

	public Kind getKind() {
		return _kind;
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
	
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars, boolean conditional) 
		throws LanguageException 
	{
		raiseValidateError("Should never be invoked in Baseclass 'Expression'", false);
	}
	
	public void validateExpression(MultiAssignmentStatement mas, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars, boolean conditional) 
		throws LanguageException 
	{
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
		if (val.equalsIgnoreCase("+"))
			return BinaryOp.PLUS;
		else if (val.equalsIgnoreCase("-"))
			return BinaryOp.MINUS;
		else if (val.equalsIgnoreCase("*"))
			return BinaryOp.MULT;
		else if (val.equalsIgnoreCase("/"))
			return BinaryOp.DIV;
		else if (val.equalsIgnoreCase("%%"))
			return BinaryOp.MODULUS;
		else if (val.equalsIgnoreCase("%/%"))
			return BinaryOp.INTDIV;
		else if (val.equalsIgnoreCase("^"))
			return BinaryOp.POW;
		else if (val.equalsIgnoreCase("%*%"))
			return BinaryOp.MATMULT;
		return BinaryOp.INVALID;
	}

	/**
	 * Convert string value to relational operator.
	 * 
	 * @param val String value ('&lt;', '&lt=', '&gt;', '&gt;=', '==', '!=')
	 * @return Relational operator ({@code RelationalOp.LESS}, {@code RelationalOp.LESSEQUAL}, 
	 * {@code RelationalOp.GREATER}, {@code RelationalOp.GREATEREQUAL}, {@code RelationalOp.EQUAL}, 
	 * {@code RelationalOp.NOTEQUAL}).
	 * Returns {@code RelationalOp.INVALID} if string value not recognized.
	 */
	public static RelationalOp getRelationalOp(String val) {
		if (val == null) 
			return null;
		else if (val.equalsIgnoreCase("<"))
			return RelationalOp.LESS;
		else if (val.equalsIgnoreCase("<="))
			return RelationalOp.LESSEQUAL;
		else if (val.equalsIgnoreCase(">"))
			return RelationalOp.GREATER;
		else if (val.equalsIgnoreCase(">="))
			return RelationalOp.GREATEREQUAL;
		else if (val.equalsIgnoreCase("=="))
			return RelationalOp.EQUAL;
		else if (val.equalsIgnoreCase("!="))
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
		if (val.equalsIgnoreCase("&&"))
			return BooleanOp.CONDITIONALAND;
		else if (val.equalsIgnoreCase("&"))
			return BooleanOp.LOGICALAND;
		else if (val.equalsIgnoreCase("||"))
			return BooleanOp.CONDITIONALOR;
		else if (val.equalsIgnoreCase("|"))
			return BooleanOp.LOGICALOR;
		else if (val.equalsIgnoreCase("!"))
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
	public static FileFormatTypes convertFormatType(String format) {
		if (format == null)
			return FileFormatTypes.TEXT;
		if (format.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_TEXT)) {
			return FileFormatTypes.TEXT;
		}
		if (format.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_BINARY)) {
			return FileFormatTypes.BINARY;
		}
		if (format.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET))  {
			return FileFormatTypes.MM;
		}
		if (format.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_CSV))  {
			return FileFormatTypes.CSV;
		}
		// ToDo : throw parse exception for invalid / unsupported format type
		return FileFormatTypes.TEXT;
	}
    
	/**
	 * Obtain temporary name ("parsertemp" + _tempId) for expression. Used to construct Hops from
	 * parse tree.
	 * 
	 * @return Temporary name of expression.
	 */
	public static String getTempName() {
		return "parsertemp" + _tempId++;
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
	 * @throws LanguageException
	 */
	public static DataType computeDataType(Expression expression1, Expression expression2, boolean cast) throws LanguageException {
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
	 * @throws LanguageException
	 */
	public static DataType computeDataType(Identifier identifier1, Identifier identifier2, boolean cast) throws LanguageException {
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
	 * @throws LanguageException
	 */
	public static ValueType computeValueType(Expression expression1, Expression expression2, boolean cast) throws LanguageException {
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
	 * @throws LanguageException
	 */
	public static ValueType computeValueType(Identifier identifier1, Identifier identifier2, boolean cast) throws LanguageException {
		ValueType v1 = identifier1.getValueType();
		ValueType v2 = identifier2.getValueType();

		if (v1 == v2)
			return v1;

		if (cast) {
			if (v1 == ValueType.DOUBLE && v2 == ValueType.INT)
				return ValueType.DOUBLE;
			if (v2 == ValueType.DOUBLE && v1 == ValueType.INT)
				return ValueType.DOUBLE;
			
			// String value type will override others
			// Primary operation involving strings is concatenation (+)
			if ( v1 == ValueType.STRING || v2 == ValueType.STRING )
				return ValueType.STRING;
		}

		//raise error with id1 location
		identifier1.raiseValidateError("Invalid Valuetypes for operation "+v1+" "+v2, false, 
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
	 * @throws LanguageException
	 */
	public void raiseValidateError( String message ) throws LanguageException {
		raiseValidateError(message, false, null);
	}
	
	/**
	 * Throw a LanguageException with the message if conditional is {@code false};
	 * otherwise log the message as a warning.
	 * 
	 * @param message the error (or warning) message
	 * @param conditional if {@code true}, display log warning message. Otherwise, the message
	 * will be thrown as a LanguageException
	 * @throws LanguageException thrown if conditional is {@code false}.
	 */
	public void raiseValidateError( String message, boolean conditional ) throws LanguageException {
		raiseValidateError(message, conditional, null);
	}
	
	/**
	 * Throw a LanguageException with the message (and optional error code) if conditional is {@code false};
	 * otherwise log the message as a warning.
	 * 
	 * @param message the error (or warning) message
	 * @param conditional if {@code true}, display log warning message. Otherwise, the message (and optional
	 * error code) will be thrown as a LanguageException
	 * @param errorCode optional error code
	 * @throws LanguageException thrown if conditional is {@code false}.
	 */
	public void raiseValidateError( String message, boolean conditional, String errorCode ) 
		throws LanguageException
	{
		if( conditional )  //warning if conditional
		{
			String fullMsg = this.printWarningLocation() + message;
			
			LOG.warn( fullMsg );
		}
		else  //error and exception if unconditional
		{
			String fullMsg = this.printErrorLocation() + message;
			
			//LOG.error( fullMsg ); //no redundant error			
			if( errorCode != null )
				throw new LanguageException( fullMsg, errorCode );
			else 
				throw new LanguageException( fullMsg );
		}
	}
	
	
	/**
	 * Returns the matrix characteristics for scalar-scalar, scalar-matrix, matrix-scalar, matrix-matrix
	 * operations. This method is aware of potentially unknowns and matrix-vector (col/row) operations.
	 * 
	 * 
	 * @param expression1 The first expression
	 * @param expression2 The second expression
	 * @return long array of 4 values, where [0] is the number of rows (rlen),
	 * [1] is the number of columns (clen), [2] is the number of rows in a block (brlen),
	 * and [3] is the number of columns in a block (bclen). Default (unknown) values are
	 * -1. Scalar values are all 0.
	 */
	public static long[] getBinaryMatrixCharacteristics(Expression expression1, Expression expression2)
	{
		long[] ret = new long[]{ -1, -1, -1, -1 };
		
		Identifier idleft = expression1.getOutput();
		Identifier idright = expression2.getOutput();
		
		if( idleft.getDataType()==DataType.SCALAR && idright.getDataType()==DataType.SCALAR ) {
			ret[0] = 0; 
			ret[1] = 0; 
			ret[2] = 0; 
			ret[3] = 0; 
		}
		else if( idleft.getDataType()==DataType.SCALAR && idright.getDataType()==DataType.MATRIX ) {
			ret[0] = idright.getDim1(); 
			ret[1] = idright.getDim2(); 
			ret[2] = idright.getRowsInBlock(); 
			ret[3] = idright.getColumnsInBlock();
		}
		else if( idleft.getDataType()==DataType.MATRIX && idright.getDataType()==DataType.SCALAR ) {
			ret[0] = idleft.getDim1(); 
			ret[1] = idleft.getDim2(); 
			ret[2] = idleft.getRowsInBlock(); 
			ret[3] = idleft.getColumnsInBlock();
		}
		else if( idleft.getDataType()==DataType.MATRIX && idright.getDataType()==DataType.MATRIX ) {
			ret[0] = idleft.getDim1(); 
			ret[1] = idleft.getDim2(); 
			ret[2] = idleft.getRowsInBlock(); 
			ret[3] = idleft.getColumnsInBlock();
			if( ret[0] < 0 && idright.getDim1() > 1 ) //robustness for row vectors
				ret[0] = idright.getDim1();
			if( ret[1] < 0 && idright.getDim2() > 1 ) //robustness for row vectors
				ret[1] = idright.getDim2();
		}
		
		return ret;
	}
	
	///////////////////////////////////////////////////////////////////////////
	// store exception info + position information for expressions
	///////////////////////////////////////////////////////////////////////////
	private String _filename;
	private int _beginLine, _beginColumn;
	private int _endLine, _endColumn;
	private ArrayList<String> _parseExceptionList = new ArrayList<String>();
	
	public void setFilename(String passed)  { _filename = passed;   }
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	public void setParseExceptionList(ArrayList<String> passed) { _parseExceptionList = passed;}
	
	/**
	 * Set the filename, the beginning line/column positions, and the ending line/column positions.
	 * 
	 * @param filename The DML/PYDML filename (if it exists)
	 * @param blp Beginning line position
	 * @param bcp Beginning column position
	 * @param elp Ending line position
	 * @param ecp Ending column position
	 */
	public void setAllPositions(String filename, int blp, int bcp, int elp, int ecp){
		_filename    = filename;
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public String getFilename()	{ return _filename;   }
	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	public ArrayList<String> getParseExceptionList() { return _parseExceptionList; }
	
	/**
	 * Return error message containing the filename, the beginning line position, and the beginning column position.
	 * 
	 * @return the error message
	 */
	public String printErrorLocation(){
		return "ERROR: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	/**
	 * Return warning message containing the filename, the beginning line position, and the beginning column position.
	 * 
	 * @return the warning message
	 */
	public String printWarningLocation(){
		return "WARNING: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
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
