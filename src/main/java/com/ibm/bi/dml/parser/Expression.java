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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.Hop.FileFormatTypes;


public abstract class Expression 
{
	
	public enum Kind {
		UnaryOp, BinaryOp, BooleanOp, BuiltinFunctionOp, ParameterizedBuiltinFunctionOp, DataOp, Data, Literal, RelationalOp, ExtBuiltinFunctionOp, FunctionCallOp
	};

	public enum BinaryOp {
		PLUS, MINUS, MULT, DIV, MODULUS, INTDIV, MATMULT, POW, INVALID
	};

	public enum RelationalOp {
		LESSEQUAL, LESS, GREATEREQUAL, GREATER, EQUAL, NOTEQUAL, INVALID
	};

	public enum BooleanOp {
		CONDITIONALAND, CONDITIONALOR, LOGICALAND, LOGICALOR, NOT, INVALID
	};

	public enum BuiltinFunctionOp {
		APPEND, 
		ABS, 
		ACOS,
		ASIN, 
		ATAN,
		AVG,
		CAST_AS_MATRIX, 
		CAST_AS_SCALAR,
		CAST_AS_DOUBLE, 
		CAST_AS_INT,
		CAST_AS_BOOLEAN,
		COLMEAN,
		COLMAX,
		COLMIN, 
		COLSUM,
		COS,
		COV, 
		CUMMAX,
		CUMMIN,
		CUMPROD,
		CUMSUM,
		DIAG,
		EXP,
		INTERQUANTILE, 
		IQM, 
		LENGTH, 
		LOG, 
		MAX,
		MEAN,
		MIN, 
		MOMENT, 
		NCOL, 
		NROW,
		OUTER,
		PPRED, 
		PROD,
		QUANTILE,
		RANGE,
		ROUND,
		ROWINDEXMAX, 
		ROWMAX,
		ROWMEAN, 
		ROWMIN,
		ROWINDEXMIN,
		ROWSUM, 
		SEQ,
		SIN, 
		SQRT,
		SUM, 
		TABLE,
		TAN,
		TRACE, 
		TRANS,
		QR,
		LU,
		EIGEN,
		SOLVE,
		CEIL,
		FLOOR,
		MEDIAN,
		INVERSE,
		SAMPLE
	};

	public enum ParameterizedBuiltinFunctionOp {
		GROUPEDAGG, RMEMPTY, REPLACE, ORDER, 
		// Distribution Functions
		CDF, INVCDF, PNORM, QNORM, PT, QT, PF, QF, PCHISQ, QCHISQ, PEXP, QEXP,
		TRANSFORM, 
		INVALID
	};
	
	public enum DataOp {
		READ, WRITE, RAND, MATRIX, INVALID	
	}

	public enum FunctCallOp {
		INTERNAL, EXTERNAL
	};
	
	public enum ExtBuiltinFunctionOp {
		EIGEN, CHOLESKY
	};

	public enum AggOp {
		SUM, MIN, MAX, INVALID
	};

	public enum ReorgOp {
		TRANSPOSE, DIAG
	};

	//public enum DataOp {
	//	READ, WRITE
	//};

	public enum DataType {
		MATRIX, SCALAR, FRAME, OBJECT, UNKNOWN
	};

	public enum ValueType {
		INT, DOUBLE, STRING, BOOLEAN, OBJECT, UNKNOWN
	};

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

	public Identifier getOutput() {
		if (_outputs != null && _outputs.length > 0)
			return _outputs[0];
		else
			return null;
	}
	
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
	 * Convert format types from parser to Hops enum : default is text
	 */
	
	public static FileFormatTypes convertFormatType(String fn) {
		if (fn == null)
			return FileFormatTypes.TEXT;
		if (fn.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_TEXT)) {
			return FileFormatTypes.TEXT;
		}
		if (fn.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_BINARY)) {
			return FileFormatTypes.BINARY;
		}
		if (fn.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET))  {
			return FileFormatTypes.MM;
		}
		if (fn.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_CSV))  {
			return FileFormatTypes.CSV;
		}
		// ToDo : throw parse exception for invalid / unsupported format type
		return FileFormatTypes.TEXT;
	}
    
	/**
	 * Construct Hops from parse tree : Create temporary views in expressions
	 */
	public static String getTempName() {
		return "parsertemp" + _tempId++;
	}

	public abstract VariableSet variablesRead();

	public abstract VariableSet variablesUpdated();

	public static DataType computeDataType(Expression e1, Expression e2, boolean cast) throws LanguageException {
		return computeDataType(e1.getOutput(), e2.getOutput(), cast);
	}

	public static DataType computeDataType(Identifier id1, Identifier id2, boolean cast) throws LanguageException {
		DataType d1 = id1.getDataType();
		DataType d2 = id2.getDataType();

		if (d1 == d2)
			return d1;

		if (cast) {
			if (d1 == DataType.MATRIX && d2 == DataType.SCALAR)
				return DataType.MATRIX;
			if (d1 == DataType.SCALAR && d2 == DataType.MATRIX)
				return DataType.MATRIX;
		}

		LOG.error(id1.printErrorLocation() + "Invalid Datatypes for operation");
		
		throw new LanguageException(id1.printErrorLocation() + "Invalid Datatypes for operation",
				LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
	}

	public static ValueType computeValueType(Expression e1, Expression e2, boolean cast) throws LanguageException {
		return computeValueType(e1.getOutput(), e2.getOutput(), cast);
	}

	public static ValueType computeValueType(Identifier id1, Identifier id2, boolean cast) throws LanguageException {
		ValueType v1 = id1.getValueType();
		ValueType v2 = id2.getValueType();

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

		LOG.error(id1.printErrorLocation() + "Invalid Valuetypes for operation");
		
		throw new LanguageException(id1.printErrorLocation() + "Invalid Valuetypes for operation",
				LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
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
	 * 
	 * @param msg
	 * @param conditional
	 * @throws LanguageException
	 */
	public void raiseValidateError( String msg, boolean conditional ) 
		throws LanguageException
	{
		raiseValidateError(msg, conditional, null);
	}
	
	/**
	 * 
	 * @param msg
	 * @param conditional
	 * @param code
	 * @throws LanguageException
	 */
	public void raiseValidateError( String msg, boolean conditional, String errorCode ) 
		throws LanguageException
	{
		if( conditional )  //warning if conditional
		{
			String fullMsg = this.printWarningLocation() + msg;
			
			LOG.warn( fullMsg );
		}
		else  //error and exception if unconditional
		{
			String fullMsg = this.printErrorLocation() + msg;
			
			LOG.error( fullMsg );			
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
	 * Format: rlen, clen, brlen, bclen.
	 * 
	 * @param left
	 * @param right
	 * @return
	 */
	public static long[] getBinaryMatrixCharacteristics( Expression left, Expression right )
	{
		long[] ret = new long[]{ -1, -1, -1, -1 };
		
		Identifier idleft = left.getOutput();
		Identifier idright = right.getOutput();
		
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
	
	public String printErrorLocation(){
		return "ERROR: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printErrorLocation(int beginLine, int beginColumn){
		return "ERROR: " + _filename + " -- line " + beginLine + ", column " + beginColumn + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printInfoLocation(){
		return "INFO: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
}
