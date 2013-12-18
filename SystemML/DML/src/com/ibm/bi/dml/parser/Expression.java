/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.Hop.FileFormatTypes;


public abstract class Expression 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum Kind {
		UnaryOp, BinaryOp, BooleanOp, BuiltinFunctionOp, ParameterizedBuiltinFunctionOp, DataOp, Data, Literal, RelationalOp, ExtBuiltinFunctionOp, FunctionCallOp
	};

	public enum BinaryOp {
		PLUS, MINUS, MULT, DIV, MODULUS, MATMULT, POW, INVALID
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
		CAST_AS_SCALAR, 
		CENTRALMOMENT, 
		COLMEAN,
		COLMAX,
		COLMIN, 
		COLSUM,
		COS,
		COVARIANCE, 
		CTABLE,
		DIAG,
		EXP,
		INTERQUANTILE, 
		IQM, 
		LENGTH, 
		LOG, 
		MAX,
		MEAN,
		MIN, 
		NCOL, 
		NROW,
		PMIN, 
		PMAX, 
		PPRED, 
		PROD,
		QUANTILE,
		RANGE,
		ROUND,
		ROWINDEXMAX, 
		ROWMAX,
		ROWMEAN, 
		ROWMIN, 
		ROWSUM, 
		SEQ,
		SIN, 
		SQRT,
		SUM, 
		TAN,
		TRACE, 
		TRANS,
		QR,
		LU
	};

	public enum ParameterizedBuiltinFunctionOp {
		CDF, GROUPEDAGG, RMEMPTY, INVALID
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
		MATRIX, SCALAR, OBJECT, UNKNOWN
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
		return _outputs[0];
	}
	
	public Identifier[] getOutputs() {
		return _outputs;
	}
	
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars) throws LanguageException {
		throw new LanguageException(this.printErrorLocation() + "Should never be invoked in Baseclass 'Expression'");
	};
	
	public void validateExpression(MultiAssignmentStatement mas, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars) throws LanguageException {
		LOG.error(this.printErrorLocation() + "Should never be invoked in Baseclass 'Expression'");
		throw new LanguageException(this.printErrorLocation() + "Should never be invoked in Baseclass 'Expression'");
	};
	
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
		if (fn.equalsIgnoreCase(Statement.FORMAT_TYPE_VALUE_TEXT)) {
			return FileFormatTypes.TEXT;
		}
		if (fn.equalsIgnoreCase(Statement.FORMAT_TYPE_VALUE_BINARY)) {
			return FileFormatTypes.BINARY;
		}
		if (fn.equalsIgnoreCase(Statement.FORMAT_TYPE_VALUE_MATRIXMARKET))  {
			return FileFormatTypes.MM;
		}
		if (fn.equalsIgnoreCase(Statement.FORMAT_TYPE_VALUE_CSV))  {
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

		throw new LanguageException(id1.printErrorLocation() + "Invalid Valuetypes for operation",
				LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
	}

	
	///////////////////////////////////////////////////////////////////////////
	// store exception info + position information for expressions
	///////////////////////////////////////////////////////////////////////////
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	private ArrayList<String> _parseExceptionList = new ArrayList<String>();
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	public void setParseExceptionList(ArrayList<String> passed) { _parseExceptionList = passed;}
	
	public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	public ArrayList<String> getParseExceptionList() { return _parseExceptionList; }
	
	public String printErrorLocation(){
		return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printInfoLocation(){
		return "INFO: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
}
