/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.json.java.JSONObject;


public class DataExpression extends DataIdentifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String RAND_ROWS 	=  "rows";	 
	public static final String RAND_COLS 	=  "cols";
	public static final String RAND_MIN  	=  "min";
	public static final String RAND_MAX  	=  "max";
	public static final String RAND_SPARSITY = "sparsity"; 
	public static final String RAND_SEED    =  "seed";
	public static final String RAND_PDF		=  "pdf";
	
	public static final String RAND_PDF_UNIFORM = "uniform";
	
	public static final String RAND_BY_ROW 	 =  "byrow";	 
	public static final String RAND_DIMNAMES =  "dimnames";
	public static final String RAND_DATA 	 =  "data";
	
	public static final String IO_FILENAME = "iofilename";
	public static final String READROWPARAM = "rows";
	public static final String READCOLPARAM = "cols";
	public static final String READNUMNONZEROPARAM = "nnz";
	
	public static final String FORMAT_TYPE 						= "format";
	public static final String FORMAT_TYPE_VALUE_TEXT 			= "text";
	public static final String FORMAT_TYPE_VALUE_BINARY 		= "binary";
	public static final String FORMAT_TYPE_VALUE_CSV			= "csv";
	public static final String FORMAT_TYPE_VALUE_MATRIXMARKET	= "mm";
	
	public static final String ROWBLOCKCOUNTPARAM = "rows_in_block";
	public static final String COLUMNBLOCKCOUNTPARAM = "cols_in_block";
	public static final String DATATYPEPARAM = "data_type";
	public static final String VALUETYPEPARAM = "value_type";
	public static final String DESCRIPTIONPARAM = "description"; 
	
	public static final String DELIM_DELIMITER = "sep";
	public static final String DELIM_HAS_HEADER_ROW = "header";
	public static final String DELIM_FILL = "fill";
	public static final String DELIM_FILL_VALUE = "default";
	public static final String DELIM_SPARSE = "sparse";
	
	public static final String[] RAND_VALID_PARAM_NAMES = 
		{ RAND_ROWS, RAND_COLS, RAND_MIN, RAND_MAX, RAND_SPARSITY, RAND_SEED, RAND_PDF}; 
	
	public static final String[] MATRIX_VALID_PARAM_NAMES = 
		{  RAND_BY_ROW, RAND_DIMNAMES, RAND_DATA, RAND_ROWS, RAND_COLS};
	
	public static final String[] READ_VALID_MTD_PARAM_NAMES = 
		{ IO_FILENAME, READROWPARAM, READCOLPARAM, READNUMNONZEROPARAM, FORMAT_TYPE, 
			ROWBLOCKCOUNTPARAM, COLUMNBLOCKCOUNTPARAM, DATATYPEPARAM, VALUETYPEPARAM, DESCRIPTIONPARAM }; 

	public static final String[] READ_VALID_PARAM_NAMES = 
	{	IO_FILENAME, READROWPARAM, READCOLPARAM, FORMAT_TYPE, DATATYPEPARAM, VALUETYPEPARAM,
			DELIM_FILL_VALUE, DELIM_DELIMITER, DELIM_FILL, DELIM_HAS_HEADER_ROW }; 
		
	/* Default Values for delimited (CSV) files */
	public static final String  DEFAULT_DELIM_DELIMITER = ",";
	public static final boolean DEFAULT_DELIM_HAS_HEADER_ROW = false;
	public static final boolean DEFAULT_DELIM_FILL = true;
	public static final double  DEFAULT_DELIM_FILL_VALUE = 0.0;
	public static final boolean DEFAULT_DELIM_SPARSE = true;
	
	private DataOp _opcode;
	private HashMap<String, Expression> _varParams;
	
		
	public static DataExpression getDataExpression(String functionName, ArrayList<ParameterExpression> passedParamExprs, int beginLine, int beginColumn) throws DMLParseException {
		
		if (functionName == null || passedParamExprs == null)
			return null;
		
		// check if the function name is built-in function
		//	 (assign built-in function op if function is built-in)
		Expression.DataOp dop = null;
		DataExpression dataExpr = null;
		if (functionName.equals("read") || functionName.equals("readMM") || functionName.equals("read.csv")){
			dop = Expression.DataOp.READ;
			dataExpr = new DataExpression(dop, new HashMap<String,Expression>());
			
			if (functionName.equals("readMM"))
				dataExpr.addVarParam(DataExpression.FORMAT_TYPE,  new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET));
			
			if (functionName.equals("read.csv"))
				dataExpr.addVarParam(DataExpression.FORMAT_TYPE,  new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_CSV));
			
			// validate the filename is the first parameter
			if (passedParamExprs.size() < 1){
				// throw exception -- must be filename as first parameter
				throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "read method must have at least filename parameter");
			}
			
			ParameterExpression pexpr = (passedParamExprs.size() == 0) ? null : passedParamExprs.get(0);
			
			if ( (pexpr != null) &&  (!(pexpr.getName() == null) || (pexpr.getName() != null && pexpr.getName().equalsIgnoreCase(DataExpression.IO_FILENAME)))){
				throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "first parameter to read statement must be filename");
			} else {
				dataExpr.addVarParam(DataExpression.IO_FILENAME, pexpr.getExpr());
			}
			
			// validate all parameters are added only once and valid name
			for (int i = 1; i < passedParamExprs.size(); i++){
				String currName = passedParamExprs.get(i).getName();
				Expression currExpr = passedParamExprs.get(i).getExpr();
				
				if (dataExpr.getVarParam(currName) != null){
					throw new DMLParseException(dataExpr.getFilename(), currExpr.printErrorLocation() + "attempted to add IOStatement parameter " + currName + " more than once");
				}
				// verify parameter names for read function
				boolean isValidName = false;
				for (String paramName : READ_VALID_PARAM_NAMES){
					if (paramName.equals(currName))
						isValidName = true;
				}
				if (!isValidName){
					throw new DMLParseException(dataExpr.getFilename(), currExpr.printErrorLocation() + "attempted to add invalid read statement parameter " + currName);
				}	
				dataExpr.addVarParam(currName, currExpr);
			}				
		}
		
		else if (functionName.equalsIgnoreCase("rand")){
			
			dop = Expression.DataOp.RAND;
			dataExpr = new DataExpression(dop, new HashMap<String,Expression>());
			
			for (ParameterExpression currExpr : passedParamExprs){
				String pname = currExpr.getName();
				Expression pexpr = currExpr.getExpr();
				if (pname == null){
					throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "for Rand Statment all arguments must be named parameters");	
					//LOG.error(dataExpr.printErrorLocation(beginLine, beginColumn) + "for Rand Statment all arguments must be named parameters");
					//throw new ParseException(dataExpr.printErrorLocation(beginLine, beginColumn) + "for Rand Statment all arguments must be named parameters");	
				}
				dataExpr.addRandExprParam(pname, pexpr); 
			}
			dataExpr.setRandDefault();
		}
		
		else if (functionName.equals("matrix")){
			dop = Expression.DataOp.MATRIX;
			dataExpr = new DataExpression(dop, new HashMap<String,Expression>());
		
			int namedParamCount = 0, unnamedParamCount = 0;
			for (ParameterExpression currExpr : passedParamExprs) {
				if (currExpr.getName() == null)
					unnamedParamCount++;
				else
					namedParamCount++;
			}

			// check whether named or unnamed parameters are used
			if (passedParamExprs.size() < 3){
				throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "for matrix statement, must specify at least 3 arguments (in order): data, rows, cols");
			}
			if (unnamedParamCount > 1){
				
				if (namedParamCount > 0)
					throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "for matrix statement, cannot mix named and unnamed parameters");
				
				if (unnamedParamCount < 3)
					throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "for matrix statement, must specify at least 3 arguments (in order): data, rows, cols");
				

				// assume: data, rows, cols, [byRow], [dimNames]
				dataExpr.addMatrixExprParam(DataExpression.RAND_DATA,passedParamExprs.get(0).getExpr());
				dataExpr.addMatrixExprParam(DataExpression.RAND_ROWS,passedParamExprs.get(1).getExpr());
				dataExpr.addMatrixExprParam(DataExpression.RAND_COLS,passedParamExprs.get(2).getExpr());
				
				if (unnamedParamCount >= 4)
					dataExpr.addMatrixExprParam(DataExpression.RAND_BY_ROW,passedParamExprs.get(3).getExpr());
				
				if (unnamedParamCount == 5)
					dataExpr.addMatrixExprParam(DataExpression.RAND_DIMNAMES,passedParamExprs.get(4).getExpr());
				
				if (unnamedParamCount > 5)
					throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "for matrix statement, at most 5 arguments supported (in order): data, rows, cols, byrow, dimname");
								   
				
			} else {
				// handle first parameter, which is data and may be unnamed
				ParameterExpression firstParam = passedParamExprs.get(0);
				if (firstParam.getName() != null && !firstParam.getName().equals(DataExpression.RAND_DATA)){
					// throw exception -- must be filename as first parameter
					throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "matrix method must have data parameter as first parameter or unnamed parameter");
				} else {
					dataExpr.addMatrixExprParam(DataExpression.RAND_DATA, passedParamExprs.get(0).getExpr());
				}
				
				for (int i=1; i<passedParamExprs.size(); i++){
					if (passedParamExprs.get(i).getName() == null){
						// throw exception -- cannot mix named and unnamed parameters
						throw new DMLParseException(dataExpr.getFilename(), dataExpr.printErrorLocation(beginLine, beginColumn) + "for matrix statement, cannot mix named and unnamed parameters, only data parameter can be unnammed");
					} else {
						dataExpr.addMatrixExprParam(passedParamExprs.get(i).getName(), passedParamExprs.get(i).getExpr()); 	
					}
				}
			}
			dataExpr.setMatrixDefault();
		} // else if (functionName.equals("matrix")){
		
		return dataExpr;
	
	} // end method getBuiltinFunctionExpression
	
	public void addRandExprParam(String paramName, Expression paramValue) throws DMLParseException
	{
		// check name is valid
		boolean found = false;
		if (paramName != null ){
			for (String name : RAND_VALID_PARAM_NAMES){
				if (name.equals(paramName)) {
					found = true;
					break;
				}			
			}
		}
		if (!found){
			
			throw new DMLParseException(paramValue.getFilename(), paramValue.printErrorLocation() + "unexpected parameter \"" + paramName +
					"\". Legal parameters for Rand statement are " 
					+ "(capitalization-sensitive): " 	+ RAND_ROWS 	
					+ ", " + RAND_COLS		+ ", " + RAND_MIN + ", " + RAND_MAX  	
					+ ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", " + RAND_PDF);
		}
		if (getVarParam(paramName) != null){
			throw new DMLParseException(paramValue.getFilename(), paramValue.printErrorLocation() + "attempted to add Rand statement parameter " + paramValue + " more than once");
		}
		// Process the case where user provides double values to rows or cols
		if (paramName.equals(RAND_ROWS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((int)((DoubleIdentifier)paramValue).getValue());
		}
		else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((int)((DoubleIdentifier)paramValue).getValue());
		}
			
		// add the parameter to expression list
		paramValue.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		addVarParam(paramName,paramValue);
		
	}
	
	public void addMatrixExprParam(String paramName, Expression paramValue) throws DMLParseException
	{
		// check name is valid
		boolean found = false;
		if (paramName != null ){
			for (String name : MATRIX_VALID_PARAM_NAMES){
				if (name.equals(paramName)) {
					found = true;
				}			
			}
		}
		
		if (!found){
			
			throw new DMLParseException(paramValue.getFilename(), paramValue.printErrorLocation() + "unexpected parameter \"" + paramName +
					"\". Legal parameters for  matrix statement are " 
					+ "(capitalization-sensitive): " 	+ RAND_DATA + ", " + RAND_ROWS 	
					+ ", " + RAND_COLS		+ ", " + RAND_BY_ROW);
		}
		if (getVarParam(paramName) != null){
			throw new DMLParseException(paramValue.getFilename(), paramValue.printErrorLocation() + "attempted to add matrix statement parameter " + paramValue + " more than once");
		}
		// Process the case where user provides double values to rows or cols
		if (paramName.equals(RAND_ROWS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((int)((DoubleIdentifier)paramValue).getValue());
		}
		else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((int)((DoubleIdentifier)paramValue).getValue());
		}
			
		// add the parameter to expression list
		paramValue.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		addVarParam(paramName,paramValue);
	}
	
	public DataExpression(DataOp op, HashMap<String,Expression> varParams) {
		_kind = Kind.DataOp;
		_opcode = op;
		_varParams = varParams;
	}

	public Expression rewriteExpression(String prefix) throws LanguageException {
		
		HashMap<String,Expression> newVarParams = new HashMap<String,Expression>();
		for (String key : _varParams.keySet()){
			Expression newExpr = _varParams.get(key).rewriteExpression(prefix);
			newVarParams.put(key, newExpr);
		}	
		DataExpression retVal = new DataExpression(_opcode, newVarParams);
		retVal.setFilename(this.getFilename());
		retVal.setBeginLine(this.getBeginLine());
		retVal.setBeginColumn(this.getBeginColumn());
		retVal.setEndLine(this.getEndLine());
		retVal.setEndColumn(this.getEndColumn());
			
		return retVal;
	}

	/**
	 * By default we use rowwise matrix reshape according to our internal dense/sparse matrix representations.
	 * ByRow specifies both input and output orientation. Note that this is different from R, where inputs are 
	 * always read by-column and the default for byRow is by-column as well.
	 */
	public void setMatrixDefault(){
		if (getVarParam(RAND_BY_ROW) == null)
			addVarParam(RAND_BY_ROW, new BooleanIdentifier(true));
	}
	
	public void setRandDefault(){
		if (getVarParam(RAND_ROWS)== null){
			IntIdentifier id = new IntIdentifier(1L);
			addVarParam(RAND_ROWS, 	id);
		}
		if (getVarParam(RAND_COLS)== null){
			IntIdentifier id = new IntIdentifier(1L);
            addVarParam(RAND_COLS, 	id);
		}
		if (getVarParam(RAND_MIN)== null){
			DoubleIdentifier id = new DoubleIdentifier(0.0);
			addVarParam(RAND_MIN, id);
		}
		if (getVarParam(RAND_MAX)== null){
			DoubleIdentifier id = new DoubleIdentifier(1.0);
			addVarParam(RAND_MAX, id);
		}
		if (getVarParam(RAND_SPARSITY)== null){
			DoubleIdentifier id = new DoubleIdentifier(1.0);
			addVarParam(RAND_SPARSITY,	id);
		}
		if (getVarParam(RAND_SEED)== null){
			IntIdentifier id = new IntIdentifier(DataGenOp.UNSPECIFIED_SEED);
			addVarParam(RAND_SEED, id);
		}
		if (getVarParam(RAND_PDF)== null){
			StringIdentifier id = new StringIdentifier(RAND_PDF_UNIFORM);
			addVarParam(RAND_PDF, id);
		}
	}
	
	
	public void setOpCode(DataOp op) {
		_opcode = op;
	}
	
	public DataOp getOpCode() {
		return _opcode;
	}
	
	public HashMap<String,Expression> getVarParams() {
		return _varParams;
	}
	
	public void setVarParams(HashMap<String, Expression> varParams) {
		_varParams = varParams;
	}
	
	public Expression getVarParam(String name) {
		return _varParams.get(name);
	}

	
	
	public void addVarParam(String name, Expression value){
		_varParams.put(name, value);
		
		// if required, initialize values
		setFilename(value.getFilename());
		if (getBeginLine() == 0) 	setBeginLine(value.getBeginLine());
		if (getBeginColumn() == 0) 	setBeginColumn(value.getBeginColumn());
		if (getEndLine() == 0) 		setEndLine(value.getEndLine());
		if (getEndColumn() == 0) 	setEndColumn(value.getEndColumn());
		
		// update values	
		if (getBeginLine() > value.getBeginLine()){
			setBeginLine(value.getBeginLine());
			setBeginColumn(value.getBeginColumn());
		}
		else if (getBeginLine() == value.getBeginLine() && getBeginColumn() > value.getBeginColumn()){
			setBeginColumn(value.getBeginColumn());
		}

		if (getEndLine() < value.getEndLine()){
			setEndLine(value.getEndLine());
			setEndColumn(value.getEndColumn());
		}
		else if (getEndLine() == value.getEndLine() && getEndColumn() < value.getEndColumn()){
			setEndColumn(value.getEndColumn());
		}		
	}
	
	public void removeVarParam(String name) {
		_varParams.remove(name);
	}
	
	/**
	 * Validate parse tree : Process Data Expression in an assignment
	 * statement
	 *  
	 * @throws LanguageException
	 * @throws ParseException 
	 * @throws IOException 
	 */
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars)
			throws LanguageException {
		
		// validate all input parameters
		Set<String> varParamKeySet = getVarParams().keySet();
		for ( String s : varParamKeySet ) {
			getVarParam(s).validateExpression(ids, currConstVars);
			if ( getVarParam(s).getOutput().getDataType() != DataType.SCALAR && !s.equals(RAND_DATA)) {
				LOG.error(this.printErrorLocation() + "Non-scalar data types are not supported for data expression.");
				throw new LanguageException(this.printErrorLocation() + "Non-scalar data types are not supported for data expression.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			// check if data parameter of matrix is scalar or matrix -- if scalar, use Rand instead
			Expression dataParam = getVarParam(RAND_DATA);
			
			// attempt to perform constant replacement on data param
			if (dataParam != null && dataParam instanceof DataIdentifier && !(dataParam instanceof IndexedIdentifier) 
					&& currConstVars.containsKey(((DataIdentifier) dataParam).getName()))
			{
				addVarParam(RAND_DATA, currConstVars.get(((DataIdentifier)dataParam).getName()));
			}
			
			if (dataParam == null && getOpCode().equals(DataOp.MATRIX)){
				LOG.error(this.printErrorLocation() + "for matrix, must define data parameter");
				throw new LanguageException(this.printErrorLocation() + "for matrix, must defined data parameter", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			if (dataParam != null && dataParam.getOutput().getDataType() == DataType.SCALAR && dataParam instanceof ConstIdentifier){
				 
				// replace DataOp MATRIX with RAND -- Rand handles matrix generation for Scalar values
				// replace data parameter with min / max within Rand case below
				this.setOpCode(DataOp.RAND);
			}		
				
		}	
					
		// IMPORTANT: for each operation, one must handle unnamed parameters
		
		switch (this.getOpCode()) {
		
		case READ:
					
			if (getVarParam(DATATYPEPARAM) != null && !(getVarParam(DATATYPEPARAM) instanceof StringIdentifier)){
				
				LOG.error(this.printErrorLocation() + "for read statement, parameter " + DATATYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.MATRIX_DATA_TYPE +", " + Statement.SCALAR_DATA_TYPE);
				
				throw new LanguageException(this.printErrorLocation() + "for read statement, parameter " + DATATYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.MATRIX_DATA_TYPE +", " + Statement.SCALAR_DATA_TYPE);
			}
			
			String dataTypeString = (getVarParam(DATATYPEPARAM) == null) ? null : getVarParam(DATATYPEPARAM).toString();
			
			// disallow certain parameters while reading a scalar
			if (dataTypeString != null && dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)){
				if ( getVarParam(READROWPARAM) != null
						|| getVarParam(READCOLPARAM) != null
						|| getVarParam(ROWBLOCKCOUNTPARAM) != null
						|| getVarParam(COLUMNBLOCKCOUNTPARAM) != null
						|| getVarParam(FORMAT_TYPE) != null
						|| getVarParam(DELIM_DELIMITER) != null	
						|| getVarParam(DELIM_HAS_HEADER_ROW) != null
						|| getVarParam(DELIM_FILL) != null
						|| getVarParam(DELIM_FILL_VALUE) != null){
					
					LOG.error(this.printErrorLocation() + "Invalid parameters in read statement of a scalar: " +
							toString() + ". Only " + VALUETYPEPARAM + " is allowed.");
					
					throw new LanguageException(this.printErrorLocation() + "Invalid parameters in read statement of a scalar: " +
							toString() + ". Only " + VALUETYPEPARAM + " is allowed.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}
			
			JSONObject configObject = null;	

			// read the configuration file
			String filename = null;
			
			if (getVarParam(IO_FILENAME) instanceof ConstIdentifier){
				filename = getVarParam(IO_FILENAME).toString() +".mtd";
				
			}
			else if (getVarParam(IO_FILENAME) instanceof BinaryExpression){
				BinaryExpression expr = (BinaryExpression)getVarParam(IO_FILENAME);
								
				if (expr.getKind()== Expression.Kind.BinaryOp){
					Expression.BinaryOp op = expr.getOpCode();
					switch (op){
					case PLUS:
							filename = "";
							filename = fileNameCat(expr, currConstVars, filename);
							// Since we have computed the value of filename, we update
							// varParams with a const string value
							StringIdentifier fileString = new StringIdentifier(filename);
							fileString.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							removeVarParam(IO_FILENAME);
							addVarParam(IO_FILENAME, fileString);
							filename = filename + ".mtd";
												
						break;
					default:
						LOG.error(this.printErrorLocation()  + "for read method, parameter " + IO_FILENAME + " can only be const string concatenations. ");
						throw new LanguageException(this.printErrorLocation()  + "for read method, parameter " + IO_FILENAME + " can only be const string concatenations. ");
					}
				}
			}
			else {
				LOG.error(this.printErrorLocation() + "for read method, parameter " + IO_FILENAME + " can only be a const string or const string concatenations. ");
				throw new LanguageException(this.printErrorLocation() + "for read method, parameter " + IO_FILENAME + " can only be a const string or const string concatenations. ");
			}
			
			// track whether should attempt to read MTD file or not
			boolean shouldReadMTD = true;
			
			if ( DMLScript.rtplatform == RUNTIME_PLATFORM.NZ)
				shouldReadMTD = false;
			
			// track whether format type has been inferred 
			boolean inferredFormatType = false;
			
			// get format type string
			String formatTypeString = (getVarParam(FORMAT_TYPE) == null) ? null : getVarParam(FORMAT_TYPE).toString();
			
			// check if file is matrix market format
			if (formatTypeString == null){
				String origFilename = getVarParam(IO_FILENAME).toString();
				boolean isMatrixMarketFormat = checkHasMatrixMarketFormat(origFilename); 
				if (isMatrixMarketFormat){
					
					formatTypeString = FORMAT_TYPE_VALUE_MATRIXMARKET;
					addVarParam(FORMAT_TYPE,new StringIdentifier(FORMAT_TYPE_VALUE_MATRIXMARKET));
					inferredFormatType = true;
					shouldReadMTD = false;
				}
			}
			
			// check if file is delimited format
			if (formatTypeString == null) {
					
				String origFilename = getVarParam(IO_FILENAME).toString();
				boolean isDelimitedFormat = checkHasDelimitedFormat(origFilename); 
				
				if (isDelimitedFormat){
					addVarParam(FORMAT_TYPE,new StringIdentifier(FORMAT_TYPE_VALUE_CSV));
					formatTypeString = FORMAT_TYPE_VALUE_CSV;
					inferredFormatType = true;
					//shouldReadMTD = false;
				}	
			}
			
				
			if (formatTypeString != null && formatTypeString.equalsIgnoreCase(FORMAT_TYPE_VALUE_MATRIXMARKET)){
				/*
				 *  handle MATRIXMARKET_FORMAT_TYPE format
				 *
				 * 1) only allow IO_FILENAME as ONLY valid parameter
				 * 
				 * 2) open the file
				 * 		A) verify header line (1st line) equals 
				 * 		B) read and discard comment lines
				 * 		C) get size information from sizing info line --- M N L
				 */
				
				for (String key : _varParams.keySet()){
					if ( !(key.equals(IO_FILENAME) || key.equals(FORMAT_TYPE) ) ){
						
						LOG.error(this.printErrorLocation() + "Invalid parameters in readMM statement: " +
								toString() + ". Only " + IO_FILENAME + " is allowed.");
						
						throw new LanguageException(this.printErrorLocation() + "Invalid parameters in readMM statement: " +
								toString() + ". Only " + IO_FILENAME + " is allowed.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
				}
				
				
				// should NOT attempt to read MTD file for MatrixMarket format
				shouldReadMTD = false;
				
				// get metadata from MatrixMarket format file
				String[] headerLines = readMatrixMarketFile(getVarParam(IO_FILENAME).toString());
				
				// process 1st line of MatrixMarket format -- must be identical to legal header
				String legalHeaderMM = "%%MatrixMarket matrix coordinate real general";
				
				if (headerLines != null && headerLines.length >= 2){
					String firstLine = headerLines[0].trim();
					if (!firstLine.equals(legalHeaderMM)){
						
						LOG.error(this.printErrorLocation() + "Unsupported format in MatrixMarket file: " +
								headerLines[0] + ". Only supported format in MatrixMarket file has header line " + legalHeaderMM);
						
						throw new LanguageException(this.printErrorLocation() + "Unsupported format in MatrixMarket file: " +
								headerLines[0] + ". Only supported format in MatrixMarket file has header line " + legalHeaderMM, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
				
					// process 2nd line of MatrixMarket format -- must have size information
				
				
					String secondLine = headerLines[1];
					String[] sizeInfo = secondLine.trim().split("\\s+");
					if (sizeInfo.length != 3){
						
						LOG.error(this.printErrorLocation() + "Unsupported size line in MatrixMarket file: " +
								headerLines[1] + ". Only supported format in MatrixMarket file has size line: <NUM ROWS> <NUM COLS> <NUM NON-ZEROS>, where each value is an integer.");
						
						throw new LanguageException(this.printErrorLocation() + "Unsupported size line in MatrixMarket file: " +
								headerLines[1] + ". Only supported format in MatrixMarket file has size line: <NUM ROWS> <NUM COLS> <NUM NON-ZEROS>, where each value is an integer.");
					}
				
					long rowsCount = -1, colsCount = -1, nnzCount = -1;
					try {
						rowsCount = Long.parseLong(sizeInfo[0]);
						if (rowsCount < 1)
							throw new Exception("invalid rows count");
						addVarParam(READROWPARAM, new IntIdentifier(rowsCount));
					}
					catch(Exception e){
						
						LOG.error(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid row count " + sizeInfo[0] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1]);
						
						throw new LanguageException(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid row count " + sizeInfo[0] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1],
								LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
				
					try {
						colsCount = Long.parseLong(sizeInfo[1]);
						if (colsCount < 1)
							throw new Exception("invalid cols count");
						addVarParam(READCOLPARAM, new IntIdentifier(colsCount));
					}
					catch(Exception e){
						
						LOG.error(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid column count " + sizeInfo[1] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1]);
						
						throw new LanguageException(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid column count " + sizeInfo[1] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1],
								LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
					
					try {
						nnzCount = Long.parseLong(sizeInfo[2]);
						if (nnzCount < 1)
							throw new Exception("invalid nnz count");
						addVarParam("nnz", new IntIdentifier(nnzCount));
					}
					catch(Exception e){
					
						LOG.error(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid number non-zeros " + sizeInfo[2] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1]);
						
						throw new LanguageException(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid number non-zeros " + sizeInfo[2] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1],
								LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);	
					}	
				}
			}
			
			configObject = null;
			
			if (shouldReadMTD){
				configObject = readMetadataFile(filename);
		        		    
		        // if the MTD file exists, check the values specified in read statement match values in metadata MTD file
		        if (configObject != null){
		        		    
		        	for (Object key : configObject.keySet()){
						
		        		boolean isValidName = false;
		        		for (String paramName : READ_VALID_MTD_PARAM_NAMES){
		    				if (paramName.equals(key))
		    					isValidName = true;
		    			}
		        		
						if (!isValidName){
							LOG.error(this.printErrorLocation() + "MTD file " + filename + " contains invalid parameter name: " + key);
							throw new LanguageException(this.printErrorLocation() + "MTD file " + filename + " contains invalid parameter name: " + key);
						}
						
						// if the read method parameter is a constant, then verify value matches MTD metadata file
						if (getVarParam(key.toString()) != null && (getVarParam(key.toString()) instanceof ConstIdentifier) 
								&& !getVarParam(key.toString()).toString().equalsIgnoreCase(configObject.get(key).toString()) ){
							
							LOG.error(this.printErrorLocation() + "parameter " + key.toString() + " has conflicting values in read statement definition and metadata. " +
									"Config file value: " + configObject.get(key).toString() + " from MTD file.  Read statement value: " + getVarParam(key.toString()));
							
							throw new LanguageException(this.printErrorLocation() + "parameter " + key.toString() + " has conflicting values in read statement definition and metadata. " +
									"Config file value: " + configObject.get(key).toString() + " from MTD file.  Read statement value: " + getVarParam(key.toString()));	
						}
						else {
							// if the read method does not specify parameter value, then add MTD metadata file value to parameter list
							if (getVarParam(key.toString()) == null){
								if ( !key.toString().equalsIgnoreCase(DESCRIPTIONPARAM) ) {
									StringIdentifier strId = new StringIdentifier(configObject.get(key).toString());
									strId.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
									addVarParam(key.toString(), strId);
								}
							}
						}
					}
		        }
		        else {
		        	LOG.warn("Metadata file: " + new Path(filename) + " not provided");
		        }
			} 
	        
			boolean isCSV = false;
			isCSV = (formatTypeString != null && formatTypeString.equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV));
			if (isCSV){
				
				 // Handle delimited file format
				 // 
				 // 1) only allow IO_FILENAME, _HEADER_ROW, FORMAT_DELIMITER, READROWPARAM, READCOLPARAM   
				 //  
				 // 2) open the file
				 //
				
				// there should be no MTD file for delimited file format
				shouldReadMTD = true;
				
				// only allow IO_FILENAME, HAS_HEADER_ROW, FORMAT_DELIMITER, READROWPARAM, READCOLPARAM   
				//		as ONLY valid parameters
				if (inferredFormatType == false){
					for (String key : _varParams.keySet()){
						if (!  (key.equals(IO_FILENAME) || key.equals(FORMAT_TYPE) 
								|| key.equals(DELIM_HAS_HEADER_ROW) || key.equals(DELIM_DELIMITER) 
								|| key.equals(DELIM_FILL) || key.equals(DELIM_FILL_VALUE)
								|| key.equals(READROWPARAM) || key.equals(READCOLPARAM)
								|| key.equals(READNUMNONZEROPARAM) || key.equals(DATATYPEPARAM) || key.equals(VALUETYPEPARAM)
								)){
							
							LOG.error(this.printErrorLocation() + "Invalid parameter " + key + " in read.csv statement: " +
									toString() + ". Only parameters allowed are: " + IO_FILENAME     + "," 
																				   + DELIM_HAS_HEADER_ROW   + "," 
																				   + DELIM_DELIMITER 	+ ","
																				   + DELIM_FILL 		+ ","
																				   + DELIM_FILL_VALUE 	+ ","
																				   + READROWPARAM     + "," 
																				   + READCOLPARAM);
							
							throw new LanguageException(this.printErrorLocation() + "Invalid parameter " + key + " in read.csv statement: " +
									toString() + ". Only parameters allowed are: " + IO_FILENAME      + "," 
																				   + DELIM_HAS_HEADER_ROW   + ","
																				   + DELIM_DELIMITER + "," 
																				   + DELIM_FILL 		+ ","
																				   + DELIM_FILL_VALUE 	+ ","
																				   + READROWPARAM     + "," 
																				   + READCOLPARAM,
																				   LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
						}
					}
				}
				
				// DEFAULT for "sep" : ","
				if (getVarParam(DELIM_DELIMITER) == null){
					addVarParam(DELIM_DELIMITER, new StringIdentifier(DEFAULT_DELIM_DELIMITER));
				}
				else {
					if ( (getVarParam(DELIM_DELIMITER) instanceof ConstIdentifier)
						&& (! (getVarParam(DELIM_DELIMITER) instanceof StringIdentifier)))
					{

						LOG.error(this.printErrorLocation() + "For delimited file " + getVarParam(DELIM_DELIMITER) 
								+  " must be a string value ");
						
						throw new LanguageException(this.printErrorLocation() + "For delimited file " + getVarParam(DELIM_DELIMITER) 
								+  " must be a string value ");
					}
				} 
				
				// DEFAULT for "default": 0
				if (getVarParam(DELIM_FILL_VALUE) == null){
					addVarParam(DELIM_FILL_VALUE, new DoubleIdentifier(DEFAULT_DELIM_FILL_VALUE));
				}
				else {
					if ( (getVarParam(DELIM_FILL_VALUE) instanceof ConstIdentifier)
							&& (! (getVarParam(DELIM_FILL_VALUE) instanceof IntIdentifier ||  getVarParam(DELIM_FILL_VALUE) instanceof DoubleIdentifier)))
					{

						LOG.error(this.printErrorLocation() + "For delimited file " + getVarParam(DELIM_HAS_HEADER_ROW) 
								+  " must be a boolean value ");
						
						throw new LanguageException(this.printErrorLocation() + "For delimited file " + getVarParam(DELIM_HAS_HEADER_ROW) 
								+  " must be a boolean value ");
					}
				} 
				
				// DEFAULT for "header": boolean false
				if (getVarParam(DELIM_HAS_HEADER_ROW) == null){
					addVarParam(DELIM_HAS_HEADER_ROW, new BooleanIdentifier(DEFAULT_DELIM_HAS_HEADER_ROW));
				}
				else {
					if ((getVarParam(DELIM_HAS_HEADER_ROW) instanceof ConstIdentifier)
						&& (! (getVarParam(DELIM_HAS_HEADER_ROW) instanceof BooleanIdentifier)))
					{
	
						LOG.error(this.printErrorLocation() + "For delimited file " + getVarParam(DELIM_HAS_HEADER_ROW) 
								+  " must be a boolean value ");
						
						throw new LanguageException(this.printErrorLocation() + "For delimited file " + getVarParam(DELIM_HAS_HEADER_ROW) 
								+  " must be a boolean value ");
					}
				}
				
				// DEFAULT for "fill": boolean false
				if (getVarParam(DELIM_FILL) == null){
					addVarParam(DELIM_FILL,new BooleanIdentifier(DEFAULT_DELIM_FILL));
				}
				else {
					
					if ((getVarParam(DELIM_FILL) instanceof ConstIdentifier)
							&& (! (getVarParam(DELIM_FILL) instanceof BooleanIdentifier)))
					{

						LOG.error(this.printErrorLocation() + "For delimited file " + getVarParam(DELIM_FILL) 
								+  " must be a boolean value ");
						
						throw new LanguageException(this.printErrorLocation() + "For delimited file " + getVarParam(DELIM_FILL) 
								+  " must be a boolean value ");
					}
				}		
			} 
	        dataTypeString = (getVarParam(DATATYPEPARAM) == null) ? null : getVarParam(DATATYPEPARAM).toString();
			
			if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(Statement.MATRIX_DATA_TYPE) ) {
				
				// set data type
		        getOutput().setDataType(DataType.MATRIX);
		        
		        // set number non-zeros
		        Expression ennz = this.getVarParam("nnz");
		        long nnz = -1;
		        if( ennz != null )
		        {
			        nnz = new Long(ennz.toString());
			        getOutput().setNnz(nnz);
		        }
		        
		        // Following dimension checks must be done when data type = MATRIX_DATA_TYPE 
				// initialize size of target data identifier to UNKNOWN
				getOutput().setDimensions(-1, -1);
				
				if ( !isCSV && (getVarParam(READROWPARAM) == null || getVarParam(READCOLPARAM) == null)){
					LOG.error(this.printErrorLocation() + "Missing or incomplete dimension information in read statement");
					throw new LanguageException(this.printErrorLocation() + "Missing or incomplete dimension information in read statement: " + filename, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				
				}
				if (getVarParam(READROWPARAM) instanceof ConstIdentifier && getVarParam(READCOLPARAM) instanceof ConstIdentifier)  {
				
					// these are strings that are long values
					Long dim1 = (getVarParam(READROWPARAM) == null) ? null : new Long (getVarParam(READROWPARAM).toString());
					Long dim2 = (getVarParam(READCOLPARAM) == null) ? null : new Long(getVarParam(READCOLPARAM).toString());
					
					if ( dim1 <= 0 || dim2 <= 0 ) {
						LOG.error(this.printErrorLocation() + "Invalid dimension information in read statement");
						throw new LanguageException(this.printErrorLocation() + "Invalid dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
					// set dim1 and dim2 values 
					if (dim1 != null && dim2 != null){
						getOutput().setDimensions(dim1, dim2);
					} else if (!isCSV && ((dim1 != null) || (dim2 != null))) {
						LOG.error(this.printErrorLocation() + "Partial dimension information in read statement");
						throw new LanguageException(this.printErrorLocation() + "Partial dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}	
				}
				
				// initialize block dimensions to UNKNOWN 
				getOutput().setBlockDimensions(-1, -1);
				
				// find "format": 1=text, 2=binary
				int format = 1; // default is "text"
				if (getVarParam(FORMAT_TYPE) == null || getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase("text")){
					format = 1;
				} else if ( getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase("binary") ) {
					format = 2;
				} else if ( getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_MATRIXMARKET) 
							|| getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV)) 
				{
					format = 1;
				} else {
					LOG.error(this.printErrorLocation() + "Invalid format in statement: " + this.toString());
					throw new LanguageException(this.printErrorLocation() + "Invalid format " + getVarParam(FORMAT_TYPE)+ " in statement: " + this.toString());
				}
				
				if (getVarParam(ROWBLOCKCOUNTPARAM) instanceof ConstIdentifier && getVarParam(COLUMNBLOCKCOUNTPARAM) instanceof ConstIdentifier)  {
				
					Long rowBlockCount = (getVarParam(ROWBLOCKCOUNTPARAM) == null) ? null : new Long(getVarParam(ROWBLOCKCOUNTPARAM).toString());
					Long columnBlockCount = (getVarParam(COLUMNBLOCKCOUNTPARAM) == null) ? null : new Long (getVarParam(COLUMNBLOCKCOUNTPARAM).toString());
		
					if ((rowBlockCount != null) && (columnBlockCount != null)) {
						getOutput().setBlockDimensions(rowBlockCount, columnBlockCount);
					} else if ((rowBlockCount != null) || (columnBlockCount != null)) {
						LOG.error(this.printErrorLocation() + "Partial block dimension information in read statement");
						throw new LanguageException(this.printErrorLocation() + "Partial block dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					} else {
						 getOutput().setBlockDimensions(-1, -1);
					}
				}
				
				// block dimensions must be -1x-1 when format="text"
				// NOTE MB: disabled validate of default blocksize for inputs w/ format="binary"
				// because we automatically introduce reblocks if blocksizes don't match
				if ( (format == 1 && (getOutput().getRowsInBlock() != -1 || getOutput().getColumnsInBlock() != -1))	){
					
					LOG.error(this.printErrorLocation() + "Invalid block dimensions (" + getOutput().getRowsInBlock() + "," + getOutput().getColumnsInBlock() + ") when format=" + getVarParam(FORMAT_TYPE) + " in \"" + this.toString() + "\".");
					throw new LanguageException(this.printErrorLocation() + "Invalid block dimensions (" + getOutput().getRowsInBlock() + "," + getOutput().getColumnsInBlock() + ") when format=" + getVarParam(FORMAT_TYPE) + " in \"" + this.toString() + "\".");
				}
			
			}
			
			else if ( dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)) {
				getOutput().setDataType(DataType.SCALAR);
				getOutput().setNnz(-1L);
			}
			
			else{		
				LOG.error(this.printErrorLocation() + "Unknown Data Type " + dataTypeString + ". Valid  values: " + Statement.SCALAR_DATA_TYPE +", " + Statement.MATRIX_DATA_TYPE);
				throw new LanguageException(this.printErrorLocation() + "Unknown Data Type " + dataTypeString + ". Valid  values: " + Statement.SCALAR_DATA_TYPE +", " + Statement.MATRIX_DATA_TYPE, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			// handle value type parameter
			if (getVarParam(VALUETYPEPARAM) != null && !(getVarParam(VALUETYPEPARAM) instanceof StringIdentifier)){
				
				LOG.error(this.printErrorLocation() + "for read method, parameter " + VALUETYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE);
				
				throw new LanguageException(this.printErrorLocation() + "for read method, parameter " + VALUETYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE,
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			// Identify the value type (used only for read method)
			String valueTypeString = getVarParam(VALUETYPEPARAM) == null ? null :  getVarParam(VALUETYPEPARAM).toString();
			if (valueTypeString != null) {
				if (valueTypeString.equalsIgnoreCase(Statement.DOUBLE_VALUE_TYPE)) {
					getOutput().setValueType(ValueType.DOUBLE);
				} else if (valueTypeString.equalsIgnoreCase(Statement.STRING_VALUE_TYPE)) {
					getOutput().setValueType(ValueType.STRING);
				} else if (valueTypeString.equalsIgnoreCase(Statement.INT_VALUE_TYPE)) {
					getOutput().setValueType(ValueType.INT);
				} else if (valueTypeString.equalsIgnoreCase(Statement.BOOLEAN_VALUE_TYPE)) {
					getOutput().setValueType(ValueType.BOOLEAN);
				} else {
					
					LOG.error(this.printErrorLocation() + "Unknown Value Type " + valueTypeString
							+ ". Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE);
					
					throw new LanguageException(this.printErrorLocation() + "Unknown Value Type " + valueTypeString
							+ ". Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE,
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
			} else {
				getOutput().setValueType(ValueType.DOUBLE);
			}

			break; 
			
		case WRITE:
			
			// for delimited format, if no delimiter specified THEN set default ","
			if (getVarParam(FORMAT_TYPE) == null || getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV)){
				if (getVarParam(DELIM_DELIMITER) == null){
					addVarParam(DELIM_DELIMITER, new StringIdentifier(DEFAULT_DELIM_DELIMITER));
				}
				if (getVarParam(DELIM_HAS_HEADER_ROW) == null){
					addVarParam(DELIM_HAS_HEADER_ROW, new BooleanIdentifier(DEFAULT_DELIM_HAS_HEADER_ROW));
				}
				if (getVarParam(DELIM_SPARSE) == null){
					addVarParam(DELIM_SPARSE, new BooleanIdentifier(DEFAULT_DELIM_SPARSE));
				}
			}
			
			if (getVarParam(IO_FILENAME) instanceof BinaryExpression){
				BinaryExpression expr = (BinaryExpression)getVarParam(IO_FILENAME);
								
				if (expr.getKind()== Expression.Kind.BinaryOp){
					Expression.BinaryOp op = expr.getOpCode();
					switch (op){
						case PLUS:
							filename = "";
							filename = fileNameCat(expr, currConstVars, filename);
							// Since we have computed the value of filename, we update
							// varParams with a const string value
							StringIdentifier fileString = new StringIdentifier(filename);
							fileString.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							removeVarParam(IO_FILENAME);
							addVarParam(IO_FILENAME, fileString);
												
							break;
						default:
							LOG.error(this.printErrorLocation() + "for OutputStatement, parameter " + IO_FILENAME + " can only be a const string or const string concatenations. ");
							throw new LanguageException(this.printErrorLocation() + "for OutputStatement, parameter " + IO_FILENAME + " can only be a const string or const string concatenations. ");
					}
				}
			}
			
			if (getVarParam(FORMAT_TYPE) == null || getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase("text"))
				getOutput().setBlockDimensions(-1, -1);
			else if (getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase("binary"))
				getOutput().setBlockDimensions(DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			else if (getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_MATRIXMARKET) || (getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV)))
				getOutput().setBlockDimensions(-1, -1);
			
			else{
				LOG.error(this.printErrorLocation() + "Invalid format " + getVarParam(FORMAT_TYPE) +  " in statement: " + this.toString());
				throw new LanguageException(this.printErrorLocation() + "Invalid format " + getVarParam(FORMAT_TYPE) + " in statement: " + this.toString());
			}
			break;

			case RAND: 
			
			Expression dataParam = getVarParam(RAND_DATA);
			if (dataParam != null){
			
				
				
				if (dataParam instanceof IntIdentifier){
					
					// update min expr with new IntIdentifier 
					long roundedValue = ((IntIdentifier)dataParam).getValue();
					Expression minExpr = new DoubleIdentifier(roundedValue);
					minExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					addVarParam(RAND_MIN, minExpr);
					addVarParam(RAND_MAX, minExpr);
				}
				// handle double constant 
				else if (dataParam instanceof DoubleIdentifier){
	
					// update col expr with new IntIdentifier (rounded down)
					double roundedValue = ((DoubleIdentifier)dataParam).getValue();
					Expression minExpr = new DoubleIdentifier(roundedValue);
					minExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					addVarParam(RAND_MIN, minExpr);
					addVarParam(RAND_MAX, minExpr);				
				}
				else {
					LOG.error(this.printErrorLocation() + "for matrix statement, parameter " 
							+ RAND_DATA + " cannot have value type String or Boolean. ");
							
					 
					throw new LanguageException(this.printErrorLocation() + "for matrix statement, parameter " 
							+ RAND_DATA + " cannot have value type String or Boolean. ");
				}
				removeVarParam(RAND_DATA);
				removeVarParam(RAND_BY_ROW);
				this.setRandDefault();
			}
			
				
				
			for (String key : _varParams.keySet()){
				boolean found = false;
				for (String name : RAND_VALID_PARAM_NAMES){
					if (name.equals(key))
					found = true;
				}
				if (!found){
					
					LOG.error(this.printErrorLocation() + "unexpected parameter \"" + key +
							"\". Legal parameters for Rand statement are " 
							+ "(capitalization-sensitive): " 	+ RAND_ROWS 	
							+ ", " + RAND_COLS		+ ", " + RAND_MIN + ", " + RAND_MAX  	
							+ ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", " + RAND_PDF);
					
					
					throw new LanguageException(this.printErrorLocation() + "unexpected parameter \"" + key +
						"\". Legal parameters for Rand statement are " 
						+ "(capitalization-sensitive): " 	+ RAND_ROWS 	
						+ ", " + RAND_COLS		+ ", " + RAND_MIN + ", " + RAND_MAX  	
						+ ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", " + RAND_PDF);
				}
			}
			//TODO: Leo Need to check with Doug about the data types
			// DoubleIdentifiers for RAND_ROWS and RAND_COLS have already been converted into IntIdentifier in RandStatment.addExprParam()  
			if (getVarParam(RAND_ROWS) instanceof StringIdentifier || getVarParam(RAND_ROWS) instanceof BooleanIdentifier){
				LOG.error(this.printErrorLocation() + "for Rand statement " + RAND_ROWS + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RAND_ROWS + " has incorrect data type");
			}
				
			if (getVarParam(RAND_COLS) instanceof StringIdentifier || getVarParam(RAND_COLS) instanceof BooleanIdentifier){
				LOG.error(this.printErrorLocation() + "for Rand statement " + RAND_COLS + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RAND_COLS + " has incorrect data type");
			}
				
			if (getVarParam(RAND_MAX) instanceof StringIdentifier || getVarParam(RAND_MAX) instanceof BooleanIdentifier) {
				LOG.error(this.printErrorLocation() + "for Rand statement " + RAND_MAX + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RAND_MAX + " has incorrect data type");
			}
			
			if (getVarParam(RAND_MIN) instanceof StringIdentifier || getVarParam(RAND_MIN) instanceof BooleanIdentifier) {
				LOG.error(this.printErrorLocation() + "for Rand statement " + RAND_MIN + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RAND_MIN + " has incorrect data type");
			}
			
			if (!(getVarParam(RAND_SPARSITY) instanceof DoubleIdentifier || getVarParam(RAND_SPARSITY) instanceof IntIdentifier)) {
				LOG.error(this.printErrorLocation() + "for Rand statement " + RAND_SPARSITY + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RAND_SPARSITY + " has incorrect data type");
			}
			
			if (!(getVarParam(RAND_SEED) instanceof IntIdentifier)) {
				LOG.error(this.printErrorLocation() + "for Rand statement " + RAND_SEED + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RAND_SEED + " has incorrect data type");
			}
			
			if (!(getVarParam(RAND_PDF) instanceof StringIdentifier)) {
				LOG.error(this.printErrorLocation() + "for Rand statement " + RAND_PDF + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RAND_PDF + " has incorrect data type");
			}
	
			long rowsLong = -1L, colsLong = -1L;

			///////////////////////////////////////////////////////////////////
			// HANDLE ROWS
			///////////////////////////////////////////////////////////////////
			Expression rowsExpr = getVarParam(RAND_ROWS);
			if (rowsExpr instanceof IntIdentifier) {
				if  (((IntIdentifier)rowsExpr).getValue() >= 1 ) {
					rowsLong = ((IntIdentifier)rowsExpr).getValue();
				}
				else {
					
					LOG.error(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue());
					
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue());
				}
			}
			else if (rowsExpr instanceof DoubleIdentifier) {
				if  (((DoubleIdentifier)rowsExpr).getValue() >= 1 ) {
					rowsLong = new Double((Math.floor(((DoubleIdentifier)rowsExpr).getValue()))).longValue();
				}
				else {
					
					LOG.error(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + rowsExpr.toString());
					
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + rowsExpr.toString());
				}		
			}
			else if (rowsExpr instanceof DataIdentifier && !(rowsExpr instanceof IndexedIdentifier)) {
				
				// check if the DataIdentifier variable is a ConstIdentifier
				String identifierName = ((DataIdentifier)rowsExpr).getName();
				if (currConstVars.containsKey(identifierName)){
					
					// handle int constant
					ConstIdentifier constValue = currConstVars.get(identifierName);
					if (constValue instanceof IntIdentifier){
						
						// check rows is >= 1 --- throw exception
						if (((IntIdentifier)constValue).getValue() < 1){
							LOG.error(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							
							throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
						// update row expr with new IntIdentifier 
						long roundedValue = ((IntIdentifier)constValue).getValue();
						rowsExpr = new IntIdentifier(roundedValue);
						rowsExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_ROWS, rowsExpr);
						rowsLong = roundedValue; 
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
						
						if (((DoubleIdentifier)constValue).getValue() < 1.0){
							LOG.error(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							
							throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
						// update row expr with new IntIdentifier (rounded down)
						long roundedValue = new Double (Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
						rowsExpr = new IntIdentifier(roundedValue);
						rowsExpr.setAllPositions(this.getFilename(),this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_ROWS, rowsExpr);
						rowsLong = roundedValue; 
						
					}
					else {
						// exception -- rows must be integer or double constant
						LOG.error(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						
						throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
					}
				}
				else {
					// handle general expression
					rowsExpr.validateExpression(ids, currConstVars);
				}
			}	
			else {
				// handle general expression
				rowsExpr.validateExpression(ids, currConstVars);
			}
				
	
			///////////////////////////////////////////////////////////////////
			// HANDLE COLUMNS
			///////////////////////////////////////////////////////////////////
			
			Expression colsExpr = getVarParam(RAND_COLS);
			if (colsExpr instanceof IntIdentifier) {
				if  (((IntIdentifier)colsExpr).getValue() >= 1 ) {
					colsLong = ((IntIdentifier)colsExpr).getValue();
				}
				else {
					LOG.error(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
					
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
				}
			}
			else if (colsExpr instanceof DoubleIdentifier) {
				if  (((DoubleIdentifier)colsExpr).getValue() >= 1 ) {
					colsLong = new Double((Math.floor(((DoubleIdentifier)colsExpr).getValue()))).longValue();
				}
				else {
					LOG.error(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
					
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
				}		
			}
			else if (colsExpr instanceof DataIdentifier && !(colsExpr instanceof IndexedIdentifier)) {
				
				// check if the DataIdentifier variable is a ConstIdentifier
				String identifierName = ((DataIdentifier)colsExpr).getName();
				if (currConstVars.containsKey(identifierName)){
					
					// handle int constant
					ConstIdentifier constValue = currConstVars.get(identifierName);
					if (constValue instanceof IntIdentifier){
						
						// check cols is >= 1 --- throw exception
						if (((IntIdentifier)constValue).getValue() < 1){
							LOG.error(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
						// update col expr with new IntIdentifier 
						long roundedValue = ((IntIdentifier)constValue).getValue();
						colsExpr = new IntIdentifier(roundedValue);
						colsExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_COLS, colsExpr);
						colsLong = roundedValue; 
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
						
						if (((DoubleIdentifier)constValue).getValue() < 1){
							LOG.error(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							
							throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
						// update col expr with new IntIdentifier (rounded down)
						long roundedValue = new Double (Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
						colsExpr = new IntIdentifier(roundedValue);
						colsExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_COLS, colsExpr);
						colsLong = roundedValue; 
						
					}
					else {
						// exception -- rows must be integer or double constant
						LOG.error(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
								"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						
						throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
								"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
					}
				}
				else {
					// handle general expression
					colsExpr.validateExpression(ids, currConstVars);
				}
					
			}	
			else {
				// handle general expression
				colsExpr.validateExpression(ids, currConstVars);
			}
			
			///////////////////////////////////////////////////////////////////
			// HANDLE MIN
			///////////////////////////////////////////////////////////////////	
			Expression minExpr = getVarParam(RAND_MIN);
			
			// perform constant propogation
			if (minExpr instanceof DataIdentifier && !(minExpr instanceof IndexedIdentifier)) {
				
				// check if the DataIdentifier variable is a ConstIdentifier
				String identifierName = ((DataIdentifier)minExpr).getName();
				if (currConstVars.containsKey(identifierName)){
					
					// handle int constant
					ConstIdentifier constValue = currConstVars.get(identifierName);
					if (constValue instanceof IntIdentifier){
						
						// update min expr with new IntIdentifier 
						long roundedValue = ((IntIdentifier)constValue).getValue();
						minExpr = new DoubleIdentifier(roundedValue);
						minExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_MIN, minExpr);
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
		
						// update col expr with new IntIdentifier (rounded down)
						double roundedValue = ((DoubleIdentifier)constValue).getValue();
						minExpr = new DoubleIdentifier(roundedValue);
						minExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_MIN, minExpr);
						
					}
					else {
						// exception -- rows must be integer or double constant
						LOG.error(this.printErrorLocation() + "In rand statement, can only assign min a numerical " +
								"value -- attempted to assign: " + constValue.toString());
						
						throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign min a numerical " +
								"value -- attempted to assign: " + constValue.toString());
					}
				}
				else {
					// handle general expression
					minExpr.validateExpression(ids, currConstVars);
				}
					
			}	
			else {
				// handle general expression
				minExpr.validateExpression(ids, currConstVars);
			}
			
			
			///////////////////////////////////////////////////////////////////
			// HANDLE MAX
			///////////////////////////////////////////////////////////////////
			Expression maxExpr = getVarParam(RAND_MAX);
			
			// perform constant propogation
			if (maxExpr instanceof DataIdentifier && !(maxExpr instanceof IndexedIdentifier)) {
				
				// check if the DataIdentifier variable is a ConstIdentifier
				String identifierName = ((DataIdentifier)maxExpr).getName();
				if (currConstVars.containsKey(identifierName)){
					
					// handle int constant
					ConstIdentifier constValue = currConstVars.get(identifierName);
					if (constValue instanceof IntIdentifier){
						
						// update min expr with new IntIdentifier 
						long roundedValue = ((IntIdentifier)constValue).getValue();
						maxExpr = new DoubleIdentifier(roundedValue);
						maxExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_MAX, maxExpr);
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
		
						// update col expr with new IntIdentifier (rounded down)
						double roundedValue = ((DoubleIdentifier)constValue).getValue();
						maxExpr = new DoubleIdentifier(roundedValue);
						maxExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_MAX, maxExpr);
						
					}
					else {
						// exception -- rows must be integer or double constant
						LOG.error(this.printErrorLocation() + "In rand statement, can only assign max a numerical " +
								"value -- attempted to assign: " + constValue.toString());
						
						throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign max a numerical " +
								"value -- attempted to assign: " + constValue.toString());
					}
				}
				else {
					// handle general expression
					maxExpr.validateExpression(ids, currConstVars);
				}		
			}	
			else {
				// handle general expression
				maxExpr.validateExpression(ids, currConstVars);
			}
		
			getOutput().setFormatType(FormatType.BINARY);
			getOutput().setDataType(DataType.MATRIX);
			getOutput().setValueType(ValueType.DOUBLE);
			getOutput().setDimensions(rowsLong, colsLong);
			
			if (getOutput() instanceof IndexedIdentifier){
				// process the "target" being indexed
				DataIdentifier targetAsSeen = ids.get(((DataIdentifier)getOutput()).getName());
				if (targetAsSeen == null){
					LOG.error(getOutput().printErrorLocation() + "cannot assign value to indexed identifier " + ((DataIdentifier)getOutput()).getName() + " without first initializing " + ((DataIdentifier)getOutput()).getName());
					throw new LanguageException(getOutput().printErrorLocation() + "cannot assign value to indexed identifier " + ((DataIdentifier)getOutput()).getName() + " without first initializing " + ((DataIdentifier)getOutput()).getName());
				}
				//_output.setProperties(targetAsSeen);
				((IndexedIdentifier) getOutput()).setOriginalDimensions(targetAsSeen.getDim1(), targetAsSeen.getDim2());
				//((IndexedIdentifier) getOutput()).setOriginalDimensions(getOutput().getDim1(), getOutput().getDim2());
			}
			//getOutput().computeDataType();

			if (getOutput() instanceof IndexedIdentifier){
				LOG.warn(this.printWarningLocation() + "Output for Rand Statement may have incorrect size information");
			}
			
			break;
			
			case MATRIX: 
			
			this.setMatrixDefault();
				
			for (String key : _varParams.keySet()){
				boolean found = false;
				for (String name : MATRIX_VALID_PARAM_NAMES){
					if (name.equals(key))
					found = true;
				}
				if (!found){
					
					LOG.error(this.printErrorLocation() + "unexpected parameter \"" + key +
							"\". Legal parameters for matrix statement are " 
							+ "(capitalization-sensitive): " 	+ RAND_DATA 	
							+ ", " + RAND_ROWS		+ ", " + RAND_COLS
							+ ", " + RAND_BY_ROW ); //  + ", " + RAND_DIMNAMES);
					
					
					throw new LanguageException(this.printErrorLocation() + "unexpected parameter \"" + key +
							"\". Legal parameters for matrix statement are " 
							+ "(capitalization-sensitive): " 	+ RAND_DATA 	
							+ ", " + RAND_ROWS		+ ", " + RAND_COLS
							+ ", " + RAND_BY_ROW );//   + ", " + RAND_DIMNAMES);
				}
			}
			//TODO: Leo Need to check with Doug about the data types
			// DoubleIdentifiers for RAND_ROWS and RAND_COLS have already been converted into IntIdentifier in RandStatment.addExprParam()  
			if (!(getVarParam(RAND_DATA) instanceof DataIdentifier)){
				LOG.error(this.printErrorLocation() + "for matrix statement " + RAND_DATA + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for matrix statement " + RAND_DATA + " has incorrect data type");
			}
			if (getVarParam(RAND_ROWS) != null && (getVarParam(RAND_ROWS) instanceof StringIdentifier || getVarParam(RAND_ROWS) instanceof BooleanIdentifier)){
				LOG.error(this.printErrorLocation() + "for matrix statement " + RAND_ROWS + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for matrix statement " + RAND_ROWS + " has incorrect data type");
			}
				
			if (getVarParam(RAND_COLS) != null && (getVarParam(RAND_COLS) instanceof StringIdentifier || getVarParam(RAND_COLS) instanceof BooleanIdentifier)){
				LOG.error(this.printErrorLocation() + "for matrix statement " + RAND_COLS + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for matrix statement " + RAND_COLS + " has incorrect data type");
			}
				
			if ( !(getVarParam(RAND_BY_ROW) instanceof BooleanIdentifier)) {
				LOG.error(this.printErrorLocation() + "for matrix statement " + RAND_BY_ROW + " has incorrect data type");
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RAND_BY_ROW + " has incorrect data type");
			}
			
			rowsLong = -1L; 
			colsLong = -1L;

			///////////////////////////////////////////////////////////////////
			// HANDLE ROWS
			///////////////////////////////////////////////////////////////////
			rowsExpr = getVarParam(RAND_ROWS);
			if (rowsExpr != null){
				if (rowsExpr instanceof IntIdentifier) {
					if  (((IntIdentifier)rowsExpr).getValue() >= 1 ) {
						rowsLong = ((IntIdentifier)rowsExpr).getValue();
					}
					else {
						
						LOG.error(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue());
						
						throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue());
					}
				}
				else if (rowsExpr instanceof DoubleIdentifier) {
					if  (((DoubleIdentifier)rowsExpr).getValue() >= 1 ) {
						rowsLong = new Double((Math.floor(((DoubleIdentifier)rowsExpr).getValue()))).longValue();
					}
					else {
						
						LOG.error(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + rowsExpr.toString());
						
						throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + rowsExpr.toString());
					}		
				}
				else if (rowsExpr instanceof DataIdentifier && !(rowsExpr instanceof IndexedIdentifier)) {
					
					// check if the DataIdentifier variable is a ConstIdentifier
					String identifierName = ((DataIdentifier)rowsExpr).getName();
					if (currConstVars.containsKey(identifierName)){
						
						// handle int constant
						ConstIdentifier constValue = currConstVars.get(identifierName);
						if (constValue instanceof IntIdentifier){
							
							// check rows is >= 1 --- throw exception
							if (((IntIdentifier)constValue).getValue() < 1){
								LOG.error(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
								
								throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							}
							// update row expr with new IntIdentifier 
							long roundedValue = ((IntIdentifier)constValue).getValue();
							rowsExpr = new IntIdentifier(roundedValue);
							rowsExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue; 
						}
						// handle double constant 
						else if (constValue instanceof DoubleIdentifier){
							
							if (((DoubleIdentifier)constValue).getValue() < 1.0){
								LOG.error(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
								
								throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							}
							// update row expr with new IntIdentifier (rounded down)
							long roundedValue = new Double (Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
							rowsExpr = new IntIdentifier(roundedValue);
							rowsExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue; 
							
						}
						else {
							// exception -- rows must be integer or double constant
							LOG.error(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							
							throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
					}
					else {
						// handle general expression
						rowsExpr.validateExpression(ids, currConstVars);
					}
				}	
				else {
					// handle general expression
					rowsExpr.validateExpression(ids, currConstVars);
				}
			}
	
			///////////////////////////////////////////////////////////////////
			// HANDLE COLUMNS
			///////////////////////////////////////////////////////////////////
			
			colsExpr = getVarParam(RAND_COLS);
			if (colsExpr != null){
				if (colsExpr instanceof IntIdentifier) {
					if  (((IntIdentifier)colsExpr).getValue() >= 1 ) {
						colsLong = ((IntIdentifier)colsExpr).getValue();
					}
					else {
						LOG.error(this.printErrorLocation() + "In matrix statement, can only assign cols a long " +
								"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
						
						throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign cols a long " +
								"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
					}
				}
				else if (colsExpr instanceof DoubleIdentifier) {
					if  (((DoubleIdentifier)colsExpr).getValue() >= 1 ) {
						colsLong = new Double((Math.floor(((DoubleIdentifier)colsExpr).getValue()))).longValue();
					}
					else {
						LOG.error(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
						
						throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
					}		
				}
				else if (colsExpr instanceof DataIdentifier && !(colsExpr instanceof IndexedIdentifier)) {
					
					// check if the DataIdentifier variable is a ConstIdentifier
					String identifierName = ((DataIdentifier)colsExpr).getName();
					if (currConstVars.containsKey(identifierName)){
						
						// handle int constant
						ConstIdentifier constValue = currConstVars.get(identifierName);
						if (constValue instanceof IntIdentifier){
							
							// check cols is >= 1 --- throw exception
							if (((IntIdentifier)constValue).getValue() < 1){
								LOG.error(this.printErrorLocation() + "In matrix statement, can only assign cols a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
								throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign cols a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							}
							// update col expr with new IntIdentifier 
							long roundedValue = ((IntIdentifier)constValue).getValue();
							colsExpr = new IntIdentifier(roundedValue);
							colsExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue; 
						}
						// handle double constant 
						else if (constValue instanceof DoubleIdentifier){
							
							if (((DoubleIdentifier)constValue).getValue() < 1){
								LOG.error(this.printErrorLocation() + "In matrix statement, can only assign cols a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
								
								throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign cols a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							}
							// update col expr with new IntIdentifier (rounded down)
							long roundedValue = new Double (Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
							colsExpr = new IntIdentifier(roundedValue);
							colsExpr.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue; 
							
						}
						else {
							// exception -- rows must be integer or double constant
							LOG.error(this.printErrorLocation() + "In matrix statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
							
							throw new LanguageException(this.printErrorLocation() + "In matrix statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
					}
					else {
						// handle general expression
						colsExpr.validateExpression(ids, currConstVars);
					}
						
				}	
				else {
					// handle general expression
					colsExpr.validateExpression(ids, currConstVars);
				}
			}	
			getOutput().setFormatType(FormatType.BINARY);
			getOutput().setDataType(DataType.MATRIX);
			getOutput().setValueType(ValueType.DOUBLE);
			getOutput().setDimensions(rowsLong, colsLong);
				
			if (getOutput() instanceof IndexedIdentifier){
				((IndexedIdentifier) getOutput()).setOriginalDimensions(getOutput().getDim1(), getOutput().getDim2());
			}
			//getOutput().computeDataType();

			if (getOutput() instanceof IndexedIdentifier){
				LOG.warn(this.printWarningLocation() + "Output for matrix Statement may have incorrect size information");
			}
			
			break;
			
	
		default:
			LOG.error(this.printErrorLocation() + "Unsupported Data expression"
					+ this.getOpCode());
			
			
			throw new LanguageException(this.printErrorLocation() + "Unsupported Data expression"
						+ this.getOpCode(),
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
		return;
	}
	
	private String fileNameCat(BinaryExpression expr, HashMap<String, ConstIdentifier> currConstVars, String filename)throws LanguageException{
		// Processing the left node first
		if (expr.getLeft() instanceof BinaryExpression 
				&& ((BinaryExpression)expr.getLeft()).getKind()== BinaryExpression.Kind.BinaryOp
				&& ((BinaryExpression)expr.getLeft()).getOpCode() == BinaryOp.PLUS){
			filename = fileNameCat((BinaryExpression)expr.getLeft(), currConstVars, filename)+ filename;
		}
		else if (expr.getLeft() instanceof StringIdentifier){
			filename = ((StringIdentifier)expr.getLeft()).getValue()+ filename;
		}
		else if (expr.getLeft() instanceof DataIdentifier 
				&& ((DataIdentifier)expr.getLeft()).getDataType() == Expression.DataType.SCALAR
				&& ((DataIdentifier)expr.getLeft()).getKind() == Expression.Kind.Data 
				&& ((DataIdentifier)expr.getLeft()).getValueType() == Expression.ValueType.STRING){
			String name = ((DataIdentifier)expr.getLeft()).getName();
			filename = ((StringIdentifier)currConstVars.get(name)).getValue() + filename;
		}
		else {
			LOG.error(this.printErrorLocation() + "Parameter " + IO_FILENAME + " only supports a const string or const string concatenations.");
			throw new LanguageException(this.printErrorLocation() + "Parameter " + IO_FILENAME + " only supports a const string or const string concatenations.");
		}
		// Now process the right node
		if (expr.getRight()instanceof BinaryExpression 
				&& ((BinaryExpression)expr.getRight()).getKind()== BinaryExpression.Kind.BinaryOp
				&& ((BinaryExpression)expr.getRight()).getOpCode() == BinaryOp.PLUS){
			filename = filename + fileNameCat((BinaryExpression)expr.getRight(), currConstVars, filename);
		}
		else if (expr.getRight() instanceof StringIdentifier){
			filename = filename + ((StringIdentifier)expr.getRight()).getValue();
		}
		else if (expr.getRight() instanceof DataIdentifier 
				&& ((DataIdentifier)expr.getRight()).getDataType() == Expression.DataType.SCALAR
				&& ((DataIdentifier)expr.getRight()).getKind() == Expression.Kind.Data 
				&& ((DataIdentifier)expr.getRight()).getValueType() == Expression.ValueType.STRING){
			String name = ((DataIdentifier)expr.getRight()).getName();
			filename =  filename + ((StringIdentifier)currConstVars.get(name)).getValue();
		}
		else {
			LOG.error(this.printErrorLocation() + "Parameter " + IO_FILENAME + " only supports a const string or const string concatenations.");
			throw new LanguageException(this.printErrorLocation() + "Parameter " + IO_FILENAME + " only supports a const string or const string concatenations.");
		}
		return filename;
			
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer(_opcode.toString() + "(");

		 for (String key : _varParams.keySet()){
			 sb.append("," + key + "=" + _varParams.get(key));
		 }
		sb.append(" )");
		return sb.toString();
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		for (String s : _varParams.keySet()) {
			result.addVariables ( _varParams.get(s).variablesRead() );
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		for (String s : _varParams.keySet()) {
			result.addVariables ( _varParams.get(s).variablesUpdated() );
		}
		result.addVariable(((DataIdentifier)this.getOutput()).getName(), (DataIdentifier)this.getOutput());
		return result;
	}

	
	public JSONObject readMetadataFile(String filename) throws LanguageException {
	
		JSONObject retVal = null;
		boolean exists = false;
		FileSystem fs = null;
		
		try {
			fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
		} catch (Exception e){
			e.printStackTrace();
			LOG.error(this.printErrorLocation() + "could not read the configuration file.");
			throw new LanguageException(this.printErrorLocation() + "could not read the configuration file.");
		}
		
		Path pt = new Path(filename);
		try {
			if (fs.exists(pt)){
				exists = true;
			}
		} catch (Exception e){
			exists = false;
		}
	
		boolean isDirBoolean = false;
		try {
			if (exists && fs.getFileStatus(pt).isDir())
				isDirBoolean = true;
			else
				isDirBoolean = false;
		}
		catch(Exception e){
			e.printStackTrace();
			LOG.error(this.printErrorLocation() + "error validing whether path " + pt.toString() + " is directory or not");
        	throw new LanguageException(this.printErrorLocation() + "error validing whether path " + pt.toString() + " is directory or not");			
		}
		
		// CASE: filename is a directory -- process as a directory
		if (exists && isDirBoolean){
			
			// read directory contents
			retVal = new JSONObject();
			
			FileStatus[] stats = null;
			
			try {
				stats = fs.listStatus(pt);
			}
			catch (Exception e){
				LOG.error(e.toString());
				LOG.error(this.printErrorLocation() + "for MTD file in directory, error reading directory with MTD file " + pt.toString() + ": " + e.toString());
				throw new LanguageException(this.printErrorLocation() + "for MTD file in directory, error reading directory with MTD file " + pt.toString() + ": " + e.toString());	
			}
			
			for(FileStatus stat : stats){
				Path childPath = stat.getPath(); // gives directory name
				if (childPath.getName().startsWith("part")){
					
					BufferedReader br = null;
					try {
						br = new BufferedReader(new InputStreamReader(fs.open(childPath)));
					}
					catch(Exception e){
						LOG.error(e.toString());
						LOG.error(this.printErrorLocation() + "for MTD file in directory, error reading part of MTD file with path " + childPath.toString() + ": " + e.toString());
						throw new LanguageException(this.printErrorLocation() + "for MTD file in directory, error reading part of MTD file with path " + childPath.toString() + e.toString());	
					}
					
					JSONObject childObj = null;
					try {
						childObj = JSONObject.parse(br);
					}
					catch(Exception e){
						LOG.error(this.printErrorLocation() + "for MTD file in directory, error parsing part of MTD file with path " + childPath.toString() + ": " + e.toString());
						throw new LanguageException(this.printErrorLocation() + "for MTD file in directory, error parsing part of MTD file with path " + childPath.toString() + ": " + e.toString());		
					}
					for (Object key : childObj.keySet()){
						retVal.put(key, childObj.get(key));
					}
				}
			} // end for 
		}
		
		// CASE: filename points to a file
		else if (exists){
			
			BufferedReader br = null;
			
			// try reading MTD file
			try {
				br=new BufferedReader(new InputStreamReader(fs.open(pt)));
			} catch (Exception e){
				LOG.error(this.printErrorLocation() + "error reading MTD file with path " + pt.toString() + ": " + e.toString());
				throw new LanguageException(this.printErrorLocation() + "error reading with path " + pt.toString() + ": " + e.toString());
	        }
			
			// try parsing MTD file
			try {
				retVal =  JSONObject.parse(br);	
			} catch (Exception e){
				LOG.error(this.printErrorLocation() + "error parsing MTD file with path " + pt.toString() + ": " + e.toString());
				throw new LanguageException(this.printErrorLocation() + "error parsing MTD with path " + pt.toString() + ": " + e.toString());
	        }
		}
			
		return retVal;
	}

	public String[] readMatrixMarketFile(String filename) 
		throws LanguageException 
	{
		String[] retVal = new String[2];
		retVal[0] = new String("");
		retVal[1] = new String("");
		boolean exists = false;
		
		try 
		{
			FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
			Path pt = new Path(filename);
			if (fs.exists(pt)){
				exists = true;
			}
			
			boolean getFileStatusIsDir = fs.getFileStatus(pt).isDir();
			
			if (exists && getFileStatusIsDir){
				LOG.error(this.printErrorLocation() + "MatrixMarket files as directories not supported");
				throw new LanguageException(this.printErrorLocation() + "MatrixMarket files as directories not supported");
			}
			else if (exists) {
				BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(pt)));
				retVal[0] = in.readLine();
				retVal[1] = in.readLine();
				
				if ( !retVal[0].startsWith("%%") ) {
					LOG.error(this.printErrorLocation() + "MatrixMarket files must begin with a header line.");
					throw new LanguageException(this.printErrorLocation() + "MatrixMarket files must begin with a header line.");
				}
			}
			else {
				LOG.error(this.printErrorLocation() + "Could not find the file: " + filename);
				throw new LanguageException(this.printErrorLocation() + "Could not find the file: " + filename);
			}
			
		} catch (IOException e){
			e.printStackTrace();
			//LOG.error(this.printErrorLocation() + "Error reading MatrixMarket file: " + filename );
			//throw new LanguageException(this.printErrorLocation() + "Error reading MatrixMarket file: " + filename );
			throw new LanguageException(e);
		}

		return retVal;
	}
	
	public boolean checkHasMatrixMarketFormat(String filename) throws LanguageException {
		
		// Check the MTD file exists. if there is an MTD file, return false.
		JSONObject mtdObject = readMetadataFile(filename +".mtd");
	    
		if (mtdObject != null)
			return false;
		
		boolean exists = false;
		FileSystem fs = null;
		
		try {
			fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
		} catch (Exception e){
			e.printStackTrace();
			LOG.error(this.printErrorLocation() + "could not read the configuration file.");
			throw new LanguageException(this.printErrorLocation() + "could not read the configuration file.");
		}
		
		Path pt = new Path(filename);
		try {
			if (fs.exists(pt)){
				exists = true;
			}
		} catch (Exception e){
			LOG.error(this.printErrorLocation() + "file " + filename + " not found");
			throw new LanguageException(this.printErrorLocation() + "file " + filename + " not found");
		}
	
		try {
			// CASE: filename is a directory -- process as a directory
			if (exists && fs.getFileStatus(pt).isDir()){
				
				// currently, only MM files as files are supported.  So, if file is directory, then infer 
				// likely not MM file
				return false;
			}
			// CASE: filename points to a file
			else if (exists){
				
				//BufferedReader in = new BufferedReader(new FileReader(filename));
				BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(pt)));
				
				String headerLine = new String("");
			
				if (in.ready())
					headerLine = in.readLine();
				in.close();
			
				// check that headerline starts with "%%"
				// will infer malformed 
				if (headerLine.startsWith("%%"))
					return true;
				else
					return false;
			}
			else {
				return false;
			}
			
		} catch (Exception e){
			return false;
		}
	}
	
	public boolean checkHasDelimitedFormat(String filename) throws LanguageException {
	 

        // if the MTD file exists, check the format is not binary 
		JSONObject mtdObject = readMetadataFile(filename + ".mtd");
        if (mtdObject != null){
        	String formatTypeString = (String)mtdObject.get(FORMAT_TYPE);
        	if (formatTypeString == null || formatTypeString.equalsIgnoreCase(FORMAT_TYPE_VALUE_BINARY) ){
        		return false;
        	}
        }
           
		boolean exists = false;
		FileSystem fs = null;
		
		try {
			fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
		} catch (Exception e){
			e.printStackTrace();
			LOG.error(this.printErrorLocation() + "could not read the configuration file.");
			throw new LanguageException(this.printErrorLocation() + "could not read the configuration file.");
		}
		
		Path pt = new Path(filename);
		try {
			if (fs.exists(pt)){
				exists = true;
			}
		} catch (Exception e){
			LOG.error(this.printErrorLocation() + "file " + filename + " not found");
			throw new LanguageException(this.printErrorLocation() + "file " + filename + " not found");
		}
	
		try {
				
			// currently only support actual file, and do not support directory for delimited file
			if (exists && fs.getFileStatus(pt).isDir()){
				return false;
			}
			
			// CASE: filename points to a file
			else if (exists){
				
				BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(pt)));
				
				String headerLine = new String("");
			
				if (in.ready())
					headerLine = in.readLine();
				in.close();
			
				// check if there are delimited values "
				if (getVarParam(DELIM_DELIMITER) != null)
					return true;
				else {
					String defaultDelimiter = ",";
					boolean lineHasDelimiters = headerLine.contains(defaultDelimiter);
					return lineHasDelimiters;
				}
			}
			else {
				return false;
			}
			
		} catch (Exception e){
			return false;
		}
	}
	
	
	
} // end class
