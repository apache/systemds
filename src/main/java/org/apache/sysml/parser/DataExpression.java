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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.parser.LanguageException.LanguageErrorCodes;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.utils.JSONHelper;


public class DataExpression extends DataIdentifier 
{
	public static final String RAND_ROWS 	=  "rows";	 
	public static final String RAND_COLS 	=  "cols";
	public static final String RAND_MIN  	=  "min";
	public static final String RAND_MAX  	=  "max";
	public static final String RAND_SPARSITY = "sparsity"; 
	public static final String RAND_SEED    =  "seed";
	public static final String RAND_PDF		=  "pdf";
	public static final String RAND_LAMBDA	=  "lambda";
	
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
	public static final String AUTHORPARAM = "author";
	public static final String SCHEMAPARAM = "schema";

	// Parameter names relevant to reading/writing delimited/csv files
	public static final String DELIM_DELIMITER = "sep";
	public static final String DELIM_HAS_HEADER_ROW = "header";
	public static final String DELIM_FILL = "fill";
	public static final String DELIM_FILL_VALUE = "default";
	//public static final String DELIM_RECODE = "recode";
	public static final String DELIM_NA_STRINGS = "na.strings";
	public static final String DELIM_NA_STRING_SEP = "\u00b7";
	
	public static final String DELIM_SPARSE = "sparse";  // applicable only for write
	
	public static final String[] RAND_VALID_PARAM_NAMES = 
		{ RAND_ROWS, RAND_COLS, RAND_MIN, RAND_MAX, RAND_SPARSITY, RAND_SEED, RAND_PDF, RAND_LAMBDA}; 
	
	public static final String[] MATRIX_VALID_PARAM_NAMES = 
		{  RAND_BY_ROW, RAND_DIMNAMES, RAND_DATA, RAND_ROWS, RAND_COLS};
	
	// Valid parameter names in a metadata file
	public static final String[] READ_VALID_MTD_PARAM_NAMES = 
		{ IO_FILENAME, READROWPARAM, READCOLPARAM, READNUMNONZEROPARAM, FORMAT_TYPE,
			ROWBLOCKCOUNTPARAM, COLUMNBLOCKCOUNTPARAM, DATATYPEPARAM, VALUETYPEPARAM, SCHEMAPARAM, DESCRIPTIONPARAM,
			// Parameters related to delimited/csv files.
			DELIM_FILL_VALUE, DELIM_DELIMITER, DELIM_FILL, DELIM_HAS_HEADER_ROW, DELIM_NA_STRINGS
		}; 

	public static final String[] READ_VALID_PARAM_NAMES = 
	{	IO_FILENAME, READROWPARAM, READCOLPARAM, FORMAT_TYPE, DATATYPEPARAM, VALUETYPEPARAM, SCHEMAPARAM,
		ROWBLOCKCOUNTPARAM, COLUMNBLOCKCOUNTPARAM, READNUMNONZEROPARAM, 
			// Parameters related to delimited/csv files.
			DELIM_FILL_VALUE, DELIM_DELIMITER, DELIM_FILL, DELIM_HAS_HEADER_ROW, DELIM_NA_STRINGS
	}; 
		
	/* Default Values for delimited (CSV) files */
	public static final String  DEFAULT_DELIM_DELIMITER = ",";
	public static final boolean DEFAULT_DELIM_HAS_HEADER_ROW = false;
	public static final boolean DEFAULT_DELIM_FILL = true;
	public static final double  DEFAULT_DELIM_FILL_VALUE = 0.0;
	public static final boolean DEFAULT_DELIM_SPARSE = false;
	
	private DataOp _opcode;
	private HashMap<String, Expression> _varParams;
	private boolean _strInit = false; //string initialize
	private boolean _checkMetadata = true; // local skip meta data reads

	public DataExpression(){
		//do nothing
	}

	
	public void setCheckMetadata(boolean checkMetadata) {
		_checkMetadata = checkMetadata;
	}


	public static DataExpression getDataExpression(String functionName, ArrayList<ParameterExpression> passedParamExprs, 
				String filename, int blp, int bcp, int elp, int ecp) throws LanguageException 
	{	
		if (functionName == null || passedParamExprs == null)
			return null;
		
		// check if the function name is built-in function
		//	 (assign built-in function op if function is built-in)
		Expression.DataOp dop = null;
		DataExpression dataExpr = null;
		if (functionName.equals("read") || functionName.equals("readMM") || functionName.equals("read.csv"))
		{
			dop = Expression.DataOp.READ;
			dataExpr = new DataExpression(dop, new HashMap<String,Expression>(), filename, blp, bcp, elp, ecp);
			
			if (functionName.equals("readMM"))
				dataExpr.addVarParam(DataExpression.FORMAT_TYPE,  
						new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET,
								filename, blp, bcp, elp, ecp));
			
			if (functionName.equals("read.csv"))
				dataExpr.addVarParam(DataExpression.FORMAT_TYPE,  
						new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_CSV, 
								filename, blp, bcp, elp, ecp));
			
			// validate the filename is the first parameter
			if (passedParamExprs.size() < 1){
				dataExpr.raiseValidateError("read method must have at least filename parameter", false);
			}
			
			ParameterExpression pexpr = (passedParamExprs.size() == 0) ? null : passedParamExprs.get(0);
			
			if ( (pexpr != null) &&  (!(pexpr.getName() == null) || (pexpr.getName() != null && pexpr.getName().equalsIgnoreCase(DataExpression.IO_FILENAME)))){
				dataExpr.raiseValidateError("first parameter to read statement must be filename");
			} else if( pexpr != null ){
				dataExpr.addVarParam(DataExpression.IO_FILENAME, pexpr.getExpr());
			}
			
			// validate all parameters are added only once and valid name
			for (int i = 1; i < passedParamExprs.size(); i++){
				String currName = passedParamExprs.get(i).getName();
				Expression currExpr = passedParamExprs.get(i).getExpr();
				
				if (dataExpr.getVarParam(currName) != null){
					dataExpr.raiseValidateError("attempted to add IOStatement parameter " + currName + " more than once");
				}
				// verify parameter names for read function
				boolean isValidName = false;
				for (String paramName : READ_VALID_PARAM_NAMES){
					if (paramName.equals(currName))
						isValidName = true;
				}
				if (!isValidName){
					dataExpr.raiseValidateError("attempted to add invalid read statement parameter " + currName);
				}	
				dataExpr.addVarParam(currName, currExpr);
			}				
		}
		
		else if (functionName.equalsIgnoreCase("rand")){
			
			dop = Expression.DataOp.RAND;
			dataExpr = new DataExpression(dop, new HashMap<String,Expression>(), 
					filename, blp, bcp, elp, ecp);
			
			for (ParameterExpression currExpr : passedParamExprs){
				String pname = currExpr.getName();
				Expression pexpr = currExpr.getExpr();
				if (pname == null){
					dataExpr.raiseValidateError("for Rand Statement all arguments must be named parameters");	
				}
				dataExpr.addRandExprParam(pname, pexpr); 
			}
			dataExpr.setRandDefault();
		}
		
		else if (functionName.equals("matrix")){
			dop = Expression.DataOp.MATRIX;
			dataExpr = new DataExpression(dop, new HashMap<String,Expression>(),
					filename, blp, bcp, elp, ecp);
		
			int namedParamCount = 0, unnamedParamCount = 0;
			for (ParameterExpression currExpr : passedParamExprs) {
				if (currExpr.getName() == null)
					unnamedParamCount++;
				else
					namedParamCount++;
			}

			// check whether named or unnamed parameters are used
			if (passedParamExprs.size() < 3){
				dataExpr.raiseValidateError("for matrix statement, must specify at least 3 arguments (in order): data, rows, cols");
			}
			
			if (unnamedParamCount > 1){
				
				if (namedParamCount > 0)
					dataExpr.raiseValidateError("for matrix statement, cannot mix named and unnamed parameters");
				
				if (unnamedParamCount < 3)
					dataExpr.raiseValidateError("for matrix statement, must specify at least 3 arguments (in order): data, rows, cols");
				

				// assume: data, rows, cols, [byRow], [dimNames]
				dataExpr.addMatrixExprParam(DataExpression.RAND_DATA,passedParamExprs.get(0).getExpr());
				dataExpr.addMatrixExprParam(DataExpression.RAND_ROWS,passedParamExprs.get(1).getExpr());
				dataExpr.addMatrixExprParam(DataExpression.RAND_COLS,passedParamExprs.get(2).getExpr());
				
				if (unnamedParamCount >= 4)
					dataExpr.addMatrixExprParam(DataExpression.RAND_BY_ROW,passedParamExprs.get(3).getExpr());
				
				if (unnamedParamCount == 5)
					dataExpr.addMatrixExprParam(DataExpression.RAND_DIMNAMES,passedParamExprs.get(4).getExpr());
				
				if (unnamedParamCount > 5)
					dataExpr.raiseValidateError("for matrix statement, at most 5 arguments supported (in order): data, rows, cols, byrow, dimname");
								   
				
			} else {
				// handle first parameter, which is data and may be unnamed
				ParameterExpression firstParam = passedParamExprs.get(0);
				if (firstParam.getName() != null && !firstParam.getName().equals(DataExpression.RAND_DATA)){
					// throw exception -- must be filename as first parameter
					dataExpr.raiseValidateError("matrix method must have data parameter as first parameter or unnamed parameter");
				} else {
					dataExpr.addMatrixExprParam(DataExpression.RAND_DATA, passedParamExprs.get(0).getExpr());
				}
				
				for (int i=1; i<passedParamExprs.size(); i++){
					if (passedParamExprs.get(i).getName() == null){
						// throw exception -- cannot mix named and unnamed parameters
						dataExpr.raiseValidateError("for matrix statement, cannot mix named and unnamed parameters, only data parameter can be unnammed");
					} else {
						dataExpr.addMatrixExprParam(passedParamExprs.get(i).getName(), passedParamExprs.get(i).getExpr()); 	
					}
				}
			}
			dataExpr.setMatrixDefault();
		} // else if (functionName.equals("matrix")){
		
		if (dataExpr != null) {
			dataExpr.setAllPositions(filename, blp, bcp, elp, ecp);
		}
		return dataExpr;
	
	} // end method getBuiltinFunctionExpression
	
	public void addRandExprParam(String paramName, Expression paramValue) 
		throws LanguageException
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
			raiseValidateError("unexpected parameter \"" + paramName +
					"\". Legal parameters for Rand statement are " 
					+ "(capitalization-sensitive): " 	+ RAND_ROWS 	
					+ ", " + RAND_COLS		+ ", " + RAND_MIN + ", " + RAND_MAX  	
					+ ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", " + RAND_PDF + ", " + RAND_LAMBDA);
		}
		if (getVarParam(paramName) != null){
			raiseValidateError("attempted to add Rand statement parameter " + paramValue + " more than once");
		}
		// Process the case where user provides double values to rows or cols
		if (paramName.equals(RAND_ROWS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((long)((DoubleIdentifier)paramValue).getValue(), 
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
		}
		else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((long)((DoubleIdentifier)paramValue).getValue(),
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
		}
			
		// add the parameter to expression list
		paramValue.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		addVarParam(paramName,paramValue);
		
	}
	
	public void addMatrixExprParam(String paramName, Expression paramValue) 
		throws LanguageException
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
			raiseValidateError("unexpected parameter \"" + paramName +
					"\". Legal parameters for  matrix statement are " 
					+ "(capitalization-sensitive): " 	+ RAND_DATA + ", " + RAND_ROWS 	
					+ ", " + RAND_COLS		+ ", " + RAND_BY_ROW);
		}
		if (getVarParam(paramName) != null) {
			raiseValidateError("attempted to add matrix statement parameter " + paramValue + " more than once");
		}
		// Process the case where user provides double values to rows or cols
		if (paramName.equals(RAND_ROWS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((long)((DoubleIdentifier)paramValue).getValue(),
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
		}
		else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((long)((DoubleIdentifier)paramValue).getValue(),
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
		}
			
		// add the parameter to expression list
		paramValue.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		addVarParam(paramName,paramValue);
	}
	
	public DataExpression(DataOp op, HashMap<String,Expression> varParams, 
			String filename, int blp, int bcp, int elp, int ecp) {
		
		_kind = Kind.DataOp;
		_opcode = op;
		_varParams = varParams;
		this.setAllPositions(filename, blp, bcp, elp, ecp);
	}

	public Expression rewriteExpression(String prefix) throws LanguageException {
		
		HashMap<String,Expression> newVarParams = new HashMap<String,Expression>();
		for( Entry<String, Expression> e : _varParams.entrySet() ){
			String key = e.getKey();
			Expression newExpr = e.getValue().rewriteExpression(prefix);
			newVarParams.put(key, newExpr);
		}	
		DataExpression retVal = new DataExpression(_opcode, newVarParams,
				this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		retVal._strInit = this._strInit;
		
		return retVal;
	}

	/**
	 * By default we use rowwise matrix reshape according to our internal dense/sparse matrix representations.
	 * ByRow specifies both input and output orientation. Note that this is different from R, where inputs are 
	 * always read by-column and the default for byRow is by-column as well.
	 */
	public void setMatrixDefault(){
		if (getVarParam(RAND_BY_ROW) == null)
			addVarParam(RAND_BY_ROW, new BooleanIdentifier(true,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn()));
	}
	
	public void setRandDefault(){
		if (getVarParam(RAND_ROWS)== null){
			IntIdentifier id = new IntIdentifier(1L,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
			addVarParam(RAND_ROWS, 	id);
		}
		if (getVarParam(RAND_COLS)== null){
			IntIdentifier id = new IntIdentifier(1L,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
            addVarParam(RAND_COLS, 	id);
		}
		if (getVarParam(RAND_MIN)== null){
			DoubleIdentifier id = new DoubleIdentifier(0.0,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
			addVarParam(RAND_MIN, id);
		}
		if (getVarParam(RAND_MAX)== null){
			DoubleIdentifier id = new DoubleIdentifier(1.0,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
			addVarParam(RAND_MAX, id);
		}
		if (getVarParam(RAND_SPARSITY)== null){
			DoubleIdentifier id = new DoubleIdentifier(1.0,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
			addVarParam(RAND_SPARSITY,	id);
		}
		if (getVarParam(RAND_SEED)== null){
			IntIdentifier id = new IntIdentifier(DataGenOp.UNSPECIFIED_SEED,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
			addVarParam(RAND_SEED, id);
		}
		if (getVarParam(RAND_PDF)== null){
			StringIdentifier id = new StringIdentifier(RAND_PDF_UNIFORM,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
			addVarParam(RAND_PDF, id);
		}
		if (getVarParam(RAND_LAMBDA)== null){
			DoubleIdentifier id = new DoubleIdentifier(1.0,
					this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
					this.getBeginLine(), this.getBeginColumn());
			addVarParam(RAND_LAMBDA, id);
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
		
	}
	
	public void removeVarParam(String name) {
		_varParams.remove(name);
	}
	
	private String getInputFileName(HashMap<String, ConstIdentifier> currConstVars, boolean conditional) 
		throws LanguageException 
	{
		String filename = null;
		
		Expression fileNameExpr = getVarParam(IO_FILENAME);
		if (fileNameExpr instanceof ConstIdentifier){
			return fileNameExpr.toString();
		}
		else if (fileNameExpr instanceof BinaryExpression){
			BinaryExpression expr = (BinaryExpression)fileNameExpr;
							
			if (expr.getKind()== Expression.Kind.BinaryOp){
				Expression.BinaryOp op = expr.getOpCode();
				switch (op){
				case PLUS:
						filename = "";
						filename = fileNameCat(expr, currConstVars, filename, conditional);
						// Since we have computed the value of filename, we update
						// varParams with a const string value
						StringIdentifier fileString = new StringIdentifier(filename, 
								this.getFilename(), this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						removeVarParam(IO_FILENAME);
						addVarParam(IO_FILENAME, fileString);
					break;
				default:
					raiseValidateError("for read method, parameter " + IO_FILENAME + " can only be const string concatenations. ", conditional);
				}
			}
		}
		else {
			raiseValidateError("for read method, parameter " + IO_FILENAME + " can only be a const string or const string concatenations. ", conditional);
		}
		
		return filename;
	}
	
	public static String getMTDFileName(String inputFileName) throws LanguageException {
		String mtdName = inputFileName + ".mtd";
		
		//validate read filename
		if( !LocalFileUtils.validateExternalFilename(mtdName, true) )
			throw new LanguageException("Invalid (non-trustworthy) hdfs read filename.");

		return mtdName;
	}
	
	/**
	 * Validate parse tree : Process Data Expression in an assignment
	 * statement
	 *  
	 * @throws LanguageException
	 * @throws ParseException 
	 * @throws IOException 
	 */
	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars, boolean conditional)
			throws LanguageException 
	{		
		// validate all input parameters
		for ( Entry<String,Expression> e : getVarParams().entrySet() ) {
			String s = e.getKey();
			Expression inputParamExpr = e.getValue();
			
			if (inputParamExpr instanceof FunctionCallIdentifier) {
				raiseValidateError("UDF function call not supported as parameter to built-in function call", false,LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			inputParamExpr.validateExpression(ids, currConstVars, conditional);
			if ( getVarParam(s).getOutput().getDataType() != DataType.SCALAR && !s.equals(RAND_DATA)) {
				raiseValidateError("Non-scalar data types are not supported for data expression.", conditional,LanguageErrorCodes.INVALID_PARAMETERS);
			}	
		}	
	
		//general data expression constant propagation
		performConstantPropagationRand( currConstVars );
		performConstantPropagationReadWrite( currConstVars );
		
		// check if data parameter of matrix is scalar or matrix -- if scalar, use Rand instead
		Expression dataParam1 = getVarParam(RAND_DATA);		
		if (dataParam1 == null && getOpCode().equals(DataOp.MATRIX)){
			raiseValidateError("for matrix, must defined data parameter", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		if (dataParam1 != null && dataParam1.getOutput().getDataType() == DataType.SCALAR /*&& dataParam instanceof ConstIdentifier*/ ){
			//MB: note we should not check for const identifiers here, because otherwise all matrix constructors with
			//variable input are routed to a reshape operation (but it works only on matrices and hence, crashes)
			
			// replace DataOp MATRIX with RAND -- Rand handles matrix generation for Scalar values
			// replace data parameter with min / max within Rand case below
			this.setOpCode(DataOp.RAND);
		}		
		
		
		// IMPORTANT: for each operation, one must handle unnamed parameters
		
		switch (this.getOpCode()) {
		
		case READ:
					
			if (getVarParam(DATATYPEPARAM) != null && !(getVarParam(DATATYPEPARAM) instanceof StringIdentifier)){
				raiseValidateError("for read statement, parameter " + DATATYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.MATRIX_DATA_TYPE +", " + Statement.SCALAR_DATA_TYPE, conditional);
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
						|| getVarParam(DELIM_FILL_VALUE) != null
						|| getVarParam(DELIM_NA_STRINGS) != null
						)
				{
					raiseValidateError("Invalid parameters in read statement of a scalar: " +
							toString() + ". Only " + VALUETYPEPARAM + " is allowed.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}
			
			JSONObject configObject = null;	

			// Process expressions in input filename
			String inputFileName = getInputFileName(currConstVars, conditional);
			
			// Obtain and validate metadata filename
			String mtdFileName = getMTDFileName(inputFileName);

			// track whether should attempt to read MTD file or not
			boolean shouldReadMTD = _checkMetadata && !ConfigurationManager
					.getCompilerConfigFlag(ConfigType.IGNORE_READ_WRITE_METADATA);

			// Check for file existence (before metadata parsing for meaningful error messages)
			if( shouldReadMTD //skip check for jmlc/mlcontext
				&& !MapReduceTool.existsFileOnHDFS(inputFileName)) 
			{
				String fsext = InfrastructureAnalyzer.isLocalMode() ? "FS (local mode)" : "HDFS";
				raiseValidateError("Read input file does not exist on "+fsext+": " + 
						inputFileName, conditional, LanguageErrorCodes.INVALID_PARAMETERS);								
			}

			// track whether format type has been inferred 
			boolean inferredFormatType = false;
			
			// get format type string
			String formatTypeString = (getVarParam(FORMAT_TYPE) == null) ? null : getVarParam(FORMAT_TYPE).toString();
			
			// check if file is matrix market format
			if (formatTypeString == null && shouldReadMTD){
				boolean isMatrixMarketFormat = checkHasMatrixMarketFormat(inputFileName, mtdFileName, conditional); 
				if (isMatrixMarketFormat){
					
					formatTypeString = FORMAT_TYPE_VALUE_MATRIXMARKET;
					addVarParam(FORMAT_TYPE,new StringIdentifier(FORMAT_TYPE_VALUE_MATRIXMARKET,
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
					inferredFormatType = true;
					shouldReadMTD = false;
				}
			}
			
			// check if file is delimited format
			if (formatTypeString == null && shouldReadMTD ) {
				boolean isDelimitedFormat = checkHasDelimitedFormat(inputFileName, conditional); 
				
				if (isDelimitedFormat){
					addVarParam(FORMAT_TYPE,new StringIdentifier(FORMAT_TYPE_VALUE_CSV,
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
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
						raiseValidateError("Invalid parameters in readMM statement: " +
								toString() + ". Only " + IO_FILENAME + " is allowed.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
				}
				
				
				// should NOT attempt to read MTD file for MatrixMarket format
				shouldReadMTD = false;
				
				// get metadata from MatrixMarket format file
				String[] headerLines = readMatrixMarketFile(inputFileName, conditional);
				
				// process 1st line of MatrixMarket format -- must be identical to legal header
				String legalHeaderMM = "%%MatrixMarket matrix coordinate real general";
				
				if (headerLines != null && headerLines.length >= 2){
					String firstLine = headerLines[0].trim();
					if (!firstLine.equals(legalHeaderMM)){
						raiseValidateError("Unsupported format in MatrixMarket file: " +
								headerLines[0] + ". Only supported format in MatrixMarket file has header line " + legalHeaderMM, 
								conditional, LanguageErrorCodes.INVALID_PARAMETERS);
						}
				
					// process 2nd line of MatrixMarket format -- must have size information
				
				
					String secondLine = headerLines[1];
					String[] sizeInfo = secondLine.trim().split("\\s+");
					if (sizeInfo.length != 3){
						raiseValidateError("Unsupported size line in MatrixMarket file: " +
								headerLines[1] + ". Only supported format in MatrixMarket file has size line: <NUM ROWS> <NUM COLS> <NUM NON-ZEROS>, where each value is an integer.", conditional);
					}
				
					long rowsCount = -1, colsCount = -1, nnzCount = -1;
					try {
						rowsCount = Long.parseLong(sizeInfo[0]);
						if (rowsCount < 1)
							throw new Exception("invalid rows count");
						addVarParam(READROWPARAM, new IntIdentifier(rowsCount,
								this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
								this.getBeginLine(), this.getBeginColumn()));
					}
					catch(Exception e){
						raiseValidateError("In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid row count " + sizeInfo[0] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1], conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
				
					try {
						colsCount = Long.parseLong(sizeInfo[1]);
						if (colsCount < 1)
							throw new Exception("invalid cols count");
						addVarParam(READCOLPARAM, new IntIdentifier(colsCount,
								this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
								this.getBeginLine(), this.getBeginColumn()));
					}
					catch(Exception e){
						raiseValidateError("In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid column count " + sizeInfo[1] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1], conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
					
					try {
						nnzCount = Long.parseLong(sizeInfo[2]);
						if (nnzCount < 1)
							throw new Exception("invalid nnz count");
						addVarParam("nnz", new IntIdentifier(nnzCount, 
								this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
								this.getBeginLine(), this.getBeginColumn()));
					}
					catch(Exception e){
						raiseValidateError("In MatrixMarket file " + getVarParam(IO_FILENAME) 
								+  " invalid number non-zeros " + sizeInfo[2] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1], conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}	
				}
			}
			
			configObject = null;
			
			if (shouldReadMTD){
				configObject = readMetadataFile(mtdFileName, conditional);
		        		    
		        // if the MTD file exists, check the values specified in read statement match values in metadata MTD file
		        if (configObject != null){
		        	parseMetaDataFileParameters(mtdFileName, configObject, conditional);
		        	inferredFormatType = true;
		        }
		        else {
		        	LOG.warn("Metadata file: " + new Path(mtdFileName) + " not provided");
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
				if( !inferredFormatType ){
					for (String key : _varParams.keySet()){
						if (!  (key.equals(IO_FILENAME) || key.equals(FORMAT_TYPE) 
								|| key.equals(DELIM_HAS_HEADER_ROW) || key.equals(DELIM_DELIMITER) 
								|| key.equals(DELIM_FILL) || key.equals(DELIM_FILL_VALUE)
								|| key.equals(READROWPARAM) || key.equals(READCOLPARAM)
								|| key.equals(READNUMNONZEROPARAM) || key.equals(DATATYPEPARAM) || key.equals(VALUETYPEPARAM)
								|| key.equals(SCHEMAPARAM)) )
						{	
							String msg = "Only parameters allowed are: " + IO_FILENAME     + "," 
									   + SCHEMAPARAM + "," 
									   + DELIM_HAS_HEADER_ROW   + "," 
									   + DELIM_DELIMITER 	+ ","
									   + DELIM_FILL 		+ ","
									   + DELIM_FILL_VALUE 	+ ","
									   + READROWPARAM     + "," 
									   + READCOLPARAM;
							
							raiseValidateError("Invalid parameter " + key + " in read statement: " +
									toString() + ". " + msg, conditional, LanguageErrorCodes.INVALID_PARAMETERS);
						}
					}
				}
				
				// DEFAULT for "sep" : ","
				if (getVarParam(DELIM_DELIMITER) == null){
					addVarParam(DELIM_DELIMITER, new StringIdentifier(DEFAULT_DELIM_DELIMITER,
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
				}
				else {
					if ( (getVarParam(DELIM_DELIMITER) instanceof ConstIdentifier)
						&& (! (getVarParam(DELIM_DELIMITER) instanceof StringIdentifier)))
					{
						raiseValidateError("For delimited file '" + getVarParam(DELIM_DELIMITER) 
								+  "' must be a string value ", conditional);
					}
				} 
				
				// DEFAULT for "default": 0
				if (getVarParam(DELIM_FILL_VALUE) == null){
					addVarParam(DELIM_FILL_VALUE, new DoubleIdentifier(DEFAULT_DELIM_FILL_VALUE,
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
				}
				else {
					if ( (getVarParam(DELIM_FILL_VALUE) instanceof ConstIdentifier)
							&& (! (getVarParam(DELIM_FILL_VALUE) instanceof IntIdentifier ||  getVarParam(DELIM_FILL_VALUE) instanceof DoubleIdentifier)))
					{
						raiseValidateError("For delimited file '" + getVarParam(DELIM_FILL_VALUE)  +  "' must be a numeric value ", conditional);
					}
				} 
				
				// DEFAULT for "header": boolean false
				if (getVarParam(DELIM_HAS_HEADER_ROW) == null){
					addVarParam(DELIM_HAS_HEADER_ROW, new BooleanIdentifier(DEFAULT_DELIM_HAS_HEADER_ROW,
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
				}
				else {
					if ((getVarParam(DELIM_HAS_HEADER_ROW) instanceof ConstIdentifier)
						&& (! (getVarParam(DELIM_HAS_HEADER_ROW) instanceof BooleanIdentifier)))
					{
						raiseValidateError("For delimited file '" + getVarParam(DELIM_HAS_HEADER_ROW) + "' must be a boolean value ", conditional);
					}
				}
				
				// DEFAULT for "fill": boolean false
				if (getVarParam(DELIM_FILL) == null){
					addVarParam(DELIM_FILL,new BooleanIdentifier(DEFAULT_DELIM_FILL,
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
				}
				else {
					
					if ((getVarParam(DELIM_FILL) instanceof ConstIdentifier)
							&& (! (getVarParam(DELIM_FILL) instanceof BooleanIdentifier)))
					{
						raiseValidateError("For delimited file '" + getVarParam(DELIM_FILL) + "' must be a boolean value ", conditional);
					}
				}		
			} 
	        dataTypeString = (getVarParam(DATATYPEPARAM) == null) ? null : getVarParam(DATATYPEPARAM).toString();
			
			if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(Statement.MATRIX_DATA_TYPE) 
					|| dataTypeString.equalsIgnoreCase(Statement.FRAME_DATA_TYPE)) {
				
				boolean isMatrix = false;
				if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(Statement.MATRIX_DATA_TYPE))
						isMatrix = true;
				
				// set data type
		        getOutput().setDataType(isMatrix ? DataType.MATRIX : DataType.FRAME);
		        
		        // set number non-zeros
		        Expression ennz = this.getVarParam("nnz");
		        long nnz = -1;
		        if( ennz != null )
		        {
			        nnz = Long.valueOf(ennz.toString());
			        getOutput().setNnz(nnz);
		        }
		        
		        // Following dimension checks must be done when data type = MATRIX_DATA_TYPE 
				// initialize size of target data identifier to UNKNOWN
				getOutput().setDimensions(-1, -1);
				
				if ( !isCSV && ConfigurationManager.getCompilerConfig()
						.getBool(ConfigType.REJECT_READ_WRITE_UNKNOWNS) //skip check for csv format / jmlc api
					&& (getVarParam(READROWPARAM) == null || getVarParam(READCOLPARAM) == null) ) {
						raiseValidateError("Missing or incomplete dimension information in read statement: " 
								+ mtdFileName, conditional, LanguageErrorCodes.INVALID_PARAMETERS);				
				}
				
				if (getVarParam(READROWPARAM) instanceof ConstIdentifier 
					&& getVarParam(READCOLPARAM) instanceof ConstIdentifier)  
				{
					// these are strings that are long values
					Long dim1 = (getVarParam(READROWPARAM) == null) ? null : Long.valueOf( getVarParam(READROWPARAM).toString());
					Long dim2 = (getVarParam(READCOLPARAM) == null) ? null : Long.valueOf( getVarParam(READCOLPARAM).toString());					
					if ( !isCSV && (dim1 <= 0 || dim2 <= 0) && ConfigurationManager
							.getCompilerConfig().getBool(ConfigType.REJECT_READ_WRITE_UNKNOWNS) ) {
						raiseValidateError("Invalid dimension information in read statement", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
					
					// set dim1 and dim2 values 
					if (dim1 != null && dim2 != null){
						getOutput().setDimensions(dim1, dim2);
					} else if (!isCSV && ((dim1 != null) || (dim2 != null))) {
						raiseValidateError("Partial dimension information in read statement", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}	
				}
				
				// initialize block dimensions to UNKNOWN 
				getOutput().setBlockDimensions(-1, -1);
				
				// find "format": 1=text, 2=binary
				int format = 1; // default is "text"
				String fmt =  (getVarParam(FORMAT_TYPE) == null ? null : getVarParam(FORMAT_TYPE).toString());
				
				if (fmt == null || fmt.equalsIgnoreCase("text")){
					getOutput().setFormatType(FormatType.TEXT);
					format = 1;
				} else if ( fmt.equalsIgnoreCase("binary") ) {
					getOutput().setFormatType(FormatType.BINARY);
					format = 2;
				} else if ( fmt.equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV)) 
				{
					getOutput().setFormatType(FormatType.CSV);
					format = 1;
				} 
				else if ( fmt.equalsIgnoreCase(FORMAT_TYPE_VALUE_MATRIXMARKET) )
				{
					getOutput().setFormatType(FormatType.MM);
					format = 1;
				} else {
					raiseValidateError("Invalid format '" + fmt+ "' in statement: " + this.toString(), conditional);
				}
				
				if (getVarParam(ROWBLOCKCOUNTPARAM) instanceof ConstIdentifier && getVarParam(COLUMNBLOCKCOUNTPARAM) instanceof ConstIdentifier)  {
				
					Long rowBlockCount = (getVarParam(ROWBLOCKCOUNTPARAM) == null) ? null : Long.valueOf(getVarParam(ROWBLOCKCOUNTPARAM).toString());
					Long columnBlockCount = (getVarParam(COLUMNBLOCKCOUNTPARAM) == null) ? null : Long.valueOf(getVarParam(COLUMNBLOCKCOUNTPARAM).toString());
		
					if ((rowBlockCount != null) && (columnBlockCount != null)) {
						getOutput().setBlockDimensions(rowBlockCount, columnBlockCount);
					} else if ((rowBlockCount != null) || (columnBlockCount != null)) {
						raiseValidateError("Partial block dimension information in read statement", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					} else {
						 getOutput().setBlockDimensions(-1, -1);
					}
				}
				
				// block dimensions must be -1x-1 when format="text"
				// NOTE MB: disabled validate of default blocksize for inputs w/ format="binary"
				// because we automatically introduce reblocks if blocksizes don't match
				if ( ( (format == 1 || !isMatrix) 
						&& (getOutput().getRowsInBlock() != -1 || getOutput().getColumnsInBlock() != -1)
					 ) ){
					raiseValidateError("Invalid block dimensions (" + getOutput().getRowsInBlock() + "," + getOutput().getColumnsInBlock() + ") when format=" + getVarParam(FORMAT_TYPE) + " in \"" + this.toString() + "\".", conditional);
				}
			
			}
			else if ( dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)) {
				getOutput().setDataType(DataType.SCALAR);
				getOutput().setNnz(-1L);
			}
			
			else{		
				raiseValidateError("Unknown Data Type " + dataTypeString + ". Valid  values: " + Statement.SCALAR_DATA_TYPE +", " + Statement.MATRIX_DATA_TYPE, conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			// handle value type parameter
			if (getVarParam(VALUETYPEPARAM) != null && !(getVarParam(VALUETYPEPARAM) instanceof StringIdentifier)){
				raiseValidateError("for read method, parameter " + VALUETYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE, conditional);
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
					raiseValidateError("Unknown Value Type " + valueTypeString
							+ ". Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE, conditional);
				}
			} else {
				getOutput().setValueType(ValueType.DOUBLE);
			}

			break; 
			
		case WRITE:
			
			// for delimited format, if no delimiter specified THEN set default ","
			if (getVarParam(FORMAT_TYPE) == null || getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV)){
				if (getVarParam(DELIM_DELIMITER) == null){
					addVarParam(DELIM_DELIMITER, new StringIdentifier(DEFAULT_DELIM_DELIMITER,
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
				}
				if (getVarParam(DELIM_HAS_HEADER_ROW) == null){
					addVarParam(DELIM_HAS_HEADER_ROW, new BooleanIdentifier(DEFAULT_DELIM_HAS_HEADER_ROW,
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
				}
				if (getVarParam(DELIM_SPARSE) == null){
					addVarParam(DELIM_SPARSE, new BooleanIdentifier(DEFAULT_DELIM_SPARSE, 
							this.getFilename(), this.getBeginLine(), this.getBeginColumn(),
							this.getBeginLine(), this.getBeginColumn()));
				}
			}
			
			/* NOTE MB: disabled filename concatenation because we now support dynamic rewrite
			if (getVarParam(IO_FILENAME) instanceof BinaryExpression){
				BinaryExpression expr = (BinaryExpression)getVarParam(IO_FILENAME);
								
				if (expr.getKind()== Expression.Kind.BinaryOp){
					Expression.BinaryOp op = expr.getOpCode();
					switch (op){
						case PLUS:
							mtdFileName = "";
							mtdFileName = fileNameCat(expr, currConstVars, mtdFileName);
							// Since we have computed the value of filename, we update
							// varParams with a const string value
							StringIdentifier fileString = new StringIdentifier(mtdFileName, 
									this.getFilename(), this.getBeginLine(), this.getBeginColumn(), 
									this.getEndLine(), this.getEndColumn());
							removeVarParam(IO_FILENAME);
							addVarParam(IO_FILENAME, fileString);
												
							break;
						default:
							raiseValidateError("for OutputStatement, parameter " + IO_FILENAME 
									+ " can only be a const string or const string concatenations. ", 
									conditional);
					}
				}
			}*/
			
			//validate read filename
			String fnameWrite = getVarParam(IO_FILENAME).toString();
			if( !LocalFileUtils.validateExternalFilename(fnameWrite, true) ) //always unconditional
				raiseValidateError("Invalid (non-trustworthy) hdfs write filename.", false);
	    	
			
			if (getVarParam(FORMAT_TYPE) == null || getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase("text"))
				getOutput().setBlockDimensions(-1, -1);
			else if (getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase("binary"))
				getOutput().setBlockDimensions(ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());
			else if (getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_MATRIXMARKET) || (getVarParam(FORMAT_TYPE).toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV)))
				getOutput().setBlockDimensions(-1, -1);
			
			else{
				raiseValidateError("Invalid format " + getVarParam(FORMAT_TYPE) +  " in statement: " + this.toString(), conditional);
			}
			break;

			case RAND: 
			
			Expression dataParam = getVarParam(RAND_DATA);
			
			if( dataParam != null ) 
			{
				// handle input variable (matrix/scalar) 
				if( dataParam instanceof DataIdentifier )
				{		
					addVarParam(RAND_MIN, dataParam);
					addVarParam(RAND_MAX, dataParam);
				}
				// handle integer constant 
				else if (dataParam instanceof IntIdentifier) {
					long roundedValue = ((IntIdentifier)dataParam).getValue();
					Expression minExpr = new DoubleIdentifier(roundedValue, getFilename(), 
							getBeginLine(), getBeginColumn(), getEndLine(), getEndColumn());
					addVarParam(RAND_MIN, minExpr);
					addVarParam(RAND_MAX, minExpr);
				}
				// handle double constant 
				else if (dataParam instanceof DoubleIdentifier) {
					double roundedValue = ((DoubleIdentifier)dataParam).getValue();
					Expression minExpr = new DoubleIdentifier(roundedValue, getFilename(), 
							getBeginLine(), getBeginColumn(), getEndLine(), getEndColumn());
					addVarParam(RAND_MIN, minExpr);
					addVarParam(RAND_MAX, minExpr);				
				}
				// handle string constant (string init) 
				else if (dataParam instanceof StringIdentifier) {
					String data = ((StringIdentifier)dataParam).getValue();
					Expression minExpr = new StringIdentifier(data, getFilename(), 
							getBeginLine(), getBeginColumn(), getEndLine(), getEndColumn());
					addVarParam(RAND_MIN, minExpr);
					addVarParam(RAND_MAX, minExpr);	
					_strInit = true;
				}
				else {
					// handle general expression
					dataParam.validateExpression(ids, currConstVars, conditional);
					addVarParam(RAND_MIN, dataParam);
					addVarParam(RAND_MAX, dataParam);
				}
				
				removeVarParam(RAND_DATA);
				removeVarParam(RAND_BY_ROW);
				this.setRandDefault();
			}
			
			//check valid parameters
			for( String key : _varParams.keySet() ) {
				boolean found = false;
				for (String name : RAND_VALID_PARAM_NAMES){
					found |= name.equals(key);
				}
				if (!found){
					raiseValidateError("unexpected parameter \"" + key +
							"\". Legal parameters for Rand statement are " 
							+ "(capitalization-sensitive): " 	+ RAND_ROWS 	
							+ ", " + RAND_COLS		+ ", " + RAND_MIN + ", " + RAND_MAX  	
							+ ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", " + RAND_PDF  + ", " + RAND_LAMBDA, conditional);
				}
			}
			
			//parameters w/ support for variable inputs
			if (getVarParam(RAND_ROWS) instanceof StringIdentifier || getVarParam(RAND_ROWS) instanceof BooleanIdentifier){
				raiseValidateError("for Rand statement " + RAND_ROWS + " has incorrect value type", conditional);
			}				
			
			if (getVarParam(RAND_COLS) instanceof StringIdentifier || getVarParam(RAND_COLS) instanceof BooleanIdentifier){
				raiseValidateError("for Rand statement " + RAND_COLS + " has incorrect value type", conditional);
			}
			
			if (getVarParam(RAND_SEED) instanceof StringIdentifier || getVarParam(RAND_SEED) instanceof BooleanIdentifier) {
				raiseValidateError("for Rand statement " + RAND_SEED + " has incorrect value type", conditional);
			}
			
			if ((getVarParam(RAND_MAX) instanceof StringIdentifier && !_strInit) || getVarParam(RAND_MAX) instanceof BooleanIdentifier) {
				raiseValidateError("for Rand statement " + RAND_MAX + " has incorrect value type", conditional);
			}
			
			if ((getVarParam(RAND_MIN) instanceof StringIdentifier && !_strInit) || getVarParam(RAND_MIN) instanceof BooleanIdentifier) {
				raiseValidateError("for Rand statement " + RAND_MIN + " has incorrect value type", conditional);
			}
			
			//parameters w/o support for variable inputs (requires double/int or string constants)
			if (!(getVarParam(RAND_SPARSITY) instanceof DoubleIdentifier || getVarParam(RAND_SPARSITY) instanceof IntIdentifier)) {
				raiseValidateError("for Rand statement " + RAND_SPARSITY + " has incorrect value type", conditional);
			}
			
			if (!(getVarParam(RAND_PDF) instanceof StringIdentifier)) {
				raiseValidateError("for Rand statement " + RAND_PDF + " has incorrect value type", conditional);
			}
	
			Expression lambda = getVarParam(RAND_LAMBDA);
			if (!( (lambda instanceof DataIdentifier 
					|| lambda instanceof ConstIdentifier) 
				&& (lambda.getOutput().getValueType() == ValueType.DOUBLE 
					|| lambda.getOutput().getValueType() == ValueType.INT) )) {
				raiseValidateError("for Rand statement " + RAND_LAMBDA + " has incorrect data type", conditional);
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
					raiseValidateError("In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue(), conditional);
				}
			}
			else if (rowsExpr instanceof DoubleIdentifier) {
				if  (((DoubleIdentifier)rowsExpr).getValue() >= 1 ) {
					rowsLong = UtilFunctions.toLong(Math.floor(((DoubleIdentifier)rowsExpr).getValue()));
				}
				else {
					raiseValidateError("In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + rowsExpr.toString(), conditional);
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
							raiseValidateError("In rand statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
						// update row expr with new IntIdentifier 
						long roundedValue = ((IntIdentifier)constValue).getValue();
						rowsExpr = new IntIdentifier(roundedValue, this.getFilename(), 
								this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_ROWS, rowsExpr);
						rowsLong = roundedValue; 
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
						
						if (((DoubleIdentifier)constValue).getValue() < 1.0){
							raiseValidateError("In rand statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
						// update row expr with new IntIdentifier (rounded down)
						long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
						rowsExpr = new IntIdentifier(roundedValue, this.getFilename(),
								this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_ROWS, rowsExpr);
						rowsLong = roundedValue; 
						
					}
					else {
						// exception -- rows must be integer or double constant
						raiseValidateError("In rand statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
					}
				}
				else {
					// handle general expression
					rowsExpr.validateExpression(ids, currConstVars, conditional);
				}
			}	
			else {
				// handle general expression
				rowsExpr.validateExpression(ids, currConstVars, conditional);
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
					raiseValidateError("In rand statement, can only assign cols a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString(), conditional);
				}
			}
			else if (colsExpr instanceof DoubleIdentifier) {
				if  (((DoubleIdentifier)colsExpr).getValue() >= 1 ) {
					colsLong = Double.valueOf((Math.floor(((DoubleIdentifier)colsExpr).getValue()))).longValue();
				}
				else {
					raiseValidateError("In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString(), conditional);
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
							raiseValidateError("In rand statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
						// update col expr with new IntIdentifier 
						long roundedValue = ((IntIdentifier)constValue).getValue();
						colsExpr = new IntIdentifier(roundedValue, this.getFilename(), 
								this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_COLS, colsExpr);
						colsLong = roundedValue; 
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
						
						if (((DoubleIdentifier)constValue).getValue() < 1){
							raiseValidateError("In rand statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
						// update col expr with new IntIdentifier (rounded down)
						long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
						colsExpr = new IntIdentifier(roundedValue, this.getFilename(), 
								this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_COLS, colsExpr);
						colsLong = roundedValue; 
						
					}
					else {
						// exception -- rows must be integer or double constant
						raiseValidateError("In rand statement, can only assign cols a long " +
								"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
					}
				}
				else {
					// handle general expression
					colsExpr.validateExpression(ids, currConstVars, conditional);
				}
					
			}	
			else {
				// handle general expression
				colsExpr.validateExpression(ids, currConstVars, conditional);
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
						minExpr = new DoubleIdentifier(roundedValue, this.getFilename(), 
								this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_MIN, minExpr);
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
		
						// update col expr with new IntIdentifier (rounded down)
						double roundedValue = ((DoubleIdentifier)constValue).getValue();
						minExpr = new DoubleIdentifier(roundedValue, this.getFilename(), 
								this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_MIN, minExpr);
						
					}
					else {
						// exception -- rows must be integer or double constant
						raiseValidateError("In rand statement, can only assign min a numerical " +
								"value -- attempted to assign: " + constValue.toString(), conditional);
					}
				}
				else {
					// handle general expression
					minExpr.validateExpression(ids, currConstVars, conditional);
				}
					
			}	
			else {
				// handle general expression
				minExpr.validateExpression(ids, currConstVars, conditional);
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
						maxExpr = new DoubleIdentifier(roundedValue, this.getFilename(), 
								this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_MAX, maxExpr);
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
		
						// update col expr with new IntIdentifier (rounded down)
						double roundedValue = ((DoubleIdentifier)constValue).getValue();
						maxExpr = new DoubleIdentifier(roundedValue, this.getFilename(), 
								this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						addVarParam(RAND_MAX, maxExpr);
						
					}
					else {
						// exception -- rows must be integer or double constant
						raiseValidateError("In rand statement, can only assign max a numerical " +
								"value -- attempted to assign: " + constValue.toString(), conditional);
					}
				}
				else {
					// handle general expression
					maxExpr.validateExpression(ids, currConstVars, conditional);
				}		
			}	
			else {
				// handle general expression
				maxExpr.validateExpression(ids, currConstVars, conditional);
			}
		
			getOutput().setFormatType(FormatType.BINARY);
			getOutput().setDataType(DataType.MATRIX);
			getOutput().setValueType(ValueType.DOUBLE);
			getOutput().setDimensions(rowsLong, colsLong);
			
			if (getOutput() instanceof IndexedIdentifier){
				// process the "target" being indexed
				DataIdentifier targetAsSeen = ids.get(((DataIdentifier)getOutput()).getName());
				if (targetAsSeen == null){
					raiseValidateError("cannot assign value to indexed identifier " + ((DataIdentifier)getOutput()).getName() + " without first initializing " + ((DataIdentifier)getOutput()).getName(), conditional);
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
			
			//handle default and input arguments
			setMatrixDefault();
			for( String key : _varParams.keySet() ) 
			{
				boolean found = false;
				for (String name : MATRIX_VALID_PARAM_NAMES) {
					found |= name.equals(key);
				}
				if( !found ) {
					raiseValidateError("unexpected parameter \"" + key + "\". "
							+ "Legal parameters for matrix statement are (case-sensitive): " 	
							+ RAND_DATA + ", " + RAND_ROWS	+ ", " + RAND_COLS + ", " + RAND_BY_ROW, conditional);
				}
			}
			
			//validate correct value types
			if (getVarParam(RAND_DATA) != null && (getVarParam(RAND_DATA) instanceof BooleanIdentifier)){
				raiseValidateError("for matrix statement " + RAND_DATA + " has incorrect value type", conditional);
			}
			if (getVarParam(RAND_ROWS) != null && (getVarParam(RAND_ROWS) instanceof StringIdentifier || getVarParam(RAND_ROWS) instanceof BooleanIdentifier)){
				raiseValidateError("for matrix statement " + RAND_ROWS + " has incorrect value type", conditional);
			}				
			if (getVarParam(RAND_COLS) != null && (getVarParam(RAND_COLS) instanceof StringIdentifier || getVarParam(RAND_COLS) instanceof BooleanIdentifier)){
				raiseValidateError("for matrix statement " + RAND_COLS + " has incorrect value type", conditional);
			}				
			if ( !(getVarParam(RAND_BY_ROW) instanceof BooleanIdentifier)) {
				raiseValidateError("for matrix statement " + RAND_BY_ROW + " has incorrect value type", conditional);
			}
			
			//validate general data expression
			getVarParam(RAND_DATA).validateExpression(ids, currConstVars, conditional);
			
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
						raiseValidateError("In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue(), conditional);
					}
				}
				else if (rowsExpr instanceof DoubleIdentifier) {
					if  (((DoubleIdentifier)rowsExpr).getValue() >= 1 ) {
						rowsLong = Double.valueOf((Math.floor(((DoubleIdentifier)rowsExpr).getValue()))).longValue();
					}
					else {
						raiseValidateError("In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + rowsExpr.toString(), conditional);
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
								raiseValidateError("In matrix statement, can only assign rows a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update row expr with new IntIdentifier 
							long roundedValue = ((IntIdentifier)constValue).getValue();
							rowsExpr = new IntIdentifier(roundedValue, this.getFilename(), 
									this.getBeginLine(), this.getBeginColumn(), 
									this.getEndLine(), this.getEndColumn());
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue; 
						}
						// handle double constant 
						else if (constValue instanceof DoubleIdentifier){
							
							if (((DoubleIdentifier)constValue).getValue() < 1.0){
								raiseValidateError("In matrix statement, can only assign rows a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update row expr with new IntIdentifier (rounded down)
							long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
							rowsExpr = new IntIdentifier(roundedValue,  this.getFilename(), 
									this.getBeginLine(), this.getBeginColumn(), 
									this.getEndLine(), this.getEndColumn());
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue; 
							
						}
						else {
							// exception -- rows must be integer or double constant
							raiseValidateError("In matrix statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
					}
					else {
						// handle general expression
						rowsExpr.validateExpression(ids, currConstVars, conditional);
					}
				}	
				else {
					// handle general expression
					rowsExpr.validateExpression(ids, currConstVars, conditional);
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
						raiseValidateError("In matrix statement, can only assign cols a long " +
								"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString(), conditional);
					}
				}
				else if (colsExpr instanceof DoubleIdentifier) {
					if  (((DoubleIdentifier)colsExpr).getValue() >= 1 ) {
						colsLong = Double.valueOf((Math.floor(((DoubleIdentifier)colsExpr).getValue()))).longValue();
					}
					else {
						raiseValidateError("In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString(), conditional);
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
								raiseValidateError("In matrix statement, can only assign cols a long " +
										"(integer) value >= 1 -- attempted to assign value: " 
										+ constValue.toString(), conditional);
							}
							// update col expr with new IntIdentifier 
							long roundedValue = ((IntIdentifier)constValue).getValue();
							colsExpr = new IntIdentifier(roundedValue, this.getFilename(), 
									this.getBeginLine(), this.getBeginColumn(), 
									this.getEndLine(), this.getEndColumn());
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue; 
						}
						// handle double constant 
						else if (constValue instanceof DoubleIdentifier){
							
							if (((DoubleIdentifier)constValue).getValue() < 1){
								raiseValidateError("In matrix statement, can only assign cols a long " +
										"(integer) value >= 1 -- attempted to assign value: " 
										+ constValue.toString(), conditional);
							}
							// update col expr with new IntIdentifier (rounded down)
							long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
							colsExpr = new IntIdentifier(roundedValue, this.getFilename(), 
									this.getBeginLine(), this.getBeginColumn(), 
									this.getEndLine(), this.getEndColumn());
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue; 
							
						}
						else {
							// exception -- rows must be integer or double constant
							raiseValidateError("In matrix statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
					}
					else {
						// handle general expression
						colsExpr.validateExpression(ids, currConstVars, conditional);
					}
						
				}	
				else {
					// handle general expression
					colsExpr.validateExpression(ids, currConstVars, conditional);
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
			raiseValidateError("Unsupported Data expression"+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS); //always unconditional
		}
		return;
	}
	
	/**
	 * 
	 * @param currConstVars
	 */
	private void performConstantPropagationRand( HashMap<String, ConstIdentifier> currConstVars )
	{
		//here, we propagate constants for all rand parameters that are required during validate.
		String[] paramNamesForEval = new String[]{RAND_DATA, RAND_SPARSITY, RAND_MIN, RAND_MAX};
		
		//replace data identifiers with const identifiers
		performConstantPropagation(currConstVars, paramNamesForEval);
	}
	
	/**
	 * 
	 * @param currConstVars
	 */
	private void performConstantPropagationReadWrite( HashMap<String, ConstIdentifier> currConstVars )
	{
		//here, we propagate constants for all read/write parameters that are required during validate.
		String[] paramNamesForEval = new String[]{FORMAT_TYPE, IO_FILENAME, READROWPARAM, READCOLPARAM, READNUMNONZEROPARAM};
		
		//replace data identifiers with const identifiers
		performConstantPropagation(currConstVars, paramNamesForEval);
	}
	
	/**
	 * 
	 * @param currConstVars
	 * @param paramNames
	 */
	private void performConstantPropagation( HashMap<String, ConstIdentifier> currConstVars, String[] paramNames )
	{
		for( String paramName : paramNames )
		{
			Expression paramExp = getVarParam(paramName);
			if (   paramExp != null && paramExp instanceof DataIdentifier && !(paramExp instanceof IndexedIdentifier) 
				&& currConstVars.containsKey(((DataIdentifier) paramExp).getName()))
			{
				addVarParam(paramName, currConstVars.get(((DataIdentifier)paramExp).getName()));
			}				
		}
	}
	
	
	private String fileNameCat(BinaryExpression expr, HashMap<String, ConstIdentifier> currConstVars, String filename, boolean conditional)
		throws LanguageException
	{
		// Processing the left node first
		if (expr.getLeft() instanceof BinaryExpression 
				&& ((BinaryExpression)expr.getLeft()).getKind()== BinaryExpression.Kind.BinaryOp
				&& ((BinaryExpression)expr.getLeft()).getOpCode() == BinaryOp.PLUS){
			filename = fileNameCat((BinaryExpression)expr.getLeft(), currConstVars, filename, conditional)+ filename;
		}
		else if (expr.getLeft() instanceof ConstIdentifier){
			filename = ((ConstIdentifier)expr.getLeft()).toString()+ filename;
		}
		else if (expr.getLeft() instanceof DataIdentifier 
				&& ((DataIdentifier)expr.getLeft()).getDataType() == Expression.DataType.SCALAR
				&& ((DataIdentifier)expr.getLeft()).getKind() == Expression.Kind.Data){ 
				//&& ((DataIdentifier)expr.getLeft()).getValueType() == Expression.ValueType.STRING){
			String name = ((DataIdentifier)expr.getLeft()).getName();
			filename = ((StringIdentifier)currConstVars.get(name)).getValue() + filename;
		}
		else {
			raiseValidateError("Parameter " + IO_FILENAME + " only supports a const string or const string concatenations.", conditional);
		}
		// Now process the right node
		if (expr.getRight()instanceof BinaryExpression 
				&& ((BinaryExpression)expr.getRight()).getKind()== BinaryExpression.Kind.BinaryOp
				&& ((BinaryExpression)expr.getRight()).getOpCode() == BinaryOp.PLUS){
			filename = filename + fileNameCat((BinaryExpression)expr.getRight(), currConstVars, filename, conditional);
		}
		// DRB: CHANGE
		else if (expr.getRight() instanceof ConstIdentifier){
			filename = filename + ((ConstIdentifier)expr.getRight()).toString();
		}
		else if (expr.getRight() instanceof DataIdentifier 
				&& ((DataIdentifier)expr.getRight()).getDataType() == Expression.DataType.SCALAR
				&& ((DataIdentifier)expr.getRight()).getKind() == Expression.Kind.Data 
				&& ((DataIdentifier)expr.getRight()).getValueType() == Expression.ValueType.STRING){
			String name = ((DataIdentifier)expr.getRight()).getName();
			filename =  filename + ((StringIdentifier)currConstVars.get(name)).getValue();
		}
		else {
			raiseValidateError("Parameter " + IO_FILENAME + " only supports a const string or const string concatenations.", conditional);
		}
		return filename;
			
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(_opcode.toString());
		sb.append("(");

		for(Entry<String,Expression> e : _varParams.entrySet()) {
			String key = e.getKey();
			Expression expr = e.getValue();
			sb.append(",");
			sb.append(key);
			sb.append("=");
			sb.append(expr);
		}
		sb.append(" )");
		return sb.toString();
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		for( Expression expr : _varParams.values() ) {
			result.addVariables ( expr.variablesRead() );
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		for( Expression expr : _varParams.values() ) {
			result.addVariables ( expr.variablesUpdated() );
		}
		result.addVariable(((DataIdentifier)this.getOutput()).getName(), (DataIdentifier)this.getOutput());
		return result;
	}
	
	@SuppressWarnings("unchecked")
	private void parseMetaDataFileParameters(String mtdFileName, JSONObject configObject, boolean conditional) 
		throws LanguageException 
	{
    	for( Object obj : configObject.entrySet() ){
			Entry<Object,Object> e = (Entry<Object, Object>) obj;
    		Object key = e.getKey();
    		Object val = e.getValue();
			
    		boolean isValidName = false;
    		for (String paramName : READ_VALID_MTD_PARAM_NAMES){
				if (paramName.equals(key))
					isValidName = true;
			}
    		
			if (!isValidName){ //wrong parameters always rejected
				raiseValidateError("MTD file " + mtdFileName + " contains invalid parameter name: " + key, false);
			}
			
			// if the read method parameter is a constant, then verify value matches MTD metadata file
			if (getVarParam(key.toString()) != null && (getVarParam(key.toString()) instanceof ConstIdentifier) 
					&& !getVarParam(key.toString()).toString().equalsIgnoreCase(val.toString()) )
			{
				raiseValidateError("parameter " + key.toString() + " has conflicting values in read statement definition and metadata. " +
						"Config file value: " + val.toString() + " from MTD file.  Read statement value: " + getVarParam(key.toString()), conditional);
			}
			else 
			{
				// if the read method does not specify parameter value, then add MTD metadata file value to parameter list
				if (getVarParam(key.toString()) == null){
					if ( !key.toString().equalsIgnoreCase(DESCRIPTIONPARAM) ) {
						StringIdentifier strId = new StringIdentifier(val.toString(),
								this.getFilename(), this.getBeginLine(), this.getBeginColumn(), 
								this.getEndLine(), this.getEndColumn());
						
						if ( key.toString().equalsIgnoreCase(DELIM_HAS_HEADER_ROW) 
								|| key.toString().equalsIgnoreCase(DELIM_FILL)
								|| key.toString().equalsIgnoreCase(DELIM_SPARSE)
								) {
							// parse these parameters as boolean values
							BooleanIdentifier boolId = null; 
							if ( strId.toString().equalsIgnoreCase("true") ) {
								boolId = new BooleanIdentifier(true, this.getFilename(), 
										this.getBeginLine(), this.getBeginColumn(), 
										this.getEndLine(), this.getEndColumn());
							}
							else if ( strId.toString().equalsIgnoreCase("false") ) {
								boolId = new BooleanIdentifier(false, this.getFilename(), 
										this.getBeginLine(), this.getBeginColumn(), 
										this.getEndLine(), this.getEndColumn());
							}
							else {
								raiseValidateError("Invalid value provided for '" + DELIM_HAS_HEADER_ROW + "' in metadata file '" + mtdFileName + "'. "
										+ "Must be either TRUE or FALSE.", conditional);
							}
							removeVarParam(key.toString());
							addVarParam(key.toString(), boolId);
						}
						else if ( key.toString().equalsIgnoreCase(DELIM_FILL_VALUE)) {
							// parse these parameters as numeric values
							DoubleIdentifier doubleId = new DoubleIdentifier( Double.parseDouble(strId.toString()),
									this.getFilename(), this.getBeginLine(), this.getBeginColumn(), 
									this.getEndLine(), this.getEndColumn());
							removeVarParam(key.toString());
							addVarParam(key.toString(), doubleId);
						}
						else if (key.toString().equalsIgnoreCase(DELIM_NA_STRINGS)) {
							String naStrings = null;
							if ( val instanceof String) {
								naStrings = val.toString();
							}
							else {
								StringBuilder sb = new StringBuilder();
								JSONArray valarr = (JSONArray)val;
								for(int naid=0; naid < valarr.size(); naid++ ) {
									sb.append( (String) valarr.get(naid) );
									if ( naid < valarr.size()-1)
										sb.append( DELIM_NA_STRING_SEP );
								}
								naStrings = sb.toString();
							}
							StringIdentifier sid = new StringIdentifier( naStrings,
									this.getFilename(), this.getBeginLine(), this.getBeginColumn(), 
									this.getEndLine(), this.getEndColumn());
							removeVarParam(key.toString());
							addVarParam(key.toString(), sid);
						}
						else {
							// by default, treat a parameter as a string
							addVarParam(key.toString(), strId);
						}
					}
				}
			}
    	}
	}
	
	/**
	 * 
	 * @param filename
	 * @return
	 * @throws LanguageException
	 */
	public JSONObject readMetadataFile(String filename, boolean conditional) 
		throws LanguageException 
	{
		JSONObject retVal = null;
		boolean exists = false;
		FileSystem fs = null;
		
		try {
			fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
		} catch (Exception e){
			raiseValidateError("could not read the configuration file: "+e.getMessage(), false);
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
			if (exists && fs.getFileStatus(pt).isDirectory())
				isDirBoolean = true;
			else
				isDirBoolean = false;
		}
		catch(Exception e){
			raiseValidateError("error validing whether path " + pt.toString() + " is directory or not: "+e.getMessage(), conditional);
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
				raiseValidateError("for MTD file in directory, error reading directory with MTD file " + pt.toString() + ": " + e.getMessage(), conditional);
			}
			
			for(FileStatus stat : stats){
				Path childPath = stat.getPath(); // gives directory name
				if (childPath.getName().startsWith("part")){
					
					BufferedReader br = null;
					try {
						br = new BufferedReader(new InputStreamReader(fs.open(childPath)));
					}
					catch(Exception e){
						raiseValidateError("for MTD file in directory, error reading part of MTD file with path " + childPath.toString() + ": " + e.getMessage(), conditional);
					}
					
					JSONObject childObj = null;
					try {
						childObj = JSONHelper.parse(br);
					}
					catch(Exception e){
						raiseValidateError("for MTD file in directory, error parsing part of MTD file with path " + childPath.toString() + ": " + e.getMessage(), conditional);
					}
					
			    	for( Object obj : childObj.entrySet() ){
						@SuppressWarnings("unchecked")
						Entry<Object,Object> e = (Entry<Object, Object>) obj;
			    		Object key = e.getKey();
			    		Object val = e.getValue();
			    		retVal.put(key, val);
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
				raiseValidateError("error reading MTD file with path " + pt.toString() + ": " + e.getMessage(), conditional);
			}
			
			// try parsing MTD file
			try {
				retVal =  JSONHelper.parse(br);	
			} catch (Exception e){
				raiseValidateError("error parsing MTD file with path " + pt.toString() + ": " + e.getMessage(), conditional);
			}
		}
			
		return retVal;
	}

	public String[] readMatrixMarketFile(String filename, boolean conditional) 
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
			
			boolean getFileStatusIsDir = fs.getFileStatus(pt).isDirectory();
			
			if (exists && getFileStatusIsDir){
				raiseValidateError("MatrixMarket files as directories not supported", conditional);
			}
			else if (exists) {
				BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(pt)));
				try
				{
					retVal[0] = in.readLine();
					// skip all commented lines
					do {
						retVal[1] = in.readLine();
					} while ( retVal[1].charAt(0) == '%' );
					
					if ( !retVal[0].startsWith("%%") ) {
						raiseValidateError("MatrixMarket files must begin with a header line.", conditional);
					}
				}
				finally
				{
					if( in != null )
						in.close();
				}
			}
			else {
				raiseValidateError("Could not find the file: " + filename, conditional);
			}
			
		} catch (IOException e){
			//LOG.error(this.printErrorLocation() + "Error reading MatrixMarket file: " + filename );
			//throw new LanguageException(this.printErrorLocation() + "Error reading MatrixMarket file: " + filename );
			throw new LanguageException(e);
		}

		return retVal;
	}
	
	public boolean checkHasMatrixMarketFormat(String inputFileName, String mtdFileName, boolean conditional) 
		throws LanguageException 
	{
		// Check the MTD file exists. if there is an MTD file, return false.
		JSONObject mtdObject = readMetadataFile(mtdFileName, conditional);
	    
		if (mtdObject != null)
			return false;
		
		boolean exists = false;
		FileSystem fs = null;
		
		try {
			fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
		} catch (Exception e){
			LOG.error(this.printErrorLocation() + "could not read the configuration file.");
			throw new LanguageException(this.printErrorLocation() + "could not read the configuration file.", e);
		}
		
		Path pt = new Path(inputFileName);
		try {
			if (fs.exists(pt)){
				exists = true;
			}
		} catch (Exception e){
			LOG.error(this.printErrorLocation() + "file " + inputFileName + " not found");
			throw new LanguageException(this.printErrorLocation() + "file " + inputFileName + " not found");
		}
	
		try {
			// CASE: filename is a directory -- process as a directory
			if (exists && fs.getFileStatus(pt).isDirectory()){
				
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
				if( headerLine !=null && headerLine.startsWith("%%") )
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
	
	public boolean checkHasDelimitedFormat(String filename, boolean conditional)
		throws LanguageException 
	{
	 
        // if the MTD file exists, check the format is not binary 
		JSONObject mtdObject = readMetadataFile(filename + ".mtd", conditional);
        if (mtdObject != null){
        	String formatTypeString = (String)JSONHelper.get(mtdObject,FORMAT_TYPE);
            if (formatTypeString != null ) {
            	if ( formatTypeString.equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV) )
            		return true;
            	else
            		return false;
        	}
        }
        return false;

        // The file format must be specified either in .mtd file or in read() statement
        // Therefore, one need not actually read the data to infer the format.
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isCSVReadWithUnknownSize()
	{
		boolean ret = false;
		
		Expression format = getVarParam(FORMAT_TYPE);
		if( _opcode == DataOp.READ && format!=null && format.toString().equalsIgnoreCase(FORMAT_TYPE_VALUE_CSV) )
		{
			Expression rows = getVarParam(READROWPARAM);
			Expression cols = getVarParam(READCOLPARAM);
			if(   (rows==null || Long.parseLong(rows.toString())<0)
				||(cols==null || Long.parseLong(cols.toString())<0) )
			{
				ret = true;
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isRead()
	{
		return (_opcode == DataOp.READ);
	}
	
} // end class
