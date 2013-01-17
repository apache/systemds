package com.ibm.bi.dml.parser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.utils.LanguageException;
import com.ibm.json.java.JSONObject;


public class DataExpression extends Expression {

	private DataOp _opcode;
	private HashMap<String, Expression> _varParams;
	
	
	public DataExpression(DataOp op, HashMap<String,Expression> varParams) {
		_kind = Kind.DataOp;
		_opcode = op;
		_varParams = varParams;
	}
	
	public DataExpression(DataOp op) {
		_kind = Kind.DataOp;
		_opcode = op;
		_varParams = new HashMap<String,Expression>();
	}

	public DataExpression() {
		_kind = Kind.DataOp;
		_opcode = DataOp.INVALID;
		_varParams = new HashMap<String,Expression>();
	}
	 
	public Expression rewriteExpression(String prefix) throws LanguageException {
		
		HashMap<String,Expression> newVarParams = new HashMap<String,Expression>();
		for (String key : _varParams.keySet()){
			Expression newExpr = _varParams.get(key).rewriteExpression(prefix);
			newVarParams.put(key, newExpr);
		}	
		DataExpression retVal = new DataExpression(_opcode, newVarParams);
		retVal._beginLine 	= this._beginLine;
		retVal._beginColumn = this._beginColumn;
		retVal._endLine 	= this._endLine;
		retVal._endColumn 	= this._endColumn;
			
		return retVal;
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
		if (_beginLine == 0) 	_beginLine 	 = value.getBeginLine();
		if (_beginColumn == 0) 	_beginColumn = value.getBeginColumn();
		if (_endLine == 0) 		_endLine 	 = value.getEndLine();
		if (_endColumn == 0) 	_endColumn 	 = value.getEndColumn();
		
		// update values	
		if (_beginLine > value.getBeginLine()){
			_beginLine = value.getBeginLine();
			_beginColumn = value.getBeginColumn();
		}
		else if (_beginLine == value.getBeginLine() &&_beginColumn > value.getBeginColumn()){
			_beginColumn = value.getBeginColumn();
		}

		if (_endLine < value.getEndLine()){
			_endLine = value.getEndLine();
			_endColumn = value.getEndColumn();
		}
		else if (_endLine == value.getEndLine() && _endColumn < value.getEndColumn()){
			_endColumn = value.getEndColumn();
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
		for ( String s : getVarParams().keySet() ) {
			getVarParam(s).validateExpression(ids, currConstVars);
			if ( getVarParam(s).getOutput().getDataType() != DataType.SCALAR ) {
				throw new LanguageException(this.printErrorLocation() + "Non-scalar data types are not supported for data expression.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}	
					
		// IMPORTANT: for each operation, one must handle unnamed parameters
		
		switch (this.getOpCode()) {
		
		case READ:
			
			
			if (getVarParam(Statement.DATATYPEPARAM) != null && !(getVarParam(Statement.DATATYPEPARAM) instanceof StringIdentifier)){
				
				throw new LanguageException(this.printErrorLocation() + "for read statement, parameter " + Statement.DATATYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.MATRIX_DATA_TYPE +", " + Statement.SCALAR_DATA_TYPE);
			}
			
			String dataTypeString = (getVarParam(Statement.DATATYPEPARAM) == null) ? null : getVarParam(Statement.DATATYPEPARAM).toString();
			
			// disallow certain parameters while reading a scalar
			if (dataTypeString != null && dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)){
				if ( getVarParam(Statement.READROWPARAM) != null
						|| getVarParam(Statement.READCOLPARAM) != null
						|| getVarParam(Statement.ROWBLOCKCOUNTPARAM) != null
						|| getVarParam(Statement.COLUMNBLOCKCOUNTPARAM) != null
						|| getVarParam(Statement.FORMAT_TYPE) != null
						|| getVarParam(Statement.FORMAT_DELIMITER) != null	
						|| getVarParam(Statement.HAS_HEADER_ROW) != null) {
					
					throw new LanguageException(this.printErrorLocation() + "Invalid parameters in read statement of a scalar: " +
							toString() + ". Only " + Statement.VALUETYPEPARAM + " is allowed.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			
				}
			}
			
			
			JSONObject configObject = null;	

			// read the configuration file
			String filename = null;
			
			if (getVarParam(Statement.IO_FILENAME) instanceof ConstIdentifier){
				filename = getVarParam(Statement.IO_FILENAME).toString() +".mtd";
				
			}
			else if (getVarParam(Statement.IO_FILENAME) instanceof BinaryExpression){
				BinaryExpression expr = (BinaryExpression)getVarParam(Statement.IO_FILENAME);
								
				if (expr.getKind()== Expression.Kind.BinaryOp){
					Expression.BinaryOp op = expr.getOpCode();
					switch (op){
					case PLUS:
							filename = "";
							filename = fileNameCat(expr, currConstVars, filename);
							// Since we have computed the value of filename, we update
							// varParams with a const string value
							StringIdentifier fileString = new StringIdentifier(filename);
							fileString.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							removeVarParam(Statement.IO_FILENAME);
							addVarParam(Statement.IO_FILENAME, fileString);
							filename = filename + ".mtd";
												
						break;
					default:
						throw new LanguageException(this.printErrorLocation()  + "for InputStatement, parameter " + Statement.IO_FILENAME + " can only be const string concatenations. ");
					}
				}
			}
			else {
				throw new LanguageException(this.printErrorLocation() + "for InputStatement, parameter " + Statement.IO_FILENAME + " can only be a const string or const string concatenations. ");
			}
			
			
			// TODO: DRB FIX -- only read MTD file if FORMAT_TYPE = Statement.DELIMITED_FORMAT_TYPE 
			//											OR FORMAT_TYPE = Statement.MATRIXMARKET_FORMAT_TYPE  
			
			boolean shouldReadMTD = true;
			String formatType = (getVarParam(Statement.FORMAT_TYPE) == null) ? null : getVarParam(Statement.FORMAT_TYPE).toString();
			
			if (formatType != null && formatType.equalsIgnoreCase(Statement.MATRIXMARKET_FORMAT_TYPE)){
				/*
				 *  handle Statement.MATRIXMARKET_FORMAT_TYPE format
				 *
				 * 1) only allow IO_FILENAME as ONLY valid parameter
				 * 
				 * 2) open the file
				 * 		A) verify header line (1st line) equals 
				 * 		B) read and discard comment lines
				 * 		C) get size information from sizing info line --- M N L
				 */
				
				// only allow IO_FILENAME as ONLY valid parameter
				for (String key : _varParams.keySet()){
					if (!key.equals(Statement.IO_FILENAME)){
						
						LOG.error(this.printErrorLocation() + "Invalid parameters in readMM statement: " +
								toString() + ". Only " + Statement.IO_FILENAME + " is allowed.");
						
						throw new LanguageException(this.printErrorLocation() + "Invalid parameters in readMM statement: " +
								toString() + ". Only " + Statement.IO_FILENAME + " is allowed.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
				}
				
				// should NOT attempt to read MTD file for MatrixMarket format
				shouldReadMTD = false;
				
				// get metadata from MatrixMarket format file
				String[] headerLines = readMatrixMarketFile(Statement.IO_FILENAME);
				
				// process 1st line of MatrixMarket format -- must be identical to legal header
				String legalHeaderMM = "%%MatrixMarket matrix coordinate real general";
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
					
					LOG.error(this.printErrorLocation() + "Unsupported format in MatrixMarket file: " +
							headerLines[0] + ". Only supported format in MatrixMarket file has header line " + legalHeaderMM);
					
					throw new LanguageException(this.printErrorLocation() + "Unsupported format in MatrixMarket file: " +
							headerLines[0] + ". Only supported format in MatrixMarket file has header line " + legalHeaderMM, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);	
				}
				
				long rowsCount = -1, colsCount = -1, nnzCount = -1;
				try {
					rowsCount = Long.parseLong(sizeInfo[0]);
					if (rowsCount < 1)
						throw new Exception("invalid rows count");
					addVarParam(Statement.READROWPARAM, new IntIdentifier(rowsCount));
				}
				catch(Exception e){
					
					LOG.error(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(Statement.IO_FILENAME) 
							+  " invalid row count " + sizeInfo[0] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1]);
					
					throw new LanguageException(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(Statement.IO_FILENAME) 
							+  " invalid row count " + sizeInfo[0] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1],
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
				
				try {
					colsCount = Long.parseLong(sizeInfo[1]);
					if (colsCount < 1)
						throw new Exception("invalid cols count");
					addVarParam(Statement.READCOLPARAM, new IntIdentifier(colsCount));
				}
				catch(Exception e){
					
					LOG.error(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(Statement.IO_FILENAME) 
							+  " invalid column count " + sizeInfo[1] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1]);
					
					throw new LanguageException(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(Statement.IO_FILENAME) 
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
				
					LOG.error(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(Statement.IO_FILENAME) 
							+  " invalid number non-zeros " + sizeInfo[2] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1]);
					
					throw new LanguageException(this.printErrorLocation() + "In MatrixMarket file " + getVarParam(Statement.IO_FILENAME) 
							+  " invalid number non-zeros " + sizeInfo[2] + " (must be long value >= 1). Sizing info line from file: " + headerLines[1],
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);	
				}	
			}
			
			else if (formatType != null && formatType.equalsIgnoreCase(Statement.DELIMITED_FORMAT_TYPE)){
			
				/* Handle delimited file format
				 * 
				 * 1) only allow IO_FILENAME, HAS_HEADER_ROW, FORMAT_DELIMITER, READROWPARAM, READCOLPARAM   
				 *  
				 * 2) open the file
				 * 		A) verify header line (1st line) equals "%%MatrixMarket matrix coordinate real general"
				 * 		B) read and discard comment lines
				 * 		C) get size information from 
				 */
				
				// there should be no MTD file for delimited file format
				shouldReadMTD = false;
				
				// only allow IO_FILENAME, HAS_HEADER_ROW, FORMAT_DELIMITER, READROWPARAM, READCOLPARAM   
				//		as ONLY valid parameters
				for (String key : _varParams.keySet()){
					if (!  (key.equals(Statement.IO_FILENAME) || key.equals(Statement.HAS_HEADER_ROW) || key.equals(Statement.FORMAT_DELIMITER) || key.equals(Statement.READROWPARAM) || key.equals(Statement.READCOLPARAM))){
						
						LOG.error(this.printErrorLocation() + "Invalid parameters in read.matrix statement: " +
								toString() + ". Only parameters allowed are: " + Statement.IO_FILENAME      + "," 
																			   + Statement.HAS_HEADER_ROW   + "," 
																			   + Statement.FORMAT_DELIMITER + "," 
																			   + Statement.READROWPARAM     + "," 
																			   + Statement.READCOLPARAM);
						
						throw new LanguageException(this.printErrorLocation() + "Invalid parameters in read.matrix statement: " +
								toString() + ". Only parameters allowed are: " + Statement.IO_FILENAME      + "," 
																			   + Statement.HAS_HEADER_ROW   + "," 
																			   + Statement.FORMAT_DELIMITER + "," 
																			   + Statement.READROWPARAM     + "," 
																			   + Statement.READCOLPARAM,
																			   LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
				}
				
				// if no delimiter mentioned, set to default value of ","
				if (getVarParam(Statement.FORMAT_DELIMITER) == null){
					addVarParam(Statement.FORMAT_DELIMITER, new StringIdentifier(","));
				}
				
				if (getVarParam(Statement.HAS_HEADER_ROW) == null){
					addVarParam(Statement.FORMAT_DELIMITER, new StringIdentifier("false"));
				}
				
				if (getVarParam(Statement.READROWPARAM) == null || getVarParam(Statement.READCOLPARAM) == null) {
					
					LOG.error(this.printErrorLocation() + "For delimited file " + getVarParam(Statement.IO_FILENAME) 
							+  " must specify both row and column dimensions ");
					
					throw new LanguageException(this.printErrorLocation() + "For delimited file " + getVarParam(Statement.IO_FILENAME) 
							+  " must specify both row and column dimensions ");
				}
			}
			
			configObject = null;
			
			if (shouldReadMTD)
				configObject = readMetadataFile(filename);
		        		    
	        // if the MTD file exists, check the values specified in read statement match values in metadata MTD file
	        if (configObject != null){
	        		    
	        	for (Object key : configObject.keySet()){
					
					if (!InputStatement.isValidParamName(key.toString(),true)){
						throw new LanguageException(this.printErrorLocation() + "MTD file " + filename + " contains invalid parameter name: " + key);
					}
					
					// if the InputStatement parameter is a constant, then verify value matches MTD metadata file
					if (getVarParam(key.toString()) != null && (getVarParam(key.toString()) instanceof ConstIdentifier) 
							&& !getVarParam(key.toString()).toString().equalsIgnoreCase(configObject.get(key).toString()) ){
						
						throw new LanguageException(this.printErrorLocation() + "parameter " + key.toString() + " has conflicting values in read statement definition and metadata. " +
								"Config file value: " + configObject.get(key).toString() + " from MTD file.  Read statement value: " + getVarParam(key.toString()));	
					}
					else {
						// if the InputStatement does not specify parameter value, then add MTD metadata file value to parameter list
						if (getVarParam(key.toString()) == null){
							StringIdentifier strId = new StringIdentifier(configObject.get(key).toString());
							strId.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							addVarParam(key.toString(), strId);
						}
					}
				}
	        }
	        else {
	        	LOG.warn("Metadata file: " + new Path(filename) + " not provided");
	        }
			
	        
	        dataTypeString = (getVarParam(Statement.DATATYPEPARAM) == null) ? null : getVarParam(Statement.DATATYPEPARAM).toString();
			
			if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(Statement.MATRIX_DATA_TYPE) ) {
				
				// set data type
		        _output.setDataType(DataType.MATRIX);
		        
		        // set number non-zeros
		        Expression ennz = this.getVarParam("nnz");
		        long nnz = -1;
		        if( ennz != null )
		        {
			        nnz = new Long(ennz.toString());
			        _output.setNnz(nnz);
		        }
		        
		        // Following dimension checks must be done when data type = MATRIX_DATA_TYPE 
				// initialize size of target data identifier to UNKNOWN
				_output.setDimensions(-1, -1);
				
				if ( getVarParam(Statement.READROWPARAM) == null || getVarParam(Statement.READCOLPARAM) == null){
					throw new LanguageException(this.printErrorLocation() + "Missing or incomplete dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				
				}
				if (getVarParam(Statement.READROWPARAM) instanceof ConstIdentifier && getVarParam(Statement.READCOLPARAM) instanceof ConstIdentifier)  {
				
					// these are strings that are long values
					Long dim1 = (getVarParam(Statement.READROWPARAM) == null) ? null : new Long (getVarParam(Statement.READROWPARAM).toString());
					Long dim2 = (getVarParam(Statement.READCOLPARAM) == null) ? null : new Long(getVarParam(Statement.READCOLPARAM).toString());
					
					if ( dim1 <= 0 || dim2 <= 0 ) {
						throw new LanguageException(this.printErrorLocation() + "Invalid dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
					// set dim1 and dim2 values 
					if (dim1 != null && dim2 != null){
						_output.setDimensions(dim1, dim2);
					} else if ((dim1 != null) || (dim2 != null)) {
						throw new LanguageException(this.printErrorLocation() + "Partial dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}	
				}
				
				// initialize block dimensions to UNKNOWN 
				_output.setBlockDimensions(-1, -1);
				
				// find "format": 1=text, 2=binary
				int format = 1; // default is "text"
				if (getVarParam(Statement.FORMAT_TYPE) == null || getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase("text")){
					format = 1;
				} else if ( getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase("binary") ) {
					format = 2;
				} else if ( getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase(Statement.MATRIXMARKET_FORMAT_TYPE) 
						|| getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase(Statement.DELIMITED_FORMAT_TYPE)) 
				{
					format = 1;
				} else {
					throw new LanguageException(this.printErrorLocation() + "Invalid format in statement: " + this.toString());
				}
				
				if (getVarParam(Statement.ROWBLOCKCOUNTPARAM) instanceof ConstIdentifier && getVarParam(Statement.COLUMNBLOCKCOUNTPARAM) instanceof ConstIdentifier)  {
				
					Long rowBlockCount = (getVarParam(Statement.ROWBLOCKCOUNTPARAM) == null) ? null : new Long(getVarParam(Statement.ROWBLOCKCOUNTPARAM).toString());
					Long columnBlockCount = (getVarParam(Statement.COLUMNBLOCKCOUNTPARAM) == null) ? null : new Long (getVarParam(Statement.COLUMNBLOCKCOUNTPARAM).toString());
		
					if ((rowBlockCount != null) && (columnBlockCount != null)) {
						_output.setBlockDimensions(rowBlockCount, columnBlockCount);
					} else if ((rowBlockCount != null) || (columnBlockCount != null)) {
						throw new LanguageException(this.printErrorLocation() + "Partial block dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					} else {
						 _output.setBlockDimensions(-1, -1);
					}
				}
				
				// block dimensions must be -1x-1 when format="text"
				// and they must be 1000x1000 when format="binary"
				if ( (format == 1 && (_output.getRowsInBlock() != -1 || _output.getColumnsInBlock() != -1))
						|| (format == 2 && (_output.getRowsInBlock() != DMLTranslator.DMLBlockSize || _output.getColumnsInBlock() != DMLTranslator.DMLBlockSize))){
					
					throw new LanguageException(this.printErrorLocation() + "Invalid block dimensions (" + _output.getRowsInBlock() + "," + _output.getColumnsInBlock() + ") when format=" + getVarParam(Statement.FORMAT_TYPE) + " in \"" + this.toString() + "\".");
				}
			
			}
			
			else if ( dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)) {
				_output.setDataType(DataType.SCALAR);
				_output.setNnz(-1L);
			}
			
			else{		
				throw new LanguageException(this.printErrorLocation() + "Unknown Data Type " + dataTypeString + ". Valid  values: " + Statement.SCALAR_DATA_TYPE +", " + Statement.MATRIX_DATA_TYPE, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			// handle value type parameter
			if (getVarParam(Statement.VALUETYPEPARAM) != null && !(getVarParam(Statement.VALUETYPEPARAM) instanceof StringIdentifier)){
				
				
				throw new LanguageException(this.printErrorLocation() + "for InputStatement, parameter " + Statement.VALUETYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE,
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			// Identify the value type (used only for InputStatement)
			String valueTypeString = getVarParam(Statement.VALUETYPEPARAM) == null ? null :  getVarParam(Statement.VALUETYPEPARAM).toString();
			if (valueTypeString != null) {
				if (valueTypeString.equalsIgnoreCase(Statement.DOUBLE_VALUE_TYPE)) {
					_output.setValueType(ValueType.DOUBLE);
				} else if (valueTypeString.equalsIgnoreCase(Statement.STRING_VALUE_TYPE)) {
					_output.setValueType(ValueType.STRING);
				} else if (valueTypeString.equalsIgnoreCase(Statement.INT_VALUE_TYPE)) {
					_output.setValueType(ValueType.INT);
				} else if (valueTypeString.equalsIgnoreCase(Statement.BOOLEAN_VALUE_TYPE)) {
					_output.setValueType(ValueType.BOOLEAN);
				} else {
					
					throw new LanguageException(this.printErrorLocation() + "Unknown Value Type " + valueTypeString
							+ ". Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE,
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
			} else {
				_output.setValueType(ValueType.DOUBLE);
			}

			break; 
			
		case WRITE:
			
			// for delimited format, if no delimiter specified THEN set default ","
			if (getVarParam(Statement.DELIMITED_FORMAT_TYPE).toString().equalsIgnoreCase(Statement.DELIMITED_FORMAT_TYPE)){
				if (getVarParam(Statement.FORMAT_DELIMITER) == null){
					addVarParam(Statement.FORMAT_DELIMITER, new StringIdentifier(","));
				}
			}
			
			if (getVarParam(Statement.IO_FILENAME) instanceof BinaryExpression){
				BinaryExpression expr = (BinaryExpression)getVarParam(Statement.IO_FILENAME);
								
				if (expr.getKind()== Expression.Kind.BinaryOp){
					Expression.BinaryOp op = expr.getOpCode();
					switch (op){
						case PLUS:
							filename = "";
							filename = fileNameCat(expr, currConstVars, filename);
							// Since we have computed the value of filename, we update
							// varParams with a const string value
							StringIdentifier fileString = new StringIdentifier(filename);
							fileString.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							removeVarParam(Statement.IO_FILENAME);
							addVarParam(Statement.IO_FILENAME, fileString);
												
							break;
						default:
							throw new LanguageException(this.printErrorLocation() + "for OutputStatement, parameter " + Statement.IO_FILENAME + " can only be a const string or const string concatenations. ");
					}
				}
			}
			
			if (getVarParam(Statement.FORMAT_TYPE) == null || getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase("text"))
				_output.setBlockDimensions(-1, -1);
			else if (getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase("binary"))
				_output.setBlockDimensions(DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			else if (getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase(Statement.DELIMITED_FORMAT_TYPE) || getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase(Statement.MATRIXMARKET_FORMAT_TYPE))
				_output.setBlockDimensions(-1, -1);
			
			else{
				throw new LanguageException(this.printErrorLocation() + "Invalid format in statement: " + this.toString());
			}
			break;

		case RAND: 
			
			for (String key : _varParams.keySet()){
				boolean found = false;
				for (String name : RandStatement.RAND_VALID_PARAM_NAMES){
					if (name.equals(key))
					found = true;
				}
				if (!found){
					
					
					throw new LanguageException(this.printErrorLocation() + "unexpected parameter \"" + key +
						"\". Legal parameters for Rand statement are " 
						+ "(capitalization-sensitive): " 	+ RandStatement.RAND_ROWS 	
						+ ", " + RandStatement.RAND_COLS		+ ", " + RandStatement.RAND_MIN + ", " + RandStatement.RAND_MAX  	
						+ ", " + RandStatement.RAND_SPARSITY + ", " + RandStatement.RAND_SEED     + ", " + RandStatement.RAND_PDF);
				}
			}
			//TODO: Leo Need to check with Doug about the data types
			// DoubleIdentifiers for RAND_ROWS and RAND_COLS have already been converted into IntIdentifier in RandStatment.addExprParam()  
			if (getVarParam(RandStatement.RAND_ROWS) instanceof StringIdentifier || getVarParam(RandStatement.RAND_ROWS) instanceof BooleanIdentifier){
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RandStatement.RAND_ROWS + " has incorrect data type");
			}
				
			if (getVarParam(RandStatement.RAND_COLS) instanceof StringIdentifier || getVarParam(RandStatement.RAND_COLS) instanceof BooleanIdentifier){
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RandStatement.RAND_COLS + " has incorrect data type");
			}
				
			if (getVarParam(RandStatement.RAND_MAX) instanceof StringIdentifier || getVarParam(RandStatement.RAND_MAX) instanceof BooleanIdentifier) {
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RandStatement.RAND_MAX + " has incorrect data type");
			}
			
			if (getVarParam(RandStatement.RAND_MIN) instanceof StringIdentifier || getVarParam(RandStatement.RAND_MIN) instanceof BooleanIdentifier) {
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RandStatement.RAND_MIN + " has incorrect data type");
			}
			
			if (!(getVarParam(RandStatement.RAND_SPARSITY) instanceof DoubleIdentifier || getVarParam(RandStatement.RAND_SPARSITY) instanceof IntIdentifier)) {
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RandStatement.RAND_SPARSITY + " has incorrect data type");
			}
			
			if (!(getVarParam(RandStatement.RAND_SEED) instanceof IntIdentifier)) {
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RandStatement.RAND_SEED + " has incorrect data type");
			}
			
			if (!(getVarParam(RandStatement.RAND_PDF) instanceof StringIdentifier)) {
				throw new LanguageException(this.printErrorLocation() + "for Rand statement " + RandStatement.RAND_PDF + " has incorrect data type");
			}
	
			long rowsLong = -1L, colsLong = -1L;

			///////////////////////////////////////////////////////////////////
			// HANDLE ROWS
			///////////////////////////////////////////////////////////////////
			Expression rowsExpr = getVarParam(RandStatement.RAND_ROWS);
			if (rowsExpr instanceof IntIdentifier) {
				if  (((IntIdentifier)rowsExpr).getValue() >= 1 ) {
					rowsLong = ((IntIdentifier)rowsExpr).getValue();
				}
				else {
					
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue());
				}
			}
			else if (rowsExpr instanceof DoubleIdentifier) {
				if  (((DoubleIdentifier)rowsExpr).getValue() >= 1 ) {
					rowsLong = new Double((Math.floor(((DoubleIdentifier)rowsExpr).getValue()))).longValue();
				}
				else {
					
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
							throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
						// update row expr with new IntIdentifier 
						long roundedValue = ((IntIdentifier)constValue).getValue();
						rowsExpr = new IntIdentifier(roundedValue);
						rowsExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RandStatement.RAND_ROWS, rowsExpr);
						rowsLong = roundedValue; 
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
						
						if (((DoubleIdentifier)constValue).getValue() < 1.0){
							
							throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
						// update row expr with new IntIdentifier (rounded down)
						long roundedValue = new Double (Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
						rowsExpr = new IntIdentifier(roundedValue);
						rowsExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RandStatement.RAND_ROWS, rowsExpr);
						rowsLong = roundedValue; 
						
					}
					else {
						// exception -- rows must be integer or double constant
						
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
			
			Expression colsExpr = getVarParam(RandStatement.RAND_COLS);
			if (colsExpr instanceof IntIdentifier) {
				if  (((IntIdentifier)colsExpr).getValue() >= 1 ) {
					colsLong = ((IntIdentifier)colsExpr).getValue();
				}
				else {
					
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString());
				}
			}
			else if (colsExpr instanceof DoubleIdentifier) {
				if  (((DoubleIdentifier)colsExpr).getValue() >= 1 ) {
					colsLong = new Double((Math.floor(((DoubleIdentifier)colsExpr).getValue()))).longValue();
				}
				else {
					
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
							throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
						// update col expr with new IntIdentifier 
						long roundedValue = ((IntIdentifier)constValue).getValue();
						colsExpr = new IntIdentifier(roundedValue);
						colsExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RandStatement.RAND_COLS, colsExpr);
						colsLong = roundedValue; 
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
						
						if (((DoubleIdentifier)constValue).getValue() < 1){
							throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
						}
						// update col expr with new IntIdentifier (rounded down)
						long roundedValue = new Double (Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
						colsExpr = new IntIdentifier(roundedValue);
						colsExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RandStatement.RAND_COLS, colsExpr);
						colsLong = roundedValue; 
						
					}
					else {
						// exception -- rows must be integer or double constant
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
			Expression minExpr = getVarParam(RandStatement.RAND_MIN);
			
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
						minExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RandStatement.RAND_MIN, minExpr);
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
		
						// update col expr with new IntIdentifier (rounded down)
						double roundedValue = ((DoubleIdentifier)constValue).getValue();
						minExpr = new DoubleIdentifier(roundedValue);
						minExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RandStatement.RAND_MIN, minExpr);
						
					}
					else {
						// exception -- rows must be integer or double constant
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
			Expression maxExpr = getVarParam(RandStatement.RAND_MAX);
			
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
						maxExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RandStatement.RAND_MAX, maxExpr);
					}
					// handle double constant 
					else if (constValue instanceof DoubleIdentifier){
		
						// update col expr with new IntIdentifier (rounded down)
						double roundedValue = ((DoubleIdentifier)constValue).getValue();
						maxExpr = new DoubleIdentifier(roundedValue);
						maxExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						addVarParam(RandStatement.RAND_MAX, maxExpr);
						
					}
					else {
						// exception -- rows must be integer or double constant
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
		
			_output.setFormatType(FormatType.BINARY);
			_output.setDataType(DataType.MATRIX);
			_output.setValueType(ValueType.DOUBLE);
			_output.setDimensions(rowsLong, colsLong);
			
			if (_output instanceof IndexedIdentifier){
				((IndexedIdentifier) _output).setOriginalDimensions(_output.getDim1(), _output.getDim2());
			}
			//_output.computeDataType();

			if (_output instanceof IndexedIdentifier){
				LOG.warn(this.printWarningLocation() + "Output for Rand Statement may have incorrect size information");
			}
			
			break;
		default:
			
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
			throw new LanguageException(this.printErrorLocation() + "Parameter " + Statement.IO_FILENAME + " only supports a const string or const string concatenations.");
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
			throw new LanguageException(this.printErrorLocation() + "Parameter " + Statement.IO_FILENAME + " only supports a const string or const string concatenations.");
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
			fs = FileSystem.get(new Configuration());
		} catch (Exception e){
			throw new LanguageException(this.printErrorLocation() + "could not read the configuration file.", e);
		}
		
		Path pt = new Path(filename);
		try {
			if (fs.exists(pt)){
				exists = true;
			}
		} catch (Exception e){
			exists = false;
		}
	
		try {
			// CASE: filename is a directory -- process as a directory
			if (exists && fs.getFileStatus(pt).isDir()){
			
				// read directory contents
				retVal = new JSONObject();
				FileStatus[] stats = fs.listStatus(pt);
				for(FileStatus stat : stats){
					Path childPath = stat.getPath(); // gives directory name
					if (childPath.getName().startsWith("part")){
						BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(childPath)));
						JSONObject childObj = JSONObject.parse(br);
						
						for (Object key : childObj.keySet()){
							retVal.put(key, childObj.get(key));
						}
					}
				} 
			}
			// CASE: filename points to a file
			else if (exists){
				BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
				retVal =  JSONObject.parse(br);
			}
			
			return retVal;
			
		} catch (Exception e){
			throw new LanguageException(this.printErrorLocation() + "error reading and/or parsing MTD file with path " + pt.toString(), e);
        }
	}
	
	
	public String[] readMatrixMarketFile(String filename) throws LanguageException {
		
		String[] retVal = new String[2];
		retVal[0] = new String("");
		retVal[1] = new String("");
		boolean exists = false;
		FileSystem fs = null;
		
		try {
			fs = FileSystem.get(new Configuration());
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
	
		try {
			// CASE: filename is a directory -- process as a directory
			if (exists && fs.getFileStatus(pt).isDir()){
			
				LOG.error(this.printErrorLocation() + "MatrixMarket files as directories not supported");
				throw new LanguageException(this.printErrorLocation() + "MatrixMarket files as directories not supported");
				/*
				// TODO: DRB FIX --- read directory contents
				retVal = new JSONObject();
				FileStatus[] stats = fs.listStatus(pt);
				for(FileStatus stat : stats){
					Path childPath = stat.getPath(); // gives directory name
					if (childPath.getName().startsWith("part")){
						BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(childPath)));
						JSONObject childObj = JSONObject.parse(br);
						
						for (Object key : childObj.keySet()){
							retVal.put(key, childObj.get(key));
						}
					}
				} 
				*/
			}
			// CASE: filename points to a file
			else if (exists){
		
				BufferedReader in = new BufferedReader(new FileReader(filename));
				boolean isDone = false;
				String headerLine = new String("");
				String sizeLine   = new String("");
				int rowCount = 0, maxRowCount = 200;
				if (in.ready())
					headerLine = in.readLine();
					
				while (in.ready() && !isDone) {
				  String currLine = in.readLine();
				  rowCount++;
				  if (!currLine.startsWith("%")){
					  sizeLine = currLine;
					  isDone = false;
				  }
				  else {
					  if (rowCount >= maxRowCount){
						  LOG.error(this.printErrorLocation() + "MatrixMarket file has too many comments -- please limit comments to <= 100 rows");
						  throw new LanguageException(this.printErrorLocation() + "MatrixMarket file has too many comments -- please limit comments to <= 100 rows");
					  }
				  }
				}
				in.close();
				
				retVal[0] = headerLine;
				retVal[1] = sizeLine;
			}
			
			return retVal;
			
		} catch (Exception e){
			LOG.error(this.printErrorLocation() + "error reading and/or parsing MatrixMarket file with path " + pt.toString());
        	throw new LanguageException(this.printErrorLocation() + "error reading and/or parsing MatrixMarket file with path " + pt.toString());
        }
	}
	
	
} // end class
