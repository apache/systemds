package com.ibm.bi.dml.parser;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;
import com.ibm.json.java.JSONObject;


public abstract class IOStatement extends Statement{
	
	protected DataIdentifier _id;
		
	// data structures to store filename and parameters (as expressions)
	protected Expression _filenameExpr;
	protected HashMap<String,Expression> _exprParams;	
	
	public IOStatement(){
		_id = null;
		_filenameExpr = null;
		_exprParams = new HashMap<String,Expression>();
	
	}

	public IOStatement(DataIdentifier t, Expression fexpr){
		_id = t;
		_filenameExpr = fexpr;
		_exprParams = new HashMap<String,Expression>();	
	}
	
	public DataIdentifier getId(){
		return _id;
	}
		
	public void setFilenameExpr(Expression expr){
		_filenameExpr = expr;
	}
	
	public Expression getFilenameExpr() {
		return _filenameExpr;
	}
		
	public void setIdentifier(DataIdentifier t) {
		_id = t;
	}
	
	public void setExprParam(String name, Expression value) {
		_exprParams.put(name, value);
	}
	
	public void setExprParams(HashMap<String,Expression> passed){
		_exprParams = passed;
	}
	
	public void addExprParam(String name, Expression value) throws ParseException
	{
		if (_exprParams.get(name) != null)
			throw new ParseException("ERROR: attempted to add IOStatement parameter " + name + " more than once");
		if (this instanceof InputStatement && !InputStatement.isValidParamName(name))
			throw new ParseException("ERROR: attempted to add invalid InputStatmement parameter " + name);
			
		_exprParams.put(name, value);
	}
	
	public Expression getExprParam(String name){
		return _exprParams.get(name);
	}
	
	
	private void processParamsForInputStatement(boolean missingdimension) throws IOException, LanguageException {
		
		// Identify the data type for input statement
		
		if (getExprParam(DATATYPEPARAM) != null && !(getExprParam(DATATYPEPARAM) instanceof StringIdentifier))
			throw new LanguageException("ERROR: for InputStatement, parameter " + DATATYPEPARAM + " can only be a string. " +
					"Valid values are: " + MATRIX_DATA_TYPE +", " + SCALAR_DATA_TYPE);
		
		String dataTypeString = (getExprParam(DATATYPEPARAM) == null) ? null : getExprParam(DATATYPEPARAM).toString();
		JSONObject configObject = null;	
		
		if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(MATRIX_DATA_TYPE) ) {
			
			// read the configuration file
			String filename = this.getFilenameExpr().toString() +".mtd";
			Path pt=new Path(filename);
	        FileSystem fs = FileSystem.get(new Configuration());
	        
	        boolean exists = false;
	        try {
	        	if (fs.exists(pt)){
	        		exists = true;
	        		
	        	}
	        } catch (Exception e){
	        	exists = false;
	        }
	        
	        // if the MTD file exists, check the values specified in read statement match values in metadata MTD file
	        if (exists){
	        
		        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
				configObject = JSONObject.parse(br);
				
				for (Object key : configObject.keySet()){
					
					if (!InputStatement.isValidParamName(key.toString()))
						throw new LanguageException("ERROR: MTD file " + filename + " contains invalid parameter name: " + key);
						
					// if the InputStatement parameter is a constant, then verify value matches MTD metadata file
					if (getExprParam(key.toString()) != null && (getExprParam(key.toString()) instanceof ConstIdentifier) 
							&& !getExprParam(key.toString()).toString().equalsIgnoreCase(configObject.get(key).toString()) ){
						throw new LanguageException("ERROR: parameter " + key.toString() + " has conflicting values in read statement definition and metadata. " +
								"Config file value: " + configObject.get(key).toString() + " from MTD file.  Read statement value: " + getExprParam(key.toString()));	
					}
					else {
						// if the InputStatement does not specify parameter value, then add MTD metadata file value to parameter list
						if (_exprParams.get(key.toString()) == null)
							_exprParams.put(key.toString(), new StringIdentifier(configObject.get(key).toString()));
					}
				}
	        }
	        else {
	        	System.out.println("INFO: could not find metadata file: " + pt);
	        }
			_id.setDataType(DataType.MATRIX);
		}
		else if ( dataTypeString.equalsIgnoreCase(SCALAR_DATA_TYPE)) {
			_id.setDataType(DataType.SCALAR);
			
			// disallow certain parameters while reading a scalar
			if ( getExprParam(READROWPARAM) != null
					|| getExprParam(READCOLPARAM) != null
					|| getExprParam(ROWBLOCKCOUNTPARAM) != null
					|| getExprParam(COLUMNBLOCKCOUNTPARAM) != null
					|| getExprParam(FORMAT_TYPE) != null )
				throw new LanguageException("ERROR: Invalid parameters in read statement of a scalar: " +
						toString() + ". Only " + VALUETYPEPARAM + " is allowed.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else{		
			throw new LanguageException("ERROR: Unknown Data Type " + dataTypeString + ". Valid  values: " + SCALAR_DATA_TYPE +", " + MATRIX_DATA_TYPE, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
		
		// handle value type parameter
		if (getExprParam(VALUETYPEPARAM) != null && !(getExprParam(VALUETYPEPARAM) instanceof StringIdentifier))
			throw new LanguageException("ERROR: for InputStatement, parameter " + VALUETYPEPARAM + " can only be a string. " +
					"Valid values are: " + DOUBLE_VALUE_TYPE +", " + INT_VALUE_TYPE + ", " + BOOLEAN_VALUE_TYPE + ", " + STRING_VALUE_TYPE,
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		
		// Identify the value type (used only for InputStatement)
		String valueTypeString = getExprParam(VALUETYPEPARAM) == null ? null :  getExprParam(VALUETYPEPARAM).toString();
		if (valueTypeString != null) {
			if (valueTypeString.equalsIgnoreCase(DOUBLE_VALUE_TYPE)) {
				_id.setValueType(ValueType.DOUBLE);
			} else if (valueTypeString.equalsIgnoreCase(STRING_VALUE_TYPE)) {
				_id.setValueType(ValueType.STRING);
			} else if (valueTypeString.equalsIgnoreCase(INT_VALUE_TYPE)) {
				_id.setValueType(ValueType.INT);
			} else if (valueTypeString.equalsIgnoreCase(BOOLEAN_VALUE_TYPE)) {
				_id.setValueType(ValueType.BOOLEAN);
			} else{
				throw new LanguageException("Unknown Value Type " + valueTypeString
						+ ". Valid values are: " + DOUBLE_VALUE_TYPE +", " + INT_VALUE_TYPE + ", " + BOOLEAN_VALUE_TYPE + ", " + STRING_VALUE_TYPE,
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
		} else {
			_id.setValueType(ValueType.DOUBLE);
				
		}
		
		// Following dimension checks must be done when data type = matrix 
		
		if ( dataTypeString == null || dataTypeString.equalsIgnoreCase("matrix") ) {
					
			// initialize size of target data identifier to UNKNOWN
			_id.setDimensions(-1, -1);
			
			if ( getExprParam(READROWPARAM) == null || getExprParam(READCOLPARAM) == null)
				throw new LanguageException("ERROR: Missing or incomplete dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			
			if (getExprParam(READROWPARAM) instanceof ConstIdentifier && getExprParam(READCOLPARAM) instanceof ConstIdentifier)  {
			
				// these are strings that are long values
				Long dim1 = (getExprParam(READROWPARAM) == null) ? null : new Long (getExprParam(READROWPARAM).toString());
				Long dim2 = (getExprParam(READCOLPARAM) == null) ? null : new Long(getExprParam(READCOLPARAM).toString());
		
				// set dim1 and dim2 values 
				if (dim1 != null && dim2 != null){
					_id.setDimensions(dim1, dim2);
				} else if ((dim1 != null) || (dim2 != null)) {
					throw new LanguageException("Partial dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}	
			}
			
			if(_id.getDim1() == -1 && _id.getDim2() == -1 && !missingdimension){
				throw new LanguageException("Missing dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
				
			
			// initialize block dimensions to UNKNOWN 
			_id.setBlockDimensions(-1, -1);
			 
			if (getExprParam(ROWBLOCKCOUNTPARAM) instanceof ConstIdentifier && getExprParam(COLUMNBLOCKCOUNTPARAM) instanceof ConstIdentifier)  {
			
				Long rowBlockCount = (getExprParam(ROWBLOCKCOUNTPARAM) == null) ? null : new Long(getExprParam(ROWBLOCKCOUNTPARAM).toString());
				Long columnBlockCount = (getExprParam(COLUMNBLOCKCOUNTPARAM) == null) ? null : new Long (getExprParam(COLUMNBLOCKCOUNTPARAM).toString());
	
				if ((rowBlockCount != null) && (columnBlockCount != null)) {
					_id.setBlockDimensions(rowBlockCount, columnBlockCount);
				} else if ((rowBlockCount != null) || (columnBlockCount != null)) {
					throw new LanguageException("Partial block dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				} else {
					 _id.setBlockDimensions(-1, -1);
				}
			}
		}
	}
	
	private void processParamsForOutputStatement()  throws LanguageException {
		// Output statements are allowed to have only "format" as its parameter
		if(_exprParams.size() > 1 || 
				(_exprParams.size() == 1 && getExprParam(FORMAT_TYPE) == null ) ) 
			throw new LanguageException("Invalid parameters in write statement: " +
					toString() + ". Only the parameter format is allowed.  Please refer to the language guide for further details", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);

		// TODO: statiko -- this logic has to be updated to remove the call to setDimensionValueProperties() in StatementBlock
		// and to support binaryBlock as well as binaryCell in read() and write() statements
		_id.setBlockDimensions(-1, -1);
	}
	
	/**
	 * processParams: provides best-effort validation of parameters to IO Statements.
	 * 	Validates that optional format, data type, value type
	 * @param missingdimension
	 * @throws LanguageException
	 * @throws IOException
	 */
	public void processParams(boolean missingdimension) throws LanguageException, IOException {
		
		if ( this instanceof InputStatement ) {
			processParamsForInputStatement(missingdimension);
		}
		else if (this instanceof OutputStatement ) {
			processParamsForOutputStatement();
		}
		
	 	if (_exprParams.containsKey(FORMAT_TYPE)){
	 	
	 		Expression formatTypeExpr = _exprParams.get(FORMAT_TYPE);  
			if (!(formatTypeExpr instanceof StringIdentifier))
				throw new LanguageException("ERROR: input statement parameter " + FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text", 
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
	 		
			String ft = formatTypeExpr.toString();
			if (ft.equalsIgnoreCase("binary")){
				_id.setFormatType(FormatType.BINARY);
			} else if (ft.equalsIgnoreCase("text")){
				_id.setFormatType(FormatType.TEXT);
			} else throw new LanguageException("ERROR: input statement parameter " + FORMAT_TYPE 
					+ " can only be a string with one of following values: binary, text", 
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		} else {
			_exprParams.put(FORMAT_TYPE, new StringIdentifier(FormatType.TEXT.toString()));
			_id.setFormatType(FormatType.TEXT);
		}
	}
	
	
	public DataIdentifier getIdentifier(){
		return _id;
	}
	
	public String getFormatName() {
		return(_exprParams.get(FORMAT_TYPE).toString());
	}
	
	@Override
	public boolean controlStatement() {
		return false;
	}
	
	public void initializeforwardLV(){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}

	@Override
	public VariableSet variablesRead() {
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
		return null;
	}
}
