package com.ibm.bi.dml.parser;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.parser.Expression.DataOp;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;
import com.ibm.json.java.JSONObject;


public abstract class IOStatement extends Statement{
	
	protected DataIdentifier _id;
		
	// data structures to store parameters as expressions

	protected DataExpression _paramsExpr;
	
	public IOStatement(){
		_id = null;
		_paramsExpr = new DataExpression();
	
	}
	
	public IOStatement(DataIdentifier t, DataOp op){
		_id = t;
		_paramsExpr = new DataExpression(op);
	
	}
	
	public IOStatement (DataOp op){
		_id  = null;
		_paramsExpr = new DataExpression(op);
		
	}

	public DataIdentifier getId(){
		return _id;
	}
	
	public void setIdentifier(DataIdentifier t) {
		_id = t;
	}
	
	public void setExprParam(String name, Expression value) {
		_paramsExpr.addVarParam(name, value);
	}
	
	public void setExprParams(DataExpression paramsExpr) {
		_paramsExpr = paramsExpr;
	}
	
	public void addExprParam(String name, Expression value, boolean fromMTDFile) throws ParseException
	{
		if (_paramsExpr.getVarParam(name) != null)
			throw new ParseException("ERROR: attempted to add IOStatement parameter " + name + " more than once");
		
		// verify parameter names for InputStatement
		if (this instanceof InputStatement && !InputStatement.isValidParamName(name, fromMTDFile))
			throw new ParseException("ERROR: attempted to add invalid read statmement parameter " + name);
		
		else if (this instanceof OutputStatement && !OutputStatement.isValidParamName(name))
			throw new ParseException("ERROR: attempted to add invalid write statmement parameter: " + name);
		
		_paramsExpr.addVarParam(name, value);
	}
	
	public Expression getExprParam(String name){
		return _paramsExpr.getVarParam(name);
	}
	
	public DataExpression getSource(){
		return _paramsExpr;
	}
	
	private void processParamsForInputStatement(boolean missingdimension) throws IOException, LanguageException {
		
		// Identify the data type for input statement
		
		if (getExprParam(DATATYPEPARAM) != null && !(getExprParam(DATATYPEPARAM) instanceof StringIdentifier))
			throw new LanguageException("ERROR: for InputStatement, parameter " + DATATYPEPARAM + " can only be a string. " +
					"Valid values are: " + MATRIX_DATA_TYPE +", " + SCALAR_DATA_TYPE);
		
		// disallow certain parameters while reading a scalar
		String dataTypeString = (getExprParam(DATATYPEPARAM) == null) ? null : getExprParam(DATATYPEPARAM).toString();
		if (dataTypeString != null && dataTypeString.equalsIgnoreCase(SCALAR_DATA_TYPE)){
			if ( getExprParam(READROWPARAM) != null
					|| getExprParam(READCOLPARAM) != null
					|| getExprParam(ROWBLOCKCOUNTPARAM) != null
					|| getExprParam(COLUMNBLOCKCOUNTPARAM) != null
					|| getExprParam(FORMAT_TYPE) != null )
				throw new LanguageException("ERROR: Invalid parameters in read statement of a scalar: " +
						toString() + ". Only " + VALUETYPEPARAM + " is allowed.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
		JSONObject configObject = null;	
		
		// check if the metadata file exists
		// read the configuration file
		boolean exists = false;
		FileSystem fs = FileSystem.get(new Configuration());
		Path pt = null;
		String filename = null;
		
		if (this._paramsExpr.getVarParam(IO_FILENAME) instanceof ConstIdentifier){
			filename = this._paramsExpr.getVarParam(IO_FILENAME).toString() +".mtd";
			pt=new Path(filename);
			try {
				if (fs.exists(pt)){
					exists = true;
				}
			} catch (Exception e){
				exists = false;
			}
		}
        // if the MTD file exists, check the values specified in read statement match values in metadata MTD file
        if (exists){
        
	        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
			configObject = JSONObject.parse(br);
			
			for (Object key : configObject.keySet()){
				
				if (!InputStatement.isValidParamName(key.toString(),true))
					throw new LanguageException("ERROR: MTD file " + filename + " contains invalid parameter name: " + key);
					
				// if the InputStatement parameter is a constant, then verify value matches MTD metadata file
				if (getExprParam(key.toString()) != null && (getExprParam(key.toString()) instanceof ConstIdentifier) 
						&& !getExprParam(key.toString()).toString().equalsIgnoreCase(configObject.get(key).toString()) ){
					throw new LanguageException("ERROR: parameter " + key.toString() + " has conflicting values in read statement definition and metadata. " +
							"Config file value: " + configObject.get(key).toString() + " from MTD file.  Read statement value: " + getExprParam(key.toString()));	
				}
				else {
					// if the InputStatement does not specify parameter value, then add MTD metadata file value to parameter list
					if (_paramsExpr.getVarParam(key.toString()) == null)
						_paramsExpr.addVarParam(key.toString(), new StringIdentifier(configObject.get(key).toString()));
				}
			}
        }
        else {
        	if (!(getExprParam(IO_FILENAME) instanceof ConstIdentifier))
        		System.out.println("INFO: non-constant expression used for filename -- no attempt to find MTD");
        	else
        		System.out.println("INFO: could not find metadata file: " + pt);
        }
		
        dataTypeString = (getExprParam(DATATYPEPARAM) == null) ? null : getExprParam(DATATYPEPARAM).toString();
		
        
		if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(MATRIX_DATA_TYPE) ) {	
			_id.setDataType(DataType.MATRIX);
		}
		else if ( dataTypeString.equalsIgnoreCase(SCALAR_DATA_TYPE)) {
			_id.setDataType(DataType.SCALAR);
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
			 
		
			// if format="text" --> verify block sizes both <= 0 OR expression
			if (getExprParam(FORMAT_TYPE) == null || getExprParam(FORMAT_TYPE).toString().equalsIgnoreCase("text")){
				
				if (getExprParam(ROWBLOCKCOUNTPARAM) != null && getExprParam(ROWBLOCKCOUNTPARAM) instanceof ConstIdentifier){
					Long rowBlockCount = (getExprParam(ROWBLOCKCOUNTPARAM) == null) ? null : new Long(getExprParam(ROWBLOCKCOUNTPARAM).toString());
					if (rowBlockCount != null && rowBlockCount > 0)
						throw new LanguageException("ERROR: Inconsistent row block value for text format data. " + 
								ROWBLOCKCOUNTPARAM + " must be 0 for format=text data.  Value was: " + rowBlockCount);		
				}
			
				if (getExprParam(COLUMNBLOCKCOUNTPARAM) != null && getExprParam(COLUMNBLOCKCOUNTPARAM) instanceof ConstIdentifier){
					Long colBlockCount = (getExprParam(COLUMNBLOCKCOUNTPARAM) == null) ? null : new Long(getExprParam(COLUMNBLOCKCOUNTPARAM).toString());
					if (colBlockCount != null && colBlockCount > 0)
						throw new LanguageException("ERROR: Inconsistent column block value for text format data. " + 
								COLUMNBLOCKCOUNTPARAM + " must be 0 for format=text data.  Value was: " + colBlockCount);		
				}
			}
			
			if (getExprParam(FORMAT_TYPE).toString().equalsIgnoreCase("binary")){
				
				if (getExprParam(ROWBLOCKCOUNTPARAM) != null && getExprParam(ROWBLOCKCOUNTPARAM) instanceof ConstIdentifier){
					Long rowBlockCount = (getExprParam(ROWBLOCKCOUNTPARAM) == null) ? null : new Long(getExprParam(ROWBLOCKCOUNTPARAM).toString());
					if (rowBlockCount != null && rowBlockCount < 1)
						throw new LanguageException("ERROR: Inconsistent row block value for binary format data. " + 
								ROWBLOCKCOUNTPARAM + " must be >= 1 for format=text data.  Value was: " + rowBlockCount);		
				}
			
				if (getExprParam(COLUMNBLOCKCOUNTPARAM) != null && getExprParam(COLUMNBLOCKCOUNTPARAM) instanceof ConstIdentifier){
					Long colBlockCount = (getExprParam(COLUMNBLOCKCOUNTPARAM) == null) ? null : new Long(getExprParam(COLUMNBLOCKCOUNTPARAM).toString());
					if (colBlockCount != null && colBlockCount < 1)
						throw new LanguageException("ERROR: Inconsistent column block value for text format data. " + 
								COLUMNBLOCKCOUNTPARAM + " must be >= 1 for format=binary data.  Value was: " + colBlockCount);		
				}
			}
			
			
			
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
		
		if (_paramsExpr.getVarParam(FORMAT_TYPE)!= null ){
		 	
	 		Expression formatTypeExpr = _paramsExpr.getVarParam(FORMAT_TYPE);  
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
			_paramsExpr.addVarParam(FORMAT_TYPE, new StringIdentifier(FormatType.TEXT.toString()));
			_id.setFormatType(FormatType.TEXT);
		}
	}
	
	
	public DataIdentifier getIdentifier(){
		return _id;
	}
	
	public String getFormatName() {
		return(_paramsExpr.getVarParam(FORMAT_TYPE).toString());
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
