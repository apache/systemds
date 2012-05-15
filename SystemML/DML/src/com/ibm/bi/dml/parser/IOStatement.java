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
	protected String _filename;
	protected HashMap<String,String> _stringParams;
	protected HashMap<String,String> _varParams;
		
	public IOStatement(){
		_id = null;
		_filename = null;
		_stringParams = new HashMap<String,String>();
		_varParams = new HashMap<String,String>();
	}

	public IOStatement(DataIdentifier t, String fname){
		_id = t;
		_filename = fname;
		_stringParams = new HashMap<String,String>();
		_varParams = new HashMap<String,String>();
	}
	
	public DataIdentifier getId(){
		return _id;
	}
		 	
	public void setFileName(String fname){
		_filename = fname;
	}
	
	public void setIdentifier(DataIdentifier t) {
		_id = t;
}
	public void addVarParam(String name, String value){
		_varParams.put(name, value);
	}
	
	public void addStringParam(String name, String value){
		_stringParams.put(name, value);
	}
		
	public String getStringParam(String name){
		if (_stringParams.containsKey(name)){
			return _stringParams.get(name);
		} else {
			return null;
		}
	}

	public String getVarParam(String name){
		if (_varParams.containsKey(name)){
			return _varParams.get(name);
		} else {
			return null;
		}
	}
	
	private void processParamsForInputStatement(boolean missingdimension) throws IOException, LanguageException {
		// Identify the data type for input statement
		String dt = getStringParam(DATATYPEPARAM);
		JSONObject configObject = null;	
	
		
		if ( dt == null || dt.equalsIgnoreCase(MATRIX_DATA_TYPE) ) {
			
			// read the configuration file
			String filename = this.getFilename()+".mtd";
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
	        
	        // if the MTD file exists, check the values specified in read statement match
	        //	values in MTD file
	        if (exists){
	        
		        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
				configObject = JSONObject.parse(br);
				
				for (Object key : configObject.keySet()){
					
					if (getStringParam(key.toString()) != null && !getStringParam(key.toString()).equalsIgnoreCase(configObject.get(key).toString()) ){
						throw new LanguageException("replacing " + key.toString() + " with value " + configObject.get(key).toString() + " from MTD file.  Read statmeent has value " + getStringParam(key.toString()));
						//_stringParams.put(key.toString(), configObject.get(key).toString());
					}
					else if (getVarParam(key.toString()) != null && !getVarParam(key.toString()).equalsIgnoreCase(configObject.get(key).toString()) ){
						throw new LanguageException("replacing " + key.toString() + " with value " + configObject.get(key).toString() + " from MTD file. Read statmeent has value " + getVarParam(key.toString()));
						//_varParams.put(key.toString(), configObject.get(key).toString());
					}
					else {
						_stringParams.put(key.toString(), configObject.get(key).toString());
					}
					
				}
	        }
	        else {
	        	System.out.println("INFO: could not find metadata file " + pt);
	        }
			_id.setDataType(DataType.MATRIX);
		}
		else if ( dt.equalsIgnoreCase(SCALAR_DATA_TYPE)) {
			_id.setDataType(DataType.SCALAR);
			
			// disallow certain parameters while reading a scalar
			if ( getStringParam(READROWPARAM) != null
					|| getStringParam(READCOLPARAM) != null
					|| getStringParam(ROWBLOCKCOUNTPARAM) != null
					|| getStringParam(COLUMNBLOCKCOUNTPARAM) != null
					|| getStringParam(FORMAT_TYPE) != null )
				throw new LanguageException("Invalid parameters in read statement of a scalar: " +
						toString() + ". Only value_type is allowed.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else{
			
			throw new LanguageException("Unknown Data Type " + dt, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
		// Identify the value type (used only for InputStatement)
		String valueType = getStringParam(VALUETYPEPARAM);
		if (valueType != null) {
			if (valueType.equalsIgnoreCase(DOUBLE_VALUE_TYPE)) {
				_id.setValueType(ValueType.DOUBLE);
			} else if (valueType.equalsIgnoreCase(STRING_VALUE_TYPE)) {
				_id.setValueType(ValueType.STRING);
			} else if (valueType.equalsIgnoreCase(INT_VALUE_TYPE)) {
				_id.setValueType(ValueType.INT);
			} else if (valueType.equalsIgnoreCase(BOOLEAN_VALUE_TYPE)) {
				_id.setValueType(ValueType.BOOLEAN);
			} else{
				throw new LanguageException("Unknown Value Type " + valueType,
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
		} else {
			_id.setValueType(ValueType.DOUBLE);
				
		}
		
		// Following checks must be done when dt = matrix 
		if ( dt == null || dt.equalsIgnoreCase("matrix") ) {
				
			// these are strings that are long values
			Long dim1 = (getStringParam(READROWPARAM) == null) ? null : new Long (getStringParam(READROWPARAM));
			Long dim2 = (getStringParam(READCOLPARAM) == null) ? null : new Long(getStringParam(READCOLPARAM));
		
		// set dim1 and dim2 values 
			if (dim1 != null && dim2 != null){
				_id.setDimensions(dim1, dim2);
			} else if ((dim1 != null) || (dim2 != null)) {
				throw new LanguageException("Partial dimension information in read statement", 
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			} else {
				if (!missingdimension){
					throw new LanguageException("Missing dimension information in read statement", 
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}
			
			Long rowBlockCount = (getStringParam(ROWBLOCKCOUNTPARAM) == null) ? null : new Long(getStringParam(ROWBLOCKCOUNTPARAM));
			Long columnBlockCount = (getStringParam(COLUMNBLOCKCOUNTPARAM) == null) ? null : new Long (getStringParam(COLUMNBLOCKCOUNTPARAM));
	
			if ((rowBlockCount != null) && (columnBlockCount != null)) {
				_id.setBlockDimensions(rowBlockCount, columnBlockCount);
			} else if ((rowBlockCount != null) || (columnBlockCount != null)) {
				throw new LanguageException("Partial block dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			} else {
				 _id.setBlockDimensions(-1, -1);
			}
		}
	
	}
	
	private void processParamsForOutputStatement()  throws LanguageException {
		// Output statements are allowed to have only "format" as its parameter
		if(_stringParams.size() > 1 || 
				(_stringParams.size() == 1 && getStringParam(FORMAT_TYPE) == null ) ) 
			throw new LanguageException("Invalid parameters in write statement: " +
					toString() + ". Only the parameter format is allowed.  Please refer to the language guide for further details", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);

		// TODO: statiko -- this logic has to be updated to remove the call to setDimensionValueProperties() in StatementBlock
		// and to support binaryBlock as well as binaryCell in read() and write() statements
		_id.setBlockDimensions(-1, -1);
	}
	
	public void processParams(boolean missingdimension) throws LanguageException, IOException {
		
		if ( this instanceof InputStatement ) {
			processParamsForInputStatement(missingdimension);
		}
		else if (this instanceof OutputStatement ) {
			processParamsForOutputStatement();
		}
		
	 	if (_stringParams.containsKey(FORMAT_TYPE)){
			String ft = _stringParams.get(FORMAT_TYPE);
			if (ft.equalsIgnoreCase("binary") || ft.equalsIgnoreCase("bin")){
				_id.setFormatType(FormatType.BINARY);
			} else if (ft.equalsIgnoreCase("text")){
				_id.setFormatType(FormatType.TEXT);
			} else throw new LanguageException("Unknown Format type " + ft,
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		} else {
			_id.setFormatType(FormatType.TEXT);
		}
	}
	

	public DataIdentifier getIdentifier(){
		return _id;
	}
	
	public String getFilename(){
		return _filename;
	}
	
	public String getFormatName() {
		return(_stringParams.get(FORMAT_TYPE));
	}
	
	@Override
	public boolean controlStatement() {
		return false;
	}
	
	public void initializeforwardLV(){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}

	
	public String toString(){
		 StringBuffer sb = new StringBuffer();
		 sb.append(_id.toString() + " = " + Statement.INPUTSTATEMENT + " ( " );
		 sb.append("\""+_filename+"\"");
		 for (String key : _stringParams.keySet()){
			 sb.append("," + key + "=" + "\"" + _stringParams.get(key) + "\"");
		 }
		 for (String key : _varParams.keySet()){
			 sb.append("," + key + "=" + _varParams.get(key));
		 }
		 sb.append(");");
		 return sb.toString();
		
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
