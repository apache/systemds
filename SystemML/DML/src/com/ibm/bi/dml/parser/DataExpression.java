package com.ibm.bi.dml.parser;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.utils.LanguageException;
import com.ibm.json.java.JSONObject;
import com.ibm.bi.dml.parser.Statement;


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
		return new DataExpression(_opcode, newVarParams);
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
	
	public Expression getVarParam(String name) {
		return _varParams.get(name);
	}

	public void addVarParam(String name, Expression value){
		_varParams.put(name, value);
	}
	
	public void removeVarParam(String name) {
		_varParams.remove(name);
	}
	
	/**
	 * Validate parse tree : Process Data Expression in an assignment
	 * statement
	 *  
	 * @throws LanguageException
	 * @throws IOException 
	 */
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars)
			throws LanguageException {
		
		// validate all input parameters
		for ( String s : getVarParams().keySet() ) {
			getVarParam(s).validateExpression(ids, currConstVars);
			
			if ( getVarParam(s).getOutput().getDataType() != DataType.SCALAR ) {
				throw new LanguageException("Non-scalar data types are not supported for data expression.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
		
		// IMPORTANT: for each operation, one must handle unnamed parameters
		
		switch (this.getOpCode()) {
		
		case READ:
			
			
			if (getVarParam(Statement.DATATYPEPARAM) != null && !(getVarParam(Statement.DATATYPEPARAM) instanceof StringIdentifier))
				throw new LanguageException("ERROR: for InputStatement, parameter " + Statement.DATATYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.MATRIX_DATA_TYPE +", " + Statement.SCALAR_DATA_TYPE);
			
			
			String dataTypeString = (getVarParam(Statement.DATATYPEPARAM) == null) ? null : getVarParam(Statement.DATATYPEPARAM).toString();
			
			// disallow certain parameters while reading a scalar
			if (dataTypeString != null && dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)){
				if ( getVarParam(Statement.READROWPARAM) != null
						|| getVarParam(Statement.READCOLPARAM) != null
						|| getVarParam(Statement.ROWBLOCKCOUNTPARAM) != null
						|| getVarParam(Statement.COLUMNBLOCKCOUNTPARAM) != null
						|| getVarParam(Statement.FORMAT_TYPE) != null )
					throw new LanguageException("ERROR: Invalid parameters in read statement of a scalar: " +
							toString() + ". Only " + Statement.VALUETYPEPARAM + " is allowed.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			
			JSONObject configObject = null;	

			// read the configuration file
			boolean exists = false;
			FileSystem fs = null;
			try {
				fs = FileSystem.get(new Configuration());
			} catch (Exception e){
				throw new LanguageException(e);
			}
			Path pt = null;
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
							removeVarParam(Statement.IO_FILENAME);
							addVarParam(Statement.IO_FILENAME, fileString);
							filename = filename + ".mtd";
												
						break;
					default:
						throw new LanguageException("Error: for InputStatement, parameter " + Statement.IO_FILENAME + " can only be const string concatenations. ");
					}
				}
			}
			else {
				throw new LanguageException("ERROR: for InputStatement, parameter " + Statement.IO_FILENAME + " can only be a const string or const string concatenations. ");
			}
			
			pt=new Path(filename);
			try {
				if (fs.exists(pt)){
					exists = true;
				}
			} catch (Exception e){
				exists = false;
			}
	        // if the MTD file exists, check the values specified in read statement match values in metadata MTD file
	        if (exists){
	        		
		        try {
		        	BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
		        	configObject = JSONObject.parse(br);
		        } catch (Exception e){
		        	throw new LanguageException(e);
		        }
		        
				for (Object key : configObject.keySet()){
					
					if (!InputStatement.isValidParamName(key.toString(),true))
						throw new LanguageException("ERROR: MTD file " + filename + " contains invalid parameter name: " + key);
						
					// if the InputStatement parameter is a constant, then verify value matches MTD metadata file
					if (getVarParam(key.toString()) != null && (getVarParam(key.toString()) instanceof ConstIdentifier) 
							&& !getVarParam(key.toString()).toString().equalsIgnoreCase(configObject.get(key).toString()) ){
						throw new LanguageException("ERROR: parameter " + key.toString() + " has conflicting values in read statement definition and metadata. " +
								"Config file value: " + configObject.get(key).toString() + " from MTD file.  Read statement value: " + getVarParam(key.toString()));	
					}
					else {
						// if the InputStatement does not specify parameter value, then add MTD metadata file value to parameter list
						if (getVarParam(key.toString()) == null)
							addVarParam(key.toString(), new StringIdentifier(configObject.get(key).toString()));
					}
				}
	        }
	        else {
	        	System.out.println("INFO: could not find metadata file: " + pt);
	        }
			
	        
	        dataTypeString = (getVarParam(Statement.DATATYPEPARAM) == null) ? null : getVarParam(Statement.DATATYPEPARAM).toString();
			
			if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(Statement.MATRIX_DATA_TYPE) ) {
				
		        _output.setDataType(DataType.MATRIX);
				
		        // Following dimension checks must be done when data type = MATRIX_DATA_TYPE 
				// initialize size of target data identifier to UNKNOWN
				_output.setDimensions(-1, -1);
				
				if ( getVarParam(Statement.READROWPARAM) == null || getVarParam(Statement.READCOLPARAM) == null)
					throw new LanguageException("ERROR: Missing or incomplete dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				
				if (getVarParam(Statement.READROWPARAM) instanceof ConstIdentifier && getVarParam(Statement.READCOLPARAM) instanceof ConstIdentifier)  {
				
					// these are strings that are long values
					Long dim1 = (getVarParam(Statement.READROWPARAM) == null) ? null : new Long (getVarParam(Statement.READROWPARAM).toString());
					Long dim2 = (getVarParam(Statement.READCOLPARAM) == null) ? null : new Long(getVarParam(Statement.READCOLPARAM).toString());
					
					if ( dim1 <= 0 || dim2 <= 0 ) {
						throw new LanguageException("Invalid dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					}
					// set dim1 and dim2 values 
					if (dim1 != null && dim2 != null){
						_output.setDimensions(dim1, dim2);
					} else if ((dim1 != null) || (dim2 != null)) {
						throw new LanguageException("Partial dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
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
				} else {
					throw new LanguageException("Invalid format in statement: " + this.toString());
				}
				
				if (getVarParam(Statement.ROWBLOCKCOUNTPARAM) instanceof ConstIdentifier && getVarParam(Statement.COLUMNBLOCKCOUNTPARAM) instanceof ConstIdentifier)  {
				
					Long rowBlockCount = (getVarParam(Statement.ROWBLOCKCOUNTPARAM) == null) ? null : new Long(getVarParam(Statement.ROWBLOCKCOUNTPARAM).toString());
					Long columnBlockCount = (getVarParam(Statement.COLUMNBLOCKCOUNTPARAM) == null) ? null : new Long (getVarParam(Statement.COLUMNBLOCKCOUNTPARAM).toString());
		
					if ((rowBlockCount != null) && (columnBlockCount != null)) {
						_output.setBlockDimensions(rowBlockCount, columnBlockCount);
					} else if ((rowBlockCount != null) || (columnBlockCount != null)) {
						throw new LanguageException("Partial block dimension information in read statement", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
					} else {
						 _output.setBlockDimensions(-1, -1);
					}
				}
				
				// block dimensions must be -1x-1 when format="text"
				// and they must be 1000x1000 when format="binary"
				if ( (format == 1 && (_output.getRowsInBlock() != -1 || _output.getColumnsInBlock() != -1))
						|| (format == 2 && (_output.getRowsInBlock() != DMLTranslator.DMLBlockSize || _output.getColumnsInBlock() != DMLTranslator.DMLBlockSize)))
					throw new LanguageException("Invalid block dimensions (" + _output.getRowsInBlock() + "," + _output.getColumnsInBlock() + ") when format=" + getVarParam(Statement.FORMAT_TYPE) + " in \"" + this.toString() + "\".");
			}
			
			else if ( dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)) {
				_output.setDataType(DataType.SCALAR);
			}
			
			else{		
				throw new LanguageException("ERROR: Unknown Data Type " + dataTypeString + ". Valid  values: " + Statement.SCALAR_DATA_TYPE +", " + Statement.MATRIX_DATA_TYPE, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			// handle value type parameter
			if (getVarParam(Statement.VALUETYPEPARAM) != null && !(getVarParam(Statement.VALUETYPEPARAM) instanceof StringIdentifier))
				throw new LanguageException("ERROR: for InputStatement, parameter " + Statement.VALUETYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE,
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			
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
				} else{
					throw new LanguageException("Unknown Value Type " + valueTypeString
							+ ". Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE,
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
			} else {
				_output.setValueType(ValueType.DOUBLE);
			}

			break; 
			
		case WRITE:
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
							removeVarParam(Statement.IO_FILENAME);
							addVarParam(Statement.IO_FILENAME, fileString);
												
							break;
						default:
							throw new LanguageException("ERROR: for OutputStatement, parameter " + Statement.IO_FILENAME + " can only be a const string or const string concatenations. ");
					}
				}
			}
			
			if (getVarParam(Statement.FORMAT_TYPE) == null || getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase("text"))
				_output.setBlockDimensions(-1, -1);
			else if (getVarParam(Statement.FORMAT_TYPE).toString().equalsIgnoreCase("binary"))
				_output.setBlockDimensions(DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			else
				throw new LanguageException("Invalid format in statement: " + this.toString());
			
			break;

		default:
			throw new LanguageException("Unsupported Data expression"
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
			throw new LanguageException("Parameter " + Statement.IO_FILENAME + " only supports a const string or const string concatenations.");
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
			throw new LanguageException("Parameter " + Statement.IO_FILENAME + " only supports a const string or const string concatenations.");
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

}
