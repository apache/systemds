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



public class ExternalFunctionStatement extends FunctionStatement
{
		
	//valid attribute names
	public static final String CLASS_NAME    = "classname";
	public static final String EXEC_TYPE     = "exectype";
	//public static final String EXEC_LOCATION = "execlocation"; 	//MB: obsolete
	public static final String CONFIG_FILE   = "configfile";
	
	//valid attribute values for execlocation and 
	public static final String FILE_BASED    = "file";
	public static final String IN_MEMORY     = "mem";
	//public static final String MASTER        = "master";	//MB: obsolete
	//public static final String WORKER        = "worker";	//MB: obsolete
	
	//default values for optional attributes
	public static final String DEFAULT_EXEC_TYPE = FILE_BASED;
	//public static final String DEFAULT_EXEC_LOCATION = MASTER; 	//MB: obsolete
	
	//all parameters
	private HashMap<String,String> _otherParams;
	
	
	public ExternalFunctionStatement(){
		super();
	}
	
	
	
	public ExternalFunctionStatement( ArrayList<DataIdentifier> functionInputs, ArrayList<DataIdentifier> functionOutputs, HashMap<String,String> map)
	{
		super();
		_inputParams = functionInputs; 
		_outputParams = (functionOutputs == null) ? new ArrayList<DataIdentifier>() : functionOutputs;
		setOtherParams(map);
	}
	
	
	public void setOtherParams(HashMap<String,String> params){
		_otherParams = params;
	}
	
	public HashMap<String,String> getOtherParams(){
		return _otherParams;
	}
	
	/**
	 * Validates all attributes and attribute values.
	 * 
	 * @throws LanguageException
	 */
	public void validateParameters(StatementBlock sb) //always unconditional  
		throws LanguageException 
	{
		
		//warnings for all not defined attributes
		for( String varName : _otherParams.keySet() )
			if( !(   varName.equals(CLASS_NAME) || varName.equals(EXEC_TYPE) 
				  || varName.equals(CONFIG_FILE) ) )                                                  
			{
				LOG.warn( printWarningLocation() + "External function specifies undefined attribute type '"+varName+"'.");
			}
		
		//class name (required)
		if( !_otherParams.containsKey(CLASS_NAME) ){
			sb.raiseValidateError("External function does not specify the required attribute '"+CLASS_NAME+"'.", false);
		}
		else if ( _otherParams.get(CLASS_NAME)==null ) {
			sb.raiseValidateError("External function specifies empty '"+CLASS_NAME+"'.", false);
		}
		
		//exec type (optional, default: file)
		if( _otherParams.containsKey( EXEC_TYPE ) )
		{
			//check specified values
			String execType = _otherParams.get(EXEC_TYPE);
			if( !(execType.equals(FILE_BASED) || execType.equals(IN_MEMORY)) ) { //always unconditional (invalid parameter)
				sb.raiseValidateError("External function specifies invalid value for (optional) attribute '"+EXEC_TYPE+"' (valid values: "+FILE_BASED+","+IN_MEMORY+").", false);
			}
		}
		else
		{
			//put default values
			_otherParams.put(EXEC_TYPE, DEFAULT_EXEC_TYPE);
		}
	}
	
	@Override
	public boolean controlStatement() 
	{
		return true;
	}
	
	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append(_name + " = ");
		
		sb.append("externalfunction ( ");
		for (int i=0; i<_inputParams.size(); i++){
			DataIdentifier curr = _inputParams.get(i);
			sb.append(curr.getName());
			if (curr.getDefaultValue() != null) sb.append(" = " + curr.getDefaultValue());
			if (i < _inputParams.size()-1) sb.append(", ");
		}
		sb.append(") return (");
		
		for (int i=0; i<_outputParams.size(); i++){
			sb.append(_outputParams.get(i).getName());
			if (i < _outputParams.size()-1) sb.append(", ");
		}
		sb.append(")\n implemented in (");
		
		int j = 0;
		for (String key : _otherParams.keySet()){
			sb.append(key + " = " + _otherParams.get(key));
			if (j < _otherParams.keySet().size()-1) sb.append(", ");
			j++;
		}
		
		sb.append(") \n");
		return sb.toString();
	}

	@Override
	public void initializeforwardLV(VariableSet activeIn) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should never call initializeforwardLV for ExternalFunctionStatement");
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for ExternalFunctionStatement");
	}
	
	@Override
	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should never call initializeforwardLV for ExternalFunctionStatement");
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for ExternalFunctionStatement");
		
	}
	
	@Override
	public VariableSet variablesRead() {
		LOG.warn(this.printWarningLocation() + "should not call variablesRead from ExternalFunctionStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		LOG.warn(this.printWarningLocation() + "should not call variablesRead from ExternalFunctionStatement ");
		return new VariableSet();
	}
}
