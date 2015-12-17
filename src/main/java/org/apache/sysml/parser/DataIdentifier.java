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


public class DataIdentifier extends Identifier 
{
	
	protected String 	_name;
	protected String 	_valueTypeString;	
	protected String 	_defaultValue;
	
	public DataIdentifier(DataIdentifier passed){
		setProperties(passed);
		_kind = Kind.Data;
		_name = passed.getName();
		_valueTypeString = passed.getValueType().toString();	
		_defaultValue = passed.getDefaultValue();
		
		// set location information
		setFilename(passed.getFilename());
		setBeginLine(passed.getBeginLine());
		setBeginColumn(passed.getBeginColumn());
		setEndLine(passed.getEndLine());
		setEndColumn(passed.getEndColumn());
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		DataIdentifier newId = new DataIdentifier(this);
		String newIdName = prefix + this._name;
		newId.setName(newIdName);
				
		return newId;
	}
	
	public DataIdentifier(String name){
		super();
		_name = name;
		_kind = Kind.Data;
		_defaultValue = null;

	}
	
	/*
	public DataIdentifier(String name, int line, int col){
		super();
		_name = name;
		_kind = Kind.Data;
		_defaultValue = null;	
	}
	*/
	public DataIdentifier(){
		_name = null;
		_kind = null;
		_defaultValue = null;
	}
	

	public void setTypeInfo( String valueType, String dataType) throws DMLParseException{
		
		if (valueType.equalsIgnoreCase("int") || valueType.equalsIgnoreCase("integer"))
			this.setValueType(ValueType.INT);
		else if (valueType.equalsIgnoreCase("double"))
			this.setValueType(ValueType.DOUBLE);
		else if (valueType.equalsIgnoreCase("string"))
			this.setValueType(ValueType.STRING);
		else if (valueType.equalsIgnoreCase("boolean"))
			this.setValueType(ValueType.BOOLEAN);
		else if (valueType.equalsIgnoreCase("object"))
			this.setValueType(ValueType.OBJECT);
		else {
			// provide location for this exception in the parser
			LOG.error(this.printErrorLocation() + "function parameter has unknown value type " + valueType);
			throw new DMLParseException(this.printErrorLocation() + "function parameter has unknown value type " + valueType);
		}
		
		if (dataType.equalsIgnoreCase("object"))
			this.setDataType(DataType.OBJECT);
		else if (dataType.equalsIgnoreCase("SCALAR"))
			this.setDataType(DataType.SCALAR);
		else if (dataType.equalsIgnoreCase("MATRIX"))
			this.setDataType(DataType.MATRIX);
		else {
			// provide location for this exception in the parser
			LOG.error(this.printErrorLocation() + "function parameter has unknown data type " + valueType);
			throw new DMLParseException(this.printErrorLocation() + "function parameter has unknown data type " + valueType);
		}
		
	}
	
	public String getName(){
		return _name;
	}
	public void setName(String name){
		_name = name;
	}
	public String getDefaultValue(){
		return _defaultValue;
	}
	public void setDefaultValue(String val){
		_defaultValue = val;
	}
	
	@Override
	public String toString() {
		return _name;
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariable(_name, this);
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		return null;
	}
	
	/**
	 * Method to specify if an expression returns multiple outputs.
	 * This method must be overridden by all child classes.
	 * @return
	 */
	public boolean multipleReturns() throws LanguageException {
		throw new LanguageException("multipleReturns() must be overridden in the subclass.");
	}
	
	@Override
	public boolean equals(Object that) 
	{
		if( !(that instanceof DataIdentifier) )
			return false;
			
		DataIdentifier target = (DataIdentifier)that;
		if(getName()!=null && !getName().equals(target.getName()))
			return false;
		if(getDataType()!=null && !getDataType().equals(target.getDataType()))
			return false;
		if(getValueType() != null && !getValueType().equals(target.getValueType()))
			return false;
		if(getFormatType()!= null && !this.getFormatType().equals(target.getFormatType()))
			return false;
		if(!(this.getDim1() == target.getDim1()))
			return false;
		if(!(this.getDim2() == target.getDim2()))
			return false;
		
		return true;
		
	}
	
	@Override
	public int hashCode()
	{
		return super.hashCode();
	}
}
