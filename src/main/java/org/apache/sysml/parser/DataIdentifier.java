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
	protected String _name;
	protected String _valueTypeString;	
	
	public DataIdentifier(DataIdentifier passed){
		setProperties(passed);
		_name = passed.getName();
		_valueTypeString = passed.getValueType().toString();	
		
		// set location information
		setFilename(passed.getFilename());
		setBeginLine(passed.getBeginLine());
		setBeginColumn(passed.getBeginColumn());
		setEndLine(passed.getEndLine());
		setEndColumn(passed.getEndColumn());
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		DataIdentifier newId = new DataIdentifier(this);
		String newIdName = prefix + _name;
		newId.setName(newIdName);
				
		return newId;
	}
	
	public DataIdentifier(String name){
		super();
		_name = name;
	}
	
	public DataIdentifier(){
		_name = null;
	}

	public String getName(){
		return _name;
	}
	public void setName(String name){
		_name = name;
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
	 * 
	 * @return true if expression returns multiple outputs
	 * @throws LanguageException if LanguageException occurs
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
