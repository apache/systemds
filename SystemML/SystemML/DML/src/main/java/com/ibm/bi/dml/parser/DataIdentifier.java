/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;


public class DataIdentifier extends Identifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	

	public void setTypeInfo( String valueType, String dataType) throws ParseException{
		
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
			throw new ParseException(this.printErrorLocation() + "function parameter has unknown value type " + valueType);
		}
		
		if (dataType.equalsIgnoreCase("object"))
			this.setDataType(DataType.OBJECT);
		else if (dataType.equalsIgnoreCase("SCALAR"))
			this.setDataType(DataType.SCALAR);
		else if (dataType.equalsIgnoreCase("MATRIX"))
			this.setDataType(DataType.MATRIX);
		else {
			// provide location for this exception in the parser
			throw new ParseException(this.printErrorLocation() + "function parameter has unknown data type " + valueType);
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
