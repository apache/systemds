package com.ibm.bi.dml.parser;

import com.ibm.bi.dml.utils.LanguageException;

public class DataIdentifier extends Identifier {

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
		_beginLine 		= passed.getBeginLine();
		_beginColumn	= passed.getBeginColumn();
		_endLine		= passed.getEndLine();
		_endColumn		= passed.getEndColumn();
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
			LOG.error(this.printErrorLocation() + "function parameter has unknown value type " + valueType);
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
			LOG.error(this.printErrorLocation() + "function parameter has unknown data type " + valueType);
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
	
	
	public String toString() {
		String retVal = new String();
		retVal += _name;		
		return retVal;
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
	
	public boolean equals(DataIdentifier target){
		
		if (!this.getName().equals(target.getName()))
			return false;
		if (!this.getDataType().equals(target.getDataType()))
			return false;
		if (!this.getValueType().equals(target.getValueType()))
			return false;
		if (this.getFormatType() != null && !this.getFormatType().equals(target.getFormatType()))
			return false;
		if (!(this.getDim1() == target.getDim1()))
			return false;
		if (!(this.getDim2() == target.getDim2()))
			return false;
		
		return true;
		
	}
	
}
