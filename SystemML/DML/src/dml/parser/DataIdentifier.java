package dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import dml.utils.LanguageException;

public class DataIdentifier extends Identifier {

	protected String _name;
	private String 	_valueTypeString;	
	private String 	_defaultValue;
	
	// Store information about where the data identifier has been defined in original DML script 
	private int _definedLine;
	private int _definedCol;
	
	private Expression _rowLowerBound = null, _rowUpperBound = null, _colLowerBound = null, _colUpperBound = null;
	
	public void setIndices(ArrayList<ArrayList<Expression>> passed) {
		if (passed.size() != 2) 
			System.out.println("[E] indices wrong length -- should be 2");
		ArrayList<Expression> rowIndices = passed.get(0);
		ArrayList<Expression> colIndices = passed.get(1);
	
		_rowLowerBound = rowIndices.get(0);
		if (rowIndices.size() == 2) _rowUpperBound = rowIndices.get(1);
		_colLowerBound = colIndices.get(0);
		if (colIndices.size() == 2) _colUpperBound = colIndices.get(1);
		
		System.out.println(this);
		
	}
	
	public Expression getRowLowerBound(){ return this._rowLowerBound; }
	public Expression getRowUpperBound(){ return this._rowUpperBound; }
	public Expression getColLowerBound(){ return this._colLowerBound; }
	public Expression getColUpperBound(){ return this._colUpperBound; }
	
	public void setRowLowerBound(Expression passed){ this._rowLowerBound = passed; }
	public void setRowUpperBound(Expression passed){ this._rowUpperBound = passed; }
	public void setColLowerBound(Expression passed){ this._colLowerBound = passed; }
	public void setColUpperBound(Expression passed){ this._colUpperBound = passed; }
	
	
	public DataIdentifier(DataIdentifier passed){
		setProperties(passed);
		_kind = Kind.Data;
		_name = passed.getName();
		_valueTypeString = passed.getValueType().toString();	
		_defaultValue = passed.getDefaultValue();
		
		_rowLowerBound = passed.getRowLowerBound();
		_rowUpperBound = passed.getRowUpperBound();
		_colLowerBound = passed.getColLowerBound();
		_colUpperBound = passed.getColUpperBound();
		
	}
		
	public Expression rewriteExpression(String prefix) throws LanguageException{
		DataIdentifier newId = new DataIdentifier(this);
		String newIdName = prefix + this._name;
		newId.setName(newIdName);
		
		if (_rowLowerBound != null || _rowUpperBound != null || _colLowerBound != null || _colUpperBound != null)
			throw new LanguageException("rewrite does not support indices yet");
		
		return newId;
	}
	
	public DataIdentifier(String name){
		super();
		_name = name;
		_kind = Kind.Data;
		_defaultValue = null;
		_definedLine = -1;
		_definedCol = -1;
	}
	
	public DataIdentifier(String name, int line, int col){
		super();
		_name = name;
		_kind = Kind.Data;
		_defaultValue = null;
		
		_definedLine = line;
		_definedCol = col;
	}
	
	public DataIdentifier(){
		_name = null;
		_kind = null;
		_defaultValue = null;
	}
	
	public int getDefinedLine(){ return _definedLine; }
	public int getDefinedCol(){ return _definedCol; }
	
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
			throw new ParseException("function parameter has unknown value type " + valueType);
		}
		
		if (dataType.equalsIgnoreCase("object"))
			this.setDataType(DataType.OBJECT);
		else if (dataType.equalsIgnoreCase("SCALAR"))
			this.setDataType(DataType.SCALAR);
		else if (dataType.equalsIgnoreCase("MATRIX"))
			this.setDataType(DataType.MATRIX);
		else {
			throw new ParseException("function parameter has unknown data type " + valueType);
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
		if (_rowLowerBound != null || _rowUpperBound != null || _colLowerBound != null || _colUpperBound != null){
				retVal += "[";
				
				if (_rowLowerBound == null && _rowUpperBound == null) 
					retVal += ":";
				else if (_rowLowerBound != null && _rowUpperBound != null)
					retVal += _rowLowerBound.toString() + ":" + _rowUpperBound.toString();
				else
					retVal += _rowLowerBound.toString();
					
				retVal += ",";
				
				if (_colLowerBound == null && _colUpperBound == null) 
					retVal += ":";
				else if (_colLowerBound != null && _colUpperBound != null)
					retVal += _colLowerBound.toString() + ":" + _colUpperBound.toString();
				else
					retVal += _colLowerBound.toString();
				
				retVal += "]";
		}
		
		return retVal;
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariable(_name, this);
		
		if (_rowLowerBound != null)
			result.addVariables(_rowLowerBound.variablesRead());
		if (_rowUpperBound != null)
			result.addVariables(_rowUpperBound.variablesRead());
		if (_colLowerBound != null)
			result.addVariables(_colLowerBound.variablesRead());
		if (_colUpperBound != null)
			result.addVariables(_colUpperBound.variablesRead());
		
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
