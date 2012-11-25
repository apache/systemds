package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.runtime.instructions.Instruction;


public class CPOperand {

	private String _name;
	private ValueType _valueType;
	private DataType _dataType;
	
	public CPOperand(String str) {
		split(str);
		//split_by_value_type_prefix(str);
	}
	
	public CPOperand() {
		_name = "";
		_valueType = ValueType.UNKNOWN;
		_dataType = DataType.UNKNOWN;
	}
	
	public CPOperand(String name, ValueType vt, DataType dt ) {
		_name = name;
		_valueType = vt;
		_dataType = dt;
	}
	
	public String get_name() {
		return _name;
	}
	
	public ValueType get_valueType() {
		return _valueType;
	}
	
	public DataType get_dataType() {
		return _dataType;
	}
	
	public void set_name(String name) {
		_name = name;
	}
	
	public void set_valueType(ValueType vt) {
		_valueType = vt;
	}
	
	public void set_dataType(DataType dt) {
		_dataType = dt;
	}
	
	public void split_by_value_type_prefix ( String str ) {
		String[] opr = str.split(Lops.VALUETYPE_PREFIX);
		_name = opr[0];
		_valueType = ValueType.valueOf(opr[1]);
	}

	public void split(String str){
		String[] opr = str.split(Instruction.VALUETYPE_PREFIX);
		if ( opr.length == 3 ) {
			_name = opr[0];
			_dataType = DataType.valueOf(opr[1]);
			_valueType = ValueType.valueOf(opr[2]);
		}
		else {
			_name = opr[0];
			_valueType = ValueType.valueOf(opr[1]);
		}
	}
	
	public void copy(CPOperand o){
		_name = o.get_name();
		_valueType = o.get_valueType();
		_dataType = o.get_dataType();
	}
}
