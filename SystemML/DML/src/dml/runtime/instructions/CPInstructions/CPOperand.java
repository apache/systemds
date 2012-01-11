package dml.runtime.instructions.CPInstructions;

import dml.lops.Lops;
import dml.parser.Expression.*;

public class CPOperand {

	private String _name;
	private ValueType _valueType;
	
	public CPOperand(String str) {
		split_by_value_type_prefix(str);
	}
	
	CPOperand(String name, ValueType vt ) {
		_name = name;
		_valueType = vt;
	}
	
	public String get_name() {
		return _name;
	}
	
	public ValueType get_valueType() {
		return _valueType;
	}
	
	public void set_name(String name) {
		_name = name;
	}
	
	public void set_valueType(ValueType vt) {
		_valueType = vt;
	}
	
	public void split_by_value_type_prefix ( String str ) {
		String[] opr = str.split(Lops.VALUETYPE_PREFIX);
		_name = opr[0];
		_valueType = ValueType.valueOf(opr[1]);
	}

	
}
