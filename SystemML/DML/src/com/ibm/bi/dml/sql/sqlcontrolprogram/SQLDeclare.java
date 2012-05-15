package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;

public class SQLDeclare implements ISQLCode {
	
	public SQLDeclare()
	{
		
	}
	
	public SQLDeclare(String varname, String type)
	{
		this._variable_name = varname;
		this._type = type;
	}
	
	String _variable_name;
	String _type;
	public String get_variable_name() {
		return _variable_name;
	}
	public void set_variable_name(String variableName) {
		_variable_name = variableName;
	}
	public String get_type() {
		return _type;
	}
	public void set_type(String type) {
		_type = type;
	}

	@Override
	public Instruction[] getInstructions() {
		return new Instruction[0];
	}

	@Override
	public String generateSQLString() {
		return get_variable_name() + " " + get_type() + ";\r\n";
	}

	@Override
	public int getStatementCount() {
		return 0;
	}
}
