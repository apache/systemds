package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;


public class SQLVariableAssignment extends SQLCreateTable {

	public SQLVariableAssignment(ValueType vt)
	{
		this.valueType = vt;
	}
	
	public SQLVariableAssignment(ValueType vt, boolean hasSelect)
	{
		this.valueType = vt;
		this.hasSelect = hasSelect;
	}
	
	boolean hasSelect = false;
	ValueType valueType;

	public ValueType getValueType() {
		return valueType;
	}

	public void setValueType(ValueType valueType) {
		this.valueType = valueType;
	}

	public String getMergedSelectStatement()
	{
		StringBuffer sb = new StringBuffer();
		if(this.get_withs().size() > 0)
			sb.append(" WITH ");
		for(int i= 0; i < this.get_withs().size(); i++)
		{
			SQLWithTable wt = this.get_withs().get(i);
			sb.append(wt.generateSQLString());
			if(i < this.get_withs().size() - 1)
				sb.append(",\r\n");
			else
				sb.append("\r\n");
		}
		sb.append(this.get_selectStatement());
		return sb.toString();
	}
	
	@Override
	public String generateSQLString() {
		
		StringBuffer sb = new StringBuffer();
		//this.get_tableName() + " := " + this.get_selectStatement() + ";\r\n"
		sb.append(this.get_tableName());
		sb.append(" := ( ");
		
		sb.append(getMergedSelectStatement());
		
		sb.append(" ) ;\r\n");
		
		return sb.toString();
	}
	
	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[1];
		instr[0] = new SQLScalarAssignInstruction(
				this.get_tableName(), this.getMergedSelectStatement(), this.valueType, hasSelect);
		return instr;
	}
	
	public int getStatementCount()
	{
		return 1;
	}
}
