package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLPrintInstruction;


public class SQLPrint extends SQLCreateTable{
	@Override
	public String generateSQLString() {
		StringBuffer sb = new StringBuffer();
		sb.append(this.get_tableName());
		sb.append(" := ");
		sb.append("( ");
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
		sb.append(");\r\n");

		sb.append("RAISE NOTICE 'Message: %', ");
		sb.append(this.get_tableName());
		sb.append(";\r\n");
		return sb.toString();
	}
	
	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[1];
		instr[0] = new SQLPrintInstruction(this.get_selectStatement());
		return instr;
	}

	@Override
	public int getStatementCount() {
		return 1;
	}
}
