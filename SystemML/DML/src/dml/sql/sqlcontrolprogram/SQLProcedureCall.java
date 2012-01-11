package dml.sql.sqlcontrolprogram;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.SQLInstructions.SQLInstruction;

public class SQLProcedureCall extends SQLCreateBase {
	String call;
	
	public SQLProcedureCall(String sql)
	{
		call = sql;
	}

	@Override
	public String generateSQLString() {
		return call;
	}

	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[1];
		instr[0] = new SQLInstruction(call);
		return instr;
	}

	@Override
	public int getStatementCount() {
		return 1;
	}
}
