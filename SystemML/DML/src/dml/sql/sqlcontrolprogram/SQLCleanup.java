package dml.sql.sqlcontrolprogram;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.SQLInstructions.SQLDropTableInstruction;
import dml.runtime.instructions.SQLInstructions.SQLInstruction;

public class SQLCleanup implements ISQLCode{
	
	public SQLCleanup()
	{
		
	}
	
	public SQLCleanup(boolean isView)
	{
		isview = isView;
	}
	
	boolean isview = false;
	
	public SQLCleanup(String tblname)
	{
		_tableName = tblname;
	}
	
	private String _tableName;

	public String get_tableName() {
		return _tableName;
	}

	public void set_tableName(String tableName) {
		_tableName = tableName;
	}
	
	public String generateSQLString()
	{
		return "call drop_if_exists('" + this.get_tableName() + "');\r\n";
		
		/*if(isview)
			return "DROP VIEW \"" + this.get_tableName() + "\";\r\n";
		else
			return "DROP TABLE \"" + this.get_tableName() + "\";\r\n";*/
	}

	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[1];
		instr[0] = new SQLDropTableInstruction(get_tableName());
		//instr[0] = new SQLInstruction("call drop_if_exists('" + this.get_tableName() + "');");
		return instr;
	}

	@Override
	public int getStatementCount() {
		return 1;
	}
}
