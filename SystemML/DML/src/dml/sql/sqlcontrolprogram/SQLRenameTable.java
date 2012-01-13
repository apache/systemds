package dml.sql.sqlcontrolprogram;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.SQLInstructions.SQLDropTableInstruction;
import dml.runtime.instructions.SQLInstructions.SQLInstruction;

public class SQLRenameTable extends SQLCreateBase {
	
	public SQLRenameTable()
	{
		
	}
	
	public SQLRenameTable(String oldName, String newName)
	{
		this.set_tableName(oldName);
		_newName = newName;
	}
	
	String _newName;
	
	public String get_newName() {
		return _newName;
	}
	public void set_newName(String newName) {
		_newName = newName;
	}
 
	@Override
	public String generateSQLString() {
		String sql = 
			"call drop_if_exists('" + _newName + "');\r\n" +
			"ALTER TABLE \"" + this.get_tableName() + "\" RENAME TO \"" + _newName + "\";\r\n";
		return sql;
	}

	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[2];
		//instr[0] = new SQLInstruction("call drop_if_exists('" + _newName + "');\r\n");
		instr[0] = new SQLDropTableInstruction(_newName);
		instr[1] = new SQLInstruction("ALTER TABLE \"" + this.get_tableName() + "\" RENAME TO \"" + _newName + "\";\r\n");
		return instr;
	}

	@Override
	public int getStatementCount() {
		return 2;
	}
	
	
}
