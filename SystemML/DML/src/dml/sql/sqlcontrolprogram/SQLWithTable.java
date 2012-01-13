package dml.sql.sqlcontrolprogram;

import dml.runtime.instructions.Instruction;

public class SQLWithTable extends SQLCreateBase {

	@Override
	public String generateSQLString() {
		return String.format("\"%s\" AS ( %s )", this.get_tableName(), this.get_selectStatement());
	}

	@Override
	public Instruction[] getInstructions() {
		return new Instruction[0];
	}

	@Override
	public int getStatementCount() {
		return 0;
	}
	
}
