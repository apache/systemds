package dml.sql.sqlcontrolprogram;

import dml.runtime.instructions.Instruction;

public interface ISQLCode extends SQLCodeGeneratable {
	public Instruction[] getInstructions();
}
