package dml.runtime.instructions.SQLInstructions;

import dml.runtime.instructions.Instruction;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.sql.sqlcontrolprogram.ExecutionResult;
import dml.utils.DMLRuntimeException;

public abstract class SQLInstructionBase extends Instruction {
	
	private static int INSTRUCTION_ID = 0;
	
	public SQLInstructionBase()
	{
		id = INSTRUCTION_ID++;
	}
	
	int id;
	
	public int getId() {
		return id;
	}

	public abstract ExecutionResult execute(ExecutionContext ec) throws DMLRuntimeException ;
}
