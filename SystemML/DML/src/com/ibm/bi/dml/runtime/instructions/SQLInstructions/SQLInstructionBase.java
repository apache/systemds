package com.ibm.bi.dml.runtime.instructions.SQLInstructions;

import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;
import com.ibm.bi.dml.utils.DMLRuntimeException;


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
