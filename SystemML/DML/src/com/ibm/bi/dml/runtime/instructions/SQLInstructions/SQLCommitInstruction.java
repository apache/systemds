package com.ibm.bi.dml.runtime.instructions.SQLInstructions;

import java.sql.SQLException;

import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class SQLCommitInstruction extends SQLInstructionBase{

	public SQLCommitInstruction()
	{
		instString = "COMMIT;";
	}
	
	@Override
	public ExecutionResult execute(ExecutionContext ec)
			throws DMLRuntimeException {
		
		try {
			if(!ec.getNzConnector().getConnection().getAutoCommit())
			ec.getNzConnector().getConnection().commit();
		} catch (SQLException e) {
			throw new DMLRuntimeException(e);
		}
		
		return new ExecutionResult();
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		// TODO Auto-generated method stub
		return null;
	}
	
}
