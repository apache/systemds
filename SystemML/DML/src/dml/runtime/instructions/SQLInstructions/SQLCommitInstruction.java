package dml.runtime.instructions.SQLInstructions;

import java.sql.SQLException;

import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.sql.sqlcontrolprogram.ExecutionResult;
import dml.utils.DMLRuntimeException;

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
