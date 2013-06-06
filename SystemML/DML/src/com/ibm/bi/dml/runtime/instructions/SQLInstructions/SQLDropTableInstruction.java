package com.ibm.bi.dml.runtime.instructions.SQLInstructions;

import java.sql.SQLException;

import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class SQLDropTableInstruction extends SQLInstructionBase {

	public SQLDropTableInstruction(String tblName)
	{
		this.tableName = tblName;
		this.instString = "DROP TABLE \"" + this.tableName + "\";";
	}
	
	public SQLDropTableInstruction()
	{
		
	}
	
	String tableName;
	
	public String getTableName() {
		return tableName;
	}

	public void setTableName(String tableName) {
		this.tableName = tableName;
	}

	@Override
	public ExecutionResult execute(ExecutionContext ec)
			throws DMLRuntimeException {
		
		//String prepSQL = "DROP TABLE \"" + this.tableName + "\";";
		
		ExecutionResult res = new ExecutionResult();
		
		long time = System.currentTimeMillis();
		try
		{
			ec.getNzConnector().executeSQL(instString);
			if(ec.isDebug())
			{
				//System.out.println("#" + this.id + "\r\n");
				System.out.println(instString);
			}
		}
		catch(SQLException e)
		{
			System.out.println("Could not drop table " + this.tableName);
		}
		catch(Exception e)
		{
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
		res.setRuntimeInMilliseconds(System.currentTimeMillis() - time);
		
		ec.addStatistic(this.getId(), res.getRuntimeInMilliseconds(), this.instString);
		
		return res;
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
