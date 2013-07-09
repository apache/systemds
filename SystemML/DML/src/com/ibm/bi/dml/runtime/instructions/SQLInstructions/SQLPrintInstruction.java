package com.ibm.bi.dml.runtime.instructions.SQLInstructions;

import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class SQLPrintInstruction extends SQLInstructionBase{

	public SQLPrintInstruction(String sql)
	{
		this.sql = sql;
		this.instString = "print(" + sql  +")";
	}
	
	String sql;
	//PrintStream stream;
	
	@Override
	public ExecutionResult execute(ExecutionContext ec) throws DMLRuntimeException {
		String s = null;
		try {
			if(ec.isDebug())
			{
				System.out.println("#" + this.id + "\r\n");
				System.out.println(sql);
			}
			s = ec.getVariableString(sql.substring(2, sql.length()-2), false);
		} catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		System.out.println(s);
		 
		return new ExecutionResult(true, 0);
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		return null;
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		return null;
	}
	
}
