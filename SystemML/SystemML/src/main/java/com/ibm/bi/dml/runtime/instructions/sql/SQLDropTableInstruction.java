/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.sql;

import java.sql.SQLException;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SQLExecutionContext;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;


public class SQLDropTableInstruction extends SQLInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
			throws DMLRuntimeException 
	{	
		//String prepSQL = "DROP TABLE \"" + this.tableName + "\";";
	
		SQLExecutionContext sec = (SQLExecutionContext)ec;
		ExecutionResult res = new ExecutionResult();
		
		long time = System.currentTimeMillis();
		try
		{
			sec.getNzConnector().executeSQL(instString);
			if(sec.isDebug())
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
			throw new DMLRuntimeException(e);
		}
		res.setRuntimeInMilliseconds(System.currentTimeMillis() - time);
		
		sec.addStatistic(this.getId(), res.getRuntimeInMilliseconds(), this.instString);
		
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
