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


public class SQLInstruction extends SQLInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public SQLInstruction(String sql)
	{
		this.sql = sql;
		this.instString = sql;
	}
	
	String sql;
	
	private String prepare(ExecutionContext ec)
	{
		SQLExecutionContext sec = (SQLExecutionContext)ec;
		
		if(!sql.contains("##"))
			return sql;
		
		String prepSQL = sql;
		
		int start = 0;
		while(true)
		{
			int from = prepSQL.indexOf("##", start);
			if(from == -1)
				break;
			int to = prepSQL.indexOf("##", from + 2);
			if(to == -1)
				break;
			start = to + 2;
			String name = prepSQL.substring(from+2, to);
			prepSQL = prepSQL.replace("##" + name + "##", sec.getVariableString(name, true));
		}
		return prepSQL;
	}
	
	@Override
	public ExecutionResult execute(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		SQLExecutionContext sec = (SQLExecutionContext)ec;
		ExecutionResult res = new ExecutionResult();
		String prepSQL = prepare(ec);

		if(sec.isDebug())
		{
			System.out.println("#" + this.id + "\r\n");
			System.out.println(prepSQL);
		}
		
		try {
			long time = System.currentTimeMillis();
			sec.getNzConnector().executeSQL(prepSQL);
			res.setRuntimeInMilliseconds(System.currentTimeMillis() - time);
			
		} catch (SQLException e) {
			res.setSuccess(false);
			throw new DMLRuntimeException(e);
		}
		System.out.println("#" + this.id + ": " + res.getRuntimeInMilliseconds() + "\r\n");
		sec.addStatistic(this.getId(), res.getRuntimeInMilliseconds(), this.instString);
		return res;
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
