/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.sql;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.cp.IntObject;
import com.ibm.bi.dml.runtime.instructions.cp.StringObject;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;


public class SQLScalarAssignInstruction extends SQLInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public SQLScalarAssignInstruction()
	{
		
	}
	
	public SQLScalarAssignInstruction(String varname, String sql, ValueType vt)
	{
		if(varname.contains("converge"))
			System.out.println();
		this.variableName = varname;
		this.sql = sql;
		this.instString = varname + " := " + sql;
		this.vt = vt;
	}
	
	public SQLScalarAssignInstruction(String varname, String sql, ValueType vt,
			boolean hasSelect)
	{
		if(varname.contains("converge"))
			System.out.println();
		this.variableName = varname;
		this.sql = sql;
		this.instString = varname + " := " + sql;
		this.vt = vt;
		this.hasSelect = hasSelect;
	}
	
	boolean hasSelect = true;
	ValueType vt;
	String variableName;
	String sql;
	String prepSQL;
	String prepName;
	
	public String getVariableName() {
		return variableName;
	}

	public void setVariableName(String variableName) {
		this.variableName = variableName;
	}

	public String getSql() {
		return sql;
	}

	public void setSql(String sql) {
		this.sql = sql;
	}

	private void prepare(ExecutionContext ec)
	{
		prepName = variableName;
		prepSQL = sql;
		
		if(prepName.startsWith("##"))
			prepName = prepName.substring(2, prepName.length() - 2);

		if(!prepSQL.contains("##"))
			return;
		
		while(true)
		{
			int from = prepSQL.indexOf("##", 0);
			if(from == -1)
				break;
			int to = prepSQL.indexOf("##", from + 2);
			if(to == -1)
				break;
			String name = prepSQL.substring(from + 2, to);
			prepSQL = prepSQL.replace("##" + name + "##", ec.getVariableString(name, hasSelect));
		}
	}
	
	@Override
	public ExecutionResult execute(ExecutionContext ec) throws DMLRuntimeException {
		prepare(ec);
		
		if(ec.isDebug())
		{
			//System.out.println("#" + this.id + "\r\n");
			System.out.println(prepName + " := " + prepSQL);
		}
		
		ExecutionResult res = new ExecutionResult();
		long start = System.currentTimeMillis();
		res.setSuccess(true);
		
		if(!hasSelect)
		{
			//TODO: the literal flag in getScalarInput() is passed as false. The value should actually come from SQLLop
			ec.setVariable(this.prepName, ec.getScalarInput(this.prepSQL, vt, false));
			return null;
		}
		try
		{
			if(this.vt == ValueType.BOOLEAN)
			{
				boolean val = ec.getNzConnector().getScalarBoolean(this.prepSQL);
				ec.setVariable(this.prepName, new BooleanObject(val));
			}
			else if(this.vt == ValueType.DOUBLE)
			{
				double val = ec.getNzConnector().getScalarDouble(this.prepSQL);
				//System.out.println("VALUE FOR VARIABLE " + prepName + " = " + val);
				ec.setVariable(this.prepName, new DoubleObject(val));
			}
			else if(this.vt == ValueType.INT)
			{
				int val = ec.getNzConnector().getScalarInteger(this.prepSQL);
				ec.setVariable(this.prepName, new IntObject(val));
			}
			else if(this.vt == ValueType.STRING)
			{
				String val = ec.getNzConnector().getScalarString(this.prepSQL);
				ec.setVariable(this.prepName, new StringObject(val));
			}
		}
		catch(Exception e)
		{
			res.setSuccess(false);
			throw new DMLRuntimeException(e);
		}
		res.setRuntimeInMilliseconds(System.currentTimeMillis() - start);

		System.out.println("#" + this.id + ": " + res.getRuntimeInMilliseconds() + "\r\n");
		
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
