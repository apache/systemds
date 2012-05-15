package com.ibm.bi.dml.runtime.instructions.SQLInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class SQLScalarAssignInstruction extends SQLInstructionBase {

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
			ec.setVariable(this.prepName, ec.getVariable(this.prepSQL, vt));
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
