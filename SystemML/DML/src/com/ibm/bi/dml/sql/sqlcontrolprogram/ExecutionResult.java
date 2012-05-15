package com.ibm.bi.dml.sql.sqlcontrolprogram;

import java.util.Date;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;


public class ExecutionResult {
	
	public ExecutionResult()
	{
	}
	
	public ExecutionResult(boolean success, long runtime, Date exeTime)
	{
		this.executionTime = exeTime;
		this.success = success;
		runtimeInMilliseconds = runtime;
	}
	
	public ExecutionResult(boolean success, long runtime)
	{
		this.executionTime = new Date();
		this.success = success;
		runtimeInMilliseconds = runtime;
	}
	
	Date executionTime;
	protected HashMap<String, Data> _variables;
	boolean success = true;
	long runtimeInMilliseconds = 0;

	public Date getExecutionTime() {
		return executionTime;
	}

	public void setExecutionTime(Date executionTime) {
		this.executionTime = executionTime;
	}

	public HashMap<String, Data> get_variables() {
		return _variables;
	}

	public boolean isSuccess() {
		return success;
	}
	public void setSuccess(boolean success) {
		this.success = success;
	}
	public long getRuntimeInMilliseconds() {
		return runtimeInMilliseconds;
	}
	public void setRuntimeInMilliseconds(long runtimeInMilliseconds) {
		this.runtimeInMilliseconds = runtimeInMilliseconds;
	}
	
	
}
