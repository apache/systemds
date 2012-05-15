package com.ibm.bi.dml.sql.sqlcontrolprogram;

public class SQLExecutionStatistics {
	
	public SQLExecutionStatistics()
	{
		
	}
	
	public SQLExecutionStatistics(String opString, int instrId, long runtime)
	{
		this.opString = opString;
		this.instructionId = instrId;
		this.runtime = runtime;
	}
	
	int timesRun = 1;
	String opString;
	int instructionId;
	long runtime;
	
	
	public int getTimesRun() {
		return timesRun;
	}

	public void setTimesRun(int timesRun) {
		this.timesRun = timesRun;
	}

	public String getOpString() {
		return opString;
	}
	public void setOpString(String opString) {
		this.opString = opString;
	}
	public int getInstructionId() {
		return instructionId;
	}
	public void setInstructionId(int instructionId) {
		this.instructionId = instructionId;
	}
	public long getRuntime() {
		return runtime;
	}
	public void setRuntime(long runtime) {
		this.runtime = runtime;
	}
	
	
}
