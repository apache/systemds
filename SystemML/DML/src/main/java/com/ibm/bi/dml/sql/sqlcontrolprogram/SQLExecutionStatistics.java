/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

public class SQLExecutionStatistics 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
