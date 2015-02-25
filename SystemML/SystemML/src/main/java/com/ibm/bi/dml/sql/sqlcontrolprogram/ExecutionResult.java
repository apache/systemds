/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import java.util.Date;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.instructions.cp.Data;


public class ExecutionResult 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
