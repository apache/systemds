/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.context;


import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.cp.StringObject;
import com.ibm.bi.dml.sql.sqlcontrolprogram.NetezzaConnector;
import com.ibm.bi.dml.sql.sqlcontrolprogram.SQLExecutionStatistics;

public class SQLExecutionContext extends ExecutionContext
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//SQL-specific attributes
	private NetezzaConnector nzConnector = null;
	private boolean debug = false;
	private ArrayList<SQLExecutionStatistics> statistics = null;
	
	protected SQLExecutionContext(NetezzaConnector nzCon)
	{
		//protected constructor to force use of ExecutionContextFactory
		super( true, null );
		
		nzConnector = nzCon;
		statistics = new ArrayList<SQLExecutionStatistics>();
	}
	
	protected SQLExecutionContext( Program prog )
	{
		//protected constructor to force use of ExecutionContextFactory
		super(true, prog);
	}
	
	protected SQLExecutionContext( boolean allocateVariableMap, Program prog )
	{
		//protected constructor to force use of ExecutionContextFactory
		super(allocateVariableMap, prog);
		
		nzConnector = null;
		statistics = null;
	}

	public String getVariableString(String name, boolean forSQL)
	{
		Data obj = _variables.get(name);
		if(obj != null)
		{
			String s = ((ScalarObject)obj).getStringValue();
			if(obj instanceof StringObject)
				s = "'" + s + "'";
			else if (obj instanceof DoubleObject && forSQL)
				s = s + "::double precision";
 			return s;
		}
		else return name;
	}
	
	public NetezzaConnector getNzConnector() {
		return nzConnector;
	}

	public boolean isDebug() {
		return debug;
	}
	public void setDebug(boolean debug) {
		this.debug = debug;
	}
	
	public ArrayList<SQLExecutionStatistics> getStatistics() {
		return statistics;
	}
		
	/**
	 * 
	 * @param instructionId
	 * @param runtime
	 * @param opString
	 */
	public void addStatistic(int instructionId, long runtime, String opString) {
		SQLExecutionStatistics s = new SQLExecutionStatistics(opString, instructionId, runtime);
		statistics.add(s);
	}
	
	/**
	 * 
	 */
	public void clearStatistics() {
		statistics.clear();
	}
	
	/**
	 * 
	 * @return
	 */
	public HashMap<Integer, SQLExecutionStatistics> getStatisticsByInstruction() 
	{
		HashMap<Integer, SQLExecutionStatistics> stats = new HashMap<Integer, SQLExecutionStatistics>();
		
		for(SQLExecutionStatistics s : statistics)
		{
			if(stats.containsKey(s.getInstructionId()))
			{
				SQLExecutionStatistics st = stats.get(s.getInstructionId());
				st.setRuntime(st.getRuntime() + s.getRuntime());
				st.setTimesRun(st.getTimesRun() + 1);
			}
			else
				stats.put(s.getInstructionId(), s);
		}
		
		return stats;
	}
	
}
