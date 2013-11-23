/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.sql.sqlcontrolprogram.NetezzaConnector;
import com.ibm.bi.dml.sql.sqlcontrolprogram.SQLExecutionStatistics;


public class ExecutionContext 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//symbol table
	private LocalVariableMap _variables;
	
	private NetezzaConnector nzConnector;
	private boolean debug;
	ArrayList<SQLExecutionStatistics> statistics;
	
	public ExecutionContext()
	{
		this( true );
	}

	public ExecutionContext(NetezzaConnector nzCon)
	{
		this( true );
		nzConnector = nzCon;
		statistics = new ArrayList<SQLExecutionStatistics>();
	}
	
	public ExecutionContext( boolean allocateVariableMap )
	{
		nzConnector = null;
		statistics = null;
		if( allocateVariableMap )
			_variables = new LocalVariableMap();
		else
			_variables = null;
	}
	
	public void addStatistic(int instructionId, long runtime, String opString)
	{
		SQLExecutionStatistics s = new SQLExecutionStatistics(opString, instructionId, runtime);
		statistics.add(s);
	}
	
	public void clearStatistics()
	{
		statistics.clear();
	}
	
	public HashMap<Integer, SQLExecutionStatistics> getStatisticsByInstruction() {
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
	
	public ArrayList<SQLExecutionStatistics> getStatistics() {
		return statistics;
	}
	
	public boolean isDebug() {
		return debug;
	}
	public void setDebug(boolean debug) {
		this.debug = debug;
	}
	
	public NetezzaConnector getNzConnector() {
		return nzConnector;
	}


	
	public LocalVariableMap getVariables() {
		return _variables;
	}
	
	public void setVariables(LocalVariableMap vars) {
		_variables = vars;
	}
	
	/* -------------------------------------------------------
	 * Methods to handle variables and associated data objects
	 * -------------------------------------------------------
	 */
	
	public Data getVariable(String name) {
		return _variables.get(name);
	}
	
	public void setVariable(String name, Data val) throws DMLRuntimeException{
		_variables.put(name, val);
	}

	public void removeVariable(String name) {
		_variables.remove(name);
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

	public void setMetaData(String fname, MetaData md) throws DMLRuntimeException {
		_variables.get(fname).setMetaData(md);
	}
	
	public MetaData getMetaData(String varname) throws DMLRuntimeException {
		return _variables.get(varname).getMetaData();
	}
	
	public void removeMetaData(String varname) throws DMLRuntimeException {
		_variables.get(varname).removeMetaData();
	}
	
	public MatrixBlock getMatrixInput(String varName) throws DMLRuntimeException {
		
		try {
			MatrixObject mobj = (MatrixObject) this.getVariable(varName);
			return mobj.acquireRead();
		} catch (CacheException e) {
			throw new DMLRuntimeException( e );
		}
	}
	
	public void releaseMatrixInput(String varName) throws DMLRuntimeException {
		try {
			((MatrixObject)this.getVariable(varName)).release();
		} catch (CacheException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	public ScalarObject getScalarInput(String name, ValueType vt, boolean isLiteral) throws DMLRuntimeException {
		if ( isLiteral ) {
			switch (vt) {
			case INT:
				int intVal = Integer.parseInt(name);
				IntObject intObj = new IntObject(intVal);
				return intObj;
			case DOUBLE:
				double doubleVal = Double.parseDouble(name);
				DoubleObject doubleObj = new DoubleObject(doubleVal);
				return doubleObj;
			case BOOLEAN:
				Boolean boolVal = Boolean.parseBoolean(name);
				BooleanObject boolObj = new BooleanObject(boolVal);
				return boolObj;
			case STRING:
				StringObject stringObj = new StringObject(name);
				return stringObj;
			default:
				throw new DMLRuntimeException("Unknown value type: " + vt + " for variable: " + name);
			}
		}
		else {
			Data obj = getVariable(name);
			if (obj == null) {
				throw new DMLRuntimeException("Unknown variable: " + name);
			}
			return (ScalarObject) obj;
		}
	}

	
	public void setScalarOutput(String varName, ScalarObject so) throws DMLRuntimeException {
		this.setVariable(varName, so);
	}
	
	public void setMatrixOutput(String varName, MatrixBlock outputData) throws DMLRuntimeException {
		MatrixObject sores = (MatrixObject) this.getVariable (varName);
        
		try {
			sores.acquireModify (outputData);
	        
	        // Output matrix is stored in cache and it is written to HDFS, on demand, at a later point in time. 
	        // downgrade() and exportData() need to be called only if we write to HDFS.
	        //sores.downgrade();
	        //sores.exportData();
	        sores.release();
	        
	        this.setVariable (varName, sores);
		
		} catch ( CacheException e ) {
			throw new DMLRuntimeException( e );
		}
	}
	
	/**
	 * Pin a given list of variables i.e., set the "clean up" state in 
	 * corresponding matrix objects, so that the cached data inside these
	 * objects is not cleared and the corresponding HDFS files are not 
	 * deleted (through rmvar instructions). 
	 * 
	 * This is necessary for: function input variables, parfor result variables, 
	 * parfor shared inputs that are passed to functions.
	 * 
	 * The function returns the OLD "clean up" state of matrix objects.
	 */
	public HashMap<String,Boolean> pinVariables(ArrayList<String> varList) 
	{
		HashMap<String, Boolean> varsState = new HashMap<String,Boolean>();
		
		for( String var : varList )
		{
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject )
			{
				//System.out.println("pin "+var);
				MatrixObject mo = (MatrixObject)dat;
				varsState.put( var, mo.isCleanupEnabled() );
				mo.enableCleanup(false); 
			}
		}
		return varsState;
	}
	
	/**
	 * Unpin the a given list of variables by setting their "cleanup" status
	 * to the values specified by <code>varsStats</code>.
	 * 
	 * Typical usage:
	 *    <code> 
	 *    oldStatus = pinVariables(varList);
	 *    ...
	 *    unpinVariables(varList, oldStatus);
	 *    </code>
	 * 
	 * i.e., a call to unpinVariables() is preceded by pinVariables(). 
	 */
	public void unpinVariables(ArrayList<String> varList, HashMap<String,Boolean> varsState)
	{
		for( String var : varList)
		{
			//System.out.println("unpin "+var);
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject )
				((MatrixObject)dat).enableCleanup(varsState.get(var));
		}
	}
	
	/**
	 * NOTE: No order guaranteed, so keep same list for pin and unpin. 
	 * 
	 * @return
	 */
	public ArrayList<String> getVarList()
	{
		ArrayList<String> varlist = new ArrayList<String>();
		varlist.addAll(_variables.keySet());	
		return varlist;
	}
}
