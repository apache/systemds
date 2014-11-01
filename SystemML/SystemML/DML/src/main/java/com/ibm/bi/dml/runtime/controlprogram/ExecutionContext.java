/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.debugger.DMLFrame;
import com.ibm.bi.dml.debugger.DMLProgramCounter;
import com.ibm.bi.dml.debugger.DebugState;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
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
	
	//program reference (e.g., function repository)
	private Program _prog;
	
	//symbol table
	private LocalVariableMap _variables;
	
	//SQL-specific (TODO should be separated)
	private NetezzaConnector nzConnector;
	private boolean debug;
	ArrayList<SQLExecutionStatistics> statistics;
	
	//debugging
	private DebugState _dbState = null;
	
	public ExecutionContext()
	{
		this( true, null );
	}
	
	public ExecutionContext(Program prog)
	{
		this( true, prog );
		
	}
	
	public ExecutionContext(LocalVariableMap vars)
	{
		this( false, null);
		_variables = vars;
	}

	public ExecutionContext(NetezzaConnector nzCon)
	{
		this( true, null );
		nzConnector = nzCon;
		statistics = new ArrayList<SQLExecutionStatistics>();
	}
	
	public ExecutionContext( boolean allocateVariableMap, Program prog )
	{
		nzConnector = null;
		statistics = null;
		if( allocateVariableMap )
			_variables = new LocalVariableMap();
		else
			_variables = null;
		_prog = prog;
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


	public Program getProgram(){
		return _prog;
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
				long intVal = Long.parseLong(name);
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
	 * 
	 * @param varName
	 * @param outputData
	 * @param inplace
	 * @throws DMLRuntimeException
	 */
	public void setMatrixOutput(String varName, MatrixBlock outputData, boolean inplace) 
		throws DMLRuntimeException 
	{
		if( inplace ) //modify metadata to prevent output serialization
		{
			MatrixObject sores = (MatrixObject) this.getVariable (varName);
	        sores.enableUpdateInPlace( true );
		}
		
		//default case
		setMatrixOutput(varName, outputData);
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
		//2-pass approach since multiple vars might refer to same matrix object
		HashMap<String, Boolean> varsState = new HashMap<String,Boolean>();
		
		//step 1) get current information
		for( String var : varList )
		{
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject )
			{
				MatrixObject mo = (MatrixObject)dat;
				varsState.put( var, mo.isCleanupEnabled() );
				//System.out.println("pre-pin "+var+" ("+mo.isCleanupEnabled()+")");
			}
		}
		
		//step 2) pin variables
		for( String var : varList )
		{
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject )
			{
				MatrixObject mo = (MatrixObject)dat;
				mo.enableCleanup(false); 
				//System.out.println("pin "+var);
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
			//System.out.println("unpin "+var+" ("+varsState.get(var)+")");
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

	
	///////////////////////////////
	// Debug State Functionality
	///////////////////////////////
	
	/**
	 * 
	 */
	public void initDebugState() 
	{
		if (DMLScript.ENABLE_DEBUG_MODE){
			_dbState = new DebugState();
		}
	}


	public void initDebugProgramCounters() 
	{
		if (DMLScript.ENABLE_DEBUG_MODE){
			_dbState.pc = new DMLProgramCounter(DMLProgram.DEFAULT_NAMESPACE, "main", 0, 0); //initialize program counter (pc)
			_dbState.prevPC = new DMLProgramCounter(DMLProgram.DEFAULT_NAMESPACE, "main", 0, 0); //initialize previous pc
		}
	}
	
	/**
	 * 
	 * @param index
	 */
	public void updateDebugState( int index ) 
	{
		if(DMLScript.ENABLE_DEBUG_MODE) {
			_dbState.getPC().setProgramBlockNumber(index);
		}
	}
	
	/**
	 * 
	 * @param currInst
	 */
	public void updateDebugState( Instruction currInst )
	{
		if (DMLScript.ENABLE_DEBUG_MODE) {
			// New change so that shell doesnot seem like it is hanging while running MR job
			// Since UI cannot accept instructions when user is submitting the program
			_dbState.nextCommand = false;
			// Change to stop before first instruction of a given line
			//update current instruction ID and line number 
			_dbState.getPC().setInstID(currInst.getInstID()); 				
			_dbState.getPC().setLineNumber(currInst.getLineNum());
			// Change to stop before first instruction of a given line
			suspendIfAskedInDebugMode(currInst);	
		}
	}
	
	/**
	 * 
	 */
	public void clearDebugProgramCounters()
	{
		if(DMLScript.ENABLE_DEBUG_MODE) {
			_dbState.pc = null;
		}
	}
	
	public void handleDebugException( Exception ex )
	{
		_dbState.getDMLStackTrace(ex);
		_dbState.suspend = true;
	}

	public void handleDebugFunctionEntry( FunctionCallCPInstruction funCallInst )
	{
		//push caller frame into call stack
		_dbState.pushFrame(getVariables(), _dbState.getPC());
		//initialize pc for callee's frame
		_dbState.pc = new DMLProgramCounter(funCallInst.getNamespace(), funCallInst.getFunctionName(), 0, 0);
	}
	
	public void handleDebugFunctionExit( FunctionCallCPInstruction funCallInst )
	{
		//pop caller frame from call stack
		DMLFrame fr = _dbState.popFrame();
		//update pc to caller's frame
		_dbState.pc = fr.getPC();
	}
	
	public DebugState getDebugState() {
		return _dbState;
	}
	
	/**
	 * This function should be called only if user has specified -debug option.
	 * In this function, if the user has issued one of the step instructions or
	 * has enabled suspend flag in previous instruction (through breakpoint),
	 * then it will wait until user issues a new debugger command.
	 * @param currInst
	 * @param ec
	 */
	@SuppressWarnings("deprecation")
	private void suspendIfAskedInDebugMode(Instruction currInst ) {
		if (!DMLScript.ENABLE_DEBUG_MODE) {
			System.err.println("ERROR: The function suspendIfAskedInDebugMode should not be called in non-debug mode.");
		}
		//check for stepping options
		if (!_dbState.suspend && _dbState.dbCommand != null) { 
			if (_dbState.dbCommand.equalsIgnoreCase("step_instruction")) {
				System.out.format("Step instruction reached at %s.\n", _dbState.getPC().toString());
				_dbState.suspend = true;
			}
			else if (_dbState.dbCommand.equalsIgnoreCase("step_line") && _dbState.prevPC.getLineNumber() != currInst.getLineNum()
					&& _dbState.prevPC.getLineNumber() != 0) {
				// Don't step into first instruction of first line
				// System.out.format("Step reached at %s.\n", this._prog.getPC().toString());
				System.out.format("Step reached at %s.\n", _dbState.getPC().toStringWithoutInstructionID());
				_dbState.suspend = true;
			}
			else if (_dbState.dbCommand.equalsIgnoreCase("step return") && currInst instanceof FunctionCallCPInstruction) {
				FunctionCallCPInstruction funCallInst = (FunctionCallCPInstruction) currInst;
				if (_dbState.dbCommandArg == null || funCallInst.getFunctionName().equalsIgnoreCase(_dbState.dbCommandArg)) {
					System.out.format("Step return reached at %s.\n", _dbState.getPC().toStringWithoutInstructionID());
					_dbState.suspend = true;
				}
			}
		}
		//check if runtime suspend signal is set
		if (_dbState.suspend) {
			//flush old commands and arguments
			_dbState.dbCommand = null;
			_dbState.dbCommandArg = null;
			//print current DML script source line
			if (currInst.getLineNum() != 0)
				_dbState.printDMLSourceLine(currInst.getLineNum());
			//save current symbol table
			_dbState.setVariables(this.getVariables());
			//send next command signal to debugger control module
			_dbState.nextCommand = true;
			//suspend runtime execution thread
			Thread.currentThread().suspend();
			//reset next command signal
			_dbState.nextCommand = false;
		}
		//reset runtime suspend signal
		_dbState.suspend = false;
		//update previous pc
		_dbState.prevPC.setFunctionName(_dbState.getPC().getFunctionName());
		_dbState.prevPC.setProgramBlockNumber(_dbState.getPC().getProgramBlockNumber());
		_dbState.prevPC.setLineNumber(currInst.getLineNum());
	}
}
