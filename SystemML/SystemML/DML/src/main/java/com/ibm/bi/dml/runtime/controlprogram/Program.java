/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Stack;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.debugger.DMLFrame;
import com.ibm.bi.dml.debugger.DMLProgramCounter;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;


public class Program 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String KEY_DELIM = "::";
	
	public ArrayList<ProgramBlock> _programBlocks;

	protected LocalVariableMap _programVariables;
	private HashMap<String, HashMap<String,FunctionProgramBlock>> _namespaceFunctions;
	
	//DML debugger variables
	protected String [] dmlScript;
	protected Stack<DMLFrame> callStack = new Stack<DMLFrame>();
	protected DMLProgramCounter pc = null, prevPC = null;
	protected LocalVariableMap frameVariables=null;
	protected String dbCommand=null;
	protected String dbCommandArg=null;
	protected boolean suspend = false;
	protected volatile boolean nextCommand = false;
	

	public Program() throws DMLRuntimeException {
		_namespaceFunctions = new HashMap<String, HashMap<String,FunctionProgramBlock>>(); 
		_programBlocks = new ArrayList<ProgramBlock>();
		_programVariables = new LocalVariableMap ();
	}

	public void addFunctionProgramBlock(String namespace, String fname, FunctionProgramBlock fpb){
		
		if (namespace == null) 
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		
		HashMap<String,FunctionProgramBlock> namespaceBlocks = null;
		
		synchronized( _namespaceFunctions )
		{
			namespaceBlocks = _namespaceFunctions.get(namespace);
			if (namespaceBlocks == null){
				namespaceBlocks = new HashMap<String,FunctionProgramBlock>();
				_namespaceFunctions.put(namespace,namespaceBlocks);
			}
		}
		
		namespaceBlocks.put(fname,fpb);
	}
	
	public HashMap<String,FunctionProgramBlock> getFunctionProgramBlocks(){
		
		HashMap<String,FunctionProgramBlock> retVal = new HashMap<String,FunctionProgramBlock>();
		
		synchronized( _namespaceFunctions )
		{
			for (String namespace : _namespaceFunctions.keySet()){
				HashMap<String,FunctionProgramBlock> namespaceFSB = _namespaceFunctions.get(namespace);
				for (String fname : namespaceFSB.keySet()){
					String fKey = DMLProgram.constructFunctionKey(namespace, fname);
					retVal.put(fKey, namespaceFSB.get(fname));
				}
			}
		}
		
		return retVal;
	}
	
	public FunctionProgramBlock getFunctionProgramBlock(String namespace, String fname) throws DMLRuntimeException{
		
		if (namespace == null) namespace = DMLProgram.DEFAULT_NAMESPACE;
		
		HashMap<String,FunctionProgramBlock> namespaceFunctBlocks = _namespaceFunctions.get(namespace);
		if (namespaceFunctBlocks == null)
			throw new DMLRuntimeException("namespace " + namespace + " is undefined");
		FunctionProgramBlock retVal = namespaceFunctBlocks.get(fname);
		if (retVal == null)
			throw new DMLRuntimeException("function " + fname + " is undefined in namespace " + namespace);
		//retVal._variables = new LocalVariableMap();
		return retVal;
	}
	
	public void addProgramBlock(ProgramBlock pb) {
		_programBlocks.add(pb);
	}

	public ArrayList<ProgramBlock> getProgramBlocks() {
		return _programBlocks;
	}

	public void execute(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		if (DMLScript.ENABLE_DEBUG_MODE) {
			pc = new DMLProgramCounter(DMLProgram.DEFAULT_NAMESPACE, "main", 0, 0); //initialize program counter (pc)
			prevPC = new DMLProgramCounter(DMLProgram.DEFAULT_NAMESPACE, "main", 0, 0); //initialize previous pc
		}
		try
		{
			for (int i=0 ; i<this._programBlocks.size() ; i++) {
				if (DMLScript.ENABLE_DEBUG_MODE) {
					this.pc.setProgramBlockNumber(i); //update pc's current block 
				}
				this._programBlocks.get(i).execute(ec);
			}
			if (DMLScript.ENABLE_DEBUG_MODE)
				pc=null;
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}
		
	/*public void cleanupCachedVariables() throws CacheStatusException
	{
		for( String var : _programVariables.keySet() )
		{
			Data dat = _programVariables.get(var);
			if( dat instanceof MatrixObjectNew )
				((MatrixObjectNew)dat).clearData();
		}
	}*/
	
	public void printMe() {
		
		/*for (String key : _functionProgramBlocks.keySet()) {
			System.out.println("function " + key);
			_functionProgramBlocks.get(key).printMe();
		}*/
		
		for (ProgramBlock pb : this._programBlocks) {
			pb.printMe();
		}
	}
	
	/////////////////////////////////
	// DML debugger public methods //
	/////////////////////////////////	
	
	/**
	 * Getter for current frame's program counter
	 * @return Current frame program counter
	 */
	public DMLProgramCounter getPC() {
		if(!DMLScript.ENABLE_DEBUG_MODE) {
			System.err.println("Error: This functionality (getPC) is available only in debug mode");
			System.exit(-1); // Fatal error to avoid unintentional bugs
		}
		return pc;
	}

	/**
	 * Getter for current frame's local variables
	 * @return Current frame local variables
	 */
	public LocalVariableMap getVariables() {
		return frameVariables;
	}
	
	/**
	 * Getter for current frame 
	 * @return Current frame
	 */
	public DMLFrame getCurrentFrame() {
		DMLFrame frame = new DMLFrame(frameVariables, pc);
		return frame;
	}
	
	/**
	 * Setter for current frame's local variables
	 * @param Current frame local variables
	 */
	public void setVariables(LocalVariableMap vars) {
		frameVariables = vars;
	}
	
	/**
	 * Is runtime ready to accept next command
	 * @return  true if the user interface can accept next command
	 */
	public boolean canAcceptNextCommand() {
		return nextCommand;
	}
	
	/**
	 * Set DML script field
	 * @param lines DML script lines of code 
	 */
	public void setDMLScript(String [] lines) {
		dmlScript = lines;
	}
	
	/**
	 * Print DML script source line
	 * @param lineNum DML script line number 
	 */
	public void printDMLSourceLine(int lineNum) {
		if (lineNum > 0 && lineNum < dmlScript.length)
			System.out.format("%-4d %s\n",lineNum, dmlScript[lineNum-1]);
	}
	
	/**
	 * Set debugger command into current runtime 
	 * @param command Debugger command
	 */
	public void setCommand(String command) {
		this.dbCommand = command;
	}

	/**
	 * Set debugger command argument into current runtime 
	 * @param cmdArg Debugger command argument
	 */
	public void setCommandArg(String cmdArg) {
		this.dbCommandArg = cmdArg;
	}
	
	/**
	 * Put current frame into stack due to function call
	 * @param vars Caller's frame symbol table
	 * @param pc Caller's frame program counter
	 */
	protected void pushFrame(LocalVariableMap vars, DMLProgramCounter pc) {		
		callStack.push(new DMLFrame(vars, pc));
	}
	
	/**
	 * Pop frame from stack as function call is done 
	 * @return Callee's frame (before function call) 
	 */
	protected DMLFrame popFrame() {
		if (callStack.isEmpty())
			return null;
		return callStack.pop();		
	}
	
	/**
	 * Get stack frame at indicated location (if any)
	 * @param location Frame position in call stack
	 * @return Stack frame at specified location
	 */
	protected DMLFrame getFrame(int location) {
		if (location < 0 || location >= callStack.size()) {
			return null;
		}
		return callStack.elementAt(location);
	}
	
	/**
	 * Get current call stack (if any) 
	 * @return Stack callStack 
	 */
	public Stack<DMLFrame> getCallStack() {
		if (callStack.isEmpty())
			return null;
		return callStack;
	}

	/**
	 * Display a full DML stack trace for a runtime exception 
	 */
	public void getDMLStackTrace(Exception e) {		
		System.err.format("Runtime exception raised %s\n", e.toString());
		System.err.println("\t at " + this.pc.toString());		
		if (this.callStack != null) {
			for (int i = callStack.size(); i > 0; i--) {			
				DMLFrame frame = callStack.get(i-1);
				System.err.println("\t at " + frame.getPC().toString());
			}
		}
	}
}
