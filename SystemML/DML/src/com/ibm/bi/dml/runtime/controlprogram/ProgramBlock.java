package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ComputationCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.VariableCPInstruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLInstructionBase;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.CacheException;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.Statistics;


public class ProgramBlock 
{	
	protected static final Log LOG = LogFactory.getLog(ProgramBlock.class.getName());
	
	protected Program _prog;		// pointer to Program this ProgramBlock is part of
	protected ArrayList<Instruction> _inst;
	protected LocalVariableMap _variables;
	
	//additional attributes for recompile
	protected StatementBlock _sb = null;
	protected long _tid = 0; //by default _t0
	
	
	public ProgramBlock(Program prog) 
		throws DMLRuntimeException 
	{	
		_prog = prog;
		_variables = new LocalVariableMap ();
		_inst = new ArrayList<Instruction>();
	}
    
	
	////////////////////////////////////////////////
	// getters, setters and similar functionality
	////////////////////////////////////////////////

	public Program getProgram(){
		return _prog;
	}
	
	public void setProgram(Program prog){
		_prog = prog;
	}
	
	public StatementBlock getStatementBlock(){
		return _sb;
	}
	
	public void setStatementBlock( StatementBlock sb ){
		_sb = sb;
	}

	public LocalVariableMap getVariables() {
		return _variables;
	}
	
	public Data getVariable(String name) {
		return _variables.get(name);
	}
	
	public void setVariables (LocalVariableMap vars){
		_variables.putAll (vars);
	}
	
	public void addVariables (LocalVariableMap vars) {
		_variables.putAll (vars);
	}

	public void setVariable(String name, Data val) throws DMLRuntimeException{
		_variables.put(name, val);
	}

	public void removeVariable(String name) {
		_variables.remove(name);
	}
	
	public  ArrayList<Instruction> getInstructions() {
		return _inst;
	}

	public Instruction getInstruction(int i) {
		return _inst.get(i);
	}
	
	public  void setInstructions( ArrayList<Instruction> inst ) {
		_inst = inst;
	}
	
	public void addInstruction(Instruction inst) {
		_inst.add(inst);
	}
	
	public int getNumInstructions() {
		return _inst.size();
	}
	
	public void setThreadID( long id ){
		_tid = id;
	}


	//////////////////////////////////////////////////////////
	// core instruction execution (program block, predicate)
	//////////////////////////////////////////////////////////

	/**
	 * Executes this program block (incl recompilation if required).
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public void execute(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		ArrayList<Instruction> tmp = _inst;
		
		//dynamically recompile instructions if enabled and required
		try {
			if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION 
				&& DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID	
				&& _sb != null 
				&& Recompiler.requiresRecompilation(_sb.get_hops()) 
				/*&& !Recompiler.containsNonRecompileInstructions(tmp)*/ )
			{
				tmp = Recompiler.recompileHopsDag(_sb.get_hops(), _variables, _tid);
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to recompile program block.", ex);
		}
		
		//actual instruction execution
		executeInstructions(tmp, ec);
	}
	
	/**
	 * Executes given predicate instructions (incl recompilation if required)
	 * 
	 * @param inst
	 * @param hops
	 * @param ec
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public ScalarObject executePredicate(ArrayList<Instruction> inst, Hops hops, ValueType retType, ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ArrayList<Instruction> tmp = inst;
	
		//System.out.println(_variables.toString());
		
		//dynamically recompile instructions if enabled and required
		try {
			if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION 
				&& DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID	
				&& Recompiler.requiresRecompilation(hops) 
				/*&& !Recompiler.containsNonRecompileInstructions(inst)*/ )
			{
				tmp = Recompiler.recompileHopsDag(hops, _variables, _tid);
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to recompile predicate instructions.", ex);
		}
		
		//actual instruction execution
		return executePredicateInstructions(tmp, retType, ec);
	}

	/**
	 * 
	 * @param inst
	 * @param ec
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected void executeInstructions(ArrayList<Instruction> inst, ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		for (int i = 0; i < inst.size(); i++) 
		{
			//indexed access required due to dynamic add
			Instruction currInst = inst.get(i);
			
			//execute instruction
			executeSingleInstruction(currInst, ec);
		}
	}
	
	/**
	 * 
	 * @param inst
	 * @param ec
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected ScalarObject executePredicateInstructions(ArrayList<Instruction> inst, ValueType retType, ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		ScalarObject ret = null;
		String retName = null;
 		boolean isSQL = false;

 		//execute all instructions
		for (Instruction currInst : inst ) 
		{
			if( !isRemoveVariableInstruction(currInst) )
			{
				//execute instruction
				executeSingleInstruction(currInst, ec);
				
				//get last return name
				if(currInst instanceof ComputationCPInstruction )
					retName = ((ComputationCPInstruction) currInst).getOutputVariableName();  
				else if(currInst instanceof VariableCPInstruction && ((VariableCPInstruction)currInst).getOutputVariableName()!=null)
					retName = ((VariableCPInstruction)currInst).getOutputVariableName();
				else if(currInst instanceof SQLScalarAssignInstruction){
					retName = ((SQLScalarAssignInstruction) currInst).getVariableName();
					isSQL = true;
				}
			}
		}
		
		//get return value
		if(!isSQL)
			ret = (ScalarObject) getScalarInput(retName, retType);
		else 
			ret = (ScalarObject) ec.getVariable(retName, retType);
		
		//execute rmvar instructions
		for (Instruction currInst : inst ) 
			if( isRemoveVariableInstruction(currInst) )
				executeSingleInstruction(currInst, ec);
		
		//check and correct scalar ret type (incl save double to int)
		if( ret.getValueType() != retType )
			switch( retType ) {
				case BOOLEAN: ret = new BooleanObject(ret.getName(),ret.getBooleanValue()); break;
				case INT:	  ret = new IntObject(ret.getName(),ret.getIntValue()); break;
				case DOUBLE:  ret = new DoubleObject(ret.getName(),ret.getDoubleValue()); break;
				case STRING:  ret = new StringObject(ret.getName(),ret.getStringValue()); break;
			}
			
		return ret;
	}
	
	/**
	 * 
	 * 
	 * @param currInst
	 * @throws DMLRuntimeException 
	 */
	private void executeSingleInstruction( Instruction currInst, ExecutionContext ec ) 
		throws DMLRuntimeException
	{
		try 
		{
			long t0 = 0, t1 = 0;
			
			if( LOG.isTraceEnabled() )
			{
				t0 = System.nanoTime();
				LOG.trace("\n Variables: " + _variables.toString());
				LOG.trace("Instruction: " + currInst.toString());
			}
		
			if (currInst instanceof MRJobInstruction) 
			{
				if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE)
					throw new DMLRuntimeException("MapReduce jobs cannot be executed when execution mode = singlenode");
				
				//execute MR job
				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
				JobReturn jb = RunMRJobs.submitJob(currMRInst, this);
				
				//specific post processing
				if ( currMRInst.getJobType() == JobType.SORT && jb.getMetaData().length > 0 ) 
				{
					/* Populate returned stats into symbol table of matrices */
					for ( int index=0; index < jb.getMetaData().length; index++) {
						String varname = currMRInst.getOutputVars()[index];
						_variables.get(varname).setMetaData(jb.getMetaData()[index]); // updateMatrixCharacteristics(mc);
					}
				}
				else if ( jb.getMetaData().length > 0 ) 
				{
					/* Populate returned stats into symbol table of matrices */
					for ( int index=0; index < jb.getMetaData().length; index++) {
						String varname = currMRInst.getOutputVars()[index];
						MatrixCharacteristics mc = ((MatrixDimensionsMetaData)jb.getMetaData(index)).getMatrixCharacteristics();
						_variables.get(varname).updateMatrixCharacteristics(mc);
					}
				}
				
				Statistics.incrementNoOfExecutedMRJobs();
				
				if (LOG.isTraceEnabled()){					
					t1 = System.nanoTime();
					LOG.trace("MRJob: " + currMRInst.getJobType() + ", duration = " + (t1-t0)/1000000);
				}
				//System.out.println("MRJob: " + currMRInst.getJobType() );
				//System.out.println(currMRInst.toString());
			} 
			else if (currInst instanceof CPInstruction) 
			{
				CPInstruction tmp = (CPInstruction)currInst;
				if( tmp.requiresLabelUpdate() ) //update labels only if required
				{
					//update labels if required
					//note: no exchange of updated instruction as labels might change in the general case
					String updInst = RunMRJobs.updateLabels(currInst.toString(), _variables);
					tmp = CPInstructionParser.parseSingleInstruction(updInst);
				}
				
				//execute original or updated instruction
				tmp.processInstruction(this); 
					
				if (LOG.isTraceEnabled()){	
					t1 = System.nanoTime();
					LOG.trace("CP Instruction: " + currInst.toString() + ", duration = " + (t1-t0)/1000000);
				}
			} 
			else if(currInst instanceof SQLInstructionBase)
			{			
				((SQLInstructionBase)currInst).execute(ec);
			}	
		}
		catch (Exception e)
		{
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating instruction: " + currInst.toString() , e);
		}
	}
	
	private boolean isRemoveVariableInstruction(Instruction inst)
	{
		return ( inst instanceof VariableCPInstruction && ((VariableCPInstruction)inst).isRemoveVariable() );
	}
	
	
	//////////////////////////////////
	// caching-related functionality 
	//////////////////////////////////
	
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
			throw new DMLRuntimeException(this.printBlockErrorLocation() , e);
		}
	}
	
	public void releaseMatrixInput(String varName) throws DMLRuntimeException {
		try {
			((MatrixObject)this.getVariable(varName)).release();
		} catch (CacheException e) {
			throw new DMLRuntimeException(this.printBlockErrorLocation() , e);
		}
	}
	
	public ScalarObject getScalarInput(String name, ValueType vt) {
		Data obj = getVariable(name);
		if (obj == null) {
			try {
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
					throw new DMLRuntimeException(this.printBlockErrorLocation() + "Unknown variable: " + name + ", or unknown value type: " + vt);
				}
			} 
			catch (Exception e) 
			{	
				e.printStackTrace();
			}
		}
		
		
		return (ScalarObject) obj;
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
			throw new DMLRuntimeException(this.printBlockErrorLocation() , e);
		}
	}
		
	/**
	 * Pin a given list of variables i.e., set the "clean up" state in 
	 * corresponding matrix objects, so that the cached data inside these
	 * objects is not cleared and the corresponding HDFS files are not 
	 * deleted (through rmvar instructions). 
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
				//System.out.println("pin ("+_ID+") "+var);
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
			//System.out.println("unpin ("+_ID+") "+var);
			
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject )
				((MatrixObject)dat).enableCleanup(varsState.get(var));
		}
	}

	
	public void printMe() {
		//System.out.println("***** INSTRUCTION BLOCK *****");
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}
	
	
	///////////////////////////////////////////////////////////////////////////
	// store position information for program blocks
	///////////////////////////////////////////////////////////////////////////
	
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	
	public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	
	public String printErrorLocation(){
		return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in program block generated from statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
}
