/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.recompile.Recompiler;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLScriptException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.cp.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.cp.ComputationCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.cp.IntObject;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.cp.StringObject;
import com.ibm.bi.dml.runtime.instructions.cp.VariableCPInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.utils.Statistics;
import com.ibm.bi.dml.yarn.DMLAppMasterUtils;


public class ProgramBlock 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected static final Log LOG = LogFactory.getLog(ProgramBlock.class.getName());
	private static final boolean CHECK_MATRIX_SPARSITY = false;
	
	protected Program _prog;		// pointer to Program this ProgramBlock is part of
	protected ArrayList<Instruction> _inst;
	
	//additional attributes for recompile
	protected StatementBlock _sb = null;
	protected long _tid = 0; //by default _t0
	
	
	public ProgramBlock(Program prog) 
		throws DMLRuntimeException 
	{	
		_prog = prog;
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
	
	public void addInstructions(ArrayList<Instruction> inst) {
		_inst.addAll(inst);
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
		try 
		{
			if( DMLScript.isActiveAM() ) //set program block specific remote memory
				DMLAppMasterUtils.setupProgramBlockRemoteMaxMemory(this);
			
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION 
				&& OptimizerUtils.isHybridExecutionMode()	
				&& _sb != null 
				&& _sb.requiresRecompilation() )
				//&& Recompiler.requiresRecompilation(_sb.get_hops()) )
			{
				tmp = Recompiler.recompileHopsDag(_sb, _sb.get_hops(), ec.getVariables(), null, false, _tid);
			}
			if( DMLScript.STATISTICS ){
				long t1 = System.nanoTime();
				Statistics.incrementHOPRecompileTime(t1-t0);
				if( tmp!=_inst )
					Statistics.incrementHOPRecompileSB();
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
	public ScalarObject executePredicate(ArrayList<Instruction> inst, Hop hops, boolean requiresRecompile, ValueType retType, ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		ArrayList<Instruction> tmp = inst;
		
		//dynamically recompile instructions if enabled and required
		try {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION 
				&& OptimizerUtils.isHybridExecutionMode()	
				&& requiresRecompile )
				//&& Recompiler.requiresRecompilation(hops)         )
			{
				tmp = Recompiler.recompileHopsDag(hops, ec.getVariables(), false, _tid);
			}
			if( DMLScript.STATISTICS ){
				long t1 = System.nanoTime();
				Statistics.incrementHOPRecompileTime(t1-t0);
				if( tmp!=inst )
					Statistics.incrementHOPRecompilePred();
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
			ec.updateDebugState(i);
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

 		//execute all instructions
 		for (int i = 0; i < inst.size(); i++)
		{
			//indexed access required due to debug mode
			Instruction currInst = inst.get(i);			
			if( !isRemoveVariableInstruction(currInst) )
			{
				//execute instruction
				ec.updateDebugState(i);
				executeSingleInstruction(currInst, ec);
				
				//get last return name
				if(currInst instanceof ComputationCPInstruction )
					retName = ((ComputationCPInstruction) currInst).getOutputVariableName();  
				else if(currInst instanceof VariableCPInstruction && ((VariableCPInstruction)currInst).getOutputVariableName()!=null)
					retName = ((VariableCPInstruction)currInst).getOutputVariableName();
			}
		}
		
		//get return value TODO: how do we differentiate literals and variables?
		ret = (ScalarObject) ec.getScalarInput(retName, retType, false);
		
		//execute rmvar instructions
		for (int i = 0; i < inst.size(); i++) {
			//indexed access required due to debug mode
			Instruction currInst = inst.get(i);
			if( isRemoveVariableInstruction(currInst) ) {
				ec.updateDebugState(i);
				executeSingleInstruction(currInst, ec);
			}
		}
		
		//check and correct scalar ret type (incl save double to int)
		if( ret.getValueType() != retType )
			switch( retType ) {
				case BOOLEAN: ret = new BooleanObject(ret.getName(),ret.getBooleanValue()); break;
				case INT:	  ret = new IntObject(ret.getName(),ret.getLongValue()); break;
				case DOUBLE:  ret = new DoubleObject(ret.getName(),ret.getDoubleValue()); break;
				case STRING:  ret = new StringObject(ret.getName(),ret.getStringValue()); break;
				default:
					//do nothing
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
			// start time measurement for statistics
			long t0 = (DMLScript.STATISTICS || LOG.isTraceEnabled()) ? 
					System.nanoTime() : 0;
			
					
			// pre-process instruction (debug state, inst patching, listeners)
			Instruction tmp = currInst.preprocessInstruction( ec );
			
			// process actual instruction
			tmp.processInstruction( ec );
			
			// post-process instruction (debug) 
			tmp.postprocessInstruction( ec );
			
			
			// maintain aggregate statistics
			if( DMLScript.STATISTICS) {
				long t1 = System.nanoTime();
				Statistics.maintainCPHeavyHitters(tmp.getOpcode(), t1-t0);
			}
				
			// optional trace information (instruction and runtime)
			if( LOG.isTraceEnabled() ) {
				long t1 = System.nanoTime();
				String time = String.format("%.3f",((double)t1-t0)/1000000000);
				LOG.trace("Instruction: "+ tmp + " (executed in " + time + "s).");
			}
			
			// optional check for correct nnz and sparse/dense representation of all 
			// variables in symbol table (for tracking source of wrong representation)
			if( CHECK_MATRIX_SPARSITY ) {
				checkSparsity( tmp, ec.getVariables() );
			}			
		}
		catch (Exception e)
		{
			if (!DMLScript.ENABLE_DEBUG_MODE) {
				if ( e instanceof DMLScriptException) 
					throw (DMLScriptException)e;
				else
					throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating instruction: " + currInst.toString() , e);
			}
			else {
				ec.handleDebugException(e);
			}
		}
	}
	
	/**
	 * 
	 * @param inst
	 * @return
	 */
	private boolean isRemoveVariableInstruction(Instruction inst)
	{
		return ( inst instanceof VariableCPInstruction && ((VariableCPInstruction)inst).isRemoveVariable() );
	}
	
	public void printMe() {
		//System.out.println("***** INSTRUCTION BLOCK *****");
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}
	
	/**
	 * 
	 * @param lastInst
	 * @param vars
	 * @throws DMLRuntimeException
	 */
	private void checkSparsity( Instruction lastInst, LocalVariableMap vars )
		throws DMLRuntimeException
	{
		for( String varname : vars.keySet() )
		{
			Data dat = vars.get(varname);
			if( dat instanceof MatrixObject )
			{
				MatrixObject mo = (MatrixObject)dat;
				if( mo.isDirty() && !mo.isPartitioned() )
				{
					MatrixBlock mb = mo.acquireRead();	
					boolean sparse1 = mb.isInSparseFormat();
					long nnz1 = mb.getNonZeros();
					synchronized( mb ) { //potential state change
						mb.recomputeNonZeros();
						mb.examSparsity();
					}
					boolean sparse2 = mb.isInSparseFormat();
					long nnz2 = mb.getNonZeros();
					mo.release();
					
					if( nnz1 != nnz2 )
						throw new DMLRuntimeException("Matrix nnz meta data was incorrect: ("+varname+", actual="+nnz1+", expected="+nnz2+", inst="+lastInst+")");
							
					
					if( sparse1 != sparse2 )
						throw new DMLRuntimeException("Matrix was in wrong data representation: ("+varname+", actual="+sparse1+", expected="+sparse2+", inst="+lastInst+")");
				}
			}
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
