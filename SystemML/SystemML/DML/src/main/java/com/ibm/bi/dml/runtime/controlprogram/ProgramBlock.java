/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ComputationCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.VariableCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLInstructionBase;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.utils.Statistics;


public class ProgramBlock 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected static final Log LOG = LogFactory.getLog(ProgramBlock.class.getName());
	
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
		try {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION 
				&& DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID	
				&& _sb != null 
				&& _sb.requiresRecompilation() )
				//&& Recompiler.requiresRecompilation(_sb.get_hops()) )
			{
				tmp = Recompiler.recompileHopsDag(_sb.get_hops(), ec.getVariables(), _tid);
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
				&& DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID	
				&& requiresRecompile )
				//&& Recompiler.requiresRecompilation(hops)         )
			{
				tmp = Recompiler.recompileHopsDag(hops, ec.getVariables(), _tid);
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
			// TODO: how do we differentiate literals and variables?
			ret = (ScalarObject) ec.getScalarInput(retName, retType, false);
		else {
			throw new DMLRuntimeException("Encountered unimplemented feature!");
			//ret = (ScalarObject) ec.getVariable(retName, retType);
		}
		
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
				LOG.trace("\n Variables: " + ec.getVariables().toString());
				LOG.trace("Instruction: " + currInst.toString());
			}
		
			if (currInst instanceof MRJobInstruction) 
			{
				if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE)
					throw new DMLRuntimeException("MapReduce jobs cannot be executed when execution mode = singlenode");
				
				//execute MR job
				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
				JobReturn jb = RunMRJobs.submitJob(currMRInst, ec);
				
				//specific post processing
				if ( currMRInst.getJobType() == JobType.SORT && jb.getMetaData().length > 0 ) 
				{
					/* Populate returned stats into symbol table of matrices */
					for ( int index=0; index < jb.getMetaData().length; index++) {
						String varname = currMRInst.getOutputVars()[index];
						ec.setMetaData(varname, jb.getMetaData()[index]);
					}
				}
				else if ( jb.getMetaData().length > 0 ) 
				{
					/* Populate returned stats into symbol table of matrices */
					for ( int index=0; index < jb.getMetaData().length; index++) {
						String varname = currMRInst.getOutputVars()[index];
						MatrixCharacteristics mc = ((MatrixDimensionsMetaData)jb.getMetaData(index)).getMatrixCharacteristics();
						ec.getVariable(varname).updateMatrixCharacteristics(mc);
					}
				}
				
				Statistics.incrementNoOfExecutedMRJobs();
				
				if (LOG.isTraceEnabled()){					
					t1 = System.nanoTime();
					LOG.trace("MRJob: " + currMRInst.getJobType() + ", duration = " + (t1-t0)/1000000);
				}
				//System.out.println("MRJob: " + currMRInst.getJobType() );
			} 
			else if (currInst instanceof CPInstruction) 
			{
				long t2 = DMLScript.STATISTICS ? System.nanoTime() : 0;
				
				CPInstruction tmp = (CPInstruction)currInst;
				if( tmp.requiresLabelUpdate() ) //update labels only if required
				{
					//update labels if required
					//note: no exchange of updated instruction as labels might change in the general case
					String updInst = RunMRJobs.updateLabels(currInst.toString(), ec.getVariables());
					tmp = CPInstructionParser.parseSingleInstruction(updInst);
				}

				//execute original or updated instruction
				if(tmp.getCPInstructionType() == CPINSTRUCTION_TYPE.External)
					tmp.processInstruction(this, ec); //FIXME
				else
					tmp.processInstruction(ec);
				
				if (LOG.isTraceEnabled()){	
					t1 = System.nanoTime();
					LOG.trace("CP Instruction: " + currInst.toString() + ", duration = " + (t1-t0)/1000000);
				}

				if( DMLScript.STATISTICS){
					long t3 = System.nanoTime();
					String opcode = InstructionUtils.getOpCode(tmp.toString());
					Statistics.maintainCPHeavyHitters(opcode, t3-t2);
				}
			} 
			else if(currInst instanceof SQLInstructionBase)
			{			
				((SQLInstructionBase)currInst).execute(null); // TODO: must pass SQLExecutionContext here!!!
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
	
	public void printMe() {
		//System.out.println("***** INSTRUCTION BLOCK *****");
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}
	
	/*
	public String explain( int level )
	{
		StringBuilder sb = new StringBuilder();
		
		//explain block
		String offset = "";
		for( int i=0; i<level; i++ )
			offset+="--";	
		sb.append("GENERIC\n");
		
		//explain instructions
		int nextlevel = level+1;
		offset += "--";		
		for( Instruction inst : _inst )
		{
			sb.append(offset);
			if( inst instanceof MRJobInstruction )
				sb.append( ((MRJobInstruction)inst).toString(nextlevel) );
			else 
				sb.append( inst );
			sb.append("\n");
		}
		
		return sb.toString();
	}
	*/
	
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
