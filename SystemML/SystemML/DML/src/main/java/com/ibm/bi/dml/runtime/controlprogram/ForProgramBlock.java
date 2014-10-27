/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLScriptException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.yarn.DMLAppMasterUtils;

public class ForProgramBlock extends ProgramBlock
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected ArrayList<Instruction> 	_fromInstructions;
	protected ArrayList<Instruction> 	_toInstructions;
	protected ArrayList<Instruction> 	_incrementInstructions;
	
	protected ArrayList <Instruction> 	_exitInstructions ;
	protected ArrayList<ProgramBlock> 	_childBlocks;

	protected String[]                  _iterablePredicateVars; //from,to,where constants/internal vars not captured via instructions
	
	public void printMe() 
	{		
		LOG.debug("***** current for block predicate inst: *****");
		LOG.debug("FROM:");
		for (Instruction cp : _fromInstructions){
			cp.printMe();
		}
		LOG.debug("TO:");
		for (Instruction cp : _toInstructions){
			cp.printMe();
		}
		LOG.debug("INCREMENT:");
		for (Instruction cp : _incrementInstructions){
			cp.printMe();
		}
		
		LOG.debug("***** children block inst: *****");
		for (ProgramBlock pb : this._childBlocks){
			pb.printMe();
		}
		
		LOG.debug("***** current block inst exit: *****");
		for (Instruction i : this._exitInstructions) {
			i.printMe();
		}
		
	}

	
	public ForProgramBlock(Program prog, String[] iterPredVars) throws DMLRuntimeException
	{
		super(prog);
		
		_exitInstructions = new ArrayList<Instruction>();
		_childBlocks = new ArrayList<ProgramBlock>();
		_iterablePredicateVars = iterPredVars;
	}
	
	public ArrayList<Instruction> getFromInstructions()
	{
		return _fromInstructions;
	}
	
	public void setFromInstructions(ArrayList<Instruction> instructions)
	{
		_fromInstructions = instructions;
	}
	
	public ArrayList<Instruction> getToInstructions()
	{
		return _toInstructions;
	}
	
	public void setToInstructions(ArrayList<Instruction> instructions)
	{
		_toInstructions = instructions;
	}
	
	public ArrayList<Instruction> getIncrementInstructions()
	{
		return _incrementInstructions;
	}
	
	public void setIncrementInstructions(ArrayList<Instruction> instructions)
	{
		_incrementInstructions = instructions;
	}
	
	public void addExitInstruction(Instruction inst){
		_exitInstructions.add(inst);
	}
	
	public ArrayList<Instruction> getExitInstructions(){
		return _exitInstructions;
	}
	
	public void setExitInstructions(ArrayList<Instruction> inst){
		_exitInstructions = inst;
	}
	

	public void addProgramBlock(ProgramBlock childBlock) {
		_childBlocks.add(childBlock);
	}
	
	public ArrayList<ProgramBlock> getChildBlocks() 
	{
		return _childBlocks;
	}
	
	public void setChildBlocks(ArrayList<ProgramBlock> pbs) 
	{
		_childBlocks = pbs;
	}
	
	public String[] getIterablePredicateVars()
	{
		return _iterablePredicateVars;
	}
	
	public void setIterablePredicateVars(String[] iterPredVars)
	{
		_iterablePredicateVars = iterPredVars;
	}
	
	@Override	
	public void execute(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, DMLScriptException
	{
		// add the iterable predicate variable to the variable set
		String iterVarName = _iterablePredicateVars[0];

		// evaluate from, to, incr only once (assumption: known at for entry)
		IntObject from = executePredicateInstructions( 1, _fromInstructions, ec );
		IntObject to   = executePredicateInstructions( 2, _toInstructions, ec );
		IntObject incr = executePredicateInstructions( 3, _incrementInstructions, ec );
		
		if ( incr.getLongValue() <= 0 ) //would produce infinite loop
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Expression for increment of variable '" + iterVarName + "' must evaluate to a positive value.");
				
		// initialize iter var to from value
		IntObject iterVar = new IntObject(iterVarName, from.getLongValue() );
		
		// execute for loop
		try 
		{
			// run for loop body as long as predicate is true 
			// (for supporting dynamic TO, move expression execution to end of while loop)
			while( iterVar.getLongValue() <= to.getLongValue() )
			{
				ec.setVariable(iterVarName, iterVar); 
				
				//for all child blocks
				for (int i=0 ; i < this._childBlocks.size() ; i++) {
					ec.updateDebugState( i );
					_childBlocks.get(i).execute(ec);
				}				
			
				// update the iterable predicate variable 
				if(ec.getVariable(iterVarName) == null || !(ec.getVariable(iterVarName) instanceof IntObject))
					throw new DMLRuntimeException("Iterable predicate variable " + iterVarName + " must remain of type scalar int.");
				
				//increment of iterVar (changes  in loop body get discarded)
				iterVar = new IntObject( iterVarName, iterVar.getLongValue()+incr.getLongValue() );
			}
		}
		catch (Exception e)
		{
			throw new DMLRuntimeException(printBlockErrorLocation() + "Error evaluating for program block", e);
		}
		
		//execute exit instructions
		try {
			executeInstructions(_exitInstructions, ec);	
		}
		catch (Exception e){
			throw new DMLRuntimeException(printBlockErrorLocation() + "Error evaluating for exit instructions", e);
		}
	}

	/**
	 * 
	 * @param pos
	 * @param instructions
	 * @param ec
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected IntObject executePredicateInstructions( int pos, ArrayList<Instruction> instructions, ExecutionContext ec ) 
		throws DMLRuntimeException
	{
		ScalarObject tmp = null;
		IntObject ret = null;
		
		try
		{
			if( _iterablePredicateVars[pos] != null )
			{
				//probe for scalar variables
				Data ldat = ec.getVariable( _iterablePredicateVars[pos] );
				if( ldat != null && ldat instanceof ScalarObject )
					tmp = (ScalarObject)ldat;
				else //handle literals
					tmp = new IntObject( UtilFunctions.parseToLong(_iterablePredicateVars[pos]) );
			}		
			else
			{
				if( _sb!=null )
				{
					ForStatementBlock fsb = (ForStatementBlock)_sb;
					Hop predHops = null;
					boolean recompile = false;
					if (pos == 1){ 
						predHops = fsb.getFromHops();
						recompile = fsb.requiresFromRecompilation();
					}
					else if (pos == 2) {
						predHops = fsb.getToHops();
						recompile = fsb.requiresToRecompilation();
					}
					else if (pos == 3){
						predHops = fsb.getIncrementHops();
						recompile = fsb.requiresIncrementRecompilation();
					}
					if( recompile && DMLScript.isActiveAM() ) //set program block specific remote memory
						DMLAppMasterUtils.setupProgramBlockRemoteMaxMemory(this);
					
					tmp = (IntObject) executePredicate(instructions, predHops, recompile, ValueType.INT, ec);
				}
				else
					tmp = (IntObject) executePredicate(instructions, null, false, ValueType.INT, ec);
			}
		}
		catch(Exception ex)
		{
			String predNameStr = null;
			if 		(pos == 1) predNameStr = "from";
			else if (pos == 2) predNameStr = "to";
			else if (pos == 3) predNameStr = "increment";
			
			throw new DMLRuntimeException(this.printBlockErrorLocation() +"Error evaluating '" + predNameStr + "' predicate", ex);
		}
		
		//final check of resulting int object
		if ( tmp == null )
		{
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Failed to evaluate the FOR predicate.");
		}
		else
		{
			if( tmp instanceof IntObject )
				ret = (IntObject)tmp;
			else //downcast to int if necessary
				ret = new IntObject(tmp.getName(),tmp.getLongValue()); 
		}
		
		return ret;
	}

	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in for program block generated from for statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
}