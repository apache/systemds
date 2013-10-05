/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ComputationCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.Instruction.INSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;


public class WhileProgramBlock extends ProgramBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ArrayList<Instruction> _predicate;
	private String _predicateResultVar;
	private ArrayList <Instruction> _exitInstructions ;
	private ArrayList<ProgramBlock> _childBlocks;

	public WhileProgramBlock(Program prog, ArrayList<Instruction> predicate) throws DMLRuntimeException{
		super(prog);
		_predicate = predicate;
		_predicateResultVar = findPredicateResultVar ();
		_exitInstructions = new ArrayList<Instruction>();
		_childBlocks = new ArrayList<ProgramBlock>(); 
	}
	
	public void printMe() {
		
		System.out.println("***** while current block predicate inst: *****");
		for (Instruction cp : _predicate){
			cp.printMe();
		}
		
		for (ProgramBlock pb : this._childBlocks){
			pb.printMe();
		}
		
		System.out.println("***** current block inst exit: *****");
		for (Instruction i : this._exitInstructions) {
			i.printMe();
		}
	}
	
	

	
	public void addProgramBlock(ProgramBlock childBlock) {
		_childBlocks.add(childBlock);
	}
	
	public void setExitInstructions2(ArrayList<Instruction> exitInstructions)
		{ _exitInstructions = exitInstructions; }

	public void setExitInstructions1(ArrayList<Instruction> predicate)
		{ _predicate = predicate; }
	
	public void addExitInstruction(Instruction inst)
		{ _exitInstructions.add(inst); }
	
	public ArrayList<Instruction> getPredicate()
		{ return _predicate; }
	
	public void setPredicate( ArrayList<Instruction> predicate )
	{ 
		_predicate = predicate;
		_predicateResultVar = findPredicateResultVar();
	}
	
	public String getPredicateResultVar()
		{ return _predicateResultVar; }
	
	public void setPredicateResultVar(String resultVar) 
		{ _predicateResultVar = resultVar; }
	
	public ArrayList<Instruction> getExitInstructions()
		{ return _exitInstructions; }
	
	private BooleanObject executePredicate(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		BooleanObject result = null;
		try
		{
			if( _predicate!=null && _predicate.size()>0 )
			{
				if( _sb!=null )
				{
					WhileStatementBlock wsb = (WhileStatementBlock)_sb;
					result = (BooleanObject) executePredicate(_predicate, wsb.getPredicateHops(), ValueType.BOOLEAN, ec);
				}
				else
					result = (BooleanObject) executePredicate(_predicate, null, ValueType.BOOLEAN, ec);
			}
			else
				result = (BooleanObject)ec.getScalarInput(_predicateResultVar, ValueType.BOOLEAN);
		}
		catch(Exception ex)
		{
			LOG.trace("\nWhile predicate variables: "+ ec.getVariables().toString());
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Failed to evaluate the WHILE predicate.", ex);
		}
		
		if ( result == null )
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Failed to evaluate the WHILE predicate.");
		
		return result;
	}
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException{

		BooleanObject predResult = executePredicate(ec); 
		
		//execute while loop
		try 
		{
			while(predResult.getBooleanValue())
			{		
				//execute all child blocks
				for( ProgramBlock pb : _childBlocks )
					pb.execute(ec);
				
				predResult = executePredicate(ec);
			}
		}
		catch(Exception e)
		{
			LOG.trace("\nWhile predicate variables: "+ ec.getVariables().toString());
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating while program block.", e);
		}
		
		//execute exit instructions
		try {
			executeInstructions(_exitInstructions, ec);
		}
		catch(Exception e)
		{
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error executing while exit instructions.", e);
		}
	}

	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
	
	public void setChildBlocks(ArrayList<ProgramBlock> childs) 
	{
		_childBlocks = childs;
	}
	
	private String findPredicateResultVar ( ) {
		String result = null;
		for ( Instruction si : _predicate ) {
			if ( si.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM && ((CPInstruction)si).getCPInstructionType() != CPINSTRUCTION_TYPE.Variable ) {
				result = ((ComputationCPInstruction) si).getOutputVariableName();  
			}
			else if(si instanceof SQLScalarAssignInstruction)
				result = ((SQLScalarAssignInstruction) si).getVariableName();
		}
		return result;
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in while program block generated from while statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
}