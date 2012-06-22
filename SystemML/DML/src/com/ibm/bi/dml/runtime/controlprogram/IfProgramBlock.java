// DOUG VERSION
package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ComputationCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.Instruction.INSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class IfProgramBlock extends ProgramBlock {

	private ArrayList<Instruction> _predicate;
	private String _predicateResultVar;
	private ArrayList <Instruction> _exitInstructions ;
	private ArrayList<ProgramBlock> _childBlocksIfBody;
	private ArrayList<ProgramBlock> _childBlocksElseBody;
	
	public ArrayList<ProgramBlock> getChildBlocksIfBody()
		{ return _childBlocksIfBody; }

	public void setChildBlocksIfBody(ArrayList<ProgramBlock> blocks)
		{ _childBlocksIfBody = blocks; }
	
	public void addProgramBlockIfBody(ProgramBlock pb)
		{ _childBlocksIfBody.add(pb); }	
	
	public ArrayList<ProgramBlock> getChildBlocksElseBody()
		{ return _childBlocksElseBody; }

	public void setChildBlocksElseBody(ArrayList<ProgramBlock> blocks)
		{ _childBlocksElseBody = blocks; }
	
	public void addProgramBlockElseBody(ProgramBlock pb)
		{ _childBlocksElseBody.add(pb); }
	
	public void printMe() {
		
		System.out.println("***** if current block predicate inst: *****");
		for (Instruction cp : _predicate){
			cp.printMe();
		}
		
		System.out.println("***** children block inst --- if body : *****");
		for (ProgramBlock pb : this._childBlocksIfBody){
			pb.printMe();
		}
	
		System.out.println("***** children block inst --- else body : *****");
		for (ProgramBlock pb: this._childBlocksElseBody){
			pb.printMe();
		}
		
		System.out.println("***** current block inst exit: *****");
		for (Instruction i : this._exitInstructions) {
			i.printMe();
		}	
	}
	
	
	public IfProgramBlock(Program prog, ArrayList<Instruction> predicate) throws DMLRuntimeException{
		super(prog);
		
		_childBlocksIfBody = new ArrayList<ProgramBlock>();
		_childBlocksElseBody = new ArrayList<ProgramBlock>();
		
		_predicate = predicate;
		_predicateResultVar = findPredicateResultVar ();
		_exitInstructions = new ArrayList<Instruction>();
	}

	public void setExitInstructions2(ArrayList<Instruction> exitInstructions){
		_exitInstructions = exitInstructions;
	}

	public void setExitInstructions1(ArrayList<Instruction> predicate){
		_predicate = predicate;
	}
	
	public void addExitInstruction(Instruction inst){
		_exitInstructions.add(inst);
	}
	
	public ArrayList<Instruction> getPredicate(){
		return _predicate;
	}
	
	public String getPredicateResultVar(){
		return _predicateResultVar;
	}
	
	public void setPredicateResultVar(String resultVar) {
		_predicateResultVar = resultVar;
	}
	
	public ArrayList<Instruction> getExitInstructions(){
		return _exitInstructions;
	}

	private String findPredicateResultVar ( ) {
		String result = null;
		for ( Instruction si : _predicate ) {
			if ( si.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM && ((CPInstruction)si).getCPInstructionType() != CPINSTRUCTION_TYPE.Variable ) {
				result = ((ComputationCPInstruction) si).getOutputVariableName();  
			}
			else if(si instanceof SQLScalarAssignInstruction)
				result = ((SQLScalarAssignInstruction)si).getVariableName();
		}
		return result;
	}
	
	private BooleanObject executePredicate(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		BooleanObject result = null;
		//TODO this has to be changed
		boolean isSQL = false;
		// Execute all scalar simple instructions (relational expressions, etc.)
		for (Instruction si : _predicate ) {
			if ( si.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM && ((CPInstruction)si).getCPInstructionType() != CPINSTRUCTION_TYPE.Variable )
				((CPInstruction)si).processInstruction(this);
			else if(si instanceof SQLScalarAssignInstruction)
			{
				((SQLScalarAssignInstruction)si).execute(ec);
				isSQL = true;
			}
		}
		
		if(!isSQL)
			result = (BooleanObject) getScalarInput(_predicateResultVar, ValueType.BOOLEAN);
		else
			result = (BooleanObject) ec.getVariable(_predicateResultVar, ValueType.BOOLEAN);
		
		// Execute all other instructions in the predicate (variableCPInstruction, etc.)
		for (Instruction si : _predicate ) {
			if ( ! (si.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM && ((CPInstruction)si).getCPInstructionType() != CPINSTRUCTION_TYPE.Variable))
				((CPInstruction)si).processInstruction(this);
		}
		
		if ( result == null )
			throw new DMLRuntimeException("Failed to evaluate the IF predicate.");
		return result;
	}
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException{

		BooleanObject predResult = executePredicate(ec); 

		if(predResult.getBooleanValue()){
			
			// for each program block
			for (ProgramBlock pb : this._childBlocksIfBody){
				
				pb.setVariables(_variables);
				pb.execute(ec);
				_variables = pb._variables;
			}
		}
		else {

			// for each program block
			for (ProgramBlock pb : this._childBlocksElseBody){
				
				pb.setVariables(_variables);
				pb.execute(ec);
				_variables = pb._variables;
			}
		}
		
		execute(_exitInstructions, ec);
		
	}
}