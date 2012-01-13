package dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import dml.parser.Expression.ValueType;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.CPInstructions.BooleanObject;
import dml.runtime.instructions.CPInstructions.CPInstruction;
import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.instructions.CPInstructions.ScalarCPInstruction;
import dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;
import dml.runtime.matrix.MetaData;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class ForProgramBlock extends ProgramBlock {

	private ArrayList<Instruction> _predicate;
	private String _predicateResultVar;
	protected ArrayList <Instruction> _exitInstructions ;
	private ArrayList<ProgramBlock> _childBlocks;
	
	public void printMe() {
		
		System.out.println("***** current for block predicate inst: *****");
		for (Instruction cp : _predicate){
			cp.printMe();
		}
		
		System.out.println("***** children block inst: *****");
		for (ProgramBlock pb : this._childBlocks){
			pb.printMe();
		}
		
		System.out.println("***** current block inst exit: *****");
		for (Instruction i : this._exitInstructions) {
			i.printMe();
		}
		
		
	}
	
	public ForProgramBlock(Program prog, ArrayList<Instruction> predicate){
		super(prog);
		_predicate = predicate;
		_predicateResultVar = findPredicateResultVar ();
		_exitInstructions = new ArrayList<Instruction>();
		_childBlocks = new ArrayList<ProgramBlock>(); 
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
			if ( si instanceof ScalarCPInstruction ) {
				result = ((ScalarCPInstruction) si).getOutputVariableName();  
			}
			else if(si instanceof SQLScalarAssignInstruction)
				result = ((SQLScalarAssignInstruction)si).getVariableName();
		}
		return result;
	}
	
	private BooleanObject executePredicate(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		BooleanObject result = null;
		//TODO change this
		boolean isSQL = false;
		// Execute all scalar simple instructions (relational expressions, etc.)
		for (Instruction si : _predicate ) {
			if ( si instanceof ScalarCPInstruction )
				((ScalarCPInstruction)si).processInstruction(this);
			else if(si instanceof SQLScalarAssignInstruction)
			{
				((SQLScalarAssignInstruction)si).execute(ec);
				isSQL = true;
			}
		}

		if(!isSQL)
			result = (BooleanObject) getScalarVariable(_predicateResultVar, ValueType.BOOLEAN);
		else
			result = (BooleanObject) ec.getVariable(_predicateResultVar, ValueType.BOOLEAN);
		
		
		// Execute all other instructions in the predicate (variableCPInstruction, etc.)
		for (Instruction si : _predicate ) {
			if ( ! (si instanceof ScalarCPInstruction) && si instanceof CPInstruction)
				((CPInstruction)si).processInstruction(this);
		}
		
		if ( result == null )
			throw new DMLRuntimeException("Failed to evaluate the FOR predicate.");
		return result;
	}
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException{
	
		BooleanObject predResult = executePredicate(ec); 

		while(predResult.getBooleanValue()){
			
			// for each program block
			for (ProgramBlock pb : this._childBlocks){
				
				pb.setVariables(_variables);
				pb.setMetaData(_matrices);
				pb.execute(ec);
				_variables = pb._variables;
				_matrices = pb.getMetaData();
			}
			
			predResult = executePredicate(ec);
		}
		
		execute(_exitInstructions, ec);	
	}

	public void addProgramBlock(ProgramBlock childBlock) {
		_childBlocks.add(childBlock);
	}
	
	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
	
}