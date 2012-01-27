package dml.runtime.controlprogram;

import java.util.ArrayList;

import dml.parser.IterablePredicate;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.CPInstructions.IntObject;
import dml.runtime.instructions.CPInstructions.ScalarObject;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class ForProgramBlock extends ProgramBlock {

	protected ArrayList<Instruction> 	_predicate;
	protected ArrayList <Instruction> 	_exitInstructions ;
	private ArrayList<ProgramBlock> 	_childBlocks;

	protected IterablePredicate			_iterablePredicate;
	
	public void printMe() {
		
		//System.out.println("***** current for block predicate inst: *****");
		//for (Instruction cp : _predicateInst){
		//	cp.printMe();
		//}
		
		System.out.println("***** children block inst: *****");
		for (ProgramBlock pb : this._childBlocks){
			pb.printMe();
		}
		
		System.out.println("***** current block inst exit: *****");
		for (Instruction i : this._exitInstructions) {
			i.printMe();
		}
		
		
	}
	
	/*
	public ForProgramBlock(Program prog, ArrayList<Instruction> predicate){
		super(prog);
		_predicate = predicate;
		//_predicateResultVar = findPredicateResultVar ();
		_exitInstructions = new ArrayList<Instruction>();
		_childBlocks = new ArrayList<ProgramBlock>(); 
	}
	*/
	
	public ForProgramBlock(Program prog, IterablePredicate iterPred){
		super(prog);
		//_predicate = null;
		//_predicateResultVar = findPredicateResultVar ();
		_exitInstructions = new ArrayList<Instruction>();
		_childBlocks = new ArrayList<ProgramBlock>(); 
		_iterablePredicate = iterPred;
		
	}
	
		
	//public void setExitInstructions1(ArrayList<Instruction> predicate){
	//	_predicate = predicate;
	//}
	
	public void addExitInstruction(Instruction inst){
		_exitInstructions.add(inst);
	}
	
	//public ArrayList<Instruction> getPredicate(){
	//	return _predicate;
	//}
	
	//public String getPredicateResultVar(){
	//	return _predicateResultVar;
	//}
	
	//public void setPredicateResultVar(String resultVar) {
	//	_predicateResultVar = resultVar;
	//}
	
	public ArrayList<Instruction> getExitInstructions(){
		return _exitInstructions;
	}

	/*
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
	*/
	
	
	/* for the current For Loop, there are no instructions to execute.  Instead, the iterable 
	 * predicate variable has it's value updated in the execute() method. 
	 * 
	 * 
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
	 */	
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException{

		// add the iterable predicate variable to the variable set
		String iterVarName = _iterablePredicate.getIterVar().getName();
		IntObject iterValue = new IntObject(iterVarName, _iterablePredicate.getFrom());
		_variables.put(iterValue.getName(), iterValue);
		ScalarObject predResult = (ScalarObject)this._variables.get(iterVarName);
		
		while(predResult.getIntValue() <= _iterablePredicate.getTo()){
						
			// for each program block
			for (ProgramBlock pb : this._childBlocks){
				
				pb.setVariables(_variables);
				pb.setMetaData(_matrices);
				pb.execute(ec);
				_variables = pb._variables;
				_matrices = pb.getMetaData();
			}
			
			// update the iterable predicate variable 
			if (_variables.get(iterVarName) == null || !(_variables.get(iterVarName) instanceof IntObject))
				throw new DMLRuntimeException("iter predicate " + iterVarName + " must be remain of type scalar int");
			
			int newValue = ((ScalarObject)_variables.get(iterVarName)).getIntValue() + _iterablePredicate.getIncrement();
			_variables.put(iterVarName, new IntObject(iterVarName,newValue));
			predResult = (ScalarObject)_variables.get(iterVarName);
			
		}
		
		//execute(_exitInstructions, ec);	
	}

	public void addProgramBlock(ProgramBlock childBlock) {
		_childBlocks.add(childBlock);
	}
	
	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
	
}