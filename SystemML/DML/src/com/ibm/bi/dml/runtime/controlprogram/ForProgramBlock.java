package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ComputationCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.Instruction.INSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

public class ForProgramBlock extends ProgramBlock
{
	protected ArrayList<Instruction> 	_fromInstructions;
	protected ArrayList<Instruction> 	_toInstructions;
	protected ArrayList<Instruction> 	_incrementInstructions;
	
	protected ArrayList <Instruction> 	_exitInstructions ;
	protected ArrayList<ProgramBlock> 	_childBlocks;

	protected String[]                  _iterablePredicateVars; //from,to,where constants/internal vars not captured via instructions
	
	public void printMe() 
	{		
		System.out.println("***** current for block predicate inst: *****");
		System.out.println("FROM:");
		for (Instruction cp : _fromInstructions){
			cp.printMe();
		}
		System.out.println("TO:");
		for (Instruction cp : _toInstructions){
			cp.printMe();
		}
		System.out.println("INCREMENT:");
		for (Instruction cp : _incrementInstructions){
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
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		// add the iterable predicate variable to the variable set
		String iterVarName = _iterablePredicateVars[0];

		// evaluate from, to, incr only once (assumption: known at for entry)
		IntObject from = executePredicateInstructions( 1, _fromInstructions, ec );
		IntObject to   = executePredicateInstructions( 2, _toInstructions, ec );
		IntObject incr = executePredicateInstructions( 3, _incrementInstructions, ec );
		
		if ( incr.getIntValue() <= 0 ) //would produce infinite loop
			throw new DMLRuntimeException("Expression for increment of variable '" + iterVarName + "' must evaluate to a positive value.");
				
		// initialize iter var to form value
		IntObject iterVar = new IntObject(iterVarName, from.getIntValue() );
		
		// run for loop body as long as predicate is true 
		// (for supporting dynamic TO, move expression execution to end of while loop)
		while( iterVar.getIntValue() <= to.getIntValue() )
		{
			_variables.put(iterVarName, iterVar); 
			
			//if( DMLScript.DEBUG )
			//	System.out.println("FOR: start iteration for "+iterVarName+"="+iterVar.getIntValue());
			
			// for each program block
			for (ProgramBlock pb : this._childBlocks)
			{	
				pb.setVariables(_variables);
				pb.execute(ec);

				_variables = pb._variables;
			}
			
			// update the iterable predicate variable 
			if (_variables.get(iterVarName) == null || !(_variables.get(iterVarName) instanceof IntObject))
				throw new DMLRuntimeException("iter predicate " + iterVarName + " must be remain of type scalar int");
			
			//increment of iterVar (changes  in loop body get discarded)
			iterVar = new IntObject( iterVarName, iterVar.getIntValue()+incr.getIntValue() );
		}
		
		execute(_exitInstructions, ec);	
	}

	protected IntObject executePredicateInstructions( int pos, ArrayList<Instruction> instructions, ExecutionContext ec ) 
		throws DMLRuntimeException
	{
		IntObject ret = null;
			
		try
		{
			if( _iterablePredicateVars[pos] != null )
			{
				//check for literals or scalar variables
				ret = (IntObject) getScalarVariable(_iterablePredicateVars[pos], ValueType.INT); 		
			}		
			else
			{
				if( instructions != null && instructions.size()>0 )
				{
					String retName = null;
					boolean isSQL = false;
					
					// Execute all scalar simple instructions (relational expressions, etc.)
					for (Instruction si : instructions ) {
						
						if ( si.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM && ((CPInstruction)si).getCPInstructionType() != CPINSTRUCTION_TYPE.Variable )
						{
							((CPInstruction)si).processInstruction(this);
							retName = ((ComputationCPInstruction) si).getOutputVariableName();  
						}
						else if(si instanceof SQLScalarAssignInstruction)
						{
							((SQLScalarAssignInstruction) si).execute(ec);
							retName = ((SQLScalarAssignInstruction) si).getVariableName();
							isSQL = true;
						}
					}
					
					if(!isSQL)
						ret = (IntObject) getScalarVariable(retName, ValueType.INT);
					else
						ret = (IntObject) ec.getVariable(retName, ValueType.INT);
					
					// Execute all other instructions in the predicate (variableCPInstruction, etc.)
					for (Instruction si : instructions ) {
						if ( !(si.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM && ((CPInstruction)si).getCPInstructionType() != CPINSTRUCTION_TYPE.Variable))
							((CPInstruction)si).processInstruction(this);
					}
				}
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		if ( ret == null )
			throw new DMLRuntimeException("Failed to evaluate the FOR predicate.");
		
		return ret;
	}
}