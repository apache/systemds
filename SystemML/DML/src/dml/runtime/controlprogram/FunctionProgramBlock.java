package dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import dml.lops.Lops;
import dml.parser.DataIdentifier;
import dml.parser.Expression.ValueType;
import dml.runtime.instructions.CPInstructionParser;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.CPInstructions.BooleanObject;
import dml.runtime.instructions.CPInstructions.CPInstruction;
import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.instructions.CPInstructions.IntObject;
import dml.runtime.instructions.CPInstructions.ScalarCPInstruction;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MetaData;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class FunctionProgramBlock extends ProgramBlock {

	protected ArrayList<ProgramBlock> _childBlocks;
	protected ArrayList<DataIdentifier> _inputParams;
	protected ArrayList<DataIdentifier> _outputParams;
	
	public void printMe() {
		
		for (ProgramBlock pb : this._childBlocks){
			pb.printMe();
		}
	}
	
	public ArrayList<DataIdentifier> getInputParams(){
		return _inputParams;
	}
	
	public ArrayList<DataIdentifier> getOutputParams(){
		return _outputParams;
	}
	
	public FunctionProgramBlock( Program prog,
								 Vector <DataIdentifier> inputParams, 
								 Vector <DataIdentifier> outputParams)
	{
		super(prog);
		_childBlocks = new ArrayList<ProgramBlock>();
		_inputParams = new ArrayList<DataIdentifier>();
		for (DataIdentifier id : inputParams){
			_inputParams.add(new DataIdentifier(id));
		}
		_outputParams = new ArrayList<DataIdentifier>();
		for (DataIdentifier id : outputParams){
			_outputParams.add(new DataIdentifier(id));
		}
	}
	
	public void addProgramBlock(ProgramBlock childBlock) {
		_childBlocks.add(childBlock);
	}
	
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException{
		
		// for each program block
		for (int i=0; i < this._childBlocks.size(); i++){
			ProgramBlock pb = this._childBlocks.get(i);
			pb.setVariables(_variables);
			pb.setMetaData(_matrices);
			pb.execute(ec);
			_variables = pb._variables;
			_matrices = pb.getMetaData();
		}
	}
	
	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
}