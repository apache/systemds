package dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import org.nimble.control.DAGQueue;

import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.matrix.MetaData;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class Program {

	public ArrayList<ProgramBlock> _programBlocks;

	protected HashMap<String, Data> _programVariables;
	private HashMap<String, FunctionProgramBlock> _functionProgramBlocks;
	
	/*
	 *  Data structure to store meta data (dimensions, etc.) associated with each matrix
	 *  String: denotes the file name on HDFS. It corresponds to a matrix.
	 *  MatrixCharacteristics: metadata 
	 */
	protected HashMap<String, MetaData> _matrices;

	//handle to the nimble dag queue
	private DAGQueue _dagQueue;
		
	public Program() {
		_functionProgramBlocks = new HashMap<String, FunctionProgramBlock>(); 
		_programBlocks = new ArrayList<ProgramBlock>();
		_programVariables = new HashMap<String, Data>();
		_matrices = new HashMap<String, MetaData>();
	}

	public void addFunctionProgramBlock(String key, FunctionProgramBlock fpb){
		_functionProgramBlocks.put(key, fpb);
	}
	
	public FunctionProgramBlock getFunctionProgramBlock(String key){
		return _functionProgramBlocks.get(key);
	}
	
	public void addProgramBlock(ProgramBlock pb) {
		_programBlocks.add(pb);
	}

	public ArrayList<ProgramBlock> getProgramBlocks() {
		return _programBlocks;
	}
	
	public void setDAGQueue(DAGQueue dagQueue)
	{
		_dagQueue = dagQueue;
	}
	
	public DAGQueue getDAGQueue()
	{
		return _dagQueue; 
	}

	public void execute(HashMap<String, Data> hashMap, ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		_programVariables.putAll(hashMap);
		for (int i=0; i<_programBlocks.size(); i++) {
			
			// execute each top-level program block
			ProgramBlock pb = _programBlocks.get(i);
			pb.setVariables(_programVariables);
			pb.setMetaData(_matrices);
			
			pb.execute(ec);
			_programVariables = pb.getVariables();
			_matrices = pb.getMetaData();
		}
	}
	
	public void printMe() {
		
		/*for (String key : _functionProgramBlocks.keySet()) {
			System.out.println("function " + key);
			_functionProgramBlocks.get(key).printMe();
		}*/
		
		for (ProgramBlock pb : this._programBlocks) {
			pb.printMe();
		}
	}
}
