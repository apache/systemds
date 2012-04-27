package dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import org.nimble.control.DAGQueue;

import dml.parser.DMLProgram;
import dml.runtime.instructions.CPInstructions.Data;
import dml.sql.sqlcontrolprogram.ExecutionContext;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class Program {

	public ArrayList<ProgramBlock> _programBlocks;

	protected HashMap<String, Data> _programVariables;
	private HashMap<String, HashMap<String,FunctionProgramBlock>> _namespaceFunctions;
	

	//handle to the nimble dag queue
	private DAGQueue _dagQueue;
		
	public Program() {
		_namespaceFunctions = new HashMap<String, HashMap<String,FunctionProgramBlock>>(); 
		_programBlocks = new ArrayList<ProgramBlock>();
		_programVariables = new HashMap<String, Data>();
	}

	public void addFunctionProgramBlock(String namespace, String fname, FunctionProgramBlock fpb){
		
		if (namespace == null) namespace = DMLProgram.DEFAULT_NAMESPACE;
		
		HashMap<String,FunctionProgramBlock> namespaceBlocks = _namespaceFunctions.get(namespace);
		if (namespaceBlocks == null){
			namespaceBlocks = new HashMap<String,FunctionProgramBlock>();
			_namespaceFunctions.put(namespace,namespaceBlocks);
		}
		namespaceBlocks.put(fname,fpb);
	}
	
	public FunctionProgramBlock getFunctionProgramBlock(String namespace, String fname) throws DMLRuntimeException{
		
		if (namespace == null) namespace = DMLProgram.DEFAULT_NAMESPACE;
		
		HashMap<String,FunctionProgramBlock> namespaceFunctBlocks = _namespaceFunctions.get(namespace);
		if (namespaceFunctBlocks == null)
			throw new DMLRuntimeException("namespace " + namespace + " is undefined");
		FunctionProgramBlock retVal = namespaceFunctBlocks.get(fname);
		if (retVal == null)
			throw new DMLRuntimeException("function " + fname + " is undefined in namespace " + namespace);
		return retVal;
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
			
			pb.execute(ec);
			_programVariables = pb.getVariables();
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
