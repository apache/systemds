package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import org.nimble.control.DAGQueue;

import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.CacheStatusException;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class Program {
	
	public static final String KEY_DELIM = "::";
	
	public ArrayList<ProgramBlock> _programBlocks;

	protected LocalVariableMap _programVariables;
	private HashMap<String, HashMap<String,FunctionProgramBlock>> _namespaceFunctions;
	

	//handle to the nimble dag queue
	private DAGQueue _dagQueue;

		
	public Program() throws DMLRuntimeException {
		_namespaceFunctions = new HashMap<String, HashMap<String,FunctionProgramBlock>>(); 
		_programBlocks = new ArrayList<ProgramBlock>();
		_programVariables = new LocalVariableMap ();
	}

	public void addFunctionProgramBlock(String namespace, String fname, FunctionProgramBlock fpb){
		
		if (namespace == null) 
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		
		HashMap<String,FunctionProgramBlock> namespaceBlocks = null;
		
		synchronized( _namespaceFunctions )
		{
			namespaceBlocks = _namespaceFunctions.get(namespace);
			if (namespaceBlocks == null){
				namespaceBlocks = new HashMap<String,FunctionProgramBlock>();
				_namespaceFunctions.put(namespace,namespaceBlocks);
			}
		}
		
		namespaceBlocks.put(fname,fpb);
	}
	
	public HashMap<String,FunctionProgramBlock> getFunctionProgramBlocks(){
		
		HashMap<String,FunctionProgramBlock> retVal = new HashMap<String,FunctionProgramBlock>();
		for (String namespace : _namespaceFunctions.keySet()){
			HashMap<String,FunctionProgramBlock> namespaceFSB = _namespaceFunctions.get(namespace);
			for (String fname : namespaceFSB.keySet()){
				retVal.put(namespace+KEY_DELIM+fname, namespaceFSB.get(fname));
			}
		}
		return retVal;
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

	public void execute(LocalVariableMap varMap, ExecutionContext ec)
	throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//populate initial symbol table
		_programVariables.putAll (varMap);
		
		for (int i=0; i<_programBlocks.size(); i++)
		{
			
			// execute each top-level program block
			ProgramBlock pb = _programBlocks.get(i);
			pb.setVariables(_programVariables);
			
			pb.execute(ec);
			_programVariables = pb.getVariables();
		}
		
		//cleanup remaining symbol table
		cleanupCachedVariables();
	}
	
	public void cleanupCachedVariables() throws CacheStatusException
	{
		for( String var : _programVariables.keySet() )
		{
			Data dat = _programVariables.get(var);
			DataType dt = dat.getDataType();
			
			if ( dt == DataType.MATRIX )
			{
				((MatrixObjectNew)dat).clearData();
			}
			else if ( dt == null || dt == DataType.UNKNOWN )
			{
				if( dat instanceof MatrixObjectNew )
					((MatrixObjectNew)dat).clearData();
			}
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
