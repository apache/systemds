/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLScriptException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;


public class Program 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String KEY_DELIM = "::";
	
	public ArrayList<ProgramBlock> _programBlocks;

	private HashMap<String, HashMap<String,FunctionProgramBlock>> _namespaceFunctions;
	
	public Program() throws DMLRuntimeException {
		_namespaceFunctions = new HashMap<String, HashMap<String,FunctionProgramBlock>>(); 
		_programBlocks = new ArrayList<ProgramBlock>();
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
		
		synchronized( _namespaceFunctions )
		{
			for (String namespace : _namespaceFunctions.keySet()){
				HashMap<String,FunctionProgramBlock> namespaceFSB = _namespaceFunctions.get(namespace);
				for (String fname : namespaceFSB.keySet()){
					String fKey = DMLProgram.constructFunctionKey(namespace, fname);
					retVal.put(fKey, namespaceFSB.get(fname));
				}
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
		//retVal._variables = new LocalVariableMap();
		return retVal;
	}
	
	public void addProgramBlock(ProgramBlock pb) {
		_programBlocks.add(pb);
	}

	public ArrayList<ProgramBlock> getProgramBlocks() {
		return _programBlocks;
	}

	public void execute(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException, DMLScriptException
	{
		ec.initDebugProgramCounters();
		
		try
		{
			for (int i=0 ; i<_programBlocks.size() ; i++) {
				ec.updateDebugState(i);
				_programBlocks.get(i).execute(ec);
			}
		}
		catch(DMLScriptException e) {
			throw e;
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
		
		ec.clearDebugProgramCounters();
	}
		
	
	public void printMe() {
		
		for (ProgramBlock pb : this._programBlocks) {
			pb.printMe();
		}
	}
}
