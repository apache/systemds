/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLScriptException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;


public class Program 
{
	
	public static final String KEY_DELIM = "::";
	
	public ArrayList<ProgramBlock> _programBlocks;

	private HashMap<String, HashMap<String,FunctionProgramBlock>> _namespaceFunctions;
	
	public Program() throws DMLRuntimeException {
		_namespaceFunctions = new HashMap<String, HashMap<String,FunctionProgramBlock>>(); 
		_programBlocks = new ArrayList<ProgramBlock>();
	}

	/**
	 * 
	 * @param namespace
	 * @param fname
	 * @param fpb
	 */
	public synchronized void addFunctionProgramBlock(String namespace, String fname, FunctionProgramBlock fpb)
	{	
		if (namespace == null) 
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		
		HashMap<String,FunctionProgramBlock> namespaceBlocks = null;
		namespaceBlocks = _namespaceFunctions.get(namespace);
		if (namespaceBlocks == null){
			namespaceBlocks = new HashMap<String,FunctionProgramBlock>();
			_namespaceFunctions.put(namespace,namespaceBlocks);
		}
		
		namespaceBlocks.put(fname,fpb);
	}
	
	/**
	 * 
	 * @param namespace
	 * @param fname
	 */
	public synchronized void removeFunctionProgramBlock(String namespace, String fname) 
	{	
		if (namespace == null) 
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		
		HashMap<String,FunctionProgramBlock> namespaceBlocks = null;
		if( _namespaceFunctions.containsKey(namespace) ){
			namespaceBlocks = _namespaceFunctions.get(namespace);
			if( namespaceBlocks.containsKey(fname) )
				namespaceBlocks.remove(fname);
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public synchronized HashMap<String,FunctionProgramBlock> getFunctionProgramBlocks(){
		
		HashMap<String,FunctionProgramBlock> retVal = new HashMap<String,FunctionProgramBlock>();
		
		//create copy of function program blocks
		for (String namespace : _namespaceFunctions.keySet()){
			HashMap<String,FunctionProgramBlock> namespaceFSB = _namespaceFunctions.get(namespace);
			for( Entry<String, FunctionProgramBlock> e: namespaceFSB.entrySet() ){
				String fname = e.getKey(); 
				FunctionProgramBlock fpb = e.getValue();
				String fKey = DMLProgram.constructFunctionKey(namespace, fname);
				retVal.put(fKey, fpb);
			}
		}
		
		return retVal;
	}
	
	/**
	 * 
	 * @param namespace
	 * @param fname
	 * @return
	 * @throws DMLRuntimeException
	 */
	public synchronized FunctionProgramBlock getFunctionProgramBlock(String namespace, String fname) throws DMLRuntimeException{
		
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

	public void execute(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException
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
