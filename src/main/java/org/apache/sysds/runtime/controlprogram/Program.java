/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;

public class Program 
{
	public static final String KEY_DELIM = "::";
	
	private DMLProgram _prog;
	private ArrayList<ProgramBlock> _programBlocks;

	private HashMap<String, HashMap<String,FunctionProgramBlock>> _namespaceFunctions;
	
	public Program() {
		_namespaceFunctions = new HashMap<>();
		_namespaceFunctions.put(DMLProgram.DEFAULT_NAMESPACE, new HashMap<>());
		_programBlocks = new ArrayList<>();
	}
	
	public Program(DMLProgram prog) {
		this();
		setDMLProg(prog);
	}

	public void setDMLProg(DMLProgram prog) {
		_prog = prog;
	}
	
	public DMLProgram getDMLProg() {
		return _prog;
	}
	
	public synchronized void addFunctionProgramBlock(String namespace, String fname, FunctionProgramBlock fpb) {
		if( fpb == null )
			throw new DMLRuntimeException("Invalid null function program block.");
		namespace = (namespace == null) ? DMLProgram.DEFAULT_NAMESPACE : namespace;
		HashMap<String,FunctionProgramBlock> namespaceBlocks = _namespaceFunctions.get(namespace);
		if (namespaceBlocks == null){
			namespaceBlocks = new HashMap<>();
			_namespaceFunctions.put(namespace,namespaceBlocks);
		}
		namespaceBlocks.put(fname,fpb);
	}

	public synchronized void removeFunctionProgramBlock(String namespace, String fname) {
		namespace = (namespace == null) ? DMLProgram.DEFAULT_NAMESPACE : namespace;
		HashMap<String,FunctionProgramBlock> namespaceBlocks = null;
		if( _namespaceFunctions.containsKey(namespace) ){
			namespaceBlocks = _namespaceFunctions.get(namespace);
			if( namespaceBlocks.containsKey(fname) )
				namespaceBlocks.remove(fname);
		}
	}

	public synchronized HashMap<String,FunctionProgramBlock> getFunctionProgramBlocks(){
		HashMap<String,FunctionProgramBlock> retVal = new HashMap<>();
		
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
	
	public synchronized boolean containsFunctionProgramBlock(String namespace, String fname) {
		namespace = (namespace == null) ? DMLProgram.DEFAULT_NAMESPACE : namespace;
		return _namespaceFunctions.containsKey(namespace)
			&& _namespaceFunctions.get(namespace).containsKey(fname);
	}
	
	public synchronized FunctionProgramBlock getFunctionProgramBlock(String namespace, String fname) {
		namespace = (namespace == null) ? DMLProgram.DEFAULT_NAMESPACE : namespace;
		HashMap<String,FunctionProgramBlock> namespaceFunctBlocks = _namespaceFunctions.get(namespace);
		if (namespaceFunctBlocks == null)
			throw new DMLRuntimeException("namespace " + namespace + " is undefined.");
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

	public void execute(ExecutionContext ec) {
		try{
			for (int i=0; i<_programBlocks.size(); i++)
				_programBlocks.get(i).execute(ec);
		}
		catch(DMLScriptException e) {
			throw e;
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	public Program clone(boolean deep) {
		if( deep )
			throw new NotImplementedException();
		Program ret = new Program(_prog);
		//shallow copy of all program blocks
		ret._programBlocks.addAll(_programBlocks);
		//shallow copy of all functions, except external 
		//functions, which require a deep copy
		for( Entry<String, HashMap<String, FunctionProgramBlock>> e1 : _namespaceFunctions.entrySet() )
			for( Entry<String, FunctionProgramBlock> e2 : e1.getValue().entrySet() ) {
				FunctionProgramBlock fpb = e2.getValue();
				ret.addFunctionProgramBlock(e1.getKey(), e2.getKey(), fpb);
			}
		return ret;
	}
	
	@Override
	public Object clone() {
		return clone(true);
	}
}
