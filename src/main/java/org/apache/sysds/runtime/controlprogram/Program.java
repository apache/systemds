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

import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.FunctionDictionary;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;

public class Program 
{
	public static final String KEY_DELIM = "::";
	
	private DMLProgram _prog;
	private final ArrayList<ProgramBlock> _programBlocks;
	private final HashMap<String, FunctionDictionary<FunctionProgramBlock>> _namespaces;
	
	public Program() {
		_namespaces = new HashMap<>();
		_namespaces.put(DMLProgram.DEFAULT_NAMESPACE, new FunctionDictionary<>());
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
	
	public synchronized void addFunctionProgramBlock(String fkey, FunctionProgramBlock fpb, boolean opt) {
		String[] parts = DMLProgram.splitFunctionKey(fkey);
		addFunctionProgramBlock(parts[0], parts[1], fpb, opt);
	}
	
	public synchronized void addFunctionProgramBlock(String namespace, String fname, FunctionProgramBlock fpb) {
		addFunctionProgramBlock(namespace, fname, fpb, true);
	}
	
	public synchronized void addFunctionProgramBlock(String namespace, String fname, FunctionProgramBlock fpb, boolean opt) {
		if( fpb == null )
			throw new DMLRuntimeException("Invalid null function program block.");
		namespace = getSafeNamespace(namespace);
		FunctionDictionary<FunctionProgramBlock> dict = _namespaces.get(namespace);
		if (dict == null)
			_namespaces.put(namespace, dict = new FunctionDictionary<>());
		dict.addFunction(fname, fpb, opt);
	}

	public synchronized void removeFunctionProgramBlock(String namespace, String fname) {
		namespace = getSafeNamespace(namespace);
		FunctionDictionary<?> dict = null;
		if( _namespaces.containsKey(namespace) ){
			dict = _namespaces.get(namespace);
			if( dict.containsFunction(fname) )
				dict.removeFunction(fname);
		}
	}

	public HashMap<String,FunctionProgramBlock> getFunctionProgramBlocks(){
		return getFunctionProgramBlocks(true);
	}
	
	public FunctionDictionary<FunctionProgramBlock> getFunctionProgramBlocks(String nsName) {
		return _namespaces.get(nsName);
	}
	
	public synchronized HashMap<String,FunctionProgramBlock> getFunctionProgramBlocks(boolean opt){
		HashMap<String,FunctionProgramBlock> retVal = new HashMap<>();
		for (Entry<String,FunctionDictionary<FunctionProgramBlock>> namespace : _namespaces.entrySet()){
			if( namespace.getValue().getFunctions(opt) != null )
				for( Entry<String, FunctionProgramBlock> e2 : namespace.getValue().getFunctions(opt).entrySet() ){
					String fKey = DMLProgram.constructFunctionKey(namespace.getKey(), e2.getKey());
					retVal.put(fKey, e2.getValue());
				}
		}
		return retVal;
	}
	
	public synchronized boolean containsFunctionProgramBlock(String namespace, String fname) {
		namespace = getSafeNamespace(namespace);
		return _namespaces.containsKey(namespace)
			&& _namespaces.get(namespace).containsFunction(fname);
	}
	
	public synchronized boolean containsFunctionProgramBlock(String fkey, boolean opt) {
		String[] parts = DMLProgram.splitFunctionKey(fkey);
		return containsFunctionProgramBlock(parts[0], parts[1], opt);
	}
	
	public synchronized boolean containsFunctionProgramBlock(String namespace, String fname, boolean opt) {
		namespace = getSafeNamespace(namespace);
		return _namespaces.containsKey(namespace)
			&& _namespaces.get(namespace).containsFunction(fname, opt);
	}
	
	public synchronized FunctionProgramBlock getFunctionProgramBlock(String namespace, String fname) {
		return getFunctionProgramBlock(namespace, fname, true);
	}
	
	public synchronized FunctionProgramBlock getFunctionProgramBlock(String fkey, boolean opt) {
		String[] parts = DMLProgram.splitFunctionKey(fkey);
		return getFunctionProgramBlock(parts[0], parts[1], opt);
	}
	
	public synchronized FunctionProgramBlock getFunctionProgramBlock(String namespace, String fname, boolean opt) {
		namespace = getSafeNamespace(namespace);
		FunctionDictionary<FunctionProgramBlock> dict = _namespaces.get(namespace);
		if (dict == null)
			throw new DMLRuntimeException("namespace " + namespace + " is undefined.");
		FunctionProgramBlock retVal = dict.getFunction(fname, opt);
		if (retVal == null)
			throw new DMLRuntimeException("function " + fname + " ("+opt+") is undefined in namespace " + namespace);
		
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

	@Override
	public Object clone() {
		Program ret = new Program(_prog);
		//shallow copy of all program blocks
		ret._programBlocks.addAll(_programBlocks);
		//shallow copy of all functions, except external 
		//functions, which require a deep copy
		for( Entry<String, FunctionDictionary<FunctionProgramBlock>> e1 : _namespaces.entrySet() )
			for( Entry<String, FunctionProgramBlock> e2 : e1.getValue().getFunctions().entrySet() )
				ret.addFunctionProgramBlock(e1.getKey(), e2.getKey(), e2.getValue());
		return ret;
	}
	
	private static String getSafeNamespace(String namespace) {
		return (namespace == null) ? DMLProgram.DEFAULT_NAMESPACE : namespace;
	}
}
