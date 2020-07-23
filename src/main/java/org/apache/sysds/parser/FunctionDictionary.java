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

package org.apache.sysds.parser;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysds.common.Types.FunctionBlock;
import org.apache.sysds.runtime.DMLRuntimeException;

/**
 * Dictionary of all functions of a namespace, represented as a simple
 * key-value map of function names and function statement blocks.
 */
public class FunctionDictionary<T extends FunctionBlock> {
	/** optimized functions **/
	private Map<String, T> _funs;
	
	/** optional unoptimized functions (no sizes/literals propagated), e.g., for eval **/
	private Map<String, T> _funsOrig;
	
	public FunctionDictionary() {
		_funs = new HashMap<>();
	}
	
	public void addFunction(String fname, T fsb) {
		if( _funs.containsKey(fname) )
			throw new DMLRuntimeException("Function '"+fname+"' already existing in namespace.");
		//add function to existing maps
		_funs.put(fname, fsb);
		if( _funsOrig != null )
			_funsOrig.put(fname, fsb);
	}
	
	public void addFunction(String fname, T fsb, boolean opt) {
		if( !opt && _funsOrig == null )
			_funsOrig = new HashMap<>();
		Map<String,T> map = opt ? _funs : _funsOrig;
		if( map.containsKey(fname) )
			throw new DMLRuntimeException("Function '"+fname+"' ("+opt+") already existing in namespace.");
		map.put(fname, fsb);
	}
	
	public void removeFunction(String fname) {
		_funs.remove(fname);
		if( _funsOrig != null )
			_funsOrig.remove(fname);
	}
	
	public T getFunction(String fname) {
		return getFunction(fname, true);
	}
	
	public T getFunction(String fname, boolean opt) {
		//check for existing unoptimized functions if necessary
		if( !opt && _funsOrig == null )
			throw new DMLRuntimeException("Requested unoptimized function "
				+ "'"+fname+"' but original function copies have not been created.");
		
		//obtain optimized or unoptimized function (null if not available)
		return opt ? _funs.get(fname) : 
			(_funsOrig != null) ? _funsOrig.get(fname) : null;
	}
	
	public boolean containsFunction(String fname) {
		return containsFunction(fname, true);
	}
	
	public boolean containsFunction(String fname, boolean opt) {
		return opt ? _funs.containsKey(fname) :
			(_funsOrig != null && _funsOrig.containsKey(fname));
	}
	
	public Map<String, T> getFunctions() {
		return getFunctions(true);
	}
	
	public Map<String, T> getFunctions(boolean opt) {
		return opt ? _funs : _funsOrig;
	}
	
	@SuppressWarnings("unchecked")
	public void copyOriginalFunctions() {
		_funsOrig = new HashMap<>();
		for( Entry<String,T> fe : _funs.entrySet() )
			_funsOrig.put(fe.getKey(), (T)fe.getValue().cloneFunctionBlock());
	}
}
