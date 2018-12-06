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
package org.apache.sysml.parser.dml;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import org.apache.sysml.runtime.DMLRuntimeException;

/** 
 * This class is a simple container class used to hold the function to be inlined.
 * It contains the function name, body and also the input and return arguments.
 * The user invokes getInlinedDML method to get the inlined DML code.
 */
public class InlineableMethods {
	ArrayList<String> _variables;
	final String _body;
	final String _fnName;
	final ArrayList<String> _inputArgs;
	final ArrayList<String> _retVariables;
	static int CALLER_ID = 1;
	
	public InlineableMethods(String fnName, String body, HashSet<String> variables, ArrayList<String> inputArgs, ArrayList<String> retVariables) {
		_fnName = fnName;
		_body = body;
		_variables = new ArrayList<String>(variables);
		_variables.sort(Comparator.comparing(String::length).reversed());
		_inputArgs = inputArgs;
		_retVariables = retVariables;
	}
	
	public ArrayList<String> getLocalVariables() {
		return _variables;
	}
	
	private String _getInlinedDML(HashMap<String, String> actualArguments) {
		String ret = _body;
		int callerID = CALLER_ID++;
		for(String var : _variables) {
			String originalVarName = var.substring(InlineHelper.ARG_PREFIX.length());
			if(actualArguments.containsKey(var)) {
				ret = ret.replaceAll(var, actualArguments.get(var));
			}
			else {
				// internal argument
				ret = ret.replaceAll(var, LOCAL_ARG_PREFIX + _fnName + "_" + callerID + "_" + originalVarName);
			}
		}
		return ret;
	}
	
	public String getInlinedDML(ArrayList<String> actualInputArgs, ArrayList<String> actualRetVariables) {
		HashMap<String, String> actualArguments = new HashMap<>();
		if(actualInputArgs.size() != _inputArgs.size()) {
			throw new DMLRuntimeException("Incorrect number of input arguments for the function " + _fnName + ": expected " 
			+ _inputArgs.size() + " (" + String.join(", ", _inputArgs) + ") but found " + actualInputArgs.size() 
			+ " (" + String.join(", ", actualInputArgs) + ")");
		}
		if(actualRetVariables.size() != _retVariables.size()) {
			throw new DMLRuntimeException("Incorrect number of return variables for the function " + _fnName + ": expected " 
			+ _retVariables.size() + " (" + String.join(", ", _retVariables) + ") but found " + actualRetVariables.size()
			+ " (" + String.join(", ", actualRetVariables) + ")");
		}
		for(int i = 0; i < _inputArgs.size(); i++) {
			actualArguments.put(_inputArgs.get(i), actualInputArgs.get(i));
		}
		for(int i = 0; i < _retVariables.size(); i++) {
			actualArguments.put(_retVariables.get(i), actualRetVariables.get(i));
		}
		return _getInlinedDML(actualArguments);
	}
	
	static final String LOCAL_ARG_PREFIX;
	static {
		Random rand = new Random();
		LOCAL_ARG_PREFIX = "LOCAL_" + Math.abs(rand.nextLong()) + "_" + Math.abs(rand.nextLong());
//		LOCAL_ARG_PREFIX = "LOCAL_";
	}
}
