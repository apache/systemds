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

package org.apache.sysds.runtime.controlprogram.federated;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;

public class ExecutionContextMap {
	private ExecutionContext _main;
	private final Map<Long, ExecutionContext> _parEc;
	
	public ExecutionContextMap() {
		_main = createExecutionContext();
		_parEc = new ConcurrentHashMap<>();
	}
	
	public synchronized ExecutionContext get(long tid) {
		//return main execution context
		if( tid <= 0 )
			return _main;
		
		//atomic probe, create if necessary, and return
		return _parEc.computeIfAbsent(tid,
			k -> deriveExecutionContext(_main));
	}
	
	public synchronized void clear() {
		//handle main symbol table (w/ tmp list for concurrent modification)
		for( String varName : new ArrayList<>(_main.getVariables().keySet()) )
			_main.cleanupDataObject(_main.removeVariable(varName));
		
		//handle parfor execution contexts
		for( ExecutionContext ec : _parEc.values() )
			for( String varName : ec.getVariables().keySet() )
				_main.cleanupDataObject(ec.removeVariable(varName));
		_parEc.clear();
	}

	public synchronized void convertToSparkCtx() {
		// set hybrid mode for global consistency
		DMLScript.setGlobalExecMode(ExecMode.HYBRID);
		
		//convert existing execution contexts
		_main = deriveExecutionContext(_main);
		_parEc.replaceAll((k,v) -> deriveExecutionContext(v));
		
	}
	private static ExecutionContext createExecutionContext() {
		ExecutionContext ec = ExecutionContextFactory.createContext();
		ec.setAutoCreateVars(true); //w/o createvar inst
		return ec;
	}
	
	private static ExecutionContext deriveExecutionContext(ExecutionContext ec) {
		//derive execution context from main to make shared variables available
		//but allow normal instruction processing and removal if necessary
		ExecutionContext ec2 = ExecutionContextFactory
			.createContext(ec.getVariables(), ec.getProgram());
		ec2.setAutoCreateVars(true); //w/o createvar inst
		return ec2;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("\nMain EC: ");
		sb.append(_main.toString());
		sb.append("ParFor ECs: ");
		sb.append(_parEc.toString());
		return sb.toString();
	}
}
