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

package org.apache.sysds.hops.fedplanner;

import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Utility class for federated planners.
 */
public class FederatedPlannerUtils {

	/**
	 * Get transient inputs from either paramMap or transientWrites.
	 * Inputs from paramMap has higher priority than inputs from transientWrites.
	 * @param currentHop hop for which inputs are read from maps
	 * @param paramMap of local parameters
	 * @param transientWrites map of transient writes
	 * @param localVariableMap map of local variables
	 * @return inputs of currentHop
	 */
	public static ArrayList<Hop> getTransientInputs(Hop currentHop, Map<String, Hop> paramMap,
		Map<String,Hop> transientWrites, LocalVariableMap localVariableMap){
		Hop tWriteHop = null;
		if ( paramMap != null)
			tWriteHop = paramMap.get(currentHop.getName());
		if ( tWriteHop == null )
			tWriteHop = transientWrites.get(currentHop.getName());
		if ( tWriteHop == null ) {
			if(localVariableMap.get(currentHop.getName()) != null)
				return null;
			else
				throw new DMLRuntimeException("Transient write not found for " + currentHop);
		}
		else
			return new ArrayList<>(Collections.singletonList(tWriteHop));
	}

	/**
	 * Return parameter map containing the mapping from parameter name to input hop
	 * for all parameters of the function hop.
	 * @param funcOp hop for which the mapping of parameter names to input hops are made
	 * @return parameter map or empty map if function has no parameters
	 */
	public static Map<String,Hop> getParamMap(FunctionOp funcOp){
		String[] inputNames = funcOp.getInputVariableNames();
		Map<String,Hop> paramMap = new HashMap<>();
		if ( inputNames != null ){
			for ( int i = 0; i < funcOp.getInput().size(); i++ )
				paramMap.put(inputNames[i],funcOp.getInput(i));
		}
		return paramMap;
	}
}
