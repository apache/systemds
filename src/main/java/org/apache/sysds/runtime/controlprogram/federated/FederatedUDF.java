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

import java.io.Serializable;

import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.Data;

public abstract class FederatedUDF implements Serializable {
	private static final long serialVersionUID = 799416525191257308L;
	
	private final long[] _inputIDs;
	
	protected FederatedUDF(long[] inIDs) {
		_inputIDs = inIDs;
	}
	
	public final long[] getInputIDs() {
		return _inputIDs;
	}
	
	/**
	 * Execute the user-defined function on a set of data objects
	 * (e.g., matrix objects, frame objects, or scalars), which are
	 * looked up by specified input IDs and passed in the same order.
	 * 
	 * Output data objects (potentially many) can be directly added
	 * to the passed execution context and its variable map.
	 * 
	 * @param ec execution context
	 * @param data one or many data objects
	 * @return federated response, with none or many output objects
	 */
	public abstract FederatedResponse execute(ExecutionContext ec, Data... data);
}
