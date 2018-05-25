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

package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.sysml.runtime.instructions.cp.Data;

public abstract class ParamServer {

	public static final String GLOBAL_PREFIX = "global_";
	public static final String GRADIENTS_PREFIX = "gradients_";
	public static final String RESULT_MODEL = "result_model";

	private Map<String, Data> _params;

	ParamServer() {
		this._params = Collections.synchronizedMap(new HashMap<>());
	}

	public Map<String, Data> getParams() {
		return _params;
	}

	public abstract void push(String key, Data value);

	public abstract Data pull(String key);
}
