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

package org.apache.sysml.runtime.instructions.flink.functions;


import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

import java.util.HashMap;

public abstract class RichFlatMapBroadcastFunction<IN, OUT> extends RichFlatMapFunction<IN, OUT> {
	protected HashMap<Long, HashMap<Long, MatrixBlock>> _pbc = null;

	@Override
	public void open(Configuration parameters) throws Exception {
		_pbc = BroadcastFunction.open(getRuntimeContext(), _pbc);
	}

	@Override
	public void close() throws Exception {
		_pbc = BroadcastFunction.close(_pbc);
	}
}
