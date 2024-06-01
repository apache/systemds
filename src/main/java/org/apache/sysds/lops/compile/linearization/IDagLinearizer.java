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

package org.apache.sysds.lops.compile.linearization;

import java.util.List;

import org.apache.sysds.lops.Lop;

/**
 * An interface for the linearization algorithms that order the DAG nodes into a 
 * sequence of instructions to execute.
 */
public abstract class IDagLinearizer {
	/**
	 * Linearized a DAG of lops into a sequence of lops that preserves all
	 * data dependencies.
	 * 
	 * @param v roots (outputs) of a DAG of lops
	 * @return list of lops (input, inner, and outputs)
	 */
	public abstract List<Lop> linearize(List<Lop> v);
}
