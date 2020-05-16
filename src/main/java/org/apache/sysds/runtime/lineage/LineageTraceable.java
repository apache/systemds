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

package org.apache.sysds.runtime.lineage;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;

public interface LineageTraceable
{
	/**
	 * Obtain meta data on number of outputs and thus, number of lineage items.
	 * 
	 * @return true iff instruction has a single output
	 */
	public default boolean hasSingleLineage() {
		return true;
	}
	
	/**
	 * Obtain lineage trace of an instruction with a single output.
	 * 
	 * @param ec execution context w/ live variables
	 * @return pair of (output variable name, output lineage item)
	 */
	public Pair<String,LineageItem> getLineageItem(ExecutionContext ec);
	
	/**
	 * Obtain lineage trace of an instruction with multiple outputs.
	 * 
	 * @param ec execution context w/ live variables
	 * @return pairs of (output variable name, output lineage item)
	 */
	public default Pair<String,LineageItem>[] getLineageItems(ExecutionContext ec) {
		throw new DMLRuntimeException("Unsupported call for instruction with single output.");
	}
}
