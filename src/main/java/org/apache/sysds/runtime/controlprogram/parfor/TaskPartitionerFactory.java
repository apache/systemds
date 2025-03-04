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

package org.apache.sysds.runtime.controlprogram.parfor;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import org.apache.sysds.runtime.instructions.cp.IntObject;

public class TaskPartitionerFactory 
{
	public static TaskPartitioner createTaskPartitioner(PTaskPartitioner type, 
		IntObject from, IntObject to, IntObject incr, long taskSize, int numThreads, String iterPredVar) 
	{
		switch( type ) {
			case FIXED:
				return new TaskPartitionerFixedsize(taskSize, iterPredVar, from, to, incr);
			case NAIVE:
				return new TaskPartitionerNaive(taskSize, iterPredVar, from, to, incr);
			case STATIC:
				return new TaskPartitionerStatic(taskSize, numThreads, iterPredVar, from, to, incr);
			case FACTORING:
				return new TaskPartitionerFactoring(taskSize, numThreads, iterPredVar, from, to, incr);
			case FACTORING_CMIN:
				return new TaskPartitionerFactoringCmin(taskSize,
					numThreads, taskSize, iterPredVar, from, to, incr);
			case FACTORING_CMAX:
				return new TaskPartitionerFactoringCmax(taskSize,
					numThreads, taskSize, iterPredVar, from, to, incr);
			default:
				throw new DMLRuntimeException("Undefined task partitioner: '"+type+"'.");
		}
	}
}
