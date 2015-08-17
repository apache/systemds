/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.instructions.cp.IntObject;

/**
 * This static task partitioner virtually iterates over the given FOR loop (from, to, incr),
 * creates iterations and group them to tasks according to a task size of numIterations/numWorkers. 
 * There, all tasks are equally sized.
 * 
 */
public class TaskPartitionerStatic extends TaskPartitionerFixedsize
{
	
	public TaskPartitionerStatic( long taskSize, int numThreads, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, iterVarName, fromVal, toVal, incrVal);
	
		_taskSize = _numIter / numThreads;
		_firstnPlus1 = (int)_numIter % numThreads;
	}	
}
