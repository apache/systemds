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

package org.apache.sysml.runtime.controlprogram.parfor;

import org.apache.sysml.runtime.instructions.cp.IntObject;

/**
 * Factoring with maximum constraint (e.g., if LIX matrix out-of-core and we need
 * to bound the maximum number of iterations per map task -> memory bounds) 
 */
public class TaskPartitionerFactoringCmax extends TaskPartitionerFactoring
{
	
	protected long _constraint = -1;
	
	public TaskPartitionerFactoringCmax( long taskSize, int numThreads, long constraint, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, numThreads, iterVarName, fromVal, toVal, incrVal);
		
		_constraint = constraint;
	}

	@Override
	protected long determineNextBatchSize(long R, int P) 
	{
		int x = 2;
		long K = (long)Math.ceil((double)R / ( x * P )); //NOTE: round creates more tasks
		
		if( K > _constraint ) //account for rounding errors
			K = _constraint;
		
		return K;
	}
	
}
