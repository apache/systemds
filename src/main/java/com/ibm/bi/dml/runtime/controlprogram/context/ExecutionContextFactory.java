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

package com.ibm.bi.dml.runtime.controlprogram.context;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.Program;

public class ExecutionContextFactory 
{
	
	/**
	 * 
	 * @return
	 */
	public static ExecutionContext createContext()
	{
		return createContext( null );
	}
	
	public static ExecutionContext createContext( Program prog )
	{
		return createContext(true, prog);
	}
	
	/**
	 * 
	 * @param platform
	 * @return
	 */
	public static ExecutionContext createContext( boolean allocateVars, Program prog )
	{
		ExecutionContext ec = null;
		
		switch( DMLScript.rtplatform )
		{
			case SINGLE_NODE:
			case HADOOP:
			case HYBRID:
				ec = new ExecutionContext(allocateVars, prog);
				break;
				
			case SPARK:
			case HYBRID_SPARK:
				ec = new SparkExecutionContext(allocateVars, prog);
				break;
		}
		
		return ec;
	}
}
