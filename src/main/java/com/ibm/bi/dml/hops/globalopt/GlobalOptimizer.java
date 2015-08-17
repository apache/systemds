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

package com.ibm.bi.dml.hops.globalopt;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFGraph;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;

/**
 * Super class for all optimizers (e.g., transformation-based, and enumeration-based)
 * 
 */
public abstract class GlobalOptimizer 
{
	
	/**
	 * Core optimizer call, to be implemented by an instance of a global
	 * data flow optimizer.
	 * 
	 * @param prog
	 * @param rtprog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws  
	 */
	public abstract GDFGraph optimize( GDFGraph gdfgraph, Summary summary )
		throws DMLRuntimeException, DMLUnsupportedOperationException, HopsException, LopsException;
}
