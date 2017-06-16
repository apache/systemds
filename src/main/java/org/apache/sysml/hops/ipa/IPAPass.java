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

package org.apache.sysml.hops.ipa;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.parser.DMLProgram;

/**
 * Base class for all IPA passes.
 */
public abstract class IPAPass 
{
	protected static final Log LOG = LogFactory.getLog(IPAPass.class.getName());
    
	/**
	 * Indicates if an IPA pass is applicable for the current
	 * configuration such as global flags or the chosen execution 
	 * mode (e.g., hybrid_spark).
	 * 
	 * @return true if applicable.
	 */
	public abstract boolean isApplicable();
	
	/**
	 * Rewrites the given program or its functions in place,
	 * with access to the read-only function call graph.
	 * 
	 * @param prog dml program
	 * @param fgraph function call graph
	 * @param fcallSizes function call size infos
	 * @throws HopsException
	 */
	public abstract void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
		throws HopsException;
}
