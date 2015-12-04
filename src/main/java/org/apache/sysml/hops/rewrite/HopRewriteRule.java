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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;

/**
 * Base class for all hop rewrites in order to enable generic
 * application of all rules.
 * 
 */
public abstract class HopRewriteRule 
{


	protected static final Log LOG = LogFactory.getLog(HopRewriteRule.class.getName());
		
	/**
	 * Handle a generic (last-level) hop DAG with multiple roots.
	 * 
	 * @param roots
	 * @param state
	 * @return
	 * @throws HopsException
	 */
	public abstract ArrayList<Hop> rewriteHopDAGs( ArrayList<Hop> roots, ProgramRewriteStatus state ) 
		throws HopsException;
	
	/**
	 * Handle a predicate hop DAG with exactly one root.
	 * 
	 * @param root
	 * @param state
	 * @return
	 * @throws HopsException
	 */
	public abstract Hop rewriteHopDAG( Hop root, ProgramRewriteStatus state ) 
		throws HopsException;
}
