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

import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.Hop.DataOpTypes;

/**
 * Rule: Eliminate for Transient Write DataHops to have no parents
 * Solution: Move parent edges of Transient Write Hop to parent of
 * its child 
 * Reason: Transient Write not being a root messes up
 * analysis for Lop's to Instruction translation (according to Amol)
 */
public class RewriteTransientWriteParentHandling extends HopRewriteRule
{

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) 
		throws HopsException
	{
		for (Hop h : roots) 
			rule_RehangTransientWriteParents(h, roots);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state)
		throws HopsException
	{
		// do nothing (does not apply to predicate hop DAGs) 	
		return root;
	}

	
	/**
	 * 
	 * @param hop
	 * @param sbHops
	 * @throws HopsException
	 */
	private void rule_RehangTransientWriteParents(Hop hop, ArrayList<Hop> sbHops) 
		throws HopsException 
	{
		if (hop instanceof DataOp && ((DataOp) hop).getDataOpType() == DataOpTypes.TRANSIENTWRITE
				&& !hop.getParent().isEmpty()) {

			// update parents inputs with data op input
			for (Hop p : hop.getParent()) {
				p.getInput().set(p.getInput().indexOf(hop), hop.getInput().get(0));
			}

			// update dataop input parent to add new parents except for
			// dataop itself
			hop.getInput().get(0).getParent().addAll(hop.getParent());

			// remove dataop parents
			hop.getParent().clear();

			// add dataop as root for this Hops DAG
			sbHops.add(hop);

			// do the same thing for my inputs (children)
			for (Hop hi : hop.getInput()) {
				rule_RehangTransientWriteParents(hi,sbHops);
			}
		}
	}
}
