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
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.utils.Explain;

/**
 * This class allows to check hop dags for validity, e.g., parent-child linking.
 * It purpose is solely for debugging purposes (enabled in ProgramRewriter).
 * 
 */
public class HopDagValidator 
{
	private static final Log LOG = LogFactory.getLog(HopDagValidator.class.getName());
	
	public static void validateHopDag(ArrayList<Hop> roots)
		throws HopsException
	{
		if( roots == null )
			return;
		try {
			Hop.resetVisitStatus(roots);
//			for( Hop hop : roots )
//				verifyNoVisit(hop);
			ValidatorState state = new ValidatorState();
			for( Hop hop : roots )
				rValidateHop(hop, state);
		}
		catch(HopsException ex) {
			try {
				LOG.error( "\n"+Explain.explainHops(roots) );
			}catch(DMLRuntimeException e){}
			throw ex;
		}
	}
	
	public static void validateHopDag(Hop root) 
		throws HopsException
	{
		if( root == null )
			return;
		try {
			root.resetVisitStatus();
//			verifyNoVisit(root);
			ValidatorState state = new ValidatorState();
			rValidateHop(root, state);
		}
		catch(HopsException ex) {
			try {
				LOG.error( "\n"+Explain.explain(root) );
			}catch(DMLRuntimeException e){}
			throw ex;
		}
	}

//	private static void verifyNoVisit(Hop hop) throws HopsException
//	{
//		HopsException.check(!hop.isVisited(), "Expected Hop should not be visited after clearing: %s", hop);
//		for (Hop child : hop.getInput())
//			verifyNoVisit(child);
//	}

	private static class ValidatorState {
		final Set<Hop> seen = Collections.newSetFromMap(new IdentityHashMap<Hop,Boolean>());
	}
	
	private static void rValidateHop(final Hop hop, final ValidatorState state)
		throws HopsException
	{
		boolean seen = !state.seen.add(hop);
		// seen ==> should be visited // true, true
		// not seen ==> should not be visited // false, false
		HopsException.check(seen == hop.isVisited(),
				"Hop seen previously is %b but hop visited previously is %b for hop %d",
				seen, !seen, hop.getHopID());
		if (seen) return;
		
		//check parent linking
		for( Hop parent : hop.getParent() )
			HopsException.check(parent.getInput().contains(hop),
					"Hop id=%d not properly linked to its parent pid=%d %s",
					hop.getHopID(), parent.getHopID(), parent.getClass().getName());
		
		//check child linking
		for( Hop child : hop.getInput() )
			HopsException.check(child.getParent().contains(hop),
					"Hop id=%d not properly linked to its child cid=%d %s",
					hop.getHopID(), child.getHopID(), child.getClass().getName());
		
		//check empty childs
		if( hop.getInput().isEmpty() )
			HopsException.check(hop instanceof DataOp || hop instanceof LiteralOp,
					"Hop id=%d is not a dataop/literal but has no children", hop.getHopID());
		
		//recursively process childs
		for( Hop child : hop.getInput() )
			rValidateHop(child, state);

//		hop.dimsKnown()
		
		hop.setVisited();
	}
}
