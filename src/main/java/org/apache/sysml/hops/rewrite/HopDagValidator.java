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
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.utils.Explain;

import static org.apache.sysml.hops.HopsException.check;

/**
 * This class allows to check hop dags for validity, e.g., parent-child linking.
 * Use it for debugging (enabled in {@link ProgramRewriter}).
 */
public class HopDagValidator {
	private static final Log LOG = LogFactory.getLog(HopDagValidator.class.getName());
	
	public static void validateHopDag(final ArrayList<Hop> roots) throws HopsException {
		if( roots == null )
			return;
		try {
			Hop.resetVisitStatus(roots);
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
	
	public static void validateHopDag(final Hop root) throws HopsException {
		if( root == null )
			return;
		try {
			root.resetVisitStatus();
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

	private static class ValidatorState {
		final Set<Long> seen = new HashSet<>();
	}
	
	private static void rValidateHop(final Hop hop, final ValidatorState state) throws HopsException {
		final long id = hop.getHopID();

		final boolean seen = !state.seen.add(id);
		check(seen == hop.isVisited(),
				"Hop seen previously is %b but hop visited previously is %b for hop %d",
				seen, !seen, id);
		if (seen) return; // we saw the Hop previously, no need to re-validate
		
		//check parent linking
		for( Hop parent : hop.getParent() )
			check(parent.getInput().contains(hop),
					"Hop id=%d not properly linked to its parent pid=%d %s",
					id, parent.getHopID(), parent.getClass().getName());

		final ArrayList<Hop> input = hop.getInput();
		final int arity = hop.getArity();
		final Expression.DataType dt = hop.getDataType();
		final Expression.ValueType vt = hop.getValueType();

		//check child linking
		for( Hop child : input )
			check(child.getParent().contains(hop),
					"Hop id=%d not properly linked to its child cid=%d %s",
					id, child.getHopID(), child.getClass().getName());

		//check empty children (other variable-length Hops must have at least one child)
		if( input.isEmpty() )
			check(hop instanceof DataOp || hop instanceof LiteralOp,
					"Hop id=%d is not a dataop/literal but has no children", id);
		// check arity matches number of children
		if (arity != -1) // for Hops with known, fixed arity
			check(input.size() == arity,
					"Hop id=%d has arity %d but has size %d", id, arity, input.size());

		// check Matrix data type Hops must have Double Value type
		if (dt == Expression.DataType.MATRIX )
			check(vt == Expression.ValueType.DOUBLE,
				"Hop id=%d has Matrix type but Value Type %s is not DOUBLE", id, vt);

		//recursively process children
		for( Hop child : input )
			rValidateHop(child, state);

		hop.setVisited();
	}
}
