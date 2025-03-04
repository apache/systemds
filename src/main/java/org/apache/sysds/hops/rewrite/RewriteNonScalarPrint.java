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

package org.apache.sysds.hops.rewrite;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.Hop;

import java.util.ArrayList;
import java.util.LinkedHashMap;

public class RewriteNonScalarPrint extends HopRewriteRule
{
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if(roots != null) {
			for(Hop h : roots)
				rewriteHopDAG(h, state);
		}
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if(root != null)
			rewritePrintNonScalar(root);
		return root;
	}

	private void rewritePrintNonScalar(Hop hop) {
		// Check if hop is a unary PRINT op
		if(HopRewriteUtils.isUnary(hop, Types.OpOp1.PRINT)) {
			// Check if child is non-scalar
			Hop child = hop.getInput().get(0);
			
			if(!child.getDataType().isScalar()) {
				LinkedHashMap<String, Hop> args = new LinkedHashMap<>();
				args.put("target", child);

				// create toString hop
				Hop toStringOp = HopRewriteUtils.createParameterizedBuiltinOp(child, args,
					Types.ParamBuiltinOp.TOSTRING);

				// Replace child with toString in hop
				HopRewriteUtils.replaceChildReference(hop, child, toStringOp, 0);
				LOG.debug("Applied non-scalar print rewrite on hop ID = " + hop.getHopID());
			}
		}
	}

}
