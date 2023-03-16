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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;

/**
 * Rule: Inject checkpointing on reading in data in all cases where the operand is used in more than one operation.
 */
public class RewriteInjectSparkPReadCheckpointing extends HopRewriteRule {
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if(!OptimizerUtils.isSparkExecutionMode())
			return roots;

		if(roots == null)
			return null;

		// top-level hops never modified
		for(Hop h : roots)
			rInjectCheckpointAfterPRead(h);

		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		// not applicable to predicates (we do not allow persistent reads there)
		return root;
	}

	private void rInjectCheckpointAfterPRead(Hop hop) {
		if(hop.isVisited())
			return;
		// Inject checkpoints after persistent reads (for binary matrices only), or
		// after reblocks that cause expensive shuffling. However, carefully avoid
		// unnecessary frame checkpoints (e.g., binary data or csv that do not cause
		// shuffle) in order to prevent excessive garbage collection due to possibly
		// many small string objects. An alternative would be serialized caching.
		boolean isMatrix = hop.getDataType().isMatrix();
		boolean isPRead = hop instanceof DataOp && ((DataOp) hop).getOp() == OpOpData.PERSISTENTREAD;
		boolean isFrameException = hop.getDataType().isFrame() && isPRead && !((DataOp) hop).getFileFormat().isIJV();

		// if the only operation performed is an action then do not add chkpoint
		if((isMatrix && isPRead) || (hop.requiresReblock() && !isFrameException)) {
			boolean isActionOnly = isActionOnly(hop, hop.getParent());
			// make given hop for checkpointing (w/ default storage level)
			// note: we do not recursively process children here in order to prevent unnecessary checkpoints
			
			if(!isActionOnly)
				hop.setRequiresCheckpoint(true);
			
		}
		else {
			if(hop.getInput() != null) {
				// process all children (prevent concurrent modification by index access)
				for(int i = 0; i < hop.getInput().size(); i++)
					rInjectCheckpointAfterPRead(hop.getInput().get(i));
			}
		}

		hop.setVisited();
	}

	private boolean isActionOnly(Hop hop, List<Hop> parents) {
		// if the number of consumers of this hop is equal to 1 and no more
		// then do not cache block unless that one operation is transient write
		if(parents.size() == 1) {
			return !( parents.get(0) instanceof DataOp && //
				((DataOp) parents.get(0)).getOp() == OpOpData.TRANSIENTWRITE);
		}
		else
			return false;

	}
}
