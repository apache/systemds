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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.utils.Explain;

public class SparsityOptimizer extends HopRewriteRule {
	protected static final Log LOG = LogFactory.getLog(SparsityOptimizer.class.getName());

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		ArrayList<Hop> ret = new ArrayList<Hop>();
		for(Hop rootHop : roots)
			ret.add(rewriteHopDAG(rootHop, state));
		return ret;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if(root.isVisited())
			return root;

		if(root instanceof DataOp && ((DataOp)root).isWrite() && root.getInput().size() > 0 &&
			HopRewriteUtils.isMatrixMultiply(root.getInput(0)) && isMMChain(root.getInput()))
			return optimizeMMChain(root);
		return root;
	}

	private static boolean isMMChain(List<Hop> hopList) {
		boolean ret = true;
		ret &= (hopList != null && hopList.size() > 0);
		for(Hop hop : hopList)
			ret &= isMMChain(hop);
		return ret;
	}

	private static boolean isMMChain(Hop hop) {
		if(HopRewriteUtils.isMatrixMultiply(hop))
			return isMMChain(hop.getInput());
		else if(hop instanceof DataOp)
			return (((DataOp)hop).getOp() == OpOpData.TRANSIENTREAD);
		return false;
	}

	public Hop optimizeMMChain(Hop rootHop) {
		List<Hop> mmChain = new ArrayList<>();
		List<Hop> mmOperators = new ArrayList<>();
		Hop mmHop = rootHop.getInput(0);
		prepareMMChain(mmHop, mmChain, mmOperators);

		Hop newMMHop = sculptMMChain(mmHop, mmChain, mmOperators);
		if(newMMHop != null) {
			HopRewriteUtils.replaceChildReference(rootHop, mmHop, newMMHop);
			mmHop = newMMHop;
		}

		// print the matrix multiplication chain
		if(LOG.isTraceEnabled()) {
			LOG.trace("SparsityOptimizer: Identified and optimized MM Chain: ");
			for(Hop h : mmChain) {
				logTraceHop(h, 1);
			}
		}

		return rootHop;
	}

	public void prepareMMChain(Hop root, List<Hop> mmChain, List<Hop> mmOperators) {
		root.resetVisitStatus();
		// initialize mmChain with inputs of root hop
		mmOperators.add(root);
		for( Hop hi : root.getInput() )
			mmChain.add(hi);

		// expand each hop to find the entire matrix multiplication chain
		int counter = 0;
		while(counter < mmChain.size()) {
			boolean expandable = true;
			Hop hop = mmChain.get(counter);

			/*
			 * Check if mmChain[i] is expandable: 
			 * 1) It must be MATMULT
			 * 2) It must not have been visited already
			 *	(one MATMULT should get expanded only in one chain)
			 * 3) Its output should not be used in multiple places
			 *	(either within chain or outside the chain)
			 */
			expandable &= HopRewriteUtils.isMatrixMultiply(hop) && !((AggBinaryOp)hop).hasLeftPMInput();
			expandable &= !hop.isVisited();
			expandable &= !(hop.getParent().size() > 1);

			hop.setVisited();

			if(!expandable) {
				counter += 1;
			}
			else {
				List<Hop> tempList = mmChain.get(counter).getInput();
				// add current operator to mmOperators and its input nodes to mm chain
				mmOperators.add(mmChain.get(counter));
				mmChain.set(counter, tempList.get(0));
				mmChain.add(counter + 1, tempList.get(1));
			}
		}
	}

	// Optimize the matrix multiplication chain by rewriting it
	// This method can modify the mm chain completely, even re-creating it from scratch is allowed.
	// NOTE: This method should return the new mm hop when re-creating it from scratch.
	private Hop sculptMMChain(Hop mmHop, List<Hop> mmChain, List<Hop> mmOperators) {
		(new RewriteMatrixMultChainOptimizationSparse()).optimizeMMChain(mmHop, mmChain, mmOperators, null);
		return null;
	}

	protected void logTraceHop(Hop hop, int level) {
		if(LOG.isTraceEnabled()) {
			String offset = Explain.getIdentation(level);
			LOG.trace(offset + "Hop " + hop.getName() +
				"(" + hop.getClass().getSimpleName() + ", " + hop.getHopID() + ")" + " " +
				hop.getDim1() + "x" + hop.getDim2() + " sp" + hop.getSparsity());
		}
	}
}
