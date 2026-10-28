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
import java.util.Arrays;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.shaded.org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.utils.Explain;

public class SparsityOptimizer extends StatementBlockRewriteRule {
	protected static final Log LOG = LogFactory.getLog(SparsityOptimizer.class.getName());

	@Override
	public boolean createsSplitDag() {
		return false;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		// optimize the matrix multiplication chain if it is isolated in its own statement block
		if(isMMChain(sb))
			sb = optimizeMMChain(sb);
		return Arrays.asList(sb);
	}

	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
		return sbs;
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

	private static boolean allChildsAreMatrixMultiply(Hop hop) {
		return hop.getInput().stream().allMatch(childHop -> HopRewriteUtils.isMatrixMultiply(childHop));
	}

	private static boolean isMMChain(StatementBlock sb) {
		boolean ret = true;
		for(Hop hop : sb.getHops()) {
			ret &= allChildsAreMatrixMultiply(hop) && hop instanceof DataOp;
			ret &= (hop.getInput().size() > 0);
			if(!ret)
				return false;
			ret &= isMMChain(hop.getInput());
		}
		return ret;
	}

	public StatementBlock optimizeMMChain(StatementBlock sb) {
		for(Hop rootHop : sb.getHops()) {
			List<Hop> mmChain = new ArrayList<>();
			List<Hop> mmOperators = new ArrayList<>();
			if(rootHop instanceof DataOp && ((DataOp)rootHop).isWrite())
				prepareMMChain(rootHop.getInput(0), mmChain, mmOperators);
			else
				throw new DMLRuntimeException("This case should not occur and is not considered.");

			throw new NotImplementedException("TODO: Optimize the MM Chain inside the isolated statement block");
		}
		return sb;
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

		// print the matrix multiplication chain
		if(LOG.isTraceEnabled()) {
			LOG.trace("Identified MM Chain: ");
			for(Hop h : mmChain) {
				logTraceHop(h, 1);
			}
		}
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
