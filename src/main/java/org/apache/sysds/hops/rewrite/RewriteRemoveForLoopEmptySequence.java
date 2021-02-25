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

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Rule: Simplify program structure by removing (par)for statements iterating over
 * an empty sequence, i.e., (par)for-loops without a single iteration.
 */
public class RewriteRemoveForLoopEmptySequence extends StatementBlockRewriteRule {

	@Override
	public boolean createsSplitDag() {
		return false;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		ArrayList<StatementBlock> ret = new ArrayList<>();
		boolean apply = false;
		
		if( sb instanceof ForStatementBlock ) {
			ForStatementBlock fsb = (ForStatementBlock) sb;
			
			//consider rewrite if the increment was specified
			Hop from = fsb.getFromHops().getInput().get(0);
			Hop to = fsb.getToHops().getInput().get(0);
			Hop incr = (fsb.getIncrementHops() != null) ?
				fsb.getIncrementHops().getInput().get(0) : new LiteralOp(1);
			
			//consider rewrite if from, to, and incr are literal ops (constant values)
			if( from instanceof LiteralOp && to instanceof LiteralOp && incr instanceof LiteralOp ) {
				double dfrom = HopRewriteUtils.getDoubleValue((LiteralOp) from);
				double dto = HopRewriteUtils.getDoubleValue((LiteralOp) to);
				double dincr = HopRewriteUtils.getDoubleValue((LiteralOp) incr);
				if( dfrom > dto && dincr == 1 )
					dincr = -1;
				apply = UtilFunctions.getSeqLength(dfrom, dto, dincr, false) <= 0;
			}
		}
		
		if(apply) //remove for-loop (add nothing)
			LOG.debug("Applied removeForLoopEmptySequence (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+").");
		else //keep original sb (no for)
			ret.add( sb );
		
		return ret;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
		return sbs;
	}
}
