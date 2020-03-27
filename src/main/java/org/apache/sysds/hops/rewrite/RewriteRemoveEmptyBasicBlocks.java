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

import org.apache.sysds.parser.StatementBlock;

/**
 * Rule: Simplify program structure by removing empty last-level blocks,
 * which may originate from the original program or due to a sequence of
 * rewrites (e.g., checkpoint injection and subsequent IPA).
 */
public class RewriteRemoveEmptyBasicBlocks extends StatementBlockRewriteRule
{
	@Override
	public boolean createsSplitDag() {
		return false;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		ArrayList<StatementBlock> ret = new ArrayList<>();
		
		//prune last level blocks with empty hops
		if( HopRewriteUtils.isLastLevelStatementBlock(sb)
			&& (sb.getHops() == null || sb.getHops().isEmpty()) ) {
			if( LOG.isDebugEnabled() )
				LOG.debug("Applied removeEmptyBasicBlocks (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+").");
		}
		else //keep original sb
			ret.add( sb );
		
		return ret;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus sate) {
		return sbs;
	}
}
