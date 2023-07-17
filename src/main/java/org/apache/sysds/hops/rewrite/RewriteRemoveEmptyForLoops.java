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

import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.StatementBlock;

/**
 * Rule: Simplify program structure by removing empty for loops,
 * which may originate from the sequence of other rewrites like
 * dead-code-elimination.
 */
public class RewriteRemoveEmptyForLoops extends StatementBlockRewriteRule
{
	@Override
	public boolean createsSplitDag() {
		return false;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		ArrayList<StatementBlock> ret = new ArrayList<>();
		
		//prune last level blocks with empty hops
		if( sb instanceof ForStatementBlock 
				&& ((ForStatement)sb.getStatement(0)).getBody().isEmpty() ) {
			if( LOG.isDebugEnabled() )
				LOG.debug("Applied removeEmptyForLopp (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+").");
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
