/* Licensed to the Apache Software Foundation (ASF) under one
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

package org.apache.sysds.lops.rewrite;

import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;

import java.util.ArrayList;
import java.util.List;

public class LopRewriter
{
	private ArrayList<LopRewriteRule> _lopSBRuleSet = null;

	public LopRewriter() {
		_lopSBRuleSet = new ArrayList<>();
		// Add rewrite rules (single and multi-statement block)
		_lopSBRuleSet.add(new RewriteAddPrefetchLop());
		_lopSBRuleSet.add(new RewriteAddBroadcastLop());
		_lopSBRuleSet.add(new RewriteAddChkpointLop());
		// TODO: A rewrite pass to remove less effective chkpoints
		// Last rewrite to reset Lop IDs in a depth-first manner
		_lopSBRuleSet.add(new RewriteFixIDs());
	}

	public void rewriteProgramLopDAGs(DMLProgram dmlp) {
		for (String namespaceKey : dmlp.getNamespaces().keySet())
			// for each namespace, handle function statement blocks
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				rewriteLopDAGsFunction(fsblock);
			}

		if (!_lopSBRuleSet.isEmpty()) {
			ArrayList<StatementBlock> sbs = rRewriteLops(dmlp.getStatementBlocks());
			dmlp.setStatementBlocks(sbs);
		}
	}

	public void rewriteLopDAGsFunction(FunctionStatementBlock fsb) {
		if( !_lopSBRuleSet.isEmpty() )
			rRewriteLop(fsb);
	}

	public ArrayList<Lop> rewriteLopDAG(ArrayList<Lop> lops) {
		StatementBlock sb = new StatementBlock();
		sb.setLops(lops);
		return rRewriteLop(sb).get(0).getLops();
	}

	public ArrayList<StatementBlock> rRewriteLops(ArrayList<StatementBlock> sbs) {
		// Apply rewrite rules to the lops of the list of statement blocks
		List<StatementBlock> tmp = sbs;
		for(LopRewriteRule r : _lopSBRuleSet)
			tmp = r.rewriteLOPinStatementBlocks(tmp);

		// Recursively rewrite lops in statement blocks
		List<StatementBlock> tmp2 = new ArrayList<>();
		for( StatementBlock sb : tmp )
			tmp2.addAll(rRewriteLop(sb));

		// Prepare output list
		sbs.clear();
		sbs.addAll(tmp2);
		return sbs;
	}

	public ArrayList<StatementBlock> rRewriteLop(StatementBlock sb) {
		ArrayList<StatementBlock> ret = new ArrayList<>();
		ret.add(sb);

		// Recursive invocation
		if (sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			fstmt.setBody(rRewriteLops(fstmt.getBody()));
		}
		else if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			wstmt.setBody(rRewriteLops(wstmt.getBody()));
		}
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			istmt.setIfBody(rRewriteLops(istmt.getIfBody()));
			istmt.setElseBody(rRewriteLops(istmt.getElseBody()));
		}
		else if (sb instanceof ForStatementBlock) { //incl parfor
			//TODO: parfor statement blocks
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fstmt.setBody(rRewriteLops(fstmt.getBody()));
		}

		// Apply rewrite rules to individual statement blocks
		for(LopRewriteRule r : _lopSBRuleSet) {
			ArrayList<StatementBlock> tmp = new ArrayList<>();
			for( StatementBlock sbc : ret )
				tmp.addAll( r.rewriteLOPinStatementBlock(sbc) );

			// Take over set of rewritten sbs
			ret.clear();
			ret.addAll(tmp);
		}

		return ret;
	}
}
