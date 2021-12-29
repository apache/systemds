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

package org.apache.sysds.hops.ipa;


import java.util.List;

import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.FunctionOp.FunctionType;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
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
import org.apache.sysds.runtime.controlprogram.Program;

/**
 * This rewrite applies static hop dag and statement block
 * rewrites such as constant folding and branch removal
 * in order to simplify statistic propagation.
 * 
 */
public class IPAPassReplaceEvalFunctionCalls extends IPAPass
{
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return fgraph.containsSecondOrderCall()
			&& OptimizerUtils.ALLOW_EVAL_FCALL_REPLACEMENT;
	}
	
	@Override
	public boolean rewriteProgram(DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes) {
		//note: we only replace eval calls that feed into a twrite/pwrite call
		//(i.e., after statement-block rewrites for splitting after val have been
		//applied) - this approach ensures that the requirements of fcalls are met
		
		// for each namespace, handle function statement blocks
		boolean ret = false;
		for (String namespaceKey : prog.getNamespaces().keySet())
			for (String fname : prog.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fsblock = prog.getFunctionStatementBlock(namespaceKey,fname);
				ret |= rewriteStatementBlock(prog, fsblock, fgraph);
			}
		
		// handle regular statement blocks in "main" method
		for(StatementBlock sb : prog.getStatementBlocks())
			ret |= rewriteStatementBlock(prog, sb, fgraph);
		
		return ret;
	}
	
	private static boolean rewriteStatementBlock(DMLProgram prog, StatementBlock sb, FunctionCallGraph fgraph) {
		boolean ret = false;
		if (sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock csb : fstmt.getBody())
				ret |= rewriteStatementBlock(prog, csb, fgraph);
		}
		else if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			for (StatementBlock csb : wstmt.getBody())
				ret |= rewriteStatementBlock(prog, csb, fgraph);
		}
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			for (StatementBlock csb : istmt.getIfBody())
				ret |= rewriteStatementBlock(prog, csb, fgraph);
			for (StatementBlock csb : istmt.getElseBody())
				ret |= rewriteStatementBlock(prog, csb, fgraph);
		}
		else if (sb instanceof ForStatementBlock) { //incl parfor
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			for (StatementBlock csb : fstmt.getBody())
				ret |= rewriteStatementBlock(prog, csb, fgraph);
		}
		else { //generic (last-level)
			ret |= checkAndReplaceEvalFunctionCall(prog, sb, fgraph);
		}
		return ret;
	}
	
	private static boolean checkAndReplaceEvalFunctionCall(DMLProgram prog, StatementBlock sb, FunctionCallGraph fgraph) {
		if( sb.getHops() == null )
			return false;
		
		List<Hop> roots = sb.getHops();
		boolean ret = false;
		for( int i=0; i<roots.size(); i++ ) {
			Hop root = roots.get(i);
			if( HopRewriteUtils.isData(root, OpOpData.TRANSIENTWRITE, OpOpData.PERSISTENTWRITE)
				&& HopRewriteUtils.isNary(root.getInput(0), OpOpN.EVAL)
				&& root.getInput(0).getInput(0) instanceof LiteralOp //constant name
				&& root.getInput(0).getParent().size() == 1)
			{
				Hop eval = root.getInput(0);
				String outvar = ((DataOp)root).getName();
				
				//get function name and namespace
				String fname = ((LiteralOp)eval.getInput(0)).getStringValue();
				String fnamespace = prog.getDefaultFunctionDictionary().containsFunction(fname) ?
					DMLProgram.DEFAULT_NAMESPACE : DMLProgram.BUILTIN_NAMESPACE;
				if( fname.contains(Program.KEY_DELIM) ) {
					String[] fparts = DMLProgram.splitFunctionKey(fname);
					fnamespace = fparts[0];
					fname = fparts[1];
				}
				fname = fnamespace.equals(DMLProgram.BUILTIN_NAMESPACE) ?
					Builtins.getInternalFName(fname, eval.getInput(1).getDataType()) : fname;
				
				//obtain functions and abort if inputs passed via list or output not a matrix
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fnamespace, fname);
				FunctionStatement fstmt = fsb!=null ? (FunctionStatement)fsb.getStatement(0) : null;
				if( eval.getInput().size() > 1 && eval.getInput(1).getDataType().isList()
					&& (fstmt==null || !fstmt.getInputParams().get(0).getDataType().isList())) {
					LOG.warn("IPA: eval("+fnamespace+"::"+fname+") "
						+ "applicable for replacement, but list inputs not yet supported.");
					continue;
				}
				if( eval.getDataType().isList() ) {
					LOG.warn("IPA: eval("+fnamespace+"::"+fname+") "
						+ "applicable for replacement, but list output not yet supported.");
					continue;
				}
				if( fstmt.getOutputParams().size() != 1 || !fstmt.getOutputParams().get(0).getDataType().isMatrix() ) {
					LOG.warn("IPA: eval("+fnamespace+"::"+fname+") "
						+ "applicable for replacement, but function output is not a matrix.");
					continue;
				}
				
				//construct direct function call
				FunctionOp fop = new FunctionOp(FunctionType.DML, fnamespace, fname,
					fstmt.getInputParamNames(), eval.getInput().subList(1, eval.getInput().size()),
					new String[]{outvar}, true);
				HopRewriteUtils.copyLineNumbers(eval, fop);
				HopRewriteUtils.removeAllChildReferences(eval);
				roots.set(i, fop); //replaced
				ret = true;
			}
		}
		return ret;
	}
}
