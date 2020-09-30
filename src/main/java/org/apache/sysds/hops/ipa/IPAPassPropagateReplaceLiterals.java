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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;

/**
 * This rewrite propagates and replaces literals into functions
 * in order to enable subsequent rewrites such as branch removal.
 * 
 */
public class IPAPassPropagateReplaceLiterals extends IPAPass
{
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return InterProceduralAnalysis.PROPAGATE_SCALAR_LITERALS;
	}
	
	@Override
	public boolean rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
	{
		//step 1: propagate final literals across main program
		rReplaceLiterals(prog.getStatementBlocks(), prog, fgraph, fcallSizes);
		
		//step 2: propagate literals into functions
		for( String fkey : fgraph.getReachableFunctions() ) {
			List<FunctionOp> flist = fgraph.getFunctionCalls(fkey);
			if( flist.isEmpty() ) //robustness removed functions
				continue;
			FunctionOp first = flist.get(0);
			
			//propagate and replace amenable literals into function
			if( fcallSizes.hasSafeLiterals(fkey) ) {
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fkey);
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				ArrayList<DataIdentifier> finputs = fstmt.getInputParams();
				
				//populate call vars with amenable literals
				LocalVariableMap callVars = new LocalVariableMap();
				for( int j=0; j<finputs.size(); j++ )
					if( fcallSizes.isSafeLiteral(fkey, j) ) {
						LiteralOp lit = (LiteralOp) first.getInput().get(j);
						String varname = (first.getInputVariableNames()!=null) ?
							first.getInputVariableNames()[j] : finputs.get(j).getName();
						callVars.put(varname, ScalarObjectFactory
							.createScalarObject(lit.getValueType(), lit));
					}
				
				//propagate constant function arguments into function
				for( StatementBlock sb : fstmt.getBody() )
					rReplaceLiterals(sb, callVars);
				
				//propagate final literals across function
				rReplaceLiterals(fstmt.getBody(), prog, fgraph, fcallSizes);
			}
		}
		return false;
	}
	
	private void rReplaceLiterals(List<StatementBlock> sbs, DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes) {
		LocalVariableMap constants = new LocalVariableMap();
		//propagate final literals across statement blocks
		for( StatementBlock sb : sbs ) {
			//delete update constant variables
			constants.removeAllIn(sb.variablesUpdated().getVariableNames());
			//literal replacement
			rReplaceLiterals(sb, constants);
			//extract literal assignments
			if( HopRewriteUtils.isLastLevelStatementBlock(sb) ) {
				for( Hop root : sb.getHops() )
					if( HopRewriteUtils.isData(root, OpOpData.TRANSIENTWRITE)
						&& root.getInput().get(0) instanceof LiteralOp) {
						constants.put(root.getName(), ScalarObjectFactory
							.createScalarObject((LiteralOp)root.getInput().get(0)));
					}
			}
		}
	}
	
	private void rReplaceLiterals(StatementBlock sb, LocalVariableMap constants) 
	{
		//remove updated literals
		for( String varname : sb.variablesUpdated().getVariableNames() )
			if( constants.keySet().contains(varname) )
				constants.remove(varname);
		
		//propagate and replace literals
		if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement ws = (WhileStatement)sb.getStatement(0);
			replaceLiterals(wsb.getPredicateHops(), constants);
			for (StatementBlock current : ws.getBody())
				rReplaceLiterals(current, constants);
		} 
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement ifs = (IfStatement) sb.getStatement(0);
			replaceLiterals(isb.getPredicateHops(), constants);
			for (StatementBlock current : ifs.getIfBody())
				rReplaceLiterals(current, constants);
			for (StatementBlock current : ifs.getElseBody())
				rReplaceLiterals(current, constants);
		} 
		else if (sb instanceof ForStatementBlock) {
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fs = (ForStatement)sb.getStatement(0);
			replaceLiterals(fsb.getFromHops(), constants);
			replaceLiterals(fsb.getToHops(), constants);
			replaceLiterals(fsb.getIncrementHops(), constants);
			for (StatementBlock current : fs.getBody())
				rReplaceLiterals(current, constants);
		}
		else {
			replaceLiterals(sb.getHops(), constants);
		}
	}
	
	private static void replaceLiterals(ArrayList<Hop> roots, LocalVariableMap constants) {
		if( roots == null )
			return;
		
		try {
			Hop.resetVisitStatus(roots);
			for( Hop root : roots )
				Recompiler.rReplaceLiterals(root, constants, true);
			Hop.resetVisitStatus(roots);
		}
		catch(Exception ex) {
			throw new HopsException(ex);
		}
	}
	
	private static void replaceLiterals(Hop root, LocalVariableMap constants) {
		if( root == null )
			return;
		
		try {
			root.resetVisitStatus();
			Recompiler.rReplaceLiterals(root, constants, true);
			root.resetVisitStatus();
		}
		catch(Exception ex) {
			throw new HopsException(ex);
		}
	}
}
