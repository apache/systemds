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

package org.apache.sysml.hops.ipa;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.StatementBlock;

/**
 * This rewrite inlines single statement block functions, which have fewer 
 * operations than an internal threshold. Function inlining happens during 
 * validate but after rewrites such as constant folding and branch removal 
 * there are additional opportunities.
 * 
 */
public class IPAPassInlineFunctions extends IPAPass
{
	@Override
	public boolean isApplicable() {
		return InterProceduralAnalysis.INLINING_MAX_NUM_OPS > 0;
	}
	
	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
		throws HopsException
	{
		//NOTE: we inline single-statement-block (i.e., last-level block) functions
		//that do not contain other functions, and either are small or called once
		
		for( String fkey : fgraph.getReachableFunctions() ) {
			FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fkey);
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			if( fstmt.getBody().size() == 1 
				&& HopRewriteUtils.isLastLevelStatementBlock(fstmt.getBody().get(0)) 
				&& !containsFunctionOp(fstmt.getBody().get(0).getHops())
				&& (fgraph.getFunctionCalls(fkey).size() == 1
					|| countOperators(fstmt.getBody().get(0).getHops()) 
						<= InterProceduralAnalysis.INLINING_MAX_NUM_OPS) )
			{
				if( LOG.isDebugEnabled() )
					LOG.debug("IPA: Inline function '"+fkey+"'");
				
				//replace all relevant function calls 
				ArrayList<Hop> hops = fstmt.getBody().get(0).getHops();
				List<FunctionOp> fcalls = fgraph.getFunctionCalls(fkey);
				List<StatementBlock> fcallsSB = fgraph.getFunctionCallsSB(fkey);
				for(int i=0; i<fcalls.size(); i++) {
					FunctionOp op = fcalls.get(i);
					
					//step 0: robustness for special cases
					if( op.getInput().size() != fstmt.getInputParams().size()
						|| op.getOutputVariableNames().length != fstmt.getOutputParams().size() )
						continue;
					
					//step 1: deep copy hop dag
					ArrayList<Hop> hops2 = Recompiler.deepCopyHopsDag(hops);
					
					//step 2: replace inputs
					HashMap<String,Hop> inMap = new HashMap<>();
					for(int j=0; j<op.getInput().size(); j++)
						inMap.put(fstmt.getInputParams().get(j).getName(), op.getInput().get(j));
					replaceTransientReads(hops2, inMap);
					
					//step 3: replace outputs
					HashMap<String,String> outMap = new HashMap<>();
					String[] opOutputs = op.getOutputVariableNames();
					for(int j=0; j<opOutputs.length; j++)
						outMap.put(fstmt.getOutputParams().get(j).getName(), opOutputs[j]);
					for(int j=0; j<hops2.size(); j++) {
						Hop out = hops2.get(j);
						if( HopRewriteUtils.isData(out, DataOpTypes.TRANSIENTWRITE) )
							out.setName(outMap.get(out.getName()));
					}
					fcallsSB.get(i).getHops().remove(op);
					fcallsSB.get(i).getHops().addAll(hops2);
				}
			}
		}
	}
	
	private static boolean containsFunctionOp(ArrayList<Hop> hops) {
		if( hops==null || hops.isEmpty() )
			return false;
		Hop.resetVisitStatus(hops);
		boolean ret = HopRewriteUtils.containsOp(hops, FunctionOp.class);
		Hop.resetVisitStatus(hops);
		return ret;
	}
	
	private static int countOperators(ArrayList<Hop> hops) {
		if( hops==null || hops.isEmpty() )
			return 0;
		Hop.resetVisitStatus(hops);
		int count = 0;
		for( Hop hop : hops )
			count += rCountOperators(hop);
		Hop.resetVisitStatus(hops);
		return count;
	}
	
	private static int rCountOperators(Hop current) {
		if( current.isVisited() )
			return 0;
		int count = !(current instanceof DataOp 
			|| current instanceof LiteralOp) ? 1 : 0;
		for( Hop c : current.getInput() )
			count += rCountOperators(c);
		current.setVisited();
		return count;
	}
	
	private static void replaceTransientReads(ArrayList<Hop> hops, HashMap<String, Hop> inMap) {
		Hop.resetVisitStatus(hops);
		for( Hop hop : hops )
			rReplaceTransientReads(hop, inMap);
		Hop.resetVisitStatus(hops);
	}
	
	private static void rReplaceTransientReads(Hop current, HashMap<String, Hop> inMap) {
		if( current.isVisited() )
			return;
		for( int i=0; i<current.getInput().size(); i++ ) {
			Hop c = current.getInput().get(i);
			rReplaceTransientReads(c, inMap);
			if( HopRewriteUtils.isData(c, DataOpTypes.TRANSIENTREAD) )
				HopRewriteUtils.replaceChildReference(current, c, inMap.get(c.getName()));
		}
		current.setVisited();
	}
}
