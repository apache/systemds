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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.stream.IntStream;

import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;

/**
 * This rewrite forwards a function call to a function with a simple
 * function call that only consumes function parameters and literals
 * into the original call location.
 */
public class IPAPassForwardFunctionCalls extends IPAPass
{
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return InterProceduralAnalysis.FORWARD_SIMPLE_FUN_CALLS;
	}
	
	@Override
	public boolean rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
	{
		for( String fkey : fgraph.getReachableFunctions() ) {
			FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fkey);
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			
			//step 1: basic application filter: simple forwarding call
			if( fstmt.getBody().size() != 1 || !singleFunctionOp(fstmt.getBody().get(0).getHops())
				|| !hasOnlySimplyArguments((FunctionOp)fstmt.getBody().get(0).getHops().get(0)))
				continue;
			if( LOG.isDebugEnabled() )
				LOG.debug("IPA: Forward-function-call candidate L1: '"+fkey+"'");
			
			//step 2: check consistent output ordering
			FunctionOp call2 = (FunctionOp)fstmt.getBody().get(0).getHops().get(0);
			if( !hasConsistentOutputOrdering(fstmt, call2)
				|| fgraph.getFunctionCalls(fkey).size() > 1)
				continue;
			if( LOG.isDebugEnabled() )
				LOG.debug("IPA: Forward-function-call candidate L2: '"+fkey+"'");
			
			//step 3: check and rewire input arguments (single call guaranteed)
			
			FunctionOp call1 = fgraph.getFunctionCalls(fkey).get(0);
			if( hasValidVariableNames(call1) && hasValidVariableNames(call2)
				&& isFirstSubsetOfSecond(call2.getInputVariableNames(), call1.getInputVariableNames())) {
				//step 4: rewire input arguments
				call1.setFunctionName(call2.getFunctionName());
				call1.setFunctionNamespace(call2.getFunctionNamespace());
				reconcileFunctionInputsInPlace(call1, call2);
				//step 5: update function call graph (old, new)
				fgraph.replaceFunctionCalls(fkey, call2.getFunctionKey());
				if( !fgraph.containsSecondOrderCall() )
					prog.removeFunctionStatementBlock(fkey);
				
				if( LOG.isDebugEnabled() )
					LOG.debug("IPA: Forward-function-call: replaced '"
						+ fkey +"' with '"+call2.getFunctionKey()+"'");
			}
		}
		return false;
	}
	
	private static boolean singleFunctionOp(ArrayList<Hop> hops) {
		if( hops==null || hops.isEmpty() || hops.size()!=1 )
			return false;
		return hops.get(0) instanceof FunctionOp;
	}
	
	private static boolean hasOnlySimplyArguments(FunctionOp fop) {
		return fop.getInput().stream().allMatch(h -> h instanceof LiteralOp 
			|| HopRewriteUtils.isData(h, OpOpData.TRANSIENTREAD));
	}
	
	private static boolean hasConsistentOutputOrdering(FunctionStatement fstmt, FunctionOp fop2) {
		int len = Math.min(fstmt.getOutputParams().size(), fop2.getOutputVariableNames().length);
		return IntStream.range(0, len).allMatch(i -> 
			fstmt.getOutputParams().get(i).getName().equals(fop2.getOutputVariableNames()[i]));
	}
	
	private static boolean hasValidVariableNames(FunctionOp fop) {
		return fop.getInputVariableNames() != null
			&& Arrays.stream(fop.getInputVariableNames()).allMatch(s -> s != null);
	}
	
	private static boolean isFirstSubsetOfSecond(String[] first, String[] second) {
		//build phase: second
		HashSet<String> probe = new HashSet<>();
		for( String s : second )
			probe.add(s);
		//probe phase: first
		return Arrays.stream(first).allMatch(s -> probe.contains(s));
	}
	
	private static void reconcileFunctionInputsInPlace(FunctionOp call1, FunctionOp call2) {
		//prepare all input of call2 for probing
		HashMap<String,Hop> probe = new HashMap<>();
		for( int i=0; i<call2.getInput().size(); i++ )
			probe.put(call2.getInputVariableNames()[i], call2.getInput().get(i));
		
		//construct new inputs for call1
		ArrayList<Hop> inputs = new ArrayList<>();
		for( int i=0; i<call1.getInput().size(); i++ )
			if( probe.containsKey(call1.getInputVariableNames()[i]) ) {
				inputs.add( (probe.get(call1.getInputVariableNames()[i]) instanceof LiteralOp) ? 
					probe.get(call1.getInputVariableNames()[i]) : call1.getInput().get(i));
			}
		HopRewriteUtils.removeAllChildReferences(call1);
		call1.addAllInputs(inputs);
		call1.setInputVariableNames(call2.getInputVariableNames());
	}
}
