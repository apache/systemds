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
import java.util.HashMap;

import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpData;

/**
 * This rewrite identifies binary operations with constant matrices 
 * such as X * ones, where ones might be created as a vector of ones
 * before a loop. Such operations frequently occur after branch removal
 * for fixed configurations or loss functions.  
 * 
 */
public class IPAPassRemoveConstantBinaryOps extends IPAPass
{
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return InterProceduralAnalysis.REMOVE_CONSTANT_BINARY_OPS;
	}
	
	@Override
	public boolean rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) {
		//approach: scan over top-level program (guaranteed to be unconditional),
		//collect ones=matrix(1,...); remove b(*)ones if not outer operation
		HashMap<String, Hop> mOnes = new HashMap<>();
		
		for( StatementBlock sb : prog.getStatementBlocks() )  {
			//pruning updated variables
			for( String var : sb.variablesUpdated().getVariableNames() )
				if( mOnes.containsKey( var ) )
					mOnes.remove( var );
			
			//replace constant binary ops
			if( !mOnes.isEmpty() )
				rRemoveConstantBinaryOp(sb, mOnes);
			
			//collect matrices of ones from last-level statement blocks
			if( !(sb instanceof IfStatementBlock || sb instanceof WhileStatementBlock 
				  || sb instanceof ForStatementBlock) )
			{
				collectMatrixOfOnes(sb.getHops(), mOnes);
			}
		}
		return false;
	}
	
	private static void collectMatrixOfOnes(ArrayList<Hop> roots, HashMap<String,Hop> mOnes)
	{
		if( roots == null )
			return;
		
		for( Hop root : roots )
			if( root instanceof DataOp && ((DataOp)root).getOp()==OpOpData.TRANSIENTWRITE
			   && root.getInput().get(0) instanceof DataGenOp
			   && ((DataGenOp)root.getInput().get(0)).getOp()==OpOpDG.RAND
			   && ((DataGenOp)root.getInput().get(0)).hasConstantValue(1.0)) 
			{
				mOnes.put(root.getName(),root.getInput().get(0));
			}
	}
	
	private static void rRemoveConstantBinaryOp(StatementBlock sb, HashMap<String,Hop> mOnes) {
		if( sb instanceof IfStatementBlock )
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			for( StatementBlock c : istmt.getIfBody() )
				rRemoveConstantBinaryOp(c, mOnes);
			if( istmt.getElseBody() != null )
				for( StatementBlock c : istmt.getElseBody() )
					rRemoveConstantBinaryOp(c, mOnes);	
		}
		else if( sb instanceof WhileStatementBlock )
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			for( StatementBlock c : wstmt.getBody() )
				rRemoveConstantBinaryOp(c, mOnes);
		}
		else if( sb instanceof ForStatementBlock )
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			for( StatementBlock c : fstmt.getBody() )
				rRemoveConstantBinaryOp(c, mOnes);	
		}
		else
		{
			if( sb.getHops() != null ){
				Hop.resetVisitStatus(sb.getHops());
				for( Hop hop : sb.getHops() )
					rRemoveConstantBinaryOp(hop, mOnes);
			}
		}
	}
	
	private static void rRemoveConstantBinaryOp(Hop hop, HashMap<String,Hop> mOnes)
	{
		if( hop.isVisited() )
			return;

		if( hop instanceof BinaryOp && ((BinaryOp)hop).getOp()==OpOp2.MULT
			&& !((BinaryOp) hop).isOuter()
			&& hop.getInput().get(0).getDataType()==DataType.MATRIX
			&& hop.getInput().get(1) instanceof DataOp
			&& mOnes.containsKey(hop.getInput().get(1).getName()) )
		{
			//replace matrix of ones with literal 1 (later on removed by
			//algebraic simplification rewrites; otherwise more complex
			//recursive processing of childs and rewiring required)
			HopRewriteUtils.removeChildReferenceByPos(hop, hop.getInput().get(1), 1);
			HopRewriteUtils.addChildReference(hop, new LiteralOp(1), 1);
		}
		
		//recursively process child nodes
		for( Hop c : hop.getInput() )
			rRemoveConstantBinaryOp(c, mOnes);
	
		hop.setVisited();
	}
}
