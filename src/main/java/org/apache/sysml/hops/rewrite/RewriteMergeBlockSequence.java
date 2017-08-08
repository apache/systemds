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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.VariableSet;
import org.apache.sysml.parser.WhileStatementBlock;

/**
 * Rule: Simplify program structure by merging sequences of last-level
 * statement blocks in order to create optimization opportunities.
 * 
 */
public class RewriteMergeBlockSequence extends StatementBlockRewriteRule
{
	private ProgramRewriter rewriter = new ProgramRewriter(
		new RewriteCommonSubexpressionElimination(true));
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, 
			ProgramRewriteStatus state) throws HopsException {
		return Arrays.asList(sb);
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, 
			ProgramRewriteStatus sate) throws HopsException 
	{
		if( sbs == null || sbs.isEmpty() )
			return sbs;
		
		//execute binary merging iterations until fixpoint 
		ArrayList<StatementBlock> tmpList = new ArrayList<>(sbs);
		boolean merged = true;
		while( merged ) {
			merged = false;
			for( int i=0; i<tmpList.size()-1; i++ ) {
				StatementBlock sb1 = tmpList.get(i);
				StatementBlock sb2 = tmpList.get(i+1);
				if( isLastLevelStatementBlock(sb1) && isLastLevelStatementBlock(sb2) 
					&& !hasFunctionOpRoot(sb1) && !sb1.isSplitDag() && !sb2.isSplitDag() 
					&& sb2.getBeginLine()!=34 ) 
				{
					ArrayList<Hop> sb1Hops = sb1.get_hops();
					ArrayList<Hop> sb2Hops = sb2.get_hops();
					
					//determine transient read inputs s2 
					Hop.resetVisitStatus(sb2Hops);
					HashMap<String,Hop> treads = new HashMap<>();
					HashMap<String,Hop> twrites = new HashMap<>();
					for( Hop root : sb2Hops )
						rCollectTransientReadWrites(root, treads, twrites);
					Hop.resetVisitStatus(sb2Hops);
					
					//merge hop dags of s1 and s2
					Hop.resetVisitStatus(sb1Hops);
					for( Hop root : sb1Hops ) {
						//connect transient writes s1 and reads s2
						if( HopRewriteUtils.isData(root, DataOpTypes.TRANSIENTWRITE) 
							&& treads.containsKey(root.getName()) ) {
							//rewire transient write and transient read
							Hop tread = treads.get(root.getName());
							Hop in = root.getInput().get(0);
							for( Hop parent : new ArrayList<Hop>(tread.getParent()) )
								HopRewriteUtils.replaceChildReference(parent, tread, in);
							HopRewriteUtils.removeAllChildReferences(root);
							//add transient write if necessary
							if( !twrites.containsKey(root.getName()) 
								&& sb2.liveOut().containsVariable(root.getName()) ) {
								sb2Hops.add(HopRewriteUtils.createDataOp(
									root.getName(), in, DataOpTypes.TRANSIENTWRITE));
							}
						}
						//add remaining roots from s1 to s2
						else if( !(HopRewriteUtils.isData(root, DataOpTypes.TRANSIENTWRITE)
							&& twrites.containsKey(root.getName())) ) {
							sb2Hops.add(root);
						}
					}
					Hop.resetVisitStatus(sb2Hops);
					
					//run common-subexpression elimination
					rewriter.rewriteHopDAGs(sb2Hops, new ProgramRewriteStatus());
					
					//modify live variable sets of s2
					sb2.setLiveIn(sb1.liveIn()); //liveOut remains unchanged
					sb2.setGen(VariableSet.minus(VariableSet.union(sb1.getGen(), sb2.getGen()), sb1.getKill()));
					sb2.setKill(VariableSet.union(sb1.getKill(), sb2.getKill()));
					sb2.setReadVariables(VariableSet.union(sb1.variablesRead(), sb2.variablesRead()));
					sb2.setUpdatedVariables(VariableSet.union(sb1.variablesUpdated(), sb2.variablesUpdated()));
					
					LOG.debug("Applied mergeStatementBlockSequences "
							+ "(blocks of lines "+sb1.getBeginLine()+"-"+sb1.getEndLine()
							+" and "+sb2.getBeginLine()+"-"+sb2.getEndLine()+").");
					
					//modify line numbers of s2
					sb2.setBeginLine(sb1.getBeginLine());
					sb2.setBeginColumn(sb1.getBeginColumn());
					
					//remove sb1 from list of statement blocks
					tmpList.remove(i);
					merged = true;
					break; //for
				}
			}
		}
		
		return tmpList;
	}
	
	private void rCollectTransientReadWrites(Hop current, HashMap<String, Hop> treads, HashMap<String, Hop> twrites) {
		if( current.isVisited() )
			return;
		//process nodes recursively
		for( Hop c : current.getInput() )
			rCollectTransientReadWrites(c, treads, twrites);
		//collect all transient reads
		if( HopRewriteUtils.isData(current, DataOpTypes.TRANSIENTREAD) )
			treads.put(current.getName(), current);
		else if( HopRewriteUtils.isData(current, DataOpTypes.TRANSIENTWRITE) )
			twrites.put(current.getName(), current);
		else if( current instanceof FunctionOp ) {
			for( String output : ((FunctionOp)current).getOutputVariableNames() )
				twrites.put(output, null); //only name lookup
		}
		current.setVisited();
	}
	
	private static boolean hasFunctionOpRoot(StatementBlock sb) 
			throws HopsException {
		if( sb == null || sb.get_hops() == null )
			return false;
		boolean ret = false;
		for( Hop root : sb.get_hops() )
			ret |= (root instanceof FunctionOp);
		return ret;
	}
	
	private static boolean isLastLevelStatementBlock(StatementBlock sb) {
		return !((sb instanceof FunctionStatementBlock)
			|| (sb instanceof WhileStatementBlock)
			|| (sb instanceof IfStatementBlock)
			|| (sb instanceof ForStatementBlock)); //incl parfor
	}
}
