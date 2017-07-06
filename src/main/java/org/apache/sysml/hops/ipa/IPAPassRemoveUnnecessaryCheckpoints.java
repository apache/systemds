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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatementBlock;

/**
 * This rewrite identifies and removes unnecessary checkpoints, i.e.,
 * persisting of Spark RDDs into a given storage level. For example,
 * in chains such as pread-checkpoint-append-checkpoint, the first
 * checkpoint is not used and creates unnecessary memory pressure.
 * 
 */
public class IPAPassRemoveUnnecessaryCheckpoints extends IPAPass
{
	@Override
	public boolean isApplicable() {
		return InterProceduralAnalysis.REMOVE_UNNECESSARY_CHECKPOINTS 
			&& OptimizerUtils.isSparkExecutionMode();
	}
	
	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
		throws HopsException
	{
		//remove unnecessary checkpoint before update 
		removeCheckpointBeforeUpdate(prog);
		
		//move necessary checkpoint after update
		moveCheckpointAfterUpdate(prog);
		
		//remove unnecessary checkpoint read-{write|uagg}
		removeCheckpointReadWrite(prog);
	}
	
	private static void removeCheckpointBeforeUpdate(DMLProgram dmlp) 
		throws HopsException
	{
		//approach: scan over top-level program (guaranteed to be unconditional),
		//collect checkpoints; determine if used before update; remove first checkpoint
		//on second checkpoint if update in between and not used before update
		
		HashMap<String, Hop> chkpointCand = new HashMap<String, Hop>();
		
		for( StatementBlock sb : dmlp.getStatementBlocks() ) 
		{
			//prune candidates (used before updated)
			Set<String> cands = new HashSet<String>(chkpointCand.keySet());
			for( String cand : cands )
				if( sb.variablesRead().containsVariable(cand) 
					&& !sb.variablesUpdated().containsVariable(cand) ) 
				{	
					//note: variableRead might include false positives due to meta 
					//data operations like nrow(X) or operations removed by rewrites 
					//double check hops on basic blocks; otherwise worst-case
					boolean skipRemove = false;
					if( sb.get_hops() !=null ) {
						Hop.resetVisitStatus(sb.get_hops());
						skipRemove = true;
						for( Hop root : sb.get_hops() )
							skipRemove &= !HopRewriteUtils.rContainsRead(root, cand, false);
					}					
					if( !skipRemove )
						chkpointCand.remove(cand);
				}
			
			//prune candidates (updated in conditional control flow)
			Set<String> cands2 = new HashSet<String>(chkpointCand.keySet());
			if( sb instanceof IfStatementBlock || sb instanceof WhileStatementBlock 
				|| sb instanceof ForStatementBlock )
			{
				for( String cand : cands2 )
					if( sb.variablesUpdated().containsVariable(cand) ) {
						chkpointCand.remove(cand);
					}
			}
			//prune candidates (updated w/ multiple reads) 
			else
			{
				for( String cand : cands2 )
					if( sb.variablesUpdated().containsVariable(cand) && sb.get_hops() != null) 
					{
						Hop.resetVisitStatus(sb.get_hops());
						for( Hop root : sb.get_hops() )
							if( root.getName().equals(cand) &&
								!HopRewriteUtils.rHasSimpleReadChain(root, cand) ) {
								chkpointCand.remove(cand);
							}
					}	
			}
		
			//collect checkpoints and remove unnecessary checkpoints
			ArrayList<Hop> tmp = collectCheckpoints(sb.get_hops());
			for( Hop chkpoint : tmp ) {
				if( chkpointCand.containsKey(chkpoint.getName()) ) {
					chkpointCand.get(chkpoint.getName()).setRequiresCheckpoint(false);		
				}
				chkpointCand.put(chkpoint.getName(), chkpoint);
			}
			
		}
	}

	private static void moveCheckpointAfterUpdate(DMLProgram dmlp) 
		throws HopsException
	{
		//approach: scan over top-level program (guaranteed to be unconditional),
		//collect checkpoints; determine if used before update; move first checkpoint
		//after update if not used before update (best effort move which often avoids
		//the second checkpoint on loops even though used in between)
		
		HashMap<String, Hop> chkpointCand = new HashMap<String, Hop>();
		
		for( StatementBlock sb : dmlp.getStatementBlocks() ) 
		{
			//prune candidates (used before updated)
			Set<String> cands = new HashSet<String>(chkpointCand.keySet());
			for( String cand : cands )
				if( sb.variablesRead().containsVariable(cand) 
					&& !sb.variablesUpdated().containsVariable(cand) ) 
				{	
					//note: variableRead might include false positives due to meta 
					//data operations like nrow(X) or operations removed by rewrites 
					//double check hops on basic blocks; otherwise worst-case
					boolean skipRemove = false;
					if( sb.get_hops() !=null ) {
						Hop.resetVisitStatus(sb.get_hops());
						skipRemove = true;
						for( Hop root : sb.get_hops() )
							skipRemove &= !HopRewriteUtils.rContainsRead(root, cand, false);
					}					
					if( !skipRemove )
						chkpointCand.remove(cand);
				}
			
			//prune candidates (updated in conditional control flow)
			Set<String> cands2 = new HashSet<String>(chkpointCand.keySet());
			if( sb instanceof IfStatementBlock || sb instanceof WhileStatementBlock 
				|| sb instanceof ForStatementBlock )
			{
				for( String cand : cands2 )
					if( sb.variablesUpdated().containsVariable(cand) ) {
						chkpointCand.remove(cand);
					}
			}
			//move checkpoint after update with simple read chain 
			//(note: right now this only applies if the checkpoints comes from a previous
			//statement block, within-dag checkpoints should be handled during injection)
			else
			{
				for( String cand : cands2 )
					if( sb.variablesUpdated().containsVariable(cand) && sb.get_hops() != null) {
						Hop.resetVisitStatus(sb.get_hops());
						for( Hop root : sb.get_hops() )
							if( root.getName().equals(cand) ) {
								if( HopRewriteUtils.rHasSimpleReadChain(root, cand) ) {
									chkpointCand.get(cand).setRequiresCheckpoint(false);
									root.getInput().get(0).setRequiresCheckpoint(true);
									chkpointCand.put(cand, root.getInput().get(0));
								}
								else
									chkpointCand.remove(cand);		
							}
					}	
			}
		
			//collect checkpoints
			ArrayList<Hop> tmp = collectCheckpoints(sb.get_hops());
			for( Hop chkpoint : tmp ) {
				chkpointCand.put(chkpoint.getName(), chkpoint);
			}
		}
	}
	
	private static void removeCheckpointReadWrite(DMLProgram dmlp) 
		throws HopsException
	{
		List<StatementBlock> sbs = dmlp.getStatementBlocks();
		
		if( sbs.size()==1 & !(sbs.get(0) instanceof IfStatementBlock 
			|| sbs.get(0) instanceof WhileStatementBlock 
			|| sbs.get(0) instanceof ForStatementBlock) ) 
		{
			//recursively process all dag roots
			if( sbs.get(0).get_hops()!=null ) {
				Hop.resetVisitStatus(sbs.get(0).get_hops());
				for( Hop root : sbs.get(0).get_hops() )
					rRemoveCheckpointReadWrite(root);
			}
		}
	}
	
	private static ArrayList<Hop> collectCheckpoints(ArrayList<Hop> roots)
	{
		ArrayList<Hop> ret = new ArrayList<Hop>();	
		if( roots != null ) {
			Hop.resetVisitStatus(roots);
			for( Hop root : roots )
				rCollectCheckpoints(root, ret);
		}
		
		return ret;
	}
	
	private static void rCollectCheckpoints(Hop hop, ArrayList<Hop> checkpoints)
	{
		if( hop.isVisited() )
			return;

		//handle leaf node for variable (checkpoint directly bound
		//to logical variable name and not used)
		if( hop.requiresCheckpoint() && hop.getParent().size()==1 
			&& hop.getParent().get(0) instanceof DataOp
			&& ((DataOp)hop.getParent().get(0)).getDataOpType()==DataOpTypes.TRANSIENTWRITE)
		{
			checkpoints.add(hop);
		}
		
		//recursively process child nodes
		for( Hop c : hop.getInput() )
			rCollectCheckpoints(c, checkpoints);
	
		hop.setVisited();
	}
	
	public static void rRemoveCheckpointReadWrite(Hop hop)
	{
		if( hop.isVisited() )
			return;

		//remove checkpoint on pread if only consumed by pwrite or uagg
		if( (hop instanceof DataOp && ((DataOp)hop).getDataOpType()==DataOpTypes.PERSISTENTWRITE)
			|| hop instanceof AggUnaryOp )	
		{
			//(pwrite|uagg) - pread
			Hop c0 = hop.getInput().get(0);
			if( c0.requiresCheckpoint() && c0.getParent().size() == 1
				&& c0 instanceof DataOp && ((DataOp)c0).getDataOpType()==DataOpTypes.PERSISTENTREAD )
			{
				c0.setRequiresCheckpoint(false);
			}
			
			//(pwrite|uagg) - frame/matri cast - pread
			if( c0 instanceof UnaryOp && c0.getParent().size() == 1 
				&& (((UnaryOp)c0).getOp()==OpOp1.CAST_AS_FRAME || ((UnaryOp)c0).getOp()==OpOp1.CAST_AS_MATRIX ) 
				&& c0.getInput().get(0).requiresCheckpoint() && c0.getInput().get(0).getParent().size() == 1
				&& c0.getInput().get(0) instanceof DataOp 
				&& ((DataOp)c0.getInput().get(0)).getDataOpType()==DataOpTypes.PERSISTENTREAD )
			{
				c0.getInput().get(0).setRequiresCheckpoint(false);
			}
		}
		
		//recursively process children
		for( Hop c : hop.getInput() )
			rRemoveCheckpointReadWrite(c);
		
		hop.setVisited();
	}
}
