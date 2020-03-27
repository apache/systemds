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
import java.util.Arrays;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;

/**
 * Rule: Split Hop DAG after CSV reads with unknown size. This is
 * important to create recompile hooks if format is read from mtd
 * (we are not able to split it on statementblock creation) and 
 * mtd has unknown size (which can only happen for CSV). 
 * 
 */
public class RewriteSplitDagUnknownCSVRead extends StatementBlockRewriteRule
{
	@Override
	public boolean createsSplitDag() {
		return true;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
	{
		//DAG splits not required for forced single node
		if( DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE
			|| !HopRewriteUtils.isLastLevelStatementBlock(sb) )
			return Arrays.asList(sb);
		
		ArrayList<StatementBlock> ret = new ArrayList<>();
		
		//collect all unknown csv reads hops
		ArrayList<Hop> cand = new ArrayList<>();
		collectCSVReadHopsUnknownSize( sb.getHops(), cand );
		
		//split hop dag on demand
		if( !cand.isEmpty() )
		{
			try
			{
				//duplicate sb incl live variable sets
				StatementBlock sb1 = new StatementBlock();
				sb1.setDMLProg(sb.getDMLProg());
				sb1.setParseInfo(sb);
				sb1.setLiveIn(new VariableSet());
				sb1.setLiveOut(new VariableSet());
				
				//move csv reads incl reblock to new statement block
				//(and replace original persistent read with transient read)
				ArrayList<Hop> sb1hops = new ArrayList<>();
				for( Hop reblock : cand )
				{
					//replace reblock inputs to avoid dangling references across dags
					//(otherwise, for instance, literal ops are shared across dags)
					for( int i=0; i<reblock.getInput().size(); i++ )
						if( reblock.getInput().get(i) instanceof LiteralOp )
							HopRewriteUtils.replaceChildReference(reblock, reblock.getInput().get(i), 
								new LiteralOp((LiteralOp)reblock.getInput().get(i)));
					
					//create new transient read
					DataOp tread = HopRewriteUtils.createTransientRead(reblock.getName(), reblock);
					
					//replace reblock with transient read
					ArrayList<Hop> parents = new ArrayList<>(reblock.getParent());
					for( int i=0; i<parents.size(); i++ ) {
						Hop parent = parents.get(i);
						HopRewriteUtils.replaceChildReference(parent, reblock, tread);
					}
					
					//add reblock sub dag to first statement block
					DataOp twrite = HopRewriteUtils.createTransientWrite(reblock.getName(), reblock);
					sb1hops.add(twrite);
					
					//update live in and out of new statement block (for piggybacking)
					DataIdentifier diVar = sb.variablesRead().getVariable(reblock.getName()); 
					if( diVar != null ){ //var read should always exist because persistent read
						sb1.liveOut().addVariable(reblock.getName(), new DataIdentifier(diVar));
						sb.liveIn().addVariable(reblock.getName(), new DataIdentifier(diVar));
					}
				}
				
				sb1.setHops(sb1hops);
				sb1.updateRecompilationFlag();
				ret.add(sb1); //statement block with csv reblocks
				ret.add(sb); //statement block with remaining hops
				sb.setSplitDag(true); //avoid later merge by other rewrites
			}
			catch(Exception ex) {
				throw new HopsException("Failed to split hops dag for csv read with unknown size.", ex);
			}
			LOG.debug("Applied splitDagUnknownCSVRead.");
		}
		//keep original hop dag
		else
		{
			ret.add(sb);
		}
		
		return ret;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus sate) {
		return sbs;
	}
	
	private void collectCSVReadHopsUnknownSize( ArrayList<Hop> roots, ArrayList<Hop> cand ) {
		if( roots == null )
			return;
		Hop.resetVisitStatus(roots);
		for( Hop root : roots )
			collectCSVReadHopsUnknownSize(root, cand);
	}
	
	private void collectCSVReadHopsUnknownSize( Hop hop, ArrayList<Hop> cand )
	{
		if( hop.isVisited() )
			return;
		
		//collect persistent reads (of type csv, with unknown size)
		if( hop instanceof DataOp )
		{
			DataOp dop = (DataOp) hop;
			if(    dop.getOp() == OpOpData.PERSISTENTREAD
				&& dop.getInputFormatType() == FileFormat.CSV
				&& !dop.dimsKnown()
				&& !HopRewriteUtils.hasOnlyWriteParents(dop, true, false) )
			{
				cand.add(dop);
			}
		}
		
		//process children
		if( hop.getInput()!=null )
			for( Hop c : hop.getInput() )
				collectCSVReadHopsUnknownSize(c, cand);
		
		hop.setVisited();
	}
}
