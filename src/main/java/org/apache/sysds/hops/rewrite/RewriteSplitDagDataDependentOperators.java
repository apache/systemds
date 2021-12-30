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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Compression.CompressConfig;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * Rule: Split Hop DAG after specific data-dependent operators. This is
 * important to create recompile hooks if output dimensions are usually
 * significantly overestimated. 
 * 
 * This is a recursive statementblock rewrite rule.
 * 
 * NOTE: Before we used AssignmentStatement.controlStatement() in order to force
 * statementblock cuts. However, this (1) cuts not only after but before-and-after
 * (which prevents certain rewrites because the input operators are unknown),
 * and (2) is statement-centric which potentially prevents the cut right after 
 * the problematic operation.
 * 
 * TODO: Cleanup runtime to never access individual statements of potentially
 * split statements blocks again (for consistency). However, currently it is
 * only used in places (e.g., parfor optimizer) that are not directly affected.
 */
public class RewriteSplitDagDataDependentOperators extends StatementBlockRewriteRule
{
	@Override
	public boolean createsSplitDag() {
		return true;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
	{
		//DAG splits not required for forced single node
		CompressConfig compress = CompressConfig.valueOf(ConfigurationManager
			.getDMLConfig().getTextValue(DMLConfig.COMPRESSED_LINALG).toUpperCase());
		if( (DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE
			&& !(compress != CompressConfig.FALSE) )
			|| !HopRewriteUtils.isLastLevelStatementBlock(sb) )
			return Arrays.asList(sb);
		
		ArrayList<StatementBlock> ret = new ArrayList<>();
	
		//collect all unknown data dependent ops
		ArrayList<Hop> cand = new ArrayList<>();
		collectDataDependentOperators( sb.getHops(), cand );
		Hop.resetVisitStatus(sb.getHops());
		
		//split hop dag on demand
		if( !cand.isEmpty() )
		{
			//collect child operators of candidates (to prevent rewrite anomalies)
			//For example, B = rmEmpty(A), C = rmEmpty(B), op(C, nrow(B)) needs to 
			//preserve the link rmEmpty(B) in the split-off DAG. Therefore, the
			//candidates themselves need to be part of this anomaly filter as well.
			HashSet<Hop> candChilds = new HashSet<>();
			collectCandidateChildOperators( cand, candChilds );
			candChilds.addAll(cand);
			
			try
			{
				//duplicate sb incl live variable sets
				StatementBlock sb1 = new StatementBlock();
				sb1.setDMLProg(sb.getDMLProg());
				sb1.setParseInfo(sb);
				sb1.setLiveIn(new VariableSet());
				sb1.setLiveOut(new VariableSet());
				
				//move data-dependent ops incl transient writes to new statement block
				//(and replace original hop with transient read)
				ArrayList<Hop> sb1hops = new ArrayList<>();
				for( Hop c : cand )
				{
					//if there are already transient writes use them and don't introduce artificial variables; 
					//unless there are transient reads w/ the same variable name in the current dag which can
					//lead to invalid reordering if variable consumers are not feeding into the candidate op.
					boolean hasTWrites = hasTransientWriteParents(c);
					boolean moveTWrite = hasTWrites ? HopRewriteUtils.rHasSimpleReadChain(
						c, getFirstTransientWriteParent(c).getName()) : false;
					
					String varname = null;
					long rlen = c.getDim1();
					long clen = c.getDim2();
					int blen = c.getBlocksize();
					
					if( hasTWrites && moveTWrite) //reuse existing transient_write
					{
						Hop twrite = getFirstTransientWriteParent(c);
						varname = twrite.getName();
						
						//create new transient read
						DataOp tread = HopRewriteUtils.createTransientRead(varname, c);
						
						//replace data-dependent operator with transient read
						ArrayList<Hop> parents = new ArrayList<>(c.getParent());
						for( int i=0; i<parents.size(); i++ ) {
							//prevent concurrent modification by index access
							Hop parent = parents.get(i);
							if( !candChilds.contains(parent) ) { //anomaly filter
								if( parent != twrite )
									HopRewriteUtils.replaceChildReference(parent, c, tread);
								else
									sb.getHops().remove(parent);
							}
						}
						
						//add data-dependent operator sub dag to first statement block
						sb1hops.add(twrite);
					}
					else //create transient write to artificial variables
					{
						varname = createCutVarName(false);
						
						//create new transient read
						DataOp tread = HopRewriteUtils.createTransientRead(varname, c);
						
						//replace data-dependent operator with transient read
						ArrayList<Hop> parents = new ArrayList<>(c.getParent());
						for( int i=0; i<parents.size(); i++ ) {
							//prevent concurrent modification by index access
							Hop parent = parents.get(i);
							if( !candChilds.contains(parent) ) { //anomaly filter
								HopRewriteUtils.replaceChildReference(parent, c, tread);
							}
						}
						
						//add data-dependent operator sub dag to first statement block
						DataOp twrite = HopRewriteUtils.createTransientWrite(varname, c);
						sb1hops.add(twrite);
					}
					
					//update live in and out of new statement block (for piggybacking)
					DataIdentifier diVar = new DataIdentifier(varname);
					diVar.setDimensions(rlen, clen);
					diVar.setBlocksize(blen);
					diVar.setDataType(c.getDataType());
					diVar.setValueType(c.getValueType());
					sb1.liveOut().addVariable(varname, new DataIdentifier(diVar));
					sb.liveIn().addVariable(varname, new DataIdentifier(diVar));
					sb.variablesRead().addVariable(varname, new DataIdentifier(diVar));
				}
				
				//ensure disjoint operators across DAGs (prevent replicated operations)
				handleReplicatedOperators( sb1hops, sb.getHops(), sb1.liveOut(), sb.liveIn() );
				
				//deep copy new dag (in order to prevent any dangling references)
				sb1.setHops(Recompiler.deepCopyHopsDag(sb1hops));
				sb1.updateRecompilationFlag();
				sb1.setSplitDag(true); //avoid later merge by other rewrites
				
				//recursive application of rewrite rule (in case of multiple data dependent operators
				//with data dependencies in between each other)
				List<StatementBlock> tmp = rewriteStatementBlock(sb1, state);
				
				//add new statement blocks to output
				ret.addAll(tmp); //statement block with data dependent hops
				ret.add(sb); //statement block with remaining hops
				sb.setSplitDag(true); //avoid later merge by other rewrites
			}
			catch(Exception ex) {
				throw new HopsException("Failed to split hops dag for data dependent operators with unknown size.", ex);
			}
			
			LOG.debug("Applied splitDagDataDependentOperators (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+").");
		}
		//keep original hop dag
		else {
			ret.add(sb);
		}
		
		return ret;
	}
	
	private void collectDataDependentOperators( ArrayList<Hop> roots, ArrayList<Hop> cand ) {
		if( roots == null )
			return;
		Hop.resetVisitStatus(roots);
		for( Hop root : roots )
			rCollectDataDependentOperators(root, cand);
	}

	private void rCollectDataDependentOperators( Hop hop, ArrayList<Hop> cand )
	{
		if( hop.isVisited() )
			return;
		
		//prevent unnecessary dag split (dims known or no consumer operations)
		boolean noSplitRequired = (HopRewriteUtils.hasOnlyWriteParents(hop, true, false)
			|| hop.dimsKnown() || DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE);
		boolean investigateChilds = true;
		
		//collect data dependent operations (to be extended as necessary)
		//#1 removeEmpty
		if( hop instanceof ParameterizedBuiltinOp 
			&& ((ParameterizedBuiltinOp) hop).getOp()==ParamBuiltinOp.RMEMPTY 
			&& !noSplitRequired
			&& !(hop.getParent().size()==1 && hop.getParent().get(0) instanceof TernaryOp 
				&& ((TernaryOp)hop.getParent().get(0)).isMatrixIgnoreZeroRewriteApplicable()))
		{
			ParameterizedBuiltinOp pbhop = (ParameterizedBuiltinOp)hop;
			cand.add(pbhop);
			investigateChilds = false;
			
			//keep interesting consumer information, flag hops accordingly 
			boolean noEmptyBlocks = true;
			boolean onlyPMM = true;
			boolean diagInput = pbhop.isTargetDiagInput();
			for( Hop p : hop.getParent() ) {
				//list of operators without need for empty blocks to be extended as needed
				noEmptyBlocks &= (   p instanceof AggBinaryOp && hop == p.getInput().get(0) 
				                  || HopRewriteUtils.isUnary(p, OpOp1.NROW) );
				onlyPMM &= (p instanceof AggBinaryOp && hop == p.getInput().get(0));
			}
			pbhop.setOutputEmptyBlocks(!noEmptyBlocks);
			
			if( onlyPMM && diagInput ){
				//configure rmEmpty to directly output selection vector
				//(only applied if dynamic recompilation enabled)
				
				if( ConfigurationManager.isDynamicRecompilation() )
					pbhop.setOutputPermutationMatrix(true);
				for( Hop p : hop.getParent() )
					((AggBinaryOp)p).setHasLeftPMInput(true);
			}
		}
		
		//#2 ctable with unknown dims
		if( HopRewriteUtils.isTernary(hop, OpOp3.CTABLE) 
			&& hop.getInput().size() < 4 //dims not provided
			&& !noSplitRequired )
		{
			cand.add(hop);
			investigateChilds = false;
			
			//keep interesting consumer information, flag hops accordingly
			boolean onlyPMM = true;
			for( Hop p : hop.getParent() ) {
				onlyPMM &= (p instanceof AggBinaryOp && hop == p.getInput().get(0));
			}
			
			if( onlyPMM && HopRewriteUtils.isBasic1NSequence(hop.getInput().get(0)) )
				hop.setOutputEmptyBlocks(false);
		}
		
		//#3 orderby childs computed in same DAG
		if( HopRewriteUtils.isReorg(hop, ReOrgOp.SORT) ){
			//params 'decreasing' / 'indexreturn'
			for( int i=2; i<=3; i++ ) {
				Hop c = hop.getInput().get(i);
				if( !(c instanceof LiteralOp || c instanceof DataOp) ){
					cand.add(c);
					c.setVisited();
					investigateChilds = false;
				}
			}
		}
		
		//#4 other data dependent operators (default handling)
		if( isBasicDataDependentOperator(hop, noSplitRequired) ) {
			cand.add(hop);
			investigateChilds = false;
		}
		
		//process children (if not already found a special operators;
		//otherwise, processed by recursive rule application)
		if( investigateChilds && hop.getInput()!=null )
			for( Hop c : hop.getInput() )
				rCollectDataDependentOperators(c, cand);
		
		hop.setVisited();
	}
	
	private static boolean isBasicDataDependentOperator(Hop hop, boolean noSplitRequired) {
		return (HopRewriteUtils.isNary(hop, OpOpN.EVAL) & !noSplitRequired)
			|| (HopRewriteUtils.isData(hop, OpOpData.SQLREAD) & !noSplitRequired)
			|| (HopRewriteUtils.isParameterBuiltinOp(hop, ParamBuiltinOp.GROUPEDAGG) 
				&& !((ParameterizedBuiltinOp)hop).isKnownNGroups() && !noSplitRequired)
			|| ((HopRewriteUtils.isUnary(hop, OpOp1.COMPRESS) || hop.requiresCompression()) &&
				(!HopRewriteUtils.hasOnlyWriteParents(hop, true, true)));
		//note: for compression we probe for write parents (part of noSplitRequired) directly
		// because we want to split even if the dimensions are known 
	}

	private static boolean hasTransientWriteParents( Hop hop ) {
		for( Hop p : hop.getParent() )
			if( p instanceof DataOp && ((DataOp)p).getOp()==OpOpData.TRANSIENTWRITE )
				return true;
		return false;
	}

	private static Hop getFirstTransientWriteParent( Hop hop ) {
		for( Hop p : hop.getParent() )
			if( p instanceof DataOp && ((DataOp)p).getOp()==OpOpData.TRANSIENTWRITE )
				return p;
		return null;
	}

	private void handleReplicatedOperators( ArrayList<Hop> rootsSB1, ArrayList<Hop> rootsSB2, VariableSet sb1out, VariableSet sb2in )
	{
		//step 1: create probe set SB1
		HashSet<Hop> probeSet = new HashSet<>();
		Hop.resetVisitStatus(rootsSB1);
		for( Hop h : rootsSB1 )
			rAddHopsToProbeSet( h, probeSet );
		
		//step 2: probe SB2 operators top-down (collect cut candidates)
		HashSet<Pair<Hop,Hop>> candSet = new HashSet<>();
		Hop.resetVisitStatus(rootsSB2);
		for( Hop h : rootsSB2 )
			rProbeAndAddHopsToCandidateSet(h, probeSet, candSet);
		
		//step 3: create additional cuts with reuse for common references
		HashMap<Long, DataOp> reuseTRead = new HashMap<>();
		for( Pair<Hop,Hop> p : candSet ) {
			Hop hop = p.getKey();
			Hop c = p.getValue();
			
			DataOp tread = reuseTRead.get(c.getHopID());
			if( tread == null ) {
				String varname = createCutVarName(false);
				
				tread = HopRewriteUtils.createTransientRead(varname, c);
				reuseTRead.put(c.getHopID(), tread);
				
				DataOp twrite = HopRewriteUtils.createTransientWrite(varname, c);
				
				//update live in and out of new statement block (for piggybacking)
				DataIdentifier diVar = new DataIdentifier(varname);
				diVar.setDimensions(c.getDim1(), c.getDim2());
				diVar.setBlocksize(c.getBlocksize());
				diVar.setDataType(c.getDataType());
				diVar.setValueType(c.getValueType());
				sb1out.addVariable(varname, new DataIdentifier(diVar));
				sb2in.addVariable(varname, new DataIdentifier(diVar));
				
				rootsSB1.add(twrite);
			}
			
			//create additional cut by rewriting both hop dags 
			int pos = HopRewriteUtils.getChildReferencePos(hop, c);
			HopRewriteUtils.removeChildReferenceByPos(hop, c, pos);
			HopRewriteUtils.addChildReference(hop, tread, pos);
		}
	}

	private void rAddHopsToProbeSet( Hop hop, HashSet<Hop> probeSet )
	{
		if( hop.isVisited() )
			return;
		
		//prevent cuts for no-ops
		if( !(   (hop instanceof DataOp && !((DataOp)hop).isPersistentReadWrite() )
			   || hop instanceof LiteralOp) )
		{
			probeSet.add(hop);
		}
			
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rAddHopsToProbeSet(c, probeSet);
	
		hop.setVisited();
	}
	
	/**
	 * NOTE: candset is a set of parent-child pairs because a parent might have 
	 * multiple references to replicated hops.
	 * 
	 * @param hop high-level operator
	 * @param probeSet probe set?
	 * @param candSet candidate set?
	 */
	private void rProbeAndAddHopsToCandidateSet( Hop hop, HashSet<Hop> probeSet, HashSet<Pair<Hop,Hop>> candSet )
	{
		if( hop.isVisited() )
			return;

		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )  {
				//probe for replicated operator, if any child is replicated, keep parent
				//for cut between parent-child; otherwise recursively descend.
				if( !probeSet.contains(c) )
					rProbeAndAddHopsToCandidateSet(c, probeSet, candSet);
				else
				{
					candSet.add(new Pair<>(hop,c));
				}
			}
		
		hop.setVisited();
	}
	
	private void collectCandidateChildOperators( ArrayList<Hop> cand, HashSet<Hop> candChilds )
	{
		Hop.resetVisitStatus(cand);
		if( cand != null )
			for( Hop root : cand )
				rCollectCandidateChildOperators(root, cand, candChilds, false);
		
		// Immediately reset the visit status because candidates might be inner nodes in the DAG.
		// Subsequent resets on the root nodes of the DAG would otherwise not necessarily reach 
		// these nodes which could lead to missing checks on subsequent passes (e.g., when checking 
		// for replicated operators).
		Hop.resetVisitStatus(cand);
	}
	
	private void rCollectCandidateChildOperators( Hop hop, ArrayList<Hop> cand, HashSet<Hop> candChilds, boolean collect )
	{
		if( hop.isVisited() )
			return;
		
		//collect operator if necessary
		if( collect ) {
			candChilds.add(hop);
		}
		
		//activate collection if we passed a candidate
		boolean passedFlag = collect;
		if( cand.contains(hop) ) {
			passedFlag = true;
		}
		
		//process childs recursively
		if( hop.getInput()!=null ) {
			for( Hop c : hop.getInput() )
				rCollectCandidateChildOperators(c, cand, candChilds, passedFlag);
		}
		
		hop.setVisited();
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus sate) {
		return sbs;
	}
}
