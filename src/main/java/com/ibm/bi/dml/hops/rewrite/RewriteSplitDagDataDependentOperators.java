/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashSet;

import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.hops.Hop.OpOp3;
import com.ibm.bi.dml.hops.Hop.ParamBuiltinOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.ReOrgOp;
import com.ibm.bi.dml.hops.Hop.VisitStatus;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.ParameterizedBuiltinOp;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.hops.TernaryOp;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.hops.recompile.Recompiler;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.VariableSet;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.matrix.data.Pair;

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

	private static String _varnamePredix = "_sbcvar";
	private static IDSequence _seq = new IDSequence();
	
	@Override
	public ArrayList<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
		throws HopsException 
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		
		//collect all unknown csv reads hops
		ArrayList<Hop> cand = new ArrayList<Hop>();
		collectDataDependentOperators( sb.get_hops(), cand );
		Hop.resetVisitStatus(sb.get_hops());
		
		//split hop dag on demand
		if( !cand.isEmpty() )
		{
			//collect child operators of candidates (to prevent rewrite anomalies)
			HashSet<Hop> candChilds = new HashSet<Hop>();
			collectCandidateChildOperators( cand, candChilds );
			
			try
			{
				//duplicate sb incl live variable sets
				StatementBlock sb1 = new StatementBlock();
				sb1.setDMLProg(sb.getDMLProg());
				sb1.setAllPositions(sb.getFilename(), sb.getBeginLine(), sb.getBeginColumn(), sb.getEndLine(), sb.getEndColumn());
				sb1.setLiveIn(new VariableSet());
				sb1.setLiveOut(new VariableSet());
				
				//move data-dependent ops incl transient writes to new statement block
				//(and replace original persistent read with transient read)
				ArrayList<Hop> sb1hops = new ArrayList<Hop>();			
				for( Hop c : cand )
				{
					//if there are already transient writes use them and don't introduce artificial variables 
					boolean hasTWrites = hasTransientWriteParents(c);
					
					String varname = null;
					long rlen = c.getDim1();
					long clen = c.getDim2();
					long nnz = c.getNnz();
					long brlen = c.getRowsInBlock();
					long bclen = c.getColsInBlock();
					
					if( hasTWrites ) //reuse existing transient_write
					{		
						Hop twrite = getFirstTransientWriteParent(c);
						varname = twrite.getName();
						
						//create new transient read
						DataOp tread = new DataOp(varname, c.getDataType(), c.getValueType(),
			                    DataOpTypes.TRANSIENTREAD, null, rlen, clen, nnz, brlen, bclen);
						tread.setVisited(VisitStatus.DONE);
						HopRewriteUtils.copyLineNumbers(c, tread);
						
						//replace data-dependent operator with transient read
						ArrayList<Hop> parents = new ArrayList<Hop>(c.getParent());
						for( int i=0; i<parents.size(); i++ ) {
							//prevent concurrent modification by index access
							Hop parent = parents.get(i);
							if( !candChilds.contains(parent) ) //anomaly filter
							{
								if( parent != twrite ) {
									int pos = HopRewriteUtils.getChildReferencePos(parent, c);
									HopRewriteUtils.removeChildReferenceByPos(parent, c, pos);
									HopRewriteUtils.addChildReference(parent, tread, pos);
								}
								else
									sb.get_hops().remove(parent);
							}
						}
						
						//add data-dependent operator sub dag to first statement block
						sb1hops.add(twrite);
					}
					else //create transient write to artificial variables
					{
						varname = _varnamePredix + _seq.getNextID();
						
						//create new transient read
						DataOp tread = new DataOp(varname, c.getDataType(), c.getValueType(),
			                    DataOpTypes.TRANSIENTREAD, null, rlen, clen, nnz, brlen, bclen);
						tread.setVisited(VisitStatus.DONE);
						HopRewriteUtils.copyLineNumbers(c, tread);
						
						//replace data-dependent operator with transient read
						ArrayList<Hop> parents = new ArrayList<Hop>(c.getParent());						
						for( int i=0; i<parents.size(); i++ ) {
							//prevent concurrent modification by index access
							Hop parent = parents.get(i);
							if( !candChilds.contains(parent) ) //anomaly filter
							{
								int pos = HopRewriteUtils.getChildReferencePos(parent, c);
								HopRewriteUtils.removeChildReferenceByPos(parent, c, pos);
								HopRewriteUtils.addChildReference(parent, tread, pos);
							}
						}
						
						//add data-dependent operator sub dag to first statement block
						DataOp twrite = new DataOp(varname, c.getDataType(), c.getValueType(),
								                   c, DataOpTypes.TRANSIENTWRITE, null);
						twrite.setVisited(VisitStatus.DONE);
						twrite.setOutputParams(rlen, clen, nnz, brlen, bclen);
						HopRewriteUtils.copyLineNumbers(c, twrite);
						sb1hops.add(twrite);	
					}
					
					//update live in and out of new statement block (for piggybacking)
					DataIdentifier diVar = new DataIdentifier(varname);
					diVar.setDimensions(rlen, clen);
					diVar.setBlockDimensions(brlen, bclen);
					diVar.setDataType(c.getDataType());
					diVar.setValueType(c.getValueType());
					sb1.liveOut().addVariable(varname, new DataIdentifier(diVar));
					sb.liveIn().addVariable(varname, new DataIdentifier(diVar));
				}
		
				//ensure disjoint operators across DAGs (prevent replicated operations)
				handleReplicatedOperators( sb1hops, sb.get_hops(), sb1.liveOut(), sb.liveIn() );
				
				//deep copy new dag (in order to prevent any dangling references)
				sb1.set_hops(Recompiler.deepCopyHopsDag(sb1hops));
				sb1.updateRecompilationFlag();
				
				//recursive application of rewrite rule (in case of multiple data dependent operators
				//with data dependencies in between each other)
				ArrayList<StatementBlock> tmp = rewriteStatementBlock( sb1, state);
				
				//add new statement blocks to output
				ret.addAll(tmp); //statement block with data dependent hops
				ret.add(sb); //statement block with remaining hops
			}
			catch(Exception ex)
			{
				throw new HopsException("Failed to split hops dag for data dependent operators with unknown size.", ex);
			}
			
			LOG.debug("Applied splitDagDataDependentOperators (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+").");
		}
		//keep original hop dag
		else
		{
			ret.add(sb);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param roots
	 * @param cand
	 */
	private void collectDataDependentOperators( ArrayList<Hop> roots, ArrayList<Hop> cand )
	{
		if( roots == null )
			return;
		
		Hop.resetVisitStatus(roots);
		for( Hop root : roots )
			rCollectDataDependentOperators(root, cand);
	}
	
	/**
	 * 
	 * @param root
	 * @param cand
	 */
	private void rCollectDataDependentOperators( Hop hop, ArrayList<Hop> cand )
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;
		
		//prevent unnecessary dag split (dims known or no consumer operations)
		boolean noSplitRequired = ( hop.dimsKnown() || HopRewriteUtils.hasOnlyWriteParents(hop, true, true) );
		boolean investigateChilds = true;
		
		//collect data dependent operations (to be extended as necessary)
		//#1 removeEmpty
		if(    hop instanceof ParameterizedBuiltinOp 
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
				                  || p instanceof UnaryOp && ((UnaryOp)p).getOp()==OpOp1.NROW);
				onlyPMM &= (p instanceof AggBinaryOp && hop == p.getInput().get(0));
			}
			pbhop.setOutputEmptyBlocks(!noEmptyBlocks);
			
			if( onlyPMM && diagInput ){
				//configure rmEmpty to directly output selection vector
				//(only applied if dynamic recompilation enabled)
				
				if( OptimizerUtils.ALLOW_DYN_RECOMPILATION  )	
					pbhop.setOutputPermutationMatrix(true);
				for( Hop p : hop.getParent() )
					((AggBinaryOp)p).setHasLeftPMInput(true);		
			}
		}
		
		//#2 ctable with unknown dims
	    if(    hop instanceof TernaryOp 
			&& ((TernaryOp) hop).getOp()==OpOp3.CTABLE 
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
	    if(   hop instanceof ReorgOp 
	       && ((ReorgOp)hop).getOp()==ReOrgOp.SORT )
	    {
	    	//params 'decreasing' / 'indexreturn'
	    	for( int i=2; i<=3; i++ ) {
	    		Hop c = hop.getInput().get(i);
	    		if( !(c instanceof LiteralOp || c instanceof DataOp) ){
		    		cand.add(c);
		    		c.setVisited(VisitStatus.DONE);
		    		investigateChilds = false;	
		    	}

	    	}	    	
	    }
		
		//process children (if not already found a special operators;
	    //otherwise, processed by recursive rule application)
		if( investigateChilds )
		    if( hop.getInput()!=null )
				for( Hop c : hop.getInput() )
					rCollectDataDependentOperators(c, cand);
		
		hop.setVisited(VisitStatus.DONE);
	}

	/**
	 * 
	 * @param hop
	 * @return
	 */
	private boolean hasTransientWriteParents( Hop hop )
	{
		for( Hop p : hop.getParent() )
			if( p instanceof DataOp && ((DataOp)p).getDataOpType()==DataOpTypes.TRANSIENTWRITE )
				return true;
		return false;
	}
	
	/**
	 * 
	 * @param hop
	 * @return
	 */
	private Hop getFirstTransientWriteParent( Hop hop )
	{
		for( Hop p : hop.getParent() )
			if( p instanceof DataOp && ((DataOp)p).getDataOpType()==DataOpTypes.TRANSIENTWRITE )
				return p;
		return null;
	}
	
	/**
	 * 
	 * @param rootsSB1
	 * @param rootsSB2
	 * @param candChilds 
	 * @param cand 
	 * @param sb2in 
	 * @param sb1out 
	 */
	private void handleReplicatedOperators( ArrayList<Hop> rootsSB1, ArrayList<Hop> rootsSB2, VariableSet sb1out, VariableSet sb2in )
	{
		//step 1: create probe set SB1
		HashSet<Hop> probeSet = new HashSet<Hop>();
		Hop.resetVisitStatus(rootsSB1);
		for( Hop h : rootsSB1 )
			rAddHopsToProbeSet( h, probeSet );
		
		//step 2: probe SB2 operators top-down (collect cut candidates)
		HashSet<Pair<Hop,Hop>> candSet = new HashSet<Pair<Hop,Hop>>();
		Hop.resetVisitStatus(rootsSB2);
		for( Hop h : rootsSB2 )
			rProbeAndAddHopsToCandidateSet(h, probeSet, candSet);
		
		//step 3: create additional cuts
		for( Pair<Hop,Hop> p : candSet ) 
		{
			String varname = _varnamePredix + _seq.getNextID();
			
			Hop hop = p.getKey();
			Hop c = p.getValue();

			DataOp tread = new DataOp(varname, c.getDataType(), c.getValueType(), DataOpTypes.TRANSIENTREAD, 
					null, c.getDim1(), c.getDim2(), c.getNnz(), c.getRowsInBlock(), c.getColsInBlock());
			tread.setVisited(VisitStatus.DONE);
			HopRewriteUtils.copyLineNumbers(c, tread);

			DataOp twrite = new DataOp(varname, c.getDataType(), c.getValueType(), c, DataOpTypes.TRANSIENTWRITE, null);
			twrite.setVisited(VisitStatus.DONE);
			twrite.setOutputParams(c.getDim1(), c.getDim2(), c.getNnz(), c.getRowsInBlock(), c.getColsInBlock());
			HopRewriteUtils.copyLineNumbers(c, twrite);
			
			//create additional cut by rewriting both hop dags 
			int pos = HopRewriteUtils.getChildReferencePos(hop, c);
			HopRewriteUtils.removeChildReferenceByPos(hop, c, pos);
			HopRewriteUtils.addChildReference(hop, tread, pos);			
		
			//update live in and out of new statement block (for piggybacking)
			DataIdentifier diVar = new DataIdentifier(varname);
			diVar.setDimensions(c.getDim1(), c.getDim2());
			diVar.setBlockDimensions(c.getRowsInBlock(), c.getColsInBlock());
			diVar.setDataType(c.getDataType());
			diVar.setValueType(c.getValueType());
			sb1out.addVariable(varname, new DataIdentifier(diVar));
			sb2in.addVariable(varname, new DataIdentifier(diVar));
			
			rootsSB1.add(twrite);
		}
	}
	
	/**
	 * 
	 * @param hop
	 * @param probeSet
	 * @param candChilds 
	 * @param cand 
	 */
	private void rAddHopsToProbeSet( Hop hop, HashSet<Hop> probeSet )
	{
		if( hop.getVisited() == VisitStatus.DONE )
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
	
		hop.setVisited(VisitStatus.DONE);	
	}
	
	/**
	 * 
	 * 
	 * NOTE: candset is a set of parent-child pairs because a parent might have 
	 * multiple references to replicated hops.
	 * 
	 * @param hop
	 * @param probeSet
	 * @param candSet
	 */
	private void rProbeAndAddHopsToCandidateSet( Hop hop, HashSet<Hop> probeSet, HashSet<Pair<Hop,Hop>> candSet )
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;

		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )  {
				//probe for replicated operator, if any child is replicated, keep parent
				//for cut between parent-child; otherwise recursively descend.
				if( !probeSet.contains(c) )
					rProbeAndAddHopsToCandidateSet(c, probeSet, candSet);
				else
				{
					candSet.add(new Pair<Hop,Hop>(hop,c)); 
				}
			}
		
		hop.setVisited(VisitStatus.DONE);	
	}
	
	/**
	 * 
	 * @param cand
	 * @param candChilds
	 */
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
	
	/**
	 * 
	 * @param hop
	 * @param cand
	 * @param candChilds
	 * @param collect
	 */
	private void rCollectCandidateChildOperators( Hop hop, ArrayList<Hop> cand, HashSet<Hop> candChilds, boolean collect )
	{
		if( hop.getVisited() == VisitStatus.DONE )
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
		
		hop.setVisited(VisitStatus.DONE);
	}
}
