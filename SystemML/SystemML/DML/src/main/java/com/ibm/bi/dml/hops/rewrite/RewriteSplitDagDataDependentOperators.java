/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashSet;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.hops.Hop.OpOp3;
import com.ibm.bi.dml.hops.Hop.ParamBuiltinOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.ParameterizedBuiltinOp;
import com.ibm.bi.dml.hops.TertiaryOp;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

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
		
		//split hop dag on demand
		if( cand.size()>0 )
		{
			try
			{
				//duplicate sb incl live variable sets
				StatementBlock sb1 = new StatementBlock();
				sb1.setAllPositions(sb.getFilename(), sb.getBeginLine(), sb.getBeginColumn(), sb.getEndLine(), sb.getEndColumn());
				sb1.setLiveIn(new VariableSet());
				sb1.setLiveOut(new VariableSet());
				
				//move csv reads incl reblock to new statement block
				//(and replace original persistent read with transient read)
				ArrayList<Hop> sb1hops = new ArrayList<Hop>();			
				for( Hop c : cand )
				{
					//if there are already transient writes use them and don't introduce artificial variables 
					boolean hasTWrites = hasTransientWriteParents(c);
					
					String varname = null;
					long rlen = c.get_dim1();
					long clen = c.get_dim2();
					long nnz = c.getNnz();
					long brlen = c.get_rows_in_block();
					long bclen = c.get_cols_in_block();
					
					if( hasTWrites ) //reuse existing transient_write
					{
						Hop twrite = getFirstTransientWriteParent(c);
						varname = twrite.get_name();
						
						//create new transient read
						DataOp tread = new DataOp(varname, DataType.MATRIX, ValueType.DOUBLE,
			                    DataOpTypes.TRANSIENTREAD, null, rlen, clen, nnz, brlen, bclen);
						tread.set_visited(VISIT_STATUS.DONE);
						HopRewriteUtils.copyLineNumbers(c, tread);
						
						//replace data-dependent operator with transient read
						ArrayList<Hop> parents = new ArrayList<Hop>(c.getParent());
						for( int i=0; i<parents.size(); i++ )
						{
							Hop parent = parents.get(i);
							if( parent != twrite ) {
								int pos = HopRewriteUtils.getChildReferencePos(parent, c);
								HopRewriteUtils.removeChildReferenceByPos(parent, c, pos);
								HopRewriteUtils.addChildReference(parent, tread, pos);
							}
							else
								sb.get_hops().remove(parent);
						}
						
						//add data-dependent operator sub dag to first statement block
						sb1hops.add(twrite);
					}
					else //create transient write to artificial variables
					{
						varname = _varnamePredix + _seq.getNextID();
						
						//create new transient read
						DataOp tread = new DataOp(varname, DataType.MATRIX, ValueType.DOUBLE,
			                    DataOpTypes.TRANSIENTREAD, null, rlen, clen, nnz, brlen, bclen);
						tread.set_visited(VISIT_STATUS.DONE);
						HopRewriteUtils.copyLineNumbers(c, tread);
						
						//replace data-dependent operator with transient read
						ArrayList<Hop> parents = new ArrayList<Hop>(c.getParent());
						for( int i=0; i<parents.size(); i++ )
						{
							Hop parent = parents.get(i);
							int pos = HopRewriteUtils.getChildReferencePos(parent, c);
							HopRewriteUtils.removeChildReferenceByPos(parent, c, pos);
							HopRewriteUtils.addChildReference(parent, tread, pos);
						}
						
						//add data-dependent operator sub dag to first statement block
						DataOp twrite = new DataOp(varname, DataType.MATRIX, ValueType.DOUBLE,
								                   c, DataOpTypes.TRANSIENTWRITE, null);
						twrite.set_visited(VISIT_STATUS.DONE);
						twrite.setOutputParams(rlen, clen, nnz, brlen, bclen);
						HopRewriteUtils.copyLineNumbers(c, twrite);
						sb1hops.add(twrite);	
					}
					
					//update live in and out of new statement block (for piggybacking)
					DataIdentifier diVar = new DataIdentifier(varname);
					diVar.setDimensions(rlen, clen);
					diVar.setBlockDimensions(brlen, bclen);
					diVar.setDataType(DataType.MATRIX);
					diVar.setValueType(ValueType.DOUBLE);
					sb1.liveOut().addVariable(varname, new DataIdentifier(diVar));
					sb.liveIn().addVariable(varname, new DataIdentifier(diVar));
				}
	
				//ensure disjoint operators across DAGs (prevent replicated operations)
				handleReplicatedOperators( sb1hops, sb.get_hops() );
				
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
			
			LOG.debug("Applied splitDagDataDependentOperators.");
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
			collectDataDependentOperators(root, cand);
	}
	
	/**
	 * 
	 * @param root
	 * @param cand
	 */
	private void collectDataDependentOperators( Hop hop, ArrayList<Hop> cand )
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		//prevent unnecessary dag split (dims known or no consumer operations)
		boolean noSplitRequired = ( hop.dimsKnown() || HopRewriteUtils.hasOnlyWriteParents(hop, true, true) );
		boolean investigateChilds = true;
		
		//collect data dependent operations (to be extended as necessary)
		//#1 removeEmpty
		if(    hop instanceof ParameterizedBuiltinOp 
			&& ((ParameterizedBuiltinOp) hop).getOp()==ParamBuiltinOp.RMEMPTY 
			&& !noSplitRequired )
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
				                  || p instanceof UnaryOp && ((UnaryOp)p).get_op()==OpOp1.NROW);
				onlyPMM &= (p instanceof AggBinaryOp && hop == p.getInput().get(0));
			}
			pbhop.setOutputEmptyBlocks(!noEmptyBlocks);
			
			if( onlyPMM && diagInput ){
				//configure rmEmpty to directly output selection vector
				//(only applied if dynamic recompilation enabled)
				
				if( DMLScript.rtplatform != RUNTIME_PLATFORM.HADOOP )	
					pbhop.setOutputPermutationMatrix(true);
				for( Hop p : hop.getParent() )
					((AggBinaryOp)p).setHasLeftPMInput(true);
					
			}
		}
		
		//#2 ctable with unknown dims
	    if(    hop instanceof TertiaryOp 
			&& ((TertiaryOp) hop).getOp()==OpOp3.CTABLE 
			&& hop.getInput().size() < 4 //dims not provided
			&& !noSplitRequired )
		{
			cand.add(hop);
			investigateChilds = false;
		}
		
		//process children (if not already found a special operators;
	    //otherwise, processed by recursive rule application)
		if( investigateChilds )
		    if( hop.getInput()!=null )
				for( Hop c : hop.getInput() )
					collectDataDependentOperators(c, cand);
		
		hop.set_visited(VISIT_STATUS.DONE);
	}

	/**
	 * 
	 * @param hop
	 * @return
	 */
	private boolean hasTransientWriteParents( Hop hop )
	{
		for( Hop p : hop.getParent() )
			if( p instanceof DataOp && ((DataOp)p).get_dataop()==DataOpTypes.TRANSIENTWRITE )
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
			if( p instanceof DataOp && ((DataOp)p).get_dataop()==DataOpTypes.TRANSIENTWRITE )
				return p;
		return null;
	}
	
	/**
	 * 
	 * @param rootsSB1
	 * @param rootsSB2
	 */
	private void handleReplicatedOperators( ArrayList<Hop> rootsSB1, ArrayList<Hop> rootsSB2 )
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
			
			DataOp tread = new DataOp(varname, c.get_dataType(), c.get_valueType(), DataOpTypes.TRANSIENTREAD, 
					null, c.get_dim1(), c.get_dim2(), c.getNnz(), c.get_rows_in_block(), c.get_cols_in_block());
			tread.set_visited(VISIT_STATUS.DONE);
			HopRewriteUtils.copyLineNumbers(c, tread);
			
			DataOp twrite = new DataOp(varname, c.get_dataType(), c.get_valueType(), c, DataOpTypes.TRANSIENTWRITE, null);
			twrite.set_visited(VISIT_STATUS.DONE);
			twrite.setOutputParams(c.get_dim1(), c.get_dim2(), c.getNnz(), c.get_rows_in_block(), c.get_cols_in_block());
			HopRewriteUtils.copyLineNumbers(c, twrite);
			
			//create additional cut by rewriting both hop dags 
			int pos = HopRewriteUtils.getChildReferencePos(hop, c);
			HopRewriteUtils.removeChildReferenceByPos(hop, c, pos);
			HopRewriteUtils.addChildReference(hop, tread, pos);			
			rootsSB1.add(twrite);
		}
	}
	
	/**
	 * 
	 * @param hop
	 * @param probeSet
	 */
	private void rAddHopsToProbeSet( Hop hop, HashSet<Hop> probeSet )
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		if( !(   (hop instanceof DataOp && !((DataOp)hop).isPersistentReadWrite() )
			   || hop instanceof LiteralOp) )
		{
			probeSet.add(hop);
		}
		
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rAddHopsToProbeSet(c, probeSet);
		
		hop.set_visited(VISIT_STATUS.DONE);	
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
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;

		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )  {
				//probe for replicated operator, if any child is replicated, keep parent
				//for cut between parent-child; otherwise recursively descend.
				if( !probeSet.contains(c) )
					rProbeAndAddHopsToCandidateSet(c, probeSet, candSet);
				else
					candSet.add(new Pair<Hop,Hop>(hop,c)); 
			}
		
		hop.set_visited(VISIT_STATUS.DONE);	
	}
}
