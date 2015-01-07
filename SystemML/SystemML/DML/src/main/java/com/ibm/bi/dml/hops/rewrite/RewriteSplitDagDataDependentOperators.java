/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.hops.Hop.OpOp3;
import com.ibm.bi.dml.hops.Hop.ParamBuiltinOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.hops.HopsException;
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

	private static String _varnamePredix = "_sbcutvar";
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
			cand.add(hop);
			investigateChilds = false;
			
			//keep interesting consumer information 
			boolean noEmptyBlocks = true; 
			for( Hop p : hop.getParent() ) {
				//list of operators without need for empty blocks to be extended as needed
				noEmptyBlocks &= (   p instanceof AggBinaryOp && hop == p.getInput().get(0) 
				                  || p instanceof UnaryOp && ((UnaryOp)p).get_op()==OpOp1.NROW);
			}
			((ParameterizedBuiltinOp) hop).setOutputEmptyBlocks(!noEmptyBlocks);
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
	
	private Hop getFirstTransientWriteParent( Hop hop )
	{
		for( Hop p : hop.getParent() )
			if( p instanceof DataOp && ((DataOp)p).get_dataop()==DataOpTypes.TRANSIENTWRITE )
				return p;
		return null;
	}
	
	/* OLD code from AssignmentStatement.controlStatement():
	 --- 
	public boolean containsIndividualStatementBlockOperations()
	{
		// if (DMLScript.ENABLE_DEBUG_MODE && !DMLScript.ENABLE_DEBUG_OPTIMIZER)
		if (DMLScript.ENABLE_DEBUG_MODE)
			return true;
		
		boolean ret = false;
		
		if( OptimizerUtils.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS )
		{
			//1) Leaf nodes with unknown size:
			//recompilation hook after reads with unknown size (this can currently only happen for csv)
			if( _source instanceof DataExpression && ((DataExpression)_source).isCSVReadWithUnknownSize() )
				ret = true;	
			
			//2) Data-dependent operations:
			
			//TODO additional candidates (currently, not enabled because worst-case estimates usually reasonable)
			//if( _source.toString().contains(Expression.DataOp.RAND.toString()) )
			//	ret = true;	
			//if( _source.toString().contains(Expression.ParameterizedBuiltinFunctionOp.GROUPEDAGG.toString()) )
			//	ret = true;	
			if( _source.toString().contains(Expression.ParameterizedBuiltinFunctionOp.RMEMPTY.toString()) )
				ret = true;	
			
			//recompilation hook after ctable because worst estimates usually too conservative 
			//(despite propagating worst-case estimates, especially if we not able to propagate sparsity)
			
			if( _source != null && _source.toString().contains(Expression.BuiltinFunctionOp.TABLE.toString()) 
				&& !isBuiltinCtableWithKnownDimensions(_source) ) //split only if unknown dimensions 
				ret = true;
		}
		//System.out.println(_source +": "+ret);
		
		return ret;
	}
	
	private static boolean isBuiltinCtableWithKnownDimensions( Expression expr )
	{
		boolean ret = false;
		
		if( expr instanceof BuiltinFunctionExpression ){
			BuiltinFunctionExpression bexpr = (BuiltinFunctionExpression) expr;
			ret = (    bexpr.getOpCode() == Expression.BuiltinFunctionOp.TABLE
					&& bexpr.getAllExpr()!=null && bexpr.getAllExpr().length>3 );
		}
		
		return ret;
	}
	 */
}
