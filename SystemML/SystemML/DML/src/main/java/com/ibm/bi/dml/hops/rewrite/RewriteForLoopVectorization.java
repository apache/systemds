/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.AggUnaryOp;
import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.AggOp;
import com.ibm.bi.dml.hops.Hop.Direction;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.LeftIndexingOp;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;

/**
 * Rule: Simplify program structure by pulling if or else statement body out
 * (removing the if statement block ifself) in order to allow intra-procedure
 * analysis to propagate exact statistics.
 * 
 */
public class RewriteForLoopVectorization extends StatementBlockRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final OpOp2[] MAP_SCALAR_AGGREGATE_SOURCE_OPS = new OpOp2[]{OpOp2.PLUS, OpOp2.MULT, OpOp2.MIN, OpOp2.MAX};
	private static final AggOp[] MAP_SCALAR_AGGREGATE_TARGET_OPS = new AggOp[]{AggOp.SUM,  AggOp.PROD, AggOp.MIN, AggOp.MAX};
	
	
	@Override
	public ArrayList<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
		throws HopsException 
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		
		if( sb instanceof ForStatementBlock )
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fs = (ForStatement) fsb.getStatement(0);
			Hop from = fsb.getFromHops();
			Hop to = fsb.getToHops();
			Hop incr = fsb.getIncrementHops();
			String iterVar = fsb.getIterPredicate().getIterVar().getName();
			
			if( fs.getBody()!=null && fs.getBody().size()==1 ) //single child block
			{
				StatementBlock csb = (StatementBlock) fs.getBody().get(0);
				if( !(   csb instanceof WhileStatementBlock  //last level block
					  || csb instanceof IfStatementBlock 
					  || csb instanceof ForStatementBlock ) )
				{
					//auto vectorzation pattern
					sb = vectorizeScalarAggregate(sb, csb, from, to, incr, iterVar);           //e.g., for(i){s = s + as.scalar(X[i,2])}
					sb = vectorizeElementwiseBinary(sb, csb, from, to, incr, iterVar);
					sb = vectorizeElementwiseUnary(sb, csb, from, to, incr, iterVar);
				}	
			}	
		}	
		
		//if no rewrite applied sb is the original for loop otherwise a last level statement block
		//that includes the equivalent vectorized operations.
		ret.add( sb );
		
		return ret;
	}
	
	/**
	 * Note: unnecessary row or column indexing then later removed via
	 * dynamic rewrites
	 * 
	 * @param sb
	 * @param csb
	 * @param from
	 * @param to
	 * @param increment
	 * @param itervar
	 * @return
	 * @throws HopsException
	 */
	private StatementBlock vectorizeScalarAggregate( StatementBlock sb, StatementBlock csb, Hop from, Hop to, Hop increment, String itervar ) 
		throws HopsException
	{
		StatementBlock ret = sb;
		
		//check supported increment values
		if( !(increment instanceof LiteralOp && ((LiteralOp)increment).getDoubleValue()==1.0) ){
			return ret;
		}
			
		//check for applicability
		boolean leftScalar = false;
		boolean rightScalar = false;
		boolean rowIx = false; //row or col
		
		if( csb.get_hops()!=null && csb.get_hops().size()==1 ){
			Hop root = csb.get_hops().get(0);
			
			if( root.get_dataType()==DataType.SCALAR && root.getInput().get(0) instanceof BinaryOp ) {
				BinaryOp bop = (BinaryOp) root.getInput().get(0);
				Hop left = bop.getInput().get(0);
				Hop right = bop.getInput().get(1);
				
				//check for left scalar plus
				if( HopRewriteUtils.isValidOp(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS) 
					&& left instanceof DataOp && left.get_dataType() == DataType.SCALAR
					&& root.get_name().equals(left.get_name()) 
					&& right instanceof UnaryOp && ((UnaryOp) right).get_op() == OpOp1.CAST_AS_SCALAR
					&& right.getInput().get(0) instanceof IndexingOp )
				{
					IndexingOp ix = (IndexingOp)right.getInput().get(0);
					if( ix.getRowLowerEqualsUpper() && ix.getInput().get(1) instanceof DataOp
						&& ix.getInput().get(1).get_name().equals(itervar) ){
						leftScalar = true;
						rowIx = true;
					}
					else if( ix.getColLowerEqualsUpper() && ix.getInput().get(3) instanceof DataOp
						&& ix.getInput().get(3).get_name().equals(itervar) ){
						leftScalar = true;
						rowIx = false;
					}
				}
				//check for right scalar plus
				else if( HopRewriteUtils.isValidOp(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS)  
					&& right instanceof DataOp && right.get_dataType() == DataType.SCALAR
					&& root.get_name().equals(right.get_name()) 
					&& left instanceof UnaryOp && ((UnaryOp) left).get_op() == OpOp1.CAST_AS_SCALAR
					&& left.getInput().get(0) instanceof IndexingOp )
				{
					IndexingOp ix = (IndexingOp)left.getInput().get(0);
					if( ix.getRowLowerEqualsUpper() && ix.getInput().get(1) instanceof DataOp
						&& ix.getInput().get(1).get_name().equals(itervar) ){
						rightScalar = true;
						rowIx = true;
					}
					else if( ix.getColLowerEqualsUpper() && ix.getInput().get(3) instanceof DataOp
						&& ix.getInput().get(3).get_name().equals(itervar) ){
						rightScalar = true;
						rowIx = false;
					}
				}
			}
		}
		
		//apply rewrite if possible
		if( leftScalar || rightScalar ) 
		{
			Hop root = csb.get_hops().get(0);
			BinaryOp bop = (BinaryOp) root.getInput().get(0);
			Hop cast = bop.getInput().get( leftScalar?1:0 );
			Hop ix = cast.getInput().get(0);
			int aggOpPos = HopRewriteUtils.getValidOpPos(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS);
			AggOp aggOp = MAP_SCALAR_AGGREGATE_TARGET_OPS[aggOpPos];
			//replace cast with sum
			AggUnaryOp newSum = new AggUnaryOp(cast.get_name(), DataType.SCALAR, ValueType.DOUBLE, aggOp, Direction.RowCol, ix);
			HopRewriteUtils.removeChildReference(cast, ix);
			HopRewriteUtils.removeChildReference(bop, cast);
			HopRewriteUtils.addChildReference(bop, newSum, leftScalar?1:0 );
			//modify indexing expression according to loop predicate from-to
			//NOTE: any redundant index operations are removed via dynamic algebraic simplification rewrites
			int index1 = rowIx ? 1 : 3;
			int index2 = rowIx ? 2 : 4;
			HopRewriteUtils.removeChildReferenceByPos(ix, ix.getInput().get(index1), index1);
			HopRewriteUtils.addChildReference(ix, from, index1);
			HopRewriteUtils.removeChildReferenceByPos(ix, ix.getInput().get(index2), index2);
			HopRewriteUtils.addChildReference(ix, to, index2);
			
			ret = csb;
			//ret.liveIn().removeVariable(itervar);
			LOG.debug("Applied vectorizeScalarSumForLoop.");
		}
		
		return ret;
	}
	
	/**
	 * Note: unnecessary row or column indexing then later removed via
	 * dynamic rewrites
	 * 
	 * @param sb
	 * @param csb
	 * @param from
	 * @param to
	 * @param increment
	 * @param itervar
	 * @return
	 * @throws HopsException
	 */
	private StatementBlock vectorizeElementwiseBinary( StatementBlock sb, StatementBlock csb, Hop from, Hop to, Hop increment, String itervar ) 
		throws HopsException
	{
		StatementBlock ret = sb;
		
		//check supported increment values
		if( !(increment instanceof LiteralOp && ((LiteralOp)increment).getDoubleValue()==1.0) ){
			return ret;
		}
			
		//check for applicability
		boolean apply = false;
		boolean rowIx = false; //row or col
		if( csb.get_hops()!=null && csb.get_hops().size()==1 )
		{
			Hop root = csb.get_hops().get(0);
			
			if( root.get_dataType()==DataType.MATRIX && root.getInput().get(0) instanceof LeftIndexingOp )
			{
				LeftIndexingOp lix = (LeftIndexingOp) root.getInput().get(0);
				Hop lixlhs = lix.getInput().get(0);
				Hop lixrhs = lix.getInput().get(1);
				
				if( lixlhs instanceof DataOp && lixrhs instanceof BinaryOp
					&& lixrhs.getInput().get(0) instanceof IndexingOp	
					&& lixrhs.getInput().get(1) instanceof IndexingOp
					&& lixrhs.getInput().get(0).getInput().get(0) instanceof DataOp
					&& lixrhs.getInput().get(1).getInput().get(0) instanceof DataOp)
				{			
					IndexingOp rix0 = (IndexingOp) lixrhs.getInput().get(0);
					IndexingOp rix1 = (IndexingOp) lixrhs.getInput().get(1);
					
					//check for rowwise
					if(    lix.getRowLowerEqualsUpper() && rix0.getRowLowerEqualsUpper() && rix1.getRowLowerEqualsUpper() 
						&& lix.getInput().get(2).get_name().equals(itervar)
						&& rix0.getInput().get(1).get_name().equals(itervar)
						&& rix1.getInput().get(1).get_name().equals(itervar))
					{
						apply = true;
						rowIx = true;
					}
					//check for colwise
					if(    lix.getColLowerEqualsUpper() && rix0.getColLowerEqualsUpper() && rix1.getColLowerEqualsUpper() 
						&& lix.getInput().get(4).get_name().equals(itervar)
						&& rix0.getInput().get(3).get_name().equals(itervar)
						&& rix1.getInput().get(3).get_name().equals(itervar))
					{
						apply = true;
						rowIx = false;
					}
				}
			}
		}	
		
		//apply rewrite if possible
		if( apply ) 
		{
			Hop root = csb.get_hops().get(0);
			LeftIndexingOp lix = (LeftIndexingOp) root.getInput().get(0);
			BinaryOp bop = (BinaryOp) lix.getInput().get(1);
			IndexingOp rix0 = (IndexingOp) bop.getInput().get(0);
			IndexingOp rix1 = (IndexingOp) bop.getInput().get(1);
			int index1 = rowIx ? 2 : 4;
			int index2 = rowIx ? 3 : 5;
			//modify left indexing bounds
			HopRewriteUtils.removeChildReferenceByPos(lix, lix.getInput().get(index1), index1 );
			HopRewriteUtils.addChildReference(lix, from, index1);
			HopRewriteUtils.removeChildReferenceByPos(lix, lix.getInput().get(index2), index2 );
			HopRewriteUtils.addChildReference(lix, to, index2);
			//modify both right indexing
			HopRewriteUtils.removeChildReferenceByPos(rix0, rix0.getInput().get(index1-1), index1-1 );
			HopRewriteUtils.addChildReference(rix0, from, index1-1);
			HopRewriteUtils.removeChildReferenceByPos(rix0, rix0.getInput().get(index2-1), index2-1 );
			HopRewriteUtils.addChildReference(rix0, to, index2-1);
			HopRewriteUtils.removeChildReferenceByPos(rix1, rix1.getInput().get(index1-1), index1-1 );
			HopRewriteUtils.addChildReference(rix1, from, index1-1);
			HopRewriteUtils.removeChildReferenceByPos(rix1, rix1.getInput().get(index2-1), index2-1 );
			HopRewriteUtils.addChildReference(rix1, to, index2-1);
			rix0.refreshSizeInformation();
			rix1.refreshSizeInformation();
			bop.refreshSizeInformation();
			lix.refreshSizeInformation();
			
			ret = csb;
			//ret.liveIn().removeVariable(itervar);
			LOG.debug("Applied vectorizeElementwiseBinaryForLoop.");
		}
		
		return ret;
	}
	
	/**
	 * Note: unnecessary row or column indexing then later removed via
	 * dynamic rewrites
	 * 
	 * @param sb
	 * @param csb
	 * @param from
	 * @param to
	 * @param increment
	 * @param itervar
	 * @return
	 * @throws HopsException
	 */
	private StatementBlock vectorizeElementwiseUnary( StatementBlock sb, StatementBlock csb, Hop from, Hop to, Hop increment, String itervar )
		throws HopsException
	{
		StatementBlock ret = sb;
		
		//check supported increment values
		if( !(increment instanceof LiteralOp && ((LiteralOp)increment).getDoubleValue()==1.0) ){
			return ret;
		}
			
		//check for applicability
		boolean apply = false;
		boolean rowIx = false; //row or col
		if( csb.get_hops()!=null && csb.get_hops().size()==1 )
		{
			Hop root = csb.get_hops().get(0);
			
			if( root.get_dataType()==DataType.MATRIX && root.getInput().get(0) instanceof LeftIndexingOp )
			{
				LeftIndexingOp lix = (LeftIndexingOp) root.getInput().get(0);
				Hop lixlhs = lix.getInput().get(0);
				Hop lixrhs = lix.getInput().get(1);
				
				if( lixlhs instanceof DataOp && lixrhs instanceof UnaryOp 
					&& lixrhs.getInput().get(0) instanceof IndexingOp
					&& lixrhs.getInput().get(0).getInput().get(0) instanceof DataOp )
				{
					IndexingOp rix = (IndexingOp) lixrhs.getInput().get(0);
					//check for rowwise
					if(    lix.getRowLowerEqualsUpper() && rix.getRowLowerEqualsUpper() 
						&& lix.getInput().get(2).get_name().equals(itervar)
						&& rix.getInput().get(1).get_name().equals(itervar) )
					{
						apply = true;
						rowIx = true;
					}
					//check for colwise
					if(    lix.getColLowerEqualsUpper() && rix.getColLowerEqualsUpper() 
						&& lix.getInput().get(4).get_name().equals(itervar)
						&& rix.getInput().get(3).get_name().equals(itervar) )
					{
						apply = true;
						rowIx = false;
					}
				}
			}
		}	
		
		//apply rewrite if possible
		if( apply ) 
		{
			Hop root = csb.get_hops().get(0);
			LeftIndexingOp lix = (LeftIndexingOp) root.getInput().get(0);
			UnaryOp uop = (UnaryOp) lix.getInput().get(1);
			IndexingOp rix = (IndexingOp) uop.getInput().get(0);
			int index1 = rowIx ? 2 : 4;
			int index2 = rowIx ? 3 : 5;
			//modify left indexing bounds
			HopRewriteUtils.removeChildReferenceByPos(lix, lix.getInput().get(index1), index1 );
			HopRewriteUtils.addChildReference(lix, from, index1);
			HopRewriteUtils.removeChildReferenceByPos(lix, lix.getInput().get(index2), index2 );
			HopRewriteUtils.addChildReference(lix, to, index2);
			//modify right indexing
			HopRewriteUtils.removeChildReferenceByPos(rix, rix.getInput().get(index1-1), index1-1 );
			HopRewriteUtils.addChildReference(rix, from, index1-1);
			HopRewriteUtils.removeChildReferenceByPos(rix, rix.getInput().get(index2-1), index2-1 );
			HopRewriteUtils.addChildReference(rix, to, index2-1);
			rix.refreshSizeInformation();
			uop.refreshSizeInformation();
			lix.refreshSizeInformation();
			
			ret = csb;
			//ret.liveIn().removeVariable(itervar);
			LOG.debug("Applied vectorizeElementwiseUnaryForLoop.");
		}
		
		return ret;
	}
	
	
}
