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

import java.util.Arrays;
import java.util.List;

import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LeftIndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpData;

/**
 * Rule: Simplify program structure by pulling if or else statement body out
 * (removing the if statement block ifself) in order to allow intra-procedure
 * analysis to propagate exact statistics.
 * 
 */
public class RewriteForLoopVectorization extends StatementBlockRewriteRule
{
	private static final OpOp2[] MAP_SCALAR_AGGREGATE_SOURCE_OPS = new OpOp2[]{OpOp2.PLUS, OpOp2.MULT, OpOp2.MIN, OpOp2.MAX};
	private static final AggOp[] MAP_SCALAR_AGGREGATE_TARGET_OPS = new AggOp[]{AggOp.SUM,  AggOp.PROD, AggOp.MIN, AggOp.MAX};
	
	@Override
	public boolean createsSplitDag() {
		return false;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
	{
		if( sb instanceof ForStatementBlock )
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fs = (ForStatement) fsb.getStatement(0);
			Hop from = unwrap(fsb.getFromHops());
			Hop to = unwrap(fsb.getToHops());
			Hop incr = unwrap(fsb.getIncrementHops());
			String iterVar = fsb.getIterPredicate().getIterVar().getName();
			
			if( fs.getBody()!=null && fs.getBody().size()==1 ) //single child block
			{
				StatementBlock csb = fs.getBody().get(0);
				if( !(   csb instanceof WhileStatementBlock  //last level block
					  || csb instanceof IfStatementBlock 
					  || csb instanceof ForStatementBlock ) )
				{
					if( !(incr==null || incr instanceof LiteralOp 
							&& ((LiteralOp)incr).getDoubleValue()==1.0) ) {
						return Arrays.asList(sb);
					}
					
					//AUTO VECTORIZATION PATTERNS
					//Note: unnecessary row or column indexing then later removed via hop rewrites
					
					//e.g., for(i in a:b){s = s + as.scalar(X[i,2])} -> s = sum(X[a:b,2])
					sb = vectorizeScalarAggregate(sb, csb, from, to, incr, iterVar);
					
					//e.g., for(i in a:b){s = s + X[i,2]} -> s = sum(X[a:b,2])
					sb = vectorizeScalarAggregate2(sb, csb, from, to, incr, iterVar);
					
					//e.g., for(i in a:b){X[i,2] = Y[i,1] + Z[i,3]} -> X[a:b,2] = Y[a:b,1] + Z[a:b,3];
					sb = vectorizeElementwiseBinary(sb, csb, from, to, incr, iterVar);
					
					//e.g., for(i in a:b){X[i,2] = abs(Y[i,1])} -> X[a:b,2] = abs(Y[a:b,1]);
					sb = vectorizeElementwiseUnary(sb, csb, from, to, incr, iterVar);
				
					//e.g., for(i in a:b){X[7,i] = Y[1,i]} -> X[7,a:b] = Y[1,a:b];
					sb = vectorizeIndexedCopy(sb, csb, from, to, incr, iterVar);
				}
			}
		}
		
		//if no rewrite applied sb is the original for loop otherwise a last level statement block
		//that includes the equivalent vectorized operations.
		return Arrays.asList(sb);
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus sate) {
		return sbs;
	}
	
	private static StatementBlock vectorizeScalarAggregate( StatementBlock sb, StatementBlock csb, Hop from, Hop to, Hop increment, String itervar ) 
	{
		StatementBlock ret = sb;
		
		//check for applicability
		boolean leftScalar = false;
		boolean rightScalar = false;
		boolean rowIx = false; //row or col
		
		if( csb.getHops()!=null && csb.getHops().size()==1 ) {
			Hop root = csb.getHops().get(0);
			
			if( root.getDataType()==DataType.SCALAR && root.getInput(0) instanceof BinaryOp ) {
				BinaryOp bop = (BinaryOp) root.getInput(0);
				Hop left = bop.getInput(0);
				Hop right = bop.getInput(1);
				
				//check for left scalar plus
				if( HopRewriteUtils.isValidOp(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS) 
					&& left instanceof DataOp && left.getDataType() == DataType.SCALAR
					&& root.getName().equals(left.getName())
					&& right instanceof UnaryOp && ((UnaryOp) right).getOp() == OpOp1.CAST_AS_SCALAR
					&& right.getInput(0) instanceof IndexingOp )
				{
					IndexingOp ix = (IndexingOp)right.getInput(0);
					if( checkItervarIndexing(ix, itervar, true) ){
						leftScalar = true;
						rowIx = true;
					}
					else if( checkItervarIndexing(ix, itervar, false) ){
						leftScalar = true;
						rowIx = false;
					}
				}
				//check for right scalar plus
				else if( HopRewriteUtils.isValidOp(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS)  
					&& right instanceof DataOp && right.getDataType() == DataType.SCALAR
					&& root.getName().equals(right.getName()) 
					&& left instanceof UnaryOp && ((UnaryOp) left).getOp() == OpOp1.CAST_AS_SCALAR
					&& left.getInput(0) instanceof IndexingOp )
				{
					IndexingOp ix = (IndexingOp)left.getInput(0);
					if( checkItervarIndexing(ix, itervar, true) ){
						rightScalar = true;
						rowIx = true;
					}
					else if( checkItervarIndexing(ix, itervar, false) ){
						rightScalar = true;
						rowIx = false;
					}
				}
			}
		}
		
		//apply rewrite if possible
		if( leftScalar || rightScalar ) {
			Hop root = csb.getHops().get(0);
			BinaryOp bop = (BinaryOp) root.getInput(0);
			Hop cast = bop.getInput().get( leftScalar?1:0 );
			Hop ix = cast.getInput(0);
			int aggOpPos = HopRewriteUtils.getValidOpPos(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS);
			AggOp aggOp = MAP_SCALAR_AGGREGATE_TARGET_OPS[aggOpPos];
			
			//replace cast with sum
			AggUnaryOp newSum = HopRewriteUtils.createAggUnaryOp(ix, aggOp, Direction.RowCol);
			HopRewriteUtils.removeChildReference(cast, ix);
			HopRewriteUtils.removeChildReference(bop, cast);
			HopRewriteUtils.addChildReference(bop, newSum, leftScalar?1:0 );
			
			//modify indexing expression according to loop predicate from-to
			//NOTE: any redundant index operations are removed via dynamic algebraic simplification rewrites
			int index1 = rowIx ? 1 : 3;
			int index2 = rowIx ? 2 : 4;
			HopRewriteUtils.replaceChildReference(ix, ix.getInput().get(index1), from, index1);
			HopRewriteUtils.replaceChildReference(ix, ix.getInput().get(index2), to, index2);
			
			//update indexing size information
			if( rowIx )
				((IndexingOp)ix).setRowLowerEqualsUpper(false);
			else
				((IndexingOp)ix).setColLowerEqualsUpper(false);
			ix.refreshSizeInformation();
			Hop.resetVisitStatus(csb.getHops(), true);
			
			ret = csb;
			LOG.debug("Applied vectorizeScalarSumForLoop.");
		}
		
		return ret;
	}
	
	private static StatementBlock vectorizeScalarAggregate2( StatementBlock sb, StatementBlock csb, Hop from, Hop to, Hop increment, String itervar ) 
	{
		StatementBlock ret = sb;
		
		//check for applicability
		boolean leftScalar = false;
		boolean rightScalar = false;
		boolean rowIx = false; //row or col
		
		if( csb.getHops()!=null && csb.getHops().size()==1 ) {
			Hop root = csb.getHops().get(0);
			
			if( root.getDataType()==DataType.SCALAR && root.getInput(0) instanceof BinaryOp ) {
				BinaryOp bop = (BinaryOp) root.getInput(0);
				Hop left = bop.getInput(0);
				Hop right = bop.getInput(1);
				
				//check for left scalar plus
				if( HopRewriteUtils.isValidOp(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS) 
					&& left instanceof DataOp && left.getDataType() == DataType.SCALAR
					&& root.getName().equals(left.getName())
					&& right instanceof IndexingOp && right.isScalar())
				{
					if( checkItervarIndexing((IndexingOp)right, itervar, true) ){
						leftScalar = true;
						rowIx = true;
					}
					else if( checkItervarIndexing((IndexingOp)right, itervar, false) ){
						leftScalar = true;
						rowIx = false;
					}
				}
				//check for right scalar plus
				else if( HopRewriteUtils.isValidOp(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS)  
					&& right instanceof DataOp && right.getDataType() == DataType.SCALAR
					&& root.getName().equals(right.getName()) 
					&& left instanceof IndexingOp && left.isScalar())
				{
					if( checkItervarIndexing((IndexingOp)left, itervar, true) ){
						rightScalar = true;
						rowIx = true;
					}
					else if( checkItervarIndexing((IndexingOp)left, itervar, false) ){
						rightScalar = true;
						rowIx = false;
					}
				}
			}
		}
		
		//apply rewrite if possible
		if( leftScalar || rightScalar ) {
			Hop root = csb.getHops().get(0);
			BinaryOp bop = (BinaryOp) root.getInput(0);
			Hop ix = bop.getInput().get( leftScalar?1:0 );
			int aggOpPos = HopRewriteUtils.getValidOpPos(bop.getOp(), MAP_SCALAR_AGGREGATE_SOURCE_OPS);
			AggOp aggOp = MAP_SCALAR_AGGREGATE_TARGET_OPS[aggOpPos];
			
			//replace cast with sum
			AggUnaryOp newSum = HopRewriteUtils.createAggUnaryOp(ix, aggOp, Direction.RowCol);
			HopRewriteUtils.removeChildReference(bop, ix);
			HopRewriteUtils.addChildReference(bop, newSum, leftScalar?1:0 );
			
			//modify indexing expression according to loop predicate from-to
			//NOTE: any redundant index operations are removed via dynamic algebraic simplification rewrites
			int index1 = rowIx ? 1 : 3;
			int index2 = rowIx ? 2 : 4;
			HopRewriteUtils.replaceChildReference(ix, ix.getInput().get(index1), from, index1);
			HopRewriteUtils.replaceChildReference(ix, ix.getInput().get(index2), to, index2);
			
			//update indexing size information
			if( rowIx )
				((IndexingOp)ix).setRowLowerEqualsUpper(false);
			else
				((IndexingOp)ix).setColLowerEqualsUpper(false);
			ix.setDataType(DataType.MATRIX);
			ix.refreshSizeInformation();
			Hop.resetVisitStatus(csb.getHops(), true);
			
			ret = csb;
			LOG.debug("Applied vectorizeScalarSumForLoop2.");
		}
		
		return ret;
	}
	
	private static StatementBlock vectorizeElementwiseBinary( StatementBlock sb, StatementBlock csb, Hop from, Hop to, Hop increment, String itervar ) 
	{
		StatementBlock ret = sb;

		//check for applicability
		boolean apply = false;
		boolean rowIx = false; //row or col
		if( csb.getHops()!=null && csb.getHops().size()==1 ) {
			Hop root = csb.getHops().get(0);

			if( root.getDataType()==DataType.MATRIX && root.getInput(0) instanceof LeftIndexingOp ) {
				LeftIndexingOp lix = (LeftIndexingOp) root.getInput(0);
				Hop lixlhs = lix.getInput(0);
				Hop lixrhs = lix.getInput(1);

				if( lixlhs instanceof DataOp && lixrhs instanceof BinaryOp
					&& lixrhs.getInput(0) instanceof IndexingOp
					&& lixrhs.getInput(1) instanceof IndexingOp
					&& lixrhs.getInput(0).getInput(0) instanceof DataOp
					&& lixrhs.getInput(1).getInput(0) instanceof DataOp)
				{
					IndexingOp rix0 = (IndexingOp) lixrhs.getInput(0);
					IndexingOp rix1 = (IndexingOp) lixrhs.getInput(1);

					//check for rowwise
					if(    lix.isRowLowerEqualsUpper() && rix0.isRowLowerEqualsUpper() && rix1.isRowLowerEqualsUpper() 
						&& lix.getInput(2).getName().equals(itervar)
						&& rix0.getInput(1).getName().equals(itervar)
						&& rix1.getInput(1).getName().equals(itervar))
					{
						apply = true;
						rowIx = true;
					}
					//check for colwise
					if(    lix.isColLowerEqualsUpper() && rix0.isColLowerEqualsUpper() && rix1.isColLowerEqualsUpper() 
						&& lix.getInput(4).getName().equals(itervar)
						&& rix0.getInput(3).getName().equals(itervar)
						&& rix1.getInput(3).getName().equals(itervar))
					{
						apply = true;
						rowIx = false;
					}
				}
			}
		}
		
		//apply rewrite if possible
		if( apply ) {
			Hop root = csb.getHops().get(0);
			LeftIndexingOp lix = (LeftIndexingOp) root.getInput(0);
			BinaryOp bop = (BinaryOp) lix.getInput(1);
			IndexingOp rix0 = (IndexingOp) bop.getInput(0);
			IndexingOp rix1 = (IndexingOp) bop.getInput(1);
			int index1 = rowIx ? 2 : 4;
			int index2 = rowIx ? 3 : 5;
			//modify left indexing bounds
			HopRewriteUtils.replaceChildReference(lix, lix.getInput().get(index1),from, index1);
			HopRewriteUtils.replaceChildReference(lix, lix.getInput().get(index2),to, index2);
			//modify both right indexing
			HopRewriteUtils.replaceChildReference(rix0, rix0.getInput(index1-1), from, index1-1);
			HopRewriteUtils.replaceChildReference(rix0, rix0.getInput(index2-1), to, index2-1);
			HopRewriteUtils.replaceChildReference(rix1, rix1.getInput(index1-1), from, index1-1);
			HopRewriteUtils.replaceChildReference(rix1, rix1.getInput(index2-1), to, index2-1);
			updateLeftAndRightIndexingSizes(rowIx, lix, rix0, rix1);
			bop.refreshSizeInformation();
			lix.refreshSizeInformation(); //after bop update
			Hop.resetVisitStatus(csb.getHops(), true);
			
			ret = csb;
			//ret.liveIn().removeVariable(itervar);
			LOG.debug("Applied vectorizeElementwiseBinaryForLoop.");
		}
		
		return ret;
	}
	
	private static StatementBlock vectorizeElementwiseUnary( StatementBlock sb, StatementBlock csb, Hop from, Hop to, Hop increment, String itervar )
	{
		StatementBlock ret = sb;
		
		//check for applicability
		boolean apply = false;
		boolean rowIx = false; //row or col
		if( csb.getHops()!=null && csb.getHops().size()==1 ) {
			Hop root = csb.getHops().get(0);
			
			if( root.getDataType()==DataType.MATRIX && root.getInput(0) instanceof LeftIndexingOp ) {
				LeftIndexingOp lix = (LeftIndexingOp) root.getInput(0);
				Hop lixlhs = lix.getInput(0);
				Hop lixrhs = lix.getInput(1);
				
				if( lixlhs instanceof DataOp && lixrhs instanceof UnaryOp 
					&& lixrhs.getInput(0) instanceof IndexingOp
					&& lixrhs.getInput(0).getInput(0) instanceof DataOp )
				{
					boolean[] tmp = checkLeftAndRightIndexing(lix, 
							(IndexingOp) lixrhs.getInput(0), itervar);
					apply = tmp[0];
					rowIx = tmp[1];
				}
			}
		}
		
		//apply rewrite if possible
		if( apply ) {
			Hop root = csb.getHops().get(0);
			LeftIndexingOp lix = (LeftIndexingOp) root.getInput(0);
			UnaryOp uop = (UnaryOp) lix.getInput(1);
			IndexingOp rix = (IndexingOp) uop.getInput(0);
			int index1 = rowIx ? 2 : 4;
			int index2 = rowIx ? 3 : 5;
			//modify left indexing bounds
			HopRewriteUtils.replaceChildReference(lix, lix.getInput(index1), from, index1);
			HopRewriteUtils.replaceChildReference(lix, lix.getInput(index2), to, index2);
			//modify right indexing
			HopRewriteUtils.replaceChildReference(rix, rix.getInput(index1-1), from, index1-1);
			HopRewriteUtils.replaceChildReference(rix, rix.getInput(index2-1), to, index2-1);
			updateLeftAndRightIndexingSizes(rowIx, lix, rix);
			uop.refreshSizeInformation();
			lix.refreshSizeInformation(); //after uop update
			Hop.resetVisitStatus(csb.getHops(), true);
			
			ret = csb;
			LOG.debug("Applied vectorizeElementwiseUnaryForLoop.");
		}
		
		return ret;
	}
	
	private static StatementBlock vectorizeIndexedCopy( StatementBlock sb, StatementBlock csb, Hop from, Hop to, Hop increment, String itervar )
	{
		StatementBlock ret = sb;
		
		//check for applicability
		boolean apply = false;
		boolean rowIx = false; //row or col
		if( csb.getHops()!=null && csb.getHops().size()==1 )
		{
			Hop root = csb.getHops().get(0);
			
			if( root.getDataType()==DataType.MATRIX && root.getInput(0) instanceof LeftIndexingOp )
			{
				LeftIndexingOp lix = (LeftIndexingOp) root.getInput(0);
				Hop lixlhs = lix.getInput(0);
				Hop lixrhs = lix.getInput(1);
				
				if( lixlhs instanceof DataOp && lixrhs instanceof IndexingOp
					&& lixrhs.getInput(0) instanceof DataOp )
				{
					boolean[] tmp = checkLeftAndRightIndexing(lix, (IndexingOp)lixrhs, itervar);
					apply = tmp[0];
					rowIx = tmp[1];
				}
			}
		}
		
		//apply rewrite if possible
		if( apply ) {
			Hop root = csb.getHops().get(0);
			LeftIndexingOp lix = (LeftIndexingOp) root.getInput(0);
			IndexingOp rix = (IndexingOp) lix.getInput(1);
			int index1 = rowIx ? 2 : 4;
			int index2 = rowIx ? 3 : 5;
			//modify left indexing bounds
			HopRewriteUtils.replaceChildReference(lix, lix.getInput(index1), from, index1);
			HopRewriteUtils.replaceChildReference(lix, lix.getInput(index2), to, index2);
			//modify right indexing
			HopRewriteUtils.replaceChildReference(rix, rix.getInput(index1-1), from, index1-1);
			HopRewriteUtils.replaceChildReference(rix, rix.getInput(index2-1), to, index2-1);
			updateLeftAndRightIndexingSizes(rowIx, lix, rix);
			Hop.resetVisitStatus(csb.getHops(), true);
			
			ret = csb;
			LOG.debug("Applied vectorizeIndexedCopy.");
		}
		
		return ret;
	}
	
	private static boolean checkItervarIndexing(IndexingOp ix, String itervar, boolean row) {
		return ix.isRowLowerEqualsUpper() 
			&& ix.getInput(row?1:3) instanceof DataOp
			&& ix.getInput(row?1:3).getName().equals(itervar);
	}
	
	private static boolean[] checkLeftAndRightIndexing(LeftIndexingOp lix, IndexingOp rix, String itervar) {
		boolean[] ret = new boolean[2]; //apply, rowIx
		
		//check for rowwise
		if(    lix.isRowLowerEqualsUpper() && rix.isRowLowerEqualsUpper()
			&& lix.getInput(2).getName().equals(itervar)
			&& rix.getInput(1).getName().equals(itervar) ) {
			ret[0] = true;
			ret[1] = true;
		}
		//check for colwise
		if(    lix.isColLowerEqualsUpper() && rix.isColLowerEqualsUpper()
			&& lix.getInput(4).getName().equals(itervar)
			&& rix.getInput(3).getName().equals(itervar) ) {
			ret[0] = true;
			ret[1] = false;
		}
		
		return ret;
	} 
	
	private static void updateLeftAndRightIndexingSizes(boolean rowIx, LeftIndexingOp lix, IndexingOp... rix) {
		//unset special flags
		if( rowIx ) {
			lix.setRowLowerEqualsUpper(false);
			for( IndexingOp rixi : rix )
				rixi.setRowLowerEqualsUpper(false);
		}
		else {
			lix.setColLowerEqualsUpper(false);
			for( IndexingOp rixi : rix )
				rixi.setColLowerEqualsUpper(false);
		}
		for( IndexingOp rixi : rix )
			rixi.refreshSizeInformation();
		lix.refreshSizeInformation();
	}
	
	private Hop unwrap(Hop hop) {
		return HopRewriteUtils.isData(hop, OpOpData.TRANSIENTWRITE) ? hop.getInput(0) : hop;
	}
}
