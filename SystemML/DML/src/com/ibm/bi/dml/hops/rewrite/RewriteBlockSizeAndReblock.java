/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.MemoTable;
import com.ibm.bi.dml.hops.ReblockOp;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.DataType;

/**
 * Rule: BlockSizeAndReblock. For all statement blocks, determine
 * "optimal" block size, and place reblock Hops. For now, we just
 * use BlockSize 1K x 1K and do reblock after Persistent Reads and
 * before Persistent Writes.
 */
public class RewriteBlockSizeAndReblock extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots)
		throws HopsException
	{
		if( roots == null )
			return null;
		
		for( Hop h : roots ) 
			rule_BlockSizeAndReblock(h, DMLTranslator.DMLBlockSize);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root) 
		throws HopsException
	{
		// do nothing (does not apply to predicate hop DAGs) 
		return root;
	}

	private void rule_BlockSizeAndReblock(Hop hop, final int GLOBAL_BLOCKSIZE) 
		throws HopsException 
	{
		// Go to the source(s) of the DAG
		for (Hop hi : hop.getInput()) {
			if (hop.get_visited() != Hop.VISIT_STATUS.DONE)
				rule_BlockSizeAndReblock(hi, GLOBAL_BLOCKSIZE);
		}

		boolean canReblock = true;
		
		if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
			canReblock = false;
		
		if (hop instanceof DataOp) {

			// if block size does not match
			if (canReblock && hop.get_dataType() != DataType.SCALAR
					&& (hop.get_rows_in_block() != GLOBAL_BLOCKSIZE || hop.get_cols_in_block() != GLOBAL_BLOCKSIZE)) {

				if (((DataOp) hop).get_dataop() == DataOp.DataOpTypes.PERSISTENTREAD) {
				
					// insert reblock after the hop
					ReblockOp r = new ReblockOp(hop, GLOBAL_BLOCKSIZE, GLOBAL_BLOCKSIZE);
					r.setAllPositions(hop.getBeginLine(), hop.getBeginColumn(), hop.getEndLine(), hop.getEndColumn());
					r.refreshMemEstimates(new MemoTable());
					r.set_visited(Hop.VISIT_STATUS.DONE);
				
				} else if (((DataOp) hop).get_dataop() == DataOp.DataOpTypes.PERSISTENTWRITE) {

					if (hop.get_rows_in_block() == -1 && hop.get_cols_in_block() == -1) {

						// if this dataop is for cell ouput, then no reblock is
						// needed as (A) all jobtypes can produce block2cell and
						// cell2cell and (B) we don't generate an explicit
						// instruction for it (the info is conveyed through
						// OutputInfo.

					} else if (hop.getInput().get(0) instanceof ReblockOp && hop.getInput().get(0).getParent().size() == 1) {

						// if a reblock is feeding into this, then use it if
						// this is
						// the only parent, otherwise new Reblock

						hop.getInput().get(0).set_rows_in_block(hop.get_rows_in_block());
						hop.getInput().get(0).set_cols_in_block(hop.get_cols_in_block());

					} else {

						ReblockOp r = new ReblockOp(hop);
						r.setAllPositions(hop.getBeginLine(), hop.getBeginColumn(), hop.getEndLine(), hop.getEndColumn());
						r.refreshMemEstimates(new MemoTable());
						r.set_visited(Hop.VISIT_STATUS.DONE);
					}

				} else if (((DataOp) hop).get_dataop() == DataOp.DataOpTypes.TRANSIENTWRITE
						|| ((DataOp) hop).get_dataop() == DataOp.DataOpTypes.TRANSIENTREAD) {
					if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE ) {
						// simply copy the values from its input
						hop.set_rows_in_block(hop.getInput().get(0).get_rows_in_block());
						hop.set_cols_in_block(hop.getInput().get(0).get_cols_in_block());
					}
					else {
						// by default, all transient reads and writes are in blocked format
						hop.set_rows_in_block(GLOBAL_BLOCKSIZE);
						hop.set_cols_in_block(GLOBAL_BLOCKSIZE);
					}

				} else {
					throw new HopsException(hop.printErrorLocation() + "unexpected non-scalar Data HOP in reblock.\n");
				}
			}
		} else {
			// TODO: following two lines are commented, and the subsequent hack is used instead!
			//set_rows_per_block(GLOBAL_BLOCKSIZE);
			//set_cols_per_block(GLOBAL_BLOCKSIZE);
			
			// TODO: this is hack!
			/*
			 * Handle hops whose output dimensions are unknown!
			 * 
			 * Constraint C1:
			 * Currently, only ctable() and groupedAggregate() fall into this category.
			 * The MR jobs for both these functions run in "cell" mode and hence make their
			 * blocking dimensions to (-1,-1).
			 * 
			 * Constraint C2:
			 * Blocking dimensions are not applicable for hops that produce scalars. 
			 * CMCOV and GroupedAgg jobs always run in "cell" mode, and hence they 
			 * produce output in cell format.
			 * 
			 * Constraint C3:
			 * Remaining hops will get their blocking dimensions from their input hops.
			 */
			
			if ( hop instanceof ReblockOp ) {
				hop.set_rows_in_block(GLOBAL_BLOCKSIZE);
				hop.set_cols_in_block(GLOBAL_BLOCKSIZE);
			}
			
			// Constraint C1:
			//else if ( (this instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)this)._op == ParamBuiltinOp.GROUPEDAGG) ) {
			//	set_rows_in_block(-1);
			//	set_cols_in_block(-1);
			//}
			
			// Constraint C2:
			else if ( hop.get_dataType() == DataType.SCALAR ) {
				hop.set_rows_in_block(-1);
				hop.set_cols_in_block(-1);
			}

			// Constraint C3:
			else {
				if ( !canReblock ) {
					hop.set_rows_in_block(-1);
					hop.set_cols_in_block(-1);
				}
				else {
					hop.set_rows_in_block(GLOBAL_BLOCKSIZE);
					hop.set_cols_in_block(GLOBAL_BLOCKSIZE);
				}
				
				// if any input is not blocked then the output of current Hop should not be blocked
				for ( Hop h : hop.getInput() ) {
					if ( h.get_dataType() == DataType.MATRIX && h.get_rows_in_block() == -1 && h.get_cols_in_block() == -1 ) {
						hop.set_rows_in_block(-1);
						hop.set_cols_in_block(-1);
						break;
					}
				}
			}
		}

		hop.set_visited(Hop.VISIT_STATUS.DONE);

	}
}
