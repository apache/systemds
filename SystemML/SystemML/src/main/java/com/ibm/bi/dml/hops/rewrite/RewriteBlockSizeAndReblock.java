/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state)
		throws HopsException
	{
		if( roots == null )
			return null;
		
		//maintain rewrite status
		if( isReblockValid() )
			state.setBlocksize(DMLTranslator.DMLBlockSize);
		
		//perform reblock and blocksize rewrite
		for( Hop h : roots ) 
			rule_BlockSizeAndReblock(h, DMLTranslator.DMLBlockSize);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
		throws HopsException
	{
		if( root == null )
			return null;
		
		//maintain rewrite status
		if( isReblockValid() )
			state.setBlocksize(DMLTranslator.DMLBlockSize);
		
		//perform reblock and blocksize rewrite
		rule_BlockSizeAndReblock(root, DMLTranslator.DMLBlockSize);
		
		return root;
	}

	private void rule_BlockSizeAndReblock(Hop hop, final int GLOBAL_BLOCKSIZE) 
		throws HopsException 
	{
		// Go to the source(s) of the DAG
		for (Hop hi : hop.getInput()) {
			if (hop.getVisited() != Hop.VisitStatus.DONE)
				rule_BlockSizeAndReblock(hi, GLOBAL_BLOCKSIZE);
		}

		boolean canReblock = isReblockValid();
		
		if (hop instanceof DataOp) 
		{
			// if block size does not match
			if(    canReblock && hop.getDataType() != DataType.SCALAR
				&& (hop.getRowsInBlock() != GLOBAL_BLOCKSIZE || hop.getColsInBlock() != GLOBAL_BLOCKSIZE) ) 
			{
				if (((DataOp) hop).getDataOpType() == DataOp.DataOpTypes.PERSISTENTREAD) 
				{
					// insert reblock after the hop
					hop.setRequiresReblock(true);
					hop.setOutputBlocksizes(GLOBAL_BLOCKSIZE, GLOBAL_BLOCKSIZE);
				} 
				else if (((DataOp) hop).getDataOpType() == DataOp.DataOpTypes.PERSISTENTWRITE) 
				{
					if (hop.getRowsInBlock() == -1 && hop.getColsInBlock() == -1) 
					{
						// if this dataop is for cell output, then no reblock is needed 
						// as (A) all jobtypes can produce block2cell and cell2cell and 
						// (B) we don't generate an explicit instruction for it (the info 
						// is conveyed through OutputInfo.
					} 
					else if (hop.getInput().get(0).requiresReblock() && hop.getInput().get(0).getParent().size() == 1) 
					{
						// if a reblock is feeding into this, then use it if this is
						// the only parent, otherwise new Reblock
						hop.getInput().get(0).setOutputBlocksizes(hop.getRowsInBlock(),hop.getColsInBlock());
					} 
					else 
					{
						// insert reblock after the hop
						hop.setRequiresReblock(true);
						hop.setOutputBlocksizes(GLOBAL_BLOCKSIZE, GLOBAL_BLOCKSIZE);
					}
				} 
				else if (((DataOp) hop).getDataOpType() == DataOp.DataOpTypes.TRANSIENTWRITE
						|| ((DataOp) hop).getDataOpType() == DataOp.DataOpTypes.TRANSIENTREAD) {
					if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE ) {
						// simply copy the values from its input
						hop.setRowsInBlock(hop.getInput().get(0).getRowsInBlock());
						hop.setColsInBlock(hop.getInput().get(0).getColsInBlock());
					}
					else {
						// by default, all transient reads and writes are in blocked format
						hop.setRowsInBlock(GLOBAL_BLOCKSIZE);
						hop.setColsInBlock(GLOBAL_BLOCKSIZE);
					}

				} else {
					throw new HopsException(hop.printErrorLocation() + "unexpected non-scalar Data HOP in reblock.\n");
				}
			}
		} 
		else //NO DATAOP 
		{
			// TODO: following two lines are commented, and the subsequent hack is used instead!
			//set_rows_per_block(GLOBAL_BLOCKSIZE);
			//set_cols_per_block(GLOBAL_BLOCKSIZE);
			
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
			
			if ( hop.requiresReblock() ) {
				hop.setRowsInBlock(GLOBAL_BLOCKSIZE);
				hop.setColsInBlock(GLOBAL_BLOCKSIZE);
			}
			
			// Constraint C1:
			//else if ( (this instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)this)._op == ParamBuiltinOp.GROUPEDAGG) ) {
			//	setRowsInBlock(-1);
			//	setColsInBlock(-1);
			//}
			
			// Constraint C2:
			else if ( hop.getDataType() == DataType.SCALAR ) {
				hop.setRowsInBlock(-1);
				hop.setColsInBlock(-1);
			}

			// Constraint C3:
			else {
				if ( !canReblock ) {
					hop.setRowsInBlock(-1);
					hop.setColsInBlock(-1);
				}
				else {
					hop.setRowsInBlock(GLOBAL_BLOCKSIZE);
					hop.setColsInBlock(GLOBAL_BLOCKSIZE);
					
					// Functions may return multiple outputs, as defined in array of outputs in FunctionOp.
					// Reblock properties need to be set for each output.
					if ( hop instanceof FunctionOp ) {
						FunctionOp fop = (FunctionOp) hop;
						if ( fop.getOutputs() != null) {
							for(Hop out : fop.getOutputs()) {
								out.setRowsInBlock(GLOBAL_BLOCKSIZE);
								out.setColsInBlock(GLOBAL_BLOCKSIZE);
							}
						}
					}
				}
				
				// if any input is not blocked then the output of current Hop should not be blocked
				for ( Hop h : hop.getInput() ) {
					if ( h.getDataType() == DataType.MATRIX && h.getRowsInBlock() == -1 && h.getColsInBlock() == -1 ) {
						hop.setRowsInBlock(-1);
						hop.setColsInBlock(-1);
						break;
					}
				}
			}
		}

		hop.setVisited(Hop.VisitStatus.DONE);
	}
	
	/**
	 * 
	 * @return
	 */
	private static boolean isReblockValid() {
		return ( DMLScript.rtplatform != RUNTIME_PLATFORM.SINGLE_NODE);
	}
}
