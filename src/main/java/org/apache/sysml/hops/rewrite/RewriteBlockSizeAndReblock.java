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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.Hop.FileFormatTypes;
import org.apache.sysml.hops.Hop.ParamBuiltinOp;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.ParameterizedBuiltinOp;
import org.apache.sysml.parser.Expression.DataType;

/**
 * Rule: BlockSizeAndReblock. For all statement blocks, determine
 * "optimal" block size, and place reblock Hops. For now, we just
 * use BlockSize 1K x 1K and do reblock after Persistent Reads and
 * before Persistent Writes.
 */
public class RewriteBlockSizeAndReblock extends HopRewriteRule
{
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state)
		throws HopsException
	{
		if( roots == null )
			return null;
		
		//maintain rewrite status
		if( isReblockValid() )
			state.setBlocksize(ConfigurationManager.getBlocksize());
		
		//perform reblock and blocksize rewrite
		for( Hop h : roots ) 
			rule_BlockSizeAndReblock(h, ConfigurationManager.getBlocksize());
		
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
			state.setBlocksize(ConfigurationManager.getBlocksize());
		
		//perform reblock and blocksize rewrite
		rule_BlockSizeAndReblock(root, ConfigurationManager.getBlocksize());
		
		return root;
	}

	private void rule_BlockSizeAndReblock(Hop hop, final int GLOBAL_BLOCKSIZE) 
		throws HopsException 
	{
		// Go to the source(s) of the DAG
		for (Hop hi : hop.getInput()) {
			if (hi.getVisited() != Hop.VisitStatus.DONE)
				rule_BlockSizeAndReblock(hi, GLOBAL_BLOCKSIZE);
		}

		boolean canReblock = isReblockValid();
		
		if (hop instanceof DataOp) 
		{
			// if block size does not match
			if( canReblock //TODO change frame condition to != BINARY once transform over frames supported
				&& ((hop.getDataType() == DataType.MATRIX && (hop.getRowsInBlock() != GLOBAL_BLOCKSIZE || hop.getColsInBlock() != GLOBAL_BLOCKSIZE)
				  ||(hop.getDataType() == DataType.FRAME && OptimizerUtils.isSparkExecutionMode() && ((DataOp)hop).getInputFormatType()==FileFormatTypes.TEXT)))) 
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
		//TODO remove once transform rebased to frames
		else if ( (hop instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)hop).getOp() == ParamBuiltinOp.TRANSFORM) ) {
			
			// check if there exists a non-csv-write output. If yes, add reblock
			boolean rblk = false;
			for(Hop out : hop.getParent()) 
			{
				if ( !(out instanceof DataOp 
						&& ((DataOp)out).getDataOpType() == DataOpTypes.PERSISTENTWRITE 
						&& ((DataOp)out).getInputFormatType() == FileFormatTypes.CSV) )
				{
					rblk = true;
					break;
				}
			}
			if ( rblk )
			{
				hop.setRequiresReblock(true);
				hop.setOutputBlocksizes(GLOBAL_BLOCKSIZE, GLOBAL_BLOCKSIZE);
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
