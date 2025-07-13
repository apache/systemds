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

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;

/**
 * Rule: BlockSizeAndReblock. For all statement blocks, determine
 * "optimal" block size, and place reblock Hops. For now, we just
 * use BlockSize 1K x 1K and do reblock after Persistent Reads and
 * before Persistent Writes.
 */
public class RewriteBlockSizeAndReblock extends HopRewriteRule
{
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return null;
		
		//maintain rewrite status
		state.setBlocksize(ConfigurationManager.getBlocksize());
		
		//perform reblock and blocksize rewrite
		for( Hop h : roots ) 
			rule_BlockSizeAndReblock(h, ConfigurationManager.getBlocksize());
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return null;
		
		//maintain rewrite status
		state.setBlocksize(ConfigurationManager.getBlocksize());
		
		//perform reblock and blocksize rewrite
		rule_BlockSizeAndReblock(root, ConfigurationManager.getBlocksize());
		
		return root;
	}

	private void rule_BlockSizeAndReblock(Hop hop, final int blocksize) 
	{
		// Go to the source(s) of the DAG
		for (Hop hi : hop.getInput()) {
			if (!hi.isVisited())
				rule_BlockSizeAndReblock(hi, blocksize);
		}
		
		if (hop instanceof DataOp) 
		{
			DataOp dop = (DataOp) hop;
			
			if( DMLScript.USE_OOC && dop.getOp() == OpOpData.PERSISTENTREAD ) {
				dop.setRequiresReblock(true);
				dop.setBlocksize(blocksize);
			}
			// if block size does not match
			else if(   (dop.getDataType() == DataType.MATRIX && (dop.getBlocksize() != blocksize))
				||(dop.getDataType() == DataType.FRAME && OptimizerUtils.isSparkExecutionMode() 
				&& (dop.getFileFormat()==FileFormat.TEXT || dop.getFileFormat()==FileFormat.CSV)) )
			{
				if( dop.getOp() == OpOpData.PERSISTENTREAD || dop.getOp() == OpOpData.FEDERATED)
				{
					// insert reblock after the hop
					dop.setRequiresReblock(true);
					dop.setBlocksize(blocksize);
				} 
				else if( dop.getOp() == OpOpData.PERSISTENTWRITE )
				{
					if (dop.getBlocksize() == -1)
					{
						// if this dataop is for cell output, then no reblock is needed 
						// as (A) all jobtypes can produce block2cell and cell2cell and 
						// (B) we don't generate an explicit instruction for it (the info 
						// is conveyed through OutputInfo.
					} 
					else if (dop.getInput().get(0).requiresReblock() && dop.getInput().get(0).getParent().size() == 1) 
					{
						// if a reblock is feeding into this, then use it if this is
						// the only parent, otherwise new Reblock
						dop.getInput().get(0).setBlocksize(dop.getBlocksize());
					}
				} 
				else if (dop.getOp().isTransient()) {
					// by default, all transient reads and writes are in blocked format
					dop.setBlocksize(blocksize);
				}
				else {
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
				hop.setBlocksize(blocksize);
			}
			
			// Constraint C1:
			
			// Constraint C2:
			else if ( hop.getDataType() == DataType.SCALAR ) {
				hop.setBlocksize(-1);
			}

			// Constraint C3:
			else {
				hop.setBlocksize(blocksize);
				
				// Functions may return multiple outputs, as defined in array of outputs in FunctionOp.
				// Reblock properties need to be set for each output.
				if ( hop instanceof FunctionOp ) {
					FunctionOp fop = (FunctionOp) hop;
					if ( fop.getOutputs() != null) {
						for(Hop out : fop.getOutputs()) {
							out.setBlocksize(blocksize);
						}
					}
				}
				
				// if any input is not blocked then the output of current Hop should not be blocked
				for ( Hop h : hop.getInput() ) {
					if ( h.getDataType() == DataType.MATRIX && h.getBlocksize() == -1) {
						hop.setBlocksize(-1);
						break;
					}
				}
			}
		}

		hop.setVisited();
	}
}
