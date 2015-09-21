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

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;

/**
 * Rule: BlockSizeAndReblock. For all statement blocks, determine
 * "optimal" block size, and place reblock Hops. For now, we just
 * use BlockSize 1K x 1K and do reblock after Persistent Reads and
 * before Persistent Writes.
 */
public class RewriteInjectSparkPReadCheckpointing extends HopRewriteRule
{
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state)
		throws HopsException
	{
		if(  !OptimizerUtils.isSparkExecutionMode()  ) 
			return roots;
		
		if( roots == null )
			return null;

		//top-level hops never modified
		for( Hop h : roots ) 
			rInjectCheckpointAfterPRead(h);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
		throws HopsException
	{
		//not applicable to predicates (we do not allow persistent reads there)
		return root;
	}

	/**
	 * 
	 * @param hop
	 * @throws HopsException
	 */
	private void rInjectCheckpointAfterPRead( Hop hop ) 
		throws HopsException 
	{
		if(hop.getVisited() == Hop.VisitStatus.DONE)
			return;
		
		if(    (hop instanceof DataOp && ((DataOp)hop).getDataOpType()==DataOpTypes.PERSISTENTREAD && !HopRewriteUtils.hasTransformParents(hop))
			|| (hop.requiresReblock())
			)
		{
			//make given hop for checkpointing (w/ default storage level)
			//note: we do not recursively process childs here in order to prevent unnecessary checkpoints
			hop.setRequiresCheckpoint(true);
		}
		else
		{
			//process childs
			if( hop.getInput() != null ) {
				//process all childs (prevent concurrent modification by index access)
				for( int i=0; i<hop.getInput().size(); i++ )
					rInjectCheckpointAfterPRead( hop.getInput().get(i) );
			}
		}
		
		hop.setVisited(Hop.VisitStatus.DONE);
	}
}
