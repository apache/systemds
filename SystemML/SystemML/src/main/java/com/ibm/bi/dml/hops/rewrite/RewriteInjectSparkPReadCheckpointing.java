/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.HopsException;

/**
 * Rule: BlockSizeAndReblock. For all statement blocks, determine
 * "optimal" block size, and place reblock Hops. For now, we just
 * use BlockSize 1K x 1K and do reblock after Persistent Reads and
 * before Persistent Writes.
 */
public class RewriteInjectSparkPReadCheckpointing extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		
		if(    (hop instanceof DataOp && ((DataOp)hop).getDataOpType()==DataOpTypes.PERSISTENTREAD)
			|| (hop.requiresReblock()) )
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
