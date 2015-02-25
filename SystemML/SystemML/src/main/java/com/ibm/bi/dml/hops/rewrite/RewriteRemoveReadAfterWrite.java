/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.parser.DataExpression;

/**
 * Rule: RemoveReadAfterWrite. If there is a persistent read with the same filename
 * as a persistent write, and read has a higher line number than the write,
 * we remove the read and consume the write input directly. This is important for two
 * reasons (1) correctness and (2) performance. Without this rewrite, we could not
 * guarantee the order of read-after-write because there is not data dependency
 * 
 */
public class RewriteRemoveReadAfterWrite extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	@SuppressWarnings("unchecked")
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state)
		throws HopsException
	{
		if( roots == null )
			return null;
		
		//collect all persistent reads and writes
		HashMap<String,Hop> reads = new HashMap<String,Hop>();
		HashMap<String,Hop> writes = new HashMap<String,Hop>();
		for( Hop h : roots ) 
			collectPersistentReadWriteOps( h, writes, reads );
		
		Hop.resetVisitStatus(roots);
		
		//check persistent reads for read-after-write pattern
		for( Entry<String, Hop> e : reads.entrySet() )
		{
			String rfname = e.getKey();
			Hop rhop = e.getValue();
			if( writes.containsKey(rfname)  //same persistent filename
				&&   (writes.get(rfname).getBeginLine()<rhop.getBeginLine() //read after write
				   || writes.get(rfname).getEndLine()<rhop.getEndLine()) ) //note: account for bug in line handling, TODO remove after line handling resolved
			{
				//rewire read consumers to write input
				Hop input = writes.get(rfname).getInput().get(0);
				ArrayList<Hop> parents = (ArrayList<Hop>) rhop.getParent().clone();
				for( Hop p : parents ) {
					int pos = HopRewriteUtils.getChildReferencePos(p, rhop);
					HopRewriteUtils.removeChildReferenceByPos(p, rhop, pos);
					HopRewriteUtils.addChildReference(p, input, pos);
				}
			}
		}
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
		throws HopsException
	{
		//do noting, read/write do not occur in predicates
		
		return root;
	}

	/**
	 * 
	 * @param hop
	 * @param pWrites
	 * @param pReads
	 * @throws HopsException
	 */
	private void collectPersistentReadWriteOps(Hop hop, HashMap<String,Hop> pWrites, HashMap<String,Hop> pReads) 
		throws HopsException 
	{
		if( hop.getVisited() == Hop.VisitStatus.DONE )
			return;
		
		//process childs
		if( !hop.getInput().isEmpty() )
			for( Hop c : hop.getInput() )
				collectPersistentReadWriteOps(c, pWrites, pReads);
		
		//process current hop
		if( hop instanceof DataOp )
		{
			DataOp dop = (DataOp)hop;
			if( dop.get_dataop()==DataOpTypes.PERSISTENTREAD )
				pReads.put(dop.getFileName(), dop);
			else if( dop.get_dataop()==DataOpTypes.PERSISTENTWRITE )
			{
				Hop fname = dop.getInput().get(dop.getParameterIndex(DataExpression.IO_FILENAME));
				if( fname instanceof LiteralOp ) //only constant writes
					pWrites.put(((LiteralOp) fname).getStringValue(), dop);	
			}
		}
		
		hop.setVisited(Hop.VisitStatus.DONE);
	}
}
