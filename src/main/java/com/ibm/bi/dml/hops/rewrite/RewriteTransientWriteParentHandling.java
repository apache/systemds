/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;

/**
 * Rule: Eliminate for Transient Write DataHops to have no parents
 * Solution: Move parent edges of Transient Write Hop to parent of
 * its child 
 * Reason: Transient Write not being a root messes up
 * analysis for Lop's to Instruction translation (according to Amol)
 */
public class RewriteTransientWriteParentHandling extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) 
		throws HopsException
	{
		for (Hop h : roots) 
			rule_RehangTransientWriteParents(h, roots);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state)
		throws HopsException
	{
		// do nothing (does not apply to predicate hop DAGs) 	
		return root;
	}

	
	/**
	 * 
	 * @param hop
	 * @param sbHops
	 * @throws HopsException
	 */
	private void rule_RehangTransientWriteParents(Hop hop, ArrayList<Hop> sbHops) 
		throws HopsException 
	{
		if (hop instanceof DataOp && ((DataOp) hop).getDataOpType() == DataOpTypes.TRANSIENTWRITE
				&& !hop.getParent().isEmpty()) {

			// update parents inputs with data op input
			for (Hop p : hop.getParent()) {
				p.getInput().set(p.getInput().indexOf(hop), hop.getInput().get(0));
			}

			// update dataop input parent to add new parents except for
			// dataop itself
			hop.getInput().get(0).getParent().addAll(hop.getParent());

			// remove dataop parents
			hop.getParent().clear();

			// add dataop as root for this Hops DAG
			sbHops.add(hop);

			// do the same thing for my inputs (children)
			for (Hop hi : hop.getInput()) {
				rule_RehangTransientWriteParents(hi,sbHops);
			}
		}
	}
}
