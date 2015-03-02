/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.gdfgraph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Set;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;

/**
 * GENERAL 'GDF GRAPH' STRUCTURE, by MB:
 *  1) Each hop is represented by an GDFNode
 *  2) Each loop is represented by a structured GDFLoopNode
 *  3) Transient Read/Write connections are represented via CrossBlockNodes,
 *     a) type PLAIN: single input crossblocknode represents unconditional data flow
 *     b) type MERGE: two inputs crossblocknode represent conditional data flow merge
 * 
 *  In detail, the graph builder essentially does a single pass over the entire program
 *  and constructs the global data flow graph bottom up. We create crossblocknodes for
 *  every transient write, loop nodes for for/while programblocks, and crossblocknodes
 *  after every if programblock. 
 *  
 */
public class GraphBuilder 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * 
	 * @param prog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws HopsException 
	 */
	public static GDFGraph constructGlobalDataFlowGraph( Program prog )
		throws DMLRuntimeException, HopsException
	{
		HashMap<String, GDFNode> roots = new HashMap<String, GDFNode>();
		
		for( ProgramBlock pb : prog.getProgramBlocks() )
			constructGDFGraph( pb, roots );
		
		//create GDF graph root nodes 
		ArrayList<GDFNode> ret = new ArrayList<GDFNode>();
		for( GDFNode root : roots.values() )
			if( !(root instanceof GDFCrossBlockNode) )
				ret.add(root);
		
		//create GDF graph
		GDFGraph graph = new GDFGraph(prog, ret);
		
		return graph;
	}
	
	/**
	 * 
	 * @param pb
	 * @param roots
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 */
	@SuppressWarnings("unchecked")
	private static void constructGDFGraph( ProgramBlock pb, HashMap<String, GDFNode> roots ) 
		throws DMLRuntimeException, HopsException
	{
		if (pb instanceof FunctionProgramBlock )
		{
			throw new DMLRuntimeException("FunctionProgramBlocks not implemented yet.");
		}
		else if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			WhileStatementBlock wsb = (WhileStatementBlock) pb.getStatementBlock();		
			//construct predicate node (conceptually sequence of from/to/incr)
			GDFNode pred = constructGDFGraph(wsb.getPredicateHops(), wpb, new HashMap<Long, GDFNode>(), roots);
			HashMap<String,GDFNode> inputs = constructLoopInputNodes(wpb, wsb, roots);
			HashMap<String,GDFNode> lroots = (HashMap<String, GDFNode>) inputs.clone();
			//process childs blocks
			for( ProgramBlock pbc : wpb.getChildBlocks() )
				constructGDFGraph(pbc, lroots);
			HashMap<String,GDFNode> outputs = constructLoopOutputNodes(wpb, wsb, lroots);
			GDFLoopNode lnode = new GDFLoopNode(wpb, pred, inputs, outputs );
			//construct crossblock nodes
			constructLoopOutputCrossBlockNodes(lnode, outputs, roots, wpb);
		}	
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			IfStatementBlock isb = (IfStatementBlock) pb.getStatementBlock();
			//construct predicate
			if( isb.getPredicateHops()!=null ) {
				Hop pred = isb.getPredicateHops();
				roots.put(pred.getName(), constructGDFGraph(pred, ipb, new HashMap<Long,GDFNode>(), roots));
			}
			//construct if and else branch separately
			HashMap<String,GDFNode> ifRoots = (HashMap<String, GDFNode>) roots.clone();
			HashMap<String,GDFNode> elseRoots = (HashMap<String, GDFNode>) roots.clone();
			for( ProgramBlock pbc : ipb.getChildBlocksIfBody() )
				constructGDFGraph(pbc, ifRoots);
			if( ipb.getChildBlocksElseBody()!=null )
				for( ProgramBlock pbc : ipb.getChildBlocksElseBody() )
					constructGDFGraph(pbc, elseRoots);
			//merge data flow roots (if no else, elseRoots refer to original roots)
			reconcileMergeIfProgramBlockOutputs(ifRoots, elseRoots, roots, ipb);
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			ForStatementBlock fsb = (ForStatementBlock)pb.getStatementBlock();
			//construct predicate node (conceptually sequence of from/to/incr)
			GDFNode pred = constructForPredicateNode(fpb, fsb, roots);
			HashMap<String,GDFNode> inputs = constructLoopInputNodes(fpb, fsb, roots);
			HashMap<String,GDFNode> lroots = (HashMap<String, GDFNode>) inputs.clone();
			//process childs blocks
			for( ProgramBlock pbc : fpb.getChildBlocks() )
				constructGDFGraph(pbc, lroots);
			HashMap<String,GDFNode> outputs = constructLoopOutputNodes(fpb, fsb, lroots);
			GDFLoopNode lnode = new GDFLoopNode(fpb, pred, inputs, outputs );
			//construct crossblock nodes
			constructLoopOutputCrossBlockNodes(lnode, outputs, roots, fpb);
		}
		else //last-level program block
		{
			StatementBlock sb = pb.getStatementBlock();
			ArrayList<Hop> hops = sb.get_hops();
			if( hops != null )
			{
				//create new local memo structure for local dag
				HashMap<Long, GDFNode> lmemo = new HashMap<Long, GDFNode>();
				for( Hop hop : hops )
				{
					//recursively construct GDF graph for hop dag root
					GDFNode root = constructGDFGraph(hop, pb, lmemo, roots);
	
					//create cross block nodes for all transient writes
					if( hop instanceof DataOp && ((DataOp)hop).get_dataop()==DataOpTypes.TRANSIENTWRITE )
						root = new GDFCrossBlockNode(hop, pb, root, hop.getName());
					
					//add GDF root node to global roots 
					roots.put(hop.getName(), root);
				}
			}
			
		}
	}
	
	/**
	 * 
	 * @param hop
	 * @param pb
	 * @param lmemo
	 * @param roots 
	 * @return
	 */
	private static GDFNode constructGDFGraph( Hop hop, ProgramBlock pb, HashMap<Long, GDFNode> lmemo, HashMap<String, GDFNode> roots )
	{
		if( lmemo.containsKey(hop.getHopID()) )
			return lmemo.get(hop.getHopID());
		
		//process childs recursively first
		ArrayList<GDFNode> inputs = new ArrayList<GDFNode>();
		for( Hop c : hop.getInput() )
			inputs.add( constructGDFGraph(c, pb, lmemo, roots) );
		
		//connect transient reads to existing roots of data flow graph 
		if( hop instanceof DataOp && ((DataOp)hop).get_dataop()==DataOpTypes.TRANSIENTREAD ){
			inputs.add(roots.get(hop.getName()));
		}
		
		//add current hop
		GDFNode gnode = new GDFNode(hop, pb, inputs);
		
		//memoize current node
		lmemo.put(hop.getHopID(), gnode);
		
		return gnode;
	}
	
	/**
	 * 
	 * @param fpb
	 * @param fsb
	 * @param roots
	 * @return
	 */
	private static GDFNode constructForPredicateNode(ForProgramBlock fpb, ForStatementBlock fsb, HashMap<String, GDFNode> roots)
	{
		HashMap<Long, GDFNode> memo = new HashMap<Long, GDFNode>();
		GDFNode from = (fsb.getFromHops()!=null)? constructGDFGraph(fsb.getFromHops(), fpb, memo, roots) : null;
		GDFNode to = (fsb.getToHops()!=null)? constructGDFGraph(fsb.getToHops(), fpb, memo, roots) : null;
		GDFNode incr = (fsb.getIncrementHops()!=null)? constructGDFGraph(fsb.getIncrementHops(), fpb, memo, roots) : null;
		ArrayList<GDFNode> inputs = new ArrayList<GDFNode>();
		inputs.add(from);
		inputs.add(to);
		inputs.add(incr);
		GDFNode pred = new GDFNode(null, fpb, inputs );
		
		return pred;
	}
	
	/**
	 * 
	 * @param fpb
	 * @param fsb
	 * @param roots
	 * @return
	 */
	private static HashMap<String, GDFNode> constructLoopInputNodes( ProgramBlock fpb, StatementBlock fsb, HashMap<String, GDFNode> roots )
	{
		HashMap<String, GDFNode> ret = new HashMap<String, GDFNode>();
		Set<String> invars = fsb.variablesRead().getVariableNames();
		for( String var : invars ) {
			GDFNode node = roots.get(var);
			ret.put(var, node);
		}
		
		return ret;
	}
	
	private static HashMap<String, GDFNode> constructLoopOutputNodes( ProgramBlock fpb, StatementBlock fsb, HashMap<String, GDFNode> roots )
	{
		HashMap<String, GDFNode> ret = new HashMap<String, GDFNode>();
		
		Set<String> outvars = fsb.variablesUpdated().getVariableNames();
		for( String var : outvars ) {
			GDFNode node = roots.get(var);
			ret.put(var, node);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param ifRoots
	 * @param elseRoots
	 * @param roots
	 * @param pb
	 */
	private static void reconcileMergeIfProgramBlockOutputs( HashMap<String, GDFNode> ifRoots, HashMap<String, GDFNode> elseRoots, HashMap<String, GDFNode> roots, IfProgramBlock pb )
	{
		//merge same variable names, different data
		//( incl add new vars from if branch if node2==null)
		for( Entry<String, GDFNode> e : ifRoots.entrySet() ){
			GDFNode node1 = e.getValue();
			GDFNode node2 = elseRoots.get(e.getKey()); //original or new
			if( node1 != node2 )
				node1 = new GDFCrossBlockNode(null, pb, node1, node2, e.getKey() );
			roots.put(e.getKey(), node1);	
		}
		
		//add new vars from else branch 
		for( Entry<String, GDFNode> e : elseRoots.entrySet() ){
			if( !ifRoots.containsKey(e.getKey()) )
				roots.put(e.getKey(), e.getValue());	
		}
	}
	
	/**
	 * 
	 * @param loop
	 * @param loutputs
	 * @param roots
	 * @param pb
	 */
	private static void constructLoopOutputCrossBlockNodes(GDFLoopNode loop, HashMap<String, GDFNode> loutputs, HashMap<String, GDFNode> roots, ProgramBlock pb)
	{
		for( Entry<String,GDFNode> e : loutputs.entrySet() ){
			GDFCrossBlockNode node = null;
			if( roots.containsKey(e.getKey()) )
				node = new GDFCrossBlockNode(null, pb, roots.get(e.getKey()), loop, e.getKey()); //MERGE
			else
				node = new GDFCrossBlockNode(null, pb, loop, e.getKey()); //PLAIN
			roots.put(e.getKey(), node);
		}
	}
}
