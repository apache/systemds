/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.HopsDag;
import com.ibm.bi.dml.hops.globalopt.LoopOp;
import com.ibm.bi.dml.hops.globalopt.MergeOp;
import com.ibm.bi.dml.hops.globalopt.PrintVisitor;
import com.ibm.bi.dml.hops.globalopt.RefreshMetaDataVisitor;
import com.ibm.bi.dml.hops.globalopt.SplitOp;
import com.ibm.bi.dml.hops.globalopt.LoopOp.LoopType;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.VariableSet;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;

/**
 * Analog version to {@link MaximalGlobalGraphCreator} which works on {@link Program} 
 * instead of {@link DMLProgram}. This is a complete code duplicate!!! 
 * TODO: Either refactor the essential logic of both classes into one or decide for one alternative.
 */
public class GlobalGraphCreator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Map<ProgramBlock, List<Hop>> blocksToHops;
	private Map<String, HopsDag> varNameToMggs;
	private static final Log LOG = LogFactory.getLog(GlobalGraphCreator.class);
	
	
	public GlobalGraphCreator() {
		this.blocksToHops = new HashMap<ProgramBlock, List<Hop>>();
		this.varNameToMggs = new HashMap<String, HopsDag>();
	}
	
	public Map<String, HopsDag> createGraph(Program program) throws HopsException {
		if(program != null ) {
			ArrayList<ProgramBlock> blocks = program.getProgramBlocks();
			Map<String,HopsDag> inVars = new HashMap<String, HopsDag>();
			for(ProgramBlock b : blocks) {
				inVars = handleBlock(b, inVars);
			}
			
			refreshMetaData(inVars);
			
			this.varNameToMggs.putAll(inVars);
		}
		return this.varNameToMggs;
	}
	
	private void refreshMetaData(Map<String, HopsDag> inVars) {
		for(HopsDag dag : inVars.values()) {
			RefreshMetaDataVisitor visitor = new RefreshMetaDataVisitor();
			visitor.initialize(dag);
		}
		
	}

	public Map<String, HopsDag> createGraph(Program program, boolean print) throws HopsException {
		this.createGraph(program);
		if(print) {
			for(String var : this.varNameToMggs.keySet()) {
				HopsDag dag = this.varNameToMggs.get(var);
				PrintVisitor v = new PrintVisitor(null);
				for(String outVarName : dag.getDagOutputs().keySet()) {
					dag.getDagOutputs().get(outVarName).accept(v);
				}
			}
		}
		return this.varNameToMggs;
	}
	

	/**
	 * @param b
	 * @throws HopsException
	 */
	public Map<String, HopsDag> handleBlock(ProgramBlock b, Map<String,HopsDag> inVars) throws HopsException {
		Map<String, HopsDag> outVars = new HashMap<String, HopsDag>();
		this.blocksToHops.put(b, b.getStatementBlock().get_hops());
			
		if(b instanceof IfProgramBlock) 
		{
			IfProgramBlock ifBlock = (IfProgramBlock)b;
			Map<String, HopsDag> outVarsIf = null;
			Map<String, HopsDag> outVarsElse = null;
			
			Map<String, SplitOp> splits = new HashMap<String, SplitOp>(); 
			appendSplitNodes(inVars, splits);
			
			for(ProgramBlock sb : ifBlock.getChildBlocksIfBody())	{
				
				outVarsIf = handleBlock(sb, inVars);
			}
			if(ifBlock.getChildBlocksElseBody() != null && ifBlock.getChildBlocksElseBody().size() > 0) {
				for(ProgramBlock sb : ifBlock.getChildBlocksElseBody()) {
					outVarsElse = handleBlock(sb, inVars);
				}
			} else {
				outVarsElse = new HashMap<String, HopsDag>(inVars);
			}
			
			outVars = mergeIfBlockOutputs(outVarsIf, outVarsElse, splits);
		}else if(b instanceof WhileProgramBlock) {
			WhileProgramBlock whileBlock = (WhileProgramBlock)b;
			outVars = mergeWhileLoop(whileBlock, inVars);
			
		} else if(b instanceof ForProgramBlock) {
			ForProgramBlock forBlock = (ForProgramBlock)b;
			outVars = mergeForLoop(forBlock, inVars);
		} else{
			outVars = mergeStatementBlock(b, inVars);
		}
		//clean outVars for unnecessary variables
		removeDeadVariables(b, outVars);
		return outVars;
	}

	private Map<String, HopsDag> mergeForLoop(ForProgramBlock forBlock,
			Map<String, HopsDag> inVars) {
		
		Map<String, HopsDag> outVars = new HashMap<String, HopsDag>(inVars);
		Map<String, HopsDag> newInvars = new HashMap<String, HopsDag>();
		
		ForStatementBlock forStatement = (ForStatementBlock) forBlock.getStatementBlock();
		//TODO: find out, how to access the predicate
		Hop incrementHops = forStatement.getIncrementHops();
		
		try {
			ProgramBlock sb = forBlock.getChildBlocks().get(0);
			Map<String, HopsDag> internalGraph = handleBlock(sb, newInvars);
			outVars = createLoopNode(forBlock, inVars, outVars, incrementHops, internalGraph, LoopType.FOR);
			

		} catch (HopsException e) {
			LOG.error(e.getMessage(), e);
		}
		
		return outVars;
	}

	/**
	 * Creates one {@link LoopOp} and inserts it into the graph for each variable that is in liveOut.
	 * @param whileBlock
	 * @param inVars
	 * @return
	 */
	private Map<String, HopsDag> mergeWhileLoop(WhileProgramBlock whileBlock,
			Map<String, HopsDag> inVars) {
		
		Map<String, HopsDag> outVars = new HashMap<String, HopsDag>(inVars);
		WhileStatementBlock wileStatementBlock = (WhileStatementBlock) whileBlock.getStatementBlock();
		Hop predicate = wileStatementBlock.getPredicateHops();
		
		Map<String, HopsDag> newInvars = new HashMap<String, HopsDag>();
		try {
//			WhileStatement statement = (WhileStatement)whileBlock.getStatement(0);
			ArrayList<ProgramBlock> body = whileBlock.getChildBlocks();
			//TODO: is there always only one sb n body
			Map<String, HopsDag> internalGraph = mergeStatementBlock(body.get(0), newInvars);
			createLoopNode(whileBlock, inVars, outVars, predicate, internalGraph, LoopType.WHILE);
		} catch (HopsException e) {
			LOG.error(e.getMessage(), e);
		}
		
		return outVars;
	}

	/**
	 * @param block
	 * @param inVars
	 * @param outVars
	 * @param predicate
	 * @param internalGraph
	 */
	private Map<String, HopsDag> createLoopNode(ProgramBlock block,
			Map<String, HopsDag> inVars, Map<String, HopsDag> outVars,
			Hop predicate, Map<String, HopsDag> internalGraph, LoopType type) {
		
		LoopOp loopNode = new LoopOp(internalGraph);
		loopNode.setPredicate(predicate);
		loopNode.setLoopType(type);
		
		VariableSet liveOut = block.getStatementBlock().liveOut();
		if(liveOut != null && liveOut.getVariables().size() > 0) {
			for(String varName : liveOut.getVariableNames()) {
				VariableSet kill = block.getStatementBlock().getKill();
				if(!kill.containsVariable(varName)) {
					//System.out.println("added dags for loop live out: " + varName);
					
					HopsDag hopsDag = inVars.get(varName);
					if(hopsDag != null) {
						Map<String, Hop> dagOutputs = hopsDag.getDagOutputs();
						//??? Hop hops = dagOutputs.get(varName);
						dagOutputs.put(varName, loopNode);
						outVars.put(varName, hopsDag);
					}else {
						HopsDag newDag = new HopsDag();
						newDag.getDagInputs().put(varName, loopNode);
						newDag.getDagOutputs().put(varName, loopNode);
						outVars.put(varName, newDag);
					}
				}
			}
		}
		
		return outVars;
	}

	private void appendSplitNodes(Map<String, HopsDag> inVars, Map<String, SplitOp> splits) {
		
		for(String dagKey : inVars.keySet()) {
			HopsDag dag = inVars.get(dagKey);
			for(Entry<String, Hop> e : dag.getDagOutputs().entrySet()) {
				String varName = e.getKey();
				Hop output = e.getValue();
				SplitOp split = new SplitOp(output, null);
				dag.getDagOutputs().put(varName, split);
				splits.put(varName, split);
			}
		}
		
	}

	/**
	 * Removes all dead variables i, i.e. that exist no longer in the liveOut set. 
	 * Except for the last block, since the liveOut set at the end of a program is empty. 
	 * @param b
	 * @param outVars
	 */
	private void removeDeadVariables(ProgramBlock b, Map<String, HopsDag> outVars) {
		VariableSet liveOut = b.getStatementBlock().liveOut();
		if(liveOut != null) {
			for(Iterator<String> oIt = outVars.keySet().iterator(); oIt.hasNext();)
			{
				String outS = oIt.next();
				
				int indexInProgram = b.getProgram().getProgramBlocks().indexOf(b);
				int size = b.getProgram().getProgramBlocks().size();
				
				if(!liveOut.containsVariable(outS) && (indexInProgram + 1) < size) {
					oIt.remove();
				}
			}
		}
	}

	@SuppressWarnings("unchecked")
	private Map<String, HopsDag> mergeIfBlockOutputs( Map<String, HopsDag> outVarsIf, 
			Map<String, HopsDag> outVarsElse, Map<String, SplitOp> splits) {
		Map<String, HopsDag> result = new HashMap<String, HopsDag>();
		Collection<String> toMerge = CollectionUtils.intersection(outVarsIf.keySet(), outVarsElse.keySet());
		for(Object m : toMerge) {
			HopsDag ifDag = outVarsIf.get(m);
			Hop ifOut = ifDag.getDagOutputs().get(m);
			HopsDag elseDag = outVarsElse.get(m);
			Hop elseOut = elseDag.getDagOutputs().get(m);
			
			MergeOp merged = new MergeOp(ifOut, elseOut, null);
			ifOut.setCrossBlockOutput(merged);
			elseOut.setCrossBlockOutput(merged);
			ifDag.getDagOutputs().put((String)m, merged);
			splits.get((String)m).setMergeNode(merged);
			result.put((String)m, ifDag);
		}
		
		Collection<String> toAdd = CollectionUtils.disjunction(outVarsIf.keySet(), outVarsElse.keySet());
		for(Object s : toAdd){
			result.put((String)s, outVarsIf.get(s)==null ? outVarsElse.get(s) : outVarsIf.get(s));
		}
	
		return result;
	}

	/**
	 * TODO: Ensure that after leaving a SB every dangling output from old hopsdags is overwritten 
	 * by stitched dags.
	 * @param programBlock
	 * @throws HopsException
	 */
	public Map<String, HopsDag> mergeStatementBlock(ProgramBlock programBlock, Map<String, HopsDag> inVars) throws HopsException {
		Map<String, HopsDag> outVars = new HashMap<String, HopsDag>(inVars);
		if(programBlock.getStatementBlock().get_hops() != null){
			for (Hop h : programBlock.getStatementBlock().get_hops()) {
				HopsDag visitor = new HopsDag();
//				visitor.putHopsToBlock(h, programBlock);
				visitor.addOriginalRootHops(h);
				if(!h.isHopsVisited(visitor)) {
					h.accept(visitor);
					//in case of the first statement block
					if(inVars.isEmpty()) {
						registerDag(visitor, outVars);
					}else {
						Map<String, Hop> dagInputs = visitor.getDagInputs();
						Set<String> inputNameCopy = new HashSet<String>();
						inputNameCopy.addAll(dagInputs.keySet());
						for(String inputName : inputNameCopy) {
							//HopsDag hopsDag = inVars.get(inputName);
							HopsDag hopsDag = outVars.get(inputName);
							if(hopsDag != null) {
								visitor = stitchUpDags(inputName, hopsDag, visitor, outVars, programBlock);
							}else {
								outVars.put(inputName, visitor);
							}
						}
						for(Entry<String, Hop> entry : visitor.getDagOutputs().entrySet()) {
							outVars.put(entry.getKey(), visitor);
						}
						
						//at this point all the dangling edges f h have been stitched up
						//no other hop h should use the same edges (except for if blocks but they 
						// are treated outside this function)
					}
					visitor.linkOutputsWithProgramBlock(programBlock);
				}
				//check if h's inputs match with any of the mggs
				//if not, add h to this.mgg
				//if, then connect h to one or more mggs
				// connection to more than one mgg merge mggs
			}
		}
		return outVars;
	}

	/**
	 * @param visitor
	 */
	private void registerDag(HopsDag visitor, Map<String, HopsDag> inVars) {
		for(String name : visitor.getDagOutputs().keySet())
		{
			inVars.put(name, visitor);
		}
	}

	/**
	 * Connects the matching edges, removes the stitched end points from origin.outputs and 
	 * add all the visitor outputs to the new dag.
	 * @param origin
	 * @param visitor
	 */
	public HopsDag stitchUpDags(String name, HopsDag origin, HopsDag visitor, Map<String, HopsDag> outVars, ProgramBlock block) {
		
		Hop source = origin.getDagOutputs().get(name);
		Hop target = visitor.getDagInputs().get(name);
		if(source != null && target != null)
		{
			source.append(target);
			origin.addStitched(name);
		}
		
		visitor.getDagInputs().remove(name);
		if(origin.getDagInputs().get(name) != null)
		{
			visitor.getDagInputs().put(name, origin.getDagInputs().get(name));
		}
		for(Entry<String, Hop> entry: origin.getDagInputs().entrySet())
		{
			if(!visitor.getDagInputs().containsKey(entry.getKey())) {
				visitor.getDagInputs().put(entry.getKey(), entry.getValue());
			}
		}
		if(visitor.getDagOutputs().containsKey(name))
		{
			outVars.put(name, visitor);
		}
		
		visitor.copyDirectory(origin.getHopsDirectory());
		visitor.copyHopsToBlocks(origin.getHopsToBlocks());
		visitor.addOriginalRootHops(origin.getOriginalRootHops());
		return visitor;
	}

	public Map<ProgramBlock, List<Hop>> getBlocksToHops() {
		return blocksToHops;
	}

	public Map<String, HopsDag> getVarNameToMggs() {
		return varNameToMggs;
	}

}
