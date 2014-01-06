/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.hops.cost.CostEstimationWrapper;
import com.ibm.bi.dml.hops.globalopt.CrossBlockOp;
import com.ibm.bi.dml.hops.globalopt.GlobalOptimizer;
import com.ibm.bi.dml.hops.globalopt.HopsDag;
import com.ibm.bi.dml.hops.globalopt.LoopOp;
import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.FormatType;
import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.InterestingPropertyType;
import com.ibm.bi.dml.hops.globalopt.enumerate.RewriteConfig.RewriteConfigType;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.Dag;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.Program;

/**
 * Global data flow optimization via enumeration-based optimizer (dynamic programming). 
 * 
 */
public class GlobalEnumerationOptimizer extends GlobalOptimizer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(GlobalEnumerationOptimizer.class);
	private static final boolean LDEBUG = false; //local debug flag
	
	
	//internal configuration parameters
	public static final int[] BLOCK_SIZES         = new int[]{-1,600,2000,3000,4000,5000};
	public static final int[] REPLICATION_FACTORS = new int[]{1,2,3,4,5};
	
	private MemoStructure _memo = null; //plan memoization table
	
	
	/** the configurations that are applied to {@link Hops}. this is a pre-computed set */
	private Collection<RewriteConfigSet> generatedConfigs;
	
	
	/** the combinations interesting that are produced by applying a given combination*/
	private Set<InterestingPropertySet> interestingPropertyCombinations;
	/** association between combinations of rewrites and combinations of interesting properties */
	private Map<RewriteConfigSet, InterestingPropertySet> configToProperties;
	/* TODO: was used in the first versions, may be replaced with this.runtimeProgram */ 
	private Program currentProgram;

	public GlobalEnumerationOptimizer( ) 
	{
		//init internal memo structure
		_memo = new MemoStructure();
		
		//init internal configuration (register rewrites, )
		HashSet<RewriteConfig> rewriteConfigs = new HashSet<RewriteConfig>();
		rewriteConfigs.add( new RewriteConfigBlocksize() );
		rewriteConfigs.add( new RewriteConfigExecType() );
		rewriteConfigs.add( new RewriteConfigFormat() );
		
		
		
		//TODO
		this.configToProperties = new HashMap<RewriteConfigSet, InterestingPropertySet>();
		this.generatedConfigs = RewriteConfigUtils.generateConfigCombinations(rewriteConfigs);
		this.interestingPropertyCombinations = generatePropertyCombinations(this.generatedConfigs);
	}
	
	/**
	 * @throws HopsException 
	 * @throws LopsException 
	 */
	@Override
	public Program optimize(DMLProgram prog, Program rtprog) 
		throws DMLRuntimeException, HopsException, LopsException 
	{		
		currentProgram = rtprog;
		Summary summary = new Summary();
		
		//init global data flow graph
		GlobalGraphCreator graphCreator = new GlobalGraphCreator();
		Map<String, HopsDag> mggSet = graphCreator.createGraph(rtprog);
	
		
		
		for(HopsDag d : mggSet.values()) {
			Map<InterestingPropertySet, MemoEntry> optPlans = new HashMap<InterestingPropertySet, MemoEntry>();
			
			for(Hop o : d.getDagOutputs().values()) {
				optPlans = enumOpt(o, summary);
			}
			
			OptimizedPlan optimalPlan = pickOptimal(optPlans);
			populatePlanToProgram(optimalPlan, d);
			
			//TODO: turn into proper logging
			//if(this.debug) 
			{
				System.out.println(summary);
			}
		}
		return rtprog;
	}

	/**
	 * Use the node configurations and put them into the respective nodes.
	 * 
	 * 
	 * 
	 * @param optimalPlan
	 * @param program
	 */
	private void populatePlanToProgram(OptimizedPlan optimalPlan,
			HopsDag dag) {
		Set<Long> addedLopIDs = new HashSet<Long>();
		
		//MB: the lops should not be replicated, all configuration changes must be made on hops level for consistency
		//Create new Dags for Lops
		//TODO: be careful, this does only work for top level blocks
		//nested blocks are not covered here, this needs to be refacted in reusing the logic of e.g. DMLProgram.getRuntimeProgram()
		//for(ProgramBlock block : this.currentProgram.getProgramBlocks()) {
		//	block.setLopsDag(new Dag<Lop>());
		//}
		
		//remove old/invalid Lops from Hops
		Set<Hop> keySet = dag.getMetaDataDirectory().keySet();
		for(Hop h : keySet) {
			if(!h.get_dataType().equals(DataType.SCALAR)) {
				h.set_lops(null);
			}
		}
		
		//Put the Lop optimal plan into the Hops
		if(optimalPlan != null && optimalPlan.getGeneratedLop() != null) {
			applyLops(optimalPlan, dag, addedLopIDs);
		}else {
			LOG.warn("No optimal plan has been generated. " + optimalPlan);
		}
		
		//generate Instructions
		generateInstructions(dag, addedLopIDs);
		
		LOG.info("populating lops done");
	}

	/**
	 * Put all the root lops into a {@link Dag} per program block and then iterate over all the touched program blocks
	 * and generate new instructions. 
	 * @param dag
	 * @param root
	 * @param addedLopIDs
	 */
	private void generateInstructions(HopsDag dag, Set<Long> addedLopIDs) {
		
		for(Hop output : dag.getDagOutputs().values()) {
			try {
				addRootLopsToDag(dag, output);
			} catch (HopsException e) {
				LOG.error(e.getMessage(), e);
			} catch (LopsException e) {
				LOG.error(e.getMessage(), e);
			}

		}
		
		//MB: the lops should not be replicated, all configuration changes must be made on hops level for consistency
		//iterate over the program block which have been modified by the algorithm
		/*
		for(ProgramBlock block : dag.getHopsToBlocks().values()) {
			Dag<Lop> lopsDag = block.getLopsDag();
			try {
				ArrayList<Instruction> instructions = lopsDag.getJobs(block.getStatementBlock(),null);
				block.setInstructions(instructions);
			} catch (LopsException e) {
				LOG.error(e.getMessage(), e);
			} catch (DMLRuntimeException e) {
				LOG.error(e.getMessage(), e);
			} catch (DMLUnsupportedOperationException e) {
				LOG.error(e.getMessage(), e);
			} catch (IOException e) {
				LOG.error(e.getMessage(), e);
			}
		}
		*/
	}

	/**
	 * Recursive method to descend in the {@link HopsDag} and put all the root lops into a {@link Dag} 
	 * (which is used for instructions generation later). Root lops are lops from transient writes/function ops 
	 * of blocks. The association between @param root and @param dag is created in @see {@link GlobalGraphCreator}.
	 * 
	 * @param dag
	 * @param addedLopIDs
	 * @param output
	 * @throws LopsException 
	 * @throws HopsException 
	 */
	private void addRootLopsToDag(HopsDag dag, Hop root) throws HopsException, LopsException {
		if(dag.containsBlockForHop(root)) {
			////MB: the lops should not be replicated, all configuration changes must be made on hops level for consistency
			/*
			ProgramBlock block = dag.getBlockForHop(root);
			Dag<Lop> lopsDag = block.getLopsDag();
			if(lopsDag == null) {
				LOG.info("lops dag is null for block: " + block.getProgram().getProgramBlocks().indexOf(block));
				lopsDag = new Dag<Lop>();
				block.setLopsDag(lopsDag);
			}
			*/
			Dag<Lop> lopsDag = new Dag<Lop>(); //MB
			
			Lop getLops = root.constructLops();
			if(getLops != null) {
				getLops.addToDag(lopsDag);
			}else {
				LOG.error("wrong configuration !!!! " + root);
			}
		}
		
		ArrayList<Hop> inputs = getInputs(root);
		for(Hop i : inputs) {
			addRootLopsToDag(dag, i);
		}
	}

	/**
	 * Recursive method to iterate over all operators from the {@link OptimizedPlan} (depth first)
	 *  and put the generated {@link Lops} of the optimal plan into the program. 
	 * @param optimalRoot
	 * @param hopsDag
	 * @param addedLopIDs
	 */
	private void applyLops(OptimizedPlan optimalRoot, HopsDag hopsDag, Set<Long> addedLopIDs) {
		Hop root = optimalRoot.getOperator();
		
		if( LDEBUG ) {
			System.out.println(" before =====================");
			System.out.println(optimalRoot.getConfig());
			System.out.println(root);
			Lop lop1 = optimalRoot.getGeneratedLop();
			System.out.println(lop1);
			if(lop1 != null) {
				System.out.println(lop1.getOutputParameters());
			}
		}
		//descend
		if(optimalRoot.getInputPlans() != null) {
			for(MemoEntry e : optimalRoot.getInputPlans()) {
				OptimizedPlan optInputPlan = e.getOptPlan();
				applyLops(optInputPlan, hopsDag, addedLopIDs);
			}	
		}
		
		//TODO: simplify as this repeats parts of the enumeration process
		try {
			generateRuntimePlan(optimalRoot, root);
		} catch (HopsException e1) {
			LOG.error(e1.getMessage(), e1);
		} catch (LopsException e1) {
			LOG.error(e1.getMessage(), e1);
		}
		
		if(LOG.isDebugEnabled()) {
			LOG.debug(root);
		}
		
	}

	/**
	 *  Main method the enumeration approach.
	 *  For given operator @param out do:
	 *  (1) check the memo table if entry then return entry
	 *  (2) check and handle loops
	 *  (3) generate candidates for optimal sub plans for each possible {@link InterestingPropertySet} enumConfig()
	 *  (4) in case of invalid configurations that can be calculated from the node itself prune away pruneInvalids()
	 *  (5) check for rewrites that can be derived for this node in isolation TODO: refactor
	 *  (6) retrieve the inputs 
	 *  (7) descend into inputs 
	 *  (8) combine candidates with optimal sub plans of inputs
	 *  (9) prune invalid combinations 
	 *  (9) check for rewrites that can be derived from the combination of input and current node, e. g. change of block size
	 *  (10) early pruning of suboptimal plans
	 *  (11) pruning of complete node plans (including every available input plan)
	 *   
	 * public due to testing
	 * @param out
	 * @return
	 * @throws LopsException 
	 * @throws HopsException 
	 */
	public Map<InterestingPropertySet, MemoEntry> enumOpt(Hop out, Summary summary) 
		throws HopsException, LopsException 
	{
		SummaryEntry record = summary.startRecording(out);
		summary.incrementDescents();
		
		if(_memo.getEntry(out) != null) {
			Map<InterestingPropertySet, MemoEntry> entry = _memo.getEntry(out);
			return entry;
		} else {
			
			if(out instanceof LoopOp) {
				enumLoop(out);
			}
			
			Map<InterestingPropertySet, Set<OptimizedPlan>> intermediates = enumConfig(out, summary, record);
			
			if(intermediates.size() > 0) {
				pruneInvalid(intermediates, summary, record);
			}
			
			checkRewrites(intermediates.values());
			ArrayList<Hop> inputs = getInputs(out);
			
			summary.increaseLevel();
			for(Hop i : inputs) {
				if(isToConfigure(i)) {
					Map<InterestingPropertySet, MemoEntry> optChildren = enumOpt(i, summary);
					//TODO: external function use case
					if(out instanceof CrossBlockOp && ((CrossBlockOp)out).getLeftInput() instanceof FunctionOp) 
					{
						System.out.println("TODO: special handling for skip reblocking here");
					}
					
					intermediates = combine(intermediates, optChildren, summary, record);
					pruneInvalid(intermediates, summary, record);
					checkRewrites(intermediates.values());
					
					pruneSuboptimal(intermediates, false, summary, record);
				}
			}
			
			summary.decreaseLevel();
			pruneSuboptimal(intermediates, true, summary, record);
			checkForContainment(intermediates);
			
			Map<InterestingPropertySet, MemoEntry> memoEntry = createMemoEntry(intermediates, out);
			_memo.add(memoEntry, out);
			if(LOG.isInfoEnabled()) {
				LOG.info("enumerated " + out.getKind() + ": " + out.get_name());
			}
		}
		
		summary.stopRecording(record);
		//lots of redundant calls here
		summary.setNumberOfConfigs(this.generatedConfigs.size());
		summary.setNumberOfInterestingProperties(this.interestingPropertyCombinations.size());
		return _memo.getEntry(out);
	}

	/**
	 * Checks if any of the optimal plans for the given {@link InterestingPropertySet}s has multiple inputs and 
	 * if those inputs share a node. In this case, the optimal plan has to have the same {@link InterestingPropertySet}.
	 * If that is not the case the memo table has to be wiped out, and enumeration has to be repeated for those two plans with fixed 
	 * ips.
	 * @param intermediates
	 * @return
	 */
	private boolean checkForContainment(Map<InterestingPropertySet, Set<OptimizedPlan>> intermediates) {
		LOG.info("checking for containment");
		for(Entry<InterestingPropertySet, Set<OptimizedPlan>> entry : intermediates.entrySet()) {
			if(!entry.getValue().isEmpty())	{
				OptimizedPlan rootPlan = entry.getValue().iterator().next();
				System.out.println("Checking for containment for " + rootPlan.getOperator() + " " + entry.getKey());
				if(rootPlan.getInputPlans().size() > 1) {
					MemoEntry firstInput = rootPlan.getInputPlans().get(0);
					MemoEntry secondInput = rootPlan.getInputPlans().get(1);
					
					HashSet<Long> visited = new HashSet<Long>();
					pairWiseFirst(firstInput, secondInput, visited);
					
					
					if(rootPlan.getInputPlans().size() > 2) {
						//FIXME: check inputs pairwise for shared nodes and containment
					}
					
				}
				
			}
		}
		return false;
	}

	/**
	 * Iterate over the first input to find shared nodes on a path of optimal plans down to the data sources. 
	 * 
	 * @param firstInput
	 * @param secondInput
	 * @param visited
	 */
	private void pairWiseFirst(MemoEntry firstInput, MemoEntry secondInput,
			HashSet<Long> visited) {
		if(!visited.contains(firstInput)) {
			pairWiseSecond(firstInput, secondInput, visited);
		}
		
		List<MemoEntry> inputPlans = firstInput.getOptPlan().getInputPlans();
		for(MemoEntry newFirstInput : inputPlans) {
			pairWiseFirst(newFirstInput, secondInput, visited);
			visited.add(newFirstInput.getRootHop().getHopID());
		}
	}

	/**
	 * Iterate over the second input to eventually find shared nodes.
	 * @param firstInput
	 * @param secondInput
	 * @param visited
	 */
	private void pairWiseSecond(MemoEntry firstInput, MemoEntry secondInput, Set<Long> visited) {
		OptimizedPlan firstPlan = firstInput.getOptPlan();
		OptimizedPlan secondPlan = secondInput.getOptPlan();
		Hop firstHop = firstPlan.getOperator();
		long firstHopID = firstHop.getHopID();
		Hop secondHop = secondPlan.getOperator();
		long secondHopID = secondHop.getHopID();
		
		//these nodes have already been visited
		if(visited.contains(firstHopID) || visited.contains(secondHopID)) {
			return;
		}
		
		if(firstHopID == secondHopID) {
			System.out.println("found net for " 
					+ firstHop + "(" + firstInput.getInterestingProperties() + ")" 
					+ "and " 
					+ secondHop + "(" + secondInput.getInterestingProperties() + ")"); 
		}else {
			List<MemoEntry> inputPlans = secondPlan.getInputPlans();
			for(MemoEntry newSecondInput : inputPlans) {
				visited.add(newSecondInput.getRootHop().getHopID());
				pairWiseSecond(firstInput, newSecondInput, visited);
			}
		}
		
	}

	/**
	 * Convenience method to retrieve the inputs whether real hops or crossblock operators
	 * @param out
	 * @return
	 */
	private ArrayList<Hop> getInputs(Hop out) {
		ArrayList<Hop> inputs = out.getInput();
		if(inputs.isEmpty() && out.getCrossBlockInput() != null) {
			inputs.add(out.getCrossBlockInput());
		}
		
		
		return inputs;
	}
	
	/**
	 * this function should do the following:
	 * 	(1) prepare the memotable (create new one versus just appending to the old one)
	 *  (2) detect LI (just a member call now)
	 *  (3) detect known predicate
	 *  (4) depending on (2) and (3) unroll, cost once or unroll 
	 *  (5) cost
	 *  (6) detect vectorization or partitioning
	 *  
	 * @param out
	 */
	private void enumLoop(Hop out) {
		// TODO Auto-generated method stub
		
	}

	/**
	 * By definition node plans is technically just a map IP->OptPlan, since pruneSuboptimal(set, true) 
	 * results in exactly one plan per IP.
	 * This method finally stores the lop per IP as a memo entry
	 * @param nodePlans
	 * @param root
	 * @return
	 */
	public Map<InterestingPropertySet, MemoEntry> createMemoEntry(
			Map<InterestingPropertySet, Set<OptimizedPlan>> nodePlans, Hop root) {
		Map<InterestingPropertySet, MemoEntry> retVal = new HashMap<InterestingPropertySet, MemoEntry>();
		for(InterestingPropertySet combi : nodePlans.keySet()) {
			Set<OptimizedPlan> subPlans = nodePlans.get(combi);
			if(subPlans.size() > 1) 
				throw new IllegalStateException("At this point, at most one best sub plan per IP combination should exist!");
			if(subPlans.size() == 1) {
				OptimizedPlan plan = subPlans.iterator().next();
				
				RewriteConfigSet config = plan.getConfig();
				Lop lop = plan.getGeneratedLop();
				Long lopId = _memo.addPlan(lop);
				
				MemoEntry e = new MemoEntry();
				e.setOptPlan(plan);
				e.setConfig(config);
				e.setInterestingProperties(combi);
				e.setRootHop(root);
				e.setLopId(lopId);
				e.setCost(plan.getCumulatedCost());
				retVal.put(combi, e);
				
			}
		}
		return retVal;
	}

	/**
	 * Assign costs to each candidate per {@link InterestingPropertySet}. Generates runtime plans for candidates and
	 * hands them over to the cost model. 
	 * 
	 * TODO: consolidate logic for checking rewrites to one place
	 * 
	 * @param nodePlans
	 * @param completeSubplan
	 * @param summary
	 * @param record
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void pruneSuboptimal(
			Map<InterestingPropertySet, Set<OptimizedPlan>> nodePlans,
			boolean completeSubplan, Summary summary, SummaryEntry record)
			throws HopsException, LopsException {
		long costCounter = 0;
		long planCounter = 0;

		if (!completeSubplan) {
			// apply heuristics
			// summary...
			// and costing with dummy hops
		} else {
			for (Entry<InterestingPropertySet, Set<OptimizedPlan>> e : nodePlans
					.entrySet()) {
				Set<OptimizedPlan> plansToCost = e.getValue();
				Set<OptimizedPlan> replacement = new HashSet<OptimizedPlan>();
				if (plansToCost != null && !plansToCost.isEmpty()) {
					Map<Double, OptimizedPlan> costMap = new HashMap<Double, OptimizedPlan>();
					for (OptimizedPlan p : plansToCost) {
						
						Hop root = p.getOperator();
						
						generateRuntimePlan(p, root);
						ArrayList<Hop> costList = new ArrayList<Hop>();
						
						costList.add(root);
						double timeEstimate = 0;
						
						try {
							
							//MB: cleanup generated lops (workaround to enable instruction generation)
							Hop.resetVisitStatus(costList);
							for( Hop hop : costList )
								rCleanupHopsDAG(hop);
							
							timeEstimate = CostEstimationWrapper
									.getTimeEstimate(costList,
											new ExecutionContext());
							p.setCost(timeEstimate);
							summary.incrementCostFunctionCalls();
						} catch (DMLRuntimeException e1) {
							LOG.error(e1.getMessage(), e1);
						} catch (DMLUnsupportedOperationException e1) {
							LOG.error(e1.getMessage(), e1);
						} catch (IOException e1) {
							LOG.error(e1.getMessage(), e1);
						} catch (NullPointerException ex) {
							LOG.error("root: " + root, ex);
						} catch (LopsException ex) {
							LOG.error("root: " + root, ex);
						}
						costCounter++;
						p.computeCumulatedCosts();
						costMap.put(p.getCumulatedCost(), p);
						
						if( LDEBUG ) {
							System.out.println(p.getConfig() + " " + p.getCumulatedCost() + " " + PrintUtil.generateLopDagString(p.getGeneratedLop()));
						}
					}
					if (costMap.size() > 0) {
						List<Double> costs = new ArrayList<Double>(costMap
								.keySet());
						Collections.sort(costs);
						Double minCost = costs.get(0);
						OptimizedPlan bestPlan = costMap.get(minCost);
						if( LDEBUG ) {
							System.out.println("best plan: "
									+ bestPlan.getConfig() + " " + bestPlan.getCumulatedCost()
									+ " "
									+ PrintUtil.generateLopDagString(bestPlan
											.getGeneratedLop()));
							System.out.println("========================");
						}
						replacement.add(bestPlan);
						planCounter++;
					}
				}
				summary.addPrunedCounter(e.getValue().size() - 1);
				e.setValue(replacement);

			}

			record.addAction("pruneSuboptimal");
			record.addEntryVal(planCounter);

			if (LOG.isInfoEnabled()) {
				LOG.info("Costed " + costCounter + " plans.");
			}
		}
	
	}

	/**
	 * MB: Fix costing with corrupted hop dags.
	 * 
	 * @param hop
	 */
	private void rCleanupHopsDAG( Hop hop )
	{
		if( hop==null || hop.get_visited()==VISIT_STATUS.DONE )
			return;
		
		//cleanup parent-child refs
		ArrayList<Hop> parents = (ArrayList<Hop>) hop.getParent().clone();
		for( Hop p : parents )
			if( !p.getInput().contains(hop) )
				hop.getParent().remove(p);
		
		//recursive invocation
		for( Hop c : hop.getInput() )
			rCleanupHopsDAG( c );
		
		hop.set_visited(VISIT_STATUS.DONE);
	}
	
	
	
	/**
	 *  (1) Puts the optimal runtime sub plans into the inputs 
	 *  (2) generates the rewrites according to the current logic and 
	 *  (3) puts them in.
	 * 
	 * @param p
	 * @param root
	 * @throws HopsException
	 * @throws LopsException
	 */
	public void generateRuntimePlan(OptimizedPlan p, Hop root)
			throws HopsException, LopsException {
		
		RewriteConfigSet config = p.getConfig();
		root.set_lops(null);
		
		configureInputLops(p, root);
		
		RewriteConfigSet extractedConfig = p.getExtractedConfig();
		if(extractedConfig != null) {
			PlanRewriter.applyToHop(root, extractedConfig);
		}else {
			throw new IllegalStateException("At this point there always be an extracted config");
		}
		
		Lop resultingSubPlan = root.constructLops();
		PlanRewriter.generateRewrites(root, p, config);
		
		p.setGeneratedLop(resultingSubPlan);
		if (resultingSubPlan == null) {
			throw new IllegalStateException(
					"generated lop is null for " + root);
		}
		p.applyRewrites();
		
		restoreInputs(p, root);
		
	}

	private void createCrossBlockOutputsForFunctionOps(OptimizedPlan p, InterestingPropertySet interestingProperties) {
		/**
		 * This function creates two memo entries one for the funcOp and one for the follwoing crossblock hop
		 * binblock is per default in DMLBlockSize
		 * costing has to be done here
		 * remember reblocks
		 */
		
		
		Hop operator = p.getOperator();
		FunctionOp funcOp = (FunctionOp)operator;
		RewriteConfigSet config = p.getConfig();
		
		Collection<CrossBlockOp> outputs = funcOp.getCrossBlockOutputs();
		for (CrossBlockOp crossBlockHop : outputs) {
			RewriteConfig formatParam = config.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
			if(formatParam.isValue(FormatType.BINARY_BLOCK.ordinal())) {
				funcOp.set_cols_in_block(DMLTranslator.DMLBlockSize);
				funcOp.set_rows_in_block(DMLTranslator.DMLBlockSize);
				
				InterestingPropertySet propertyCombination = null;
				OptimizedPlan plan = new OptimizedPlan();
				for(RewriteConfigSet c : this.generatedConfigs)  {
					RewriteConfig blockParam = c.getConfigByType(RewriteConfigType.BLOCK_SIZE);
					if(blockParam.getValue() == (long)DMLTranslator.DMLBlockSize) {
						plan.setConfig(c);
						propertyCombination = this.configToProperties.get(c);
						break;
					}
				}
				
				//this might be done already by cost function
				p.computeCumulatedCosts();
				plan.setCumulatedCost(p.getCumulatedCost());
				plan.setOperator(crossBlockHop);
				Map<InterestingPropertySet, MemoEntry> entry = _memo.getEntry(crossBlockHop);
				if(entry == null) {
					Map<InterestingPropertySet, Set<OptimizedPlan>> nodePlans = new HashMap<InterestingPropertySet, Set<OptimizedPlan>>();
					Set<OptimizedPlan> planSet = new HashSet<OptimizedPlan>();
					planSet.add(plan);
					nodePlans.put(propertyCombination, planSet);
					this.createMemoEntry(nodePlans, crossBlockHop);
				}else {
					MemoEntry newEntry = new MemoEntry();
					newEntry.setConfig(plan.getConfig());
					newEntry.setCost(plan.getCumulatedCost());
					newEntry.setInterestingProperties(propertyCombination);
					newEntry.setRootHop(crossBlockHop);
					
					
				}
				
			}
		}
		
		
	}

	private void restoreInputs(OptimizedPlan p, Hop root) {
		for (int i = 0; i < root.getInput().size(); i++) {
			Hop input = root.getInput().get(i);
			if (!(input instanceof LiteralOp)) {
				List<MemoEntry> inputPlans = p
						.getInputPlans();
				if (inputPlans != null
						&& inputPlans.size() > i) {
					
					MemoEntry relevantSubplan = inputPlans
							.get(i);
					OptimizedPlan optPlan = relevantSubplan.getOptPlan();
					RewriteConfigSet extractedConfig = optPlan.getExtractedConfig();
					PlanRewriter.applyToHop(input, extractedConfig);
				}
			}
		}
	}

	/**
	 * Configure the inputs according to the stored parameters in the memotable.
	 * 
	 * @param p
	 * @param root
	 */
	private void configureInputLops(OptimizedPlan p, Hop root) {
		for (int i = 0; i < root.getInput().size(); i++) {
			Hop input = root.getInput().get(i);
			if (!(input instanceof LiteralOp)) {
				List<MemoEntry> inputPlans = p
						.getInputPlans();
				if (inputPlans != null
						&& inputPlans.size() > i) {
					
					MemoEntry relevantSubplan = inputPlans
							.get(i);
					RewriteConfigSet inputConfig = relevantSubplan.getConfig();
					Long lopId = relevantSubplan.getLopId();
					Lop lop = _memo.getPlan(lopId);
					input.set_lops(lop);
					
					/*
					 * TODO: generalize that to all operators
					 * basically, determine if a sibling of lop is executed in MR 
					 * and pick that block size and configure lop with that block size
					 * 
					*/
					if(inputConfig.getConfigByType(RewriteConfigType.EXEC_TYPE).getValue()==ExecType.CP.ordinal()) {
						boolean fromSibling = false;
						for(Hop in : root.getInput()) {
							if(!(in instanceof LiteralOp) && !in.equals(input)) {
								int getColsInBlock = (int)in.get_cols_in_block();
								//TODO: use for rectangular block shapes later
								int getRowsInBlock = (int)in.get_rows_in_block();
								
								//this is a workaround to prevent a permanent change in the inputConfig
								//due to reference change
								RewriteConfigBlocksize param = new RewriteConfigBlocksize();
								param.setValue(getColsInBlock);
								
								input.set_lops(null);
								PlanRewriter.applyToHop(input, inputConfig);
								PlanRewriter.applyToHop(input, param);
								try {
									lop = input.constructLops();
									_memo.setPlan(lopId, lop);
								} catch (HopsException e) {
									LOG.error(e.getMessage(), e);
								} catch (LopsException e) {
									LOG.error(e.getMessage(), e);
								}
								fromSibling = true;
								break;
							}
						}
						//export whatever block size is needed by parent node
						if(!fromSibling) {
							RewriteConfigBlocksize bsize = new RewriteConfigBlocksize((int)root.get_cols_in_block());
							input.set_lops(null);
							PlanRewriter.applyToHop(input, inputConfig);
							PlanRewriter.applyToHop(input, bsize);
							try {
								lop = input.constructLops();
								_memo.setPlan(lopId, lop);
							} catch (HopsException e) {
								LOG.error(e.getMessage(), e);
							} catch (LopsException e) {
								LOG.error(e.getMessage(), e);
							}
						}
						
						
					}else {
						PlanRewriter.applyToHop(input, inputConfig);
					}
				}
			}
		}
	}

	/**
	 * Prune invalid (combinations of) configurations/rewrites from the set of intermediates.
	 * Such could be e.g. Configurations for a persistent read of a matrix in CP or combinations of n-ary ops with inputs of different 
	 * block sizes.
	 * 
	 * TODO: simplify the logic here. The more rewrites and configuration parameters the more complex this method got
	 * @param combined
	 * @param record 
	 */
	private void pruneInvalid(Map<InterestingPropertySet, Set<OptimizedPlan>> combined, Summary summary, SummaryEntry record) {
		long pruneCounter = 0;
		long planCounter = 0;
		Set<InterestingPropertySet> removeSet = new HashSet<InterestingPropertySet>();
		
		for(Set<OptimizedPlan> plans : combined.values()){
		
			for(Iterator<OptimizedPlan> ni = plans.iterator(); ni.hasNext(); ) {
				OptimizedPlan plan = ni.next();
				boolean isIntermediateValid = true;
				
				Hop operator = plan.getOperator();
				
				RewriteConfigSet config = plan.getConfig();
				
				//check for CP memory limit 
				if(!PlanRewriter.isValidForOperator(operator, config) )
				{
					isIntermediateValid = false;
				}
				List<MemoEntry> inputPlans = plan.getInputPlans();
				
				//check for non matching interesting properties among the n inputs (n > 1); e.g. different binblock sizes
				isIntermediateValid = isIntermediateValid && checkValidityOfInputs(config, inputPlans);
				//if crossblock is involved the configs have to be the same, no enumeration/rewrites here
				isIntermediateValid = isIntermediateValid && isCrossBlockHopWiringValid(plan);
				//check if the root is capable of dealing with the configured format
				isIntermediateValid = isIntermediateValid && isFormatValid(plan);
				
				//finally add the candidate to the prune set if invalid
				if(!isIntermediateValid)
				{
					isIntermediateValid = true;
					ni.remove();
					summary.incrementPrunedInvalid();
					pruneCounter++;
				}else {
				}
			}
		}
		
		//TODO: this is very inefficient, remove unnecessary loops
		for(InterestingPropertySet ipc : combined.keySet()) {
			if(combined.get(ipc).isEmpty()) {
				removeSet.add(ipc);
			}
			planCounter += combined.get(ipc).size();
		}
		
		for(InterestingPropertySet ipc : removeSet) {
			combined.remove(ipc);
		}
		
		record.addAction("pruneInvalid");
		record.addEntryVal(planCounter);
		
		if(LOG.isInfoEnabled()) {
			LOG.info("Pruned " + pruneCounter);
		}
	}

	/**
	 * TODO: refactor to {@link FormatProperty}
	 * 
	 * Should check if the current operator is something else than a {@link DataOp} or FunctionOp 
	 * and if one of its inputs delivers text
	 * @param plan
	 * @return
	 */
	private boolean isFormatValid(OptimizedPlan plan) {
		Hop operator = plan.getOperator();
		RewriteConfigSet rootConfig = plan.getConfig();
		RewriteConfig rootFormat = rootConfig.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
		List<MemoEntry> inputPlans = plan.getInputPlans();
		if(inputPlans != null && inputPlans.size() > 0) {
			MemoEntry firstInput = inputPlans.get(0);
			RewriteConfigSet inputConfig = firstInput.getConfig();
			RewriteConfigFormat inputFormat = (RewriteConfigFormat)inputConfig.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
		
			if(!PlanRewriter.isFormatValid(operator, inputFormat)) {
				return false;
			}
		}
		
		return true;
	}

	/**
	 * Compare the values of input matrices to have matching block size and format.
	 * 
	 * @param inputPlans
	 * @return
	 */
	private boolean checkValidityOfInputs(RewriteConfigSet rootConfig, List<MemoEntry> inputPlans) {
		boolean areInputsValid = true;
		
		if(inputPlans != null && inputPlans.size() > 1) {
			for(int i = 0; i < inputPlans.size() - 1; i++) {
				Hop op1 = inputPlans.get(i).getRootHop();
				
				if(op1.get_dataType().equals(DataType.MATRIX)) {
					RewriteConfigSet firstConfig = inputPlans.get(i).getConfig();
					
					Hop op2 = inputPlans.get(i + 1).getRootHop();
					
					if(op2.get_dataType().equals(DataType.MATRIX)) {
						RewriteConfigSet followConfig = inputPlans.get(i + 1).getConfig();
						if(!RewriteConfigUtils.isCompatible(firstConfig, followConfig)) {
							areInputsValid = false;
							break;
						}
					}
				}
			}
		}
		return areInputsValid;
	}

	/**
	 * Checks if 
	 * (1) this is a plan for a CBH 
	 * (2) the input to this plan is a CBH
	 * (3) this a plan for a transient write.
	 * In any case the formats and block sizes need to match. 
	 * @param plan
	 * @return
	 */
	private boolean isCrossBlockHopWiringValid(OptimizedPlan plan) {
		
		Hop operator = plan.getOperator();
		RewriteConfigSet config = plan.getConfig();
		List<MemoEntry> inputs = plan.getInputPlans();
		Integer rootBlockSize = config.getConfigByType(RewriteConfigType.BLOCK_SIZE).getValue();
		Integer rootFormat = config.getConfigByType(RewriteConfigType.FORMAT_CHANGE).getValue();
		Integer rootLocation = config.getConfigByType(RewriteConfigType.EXEC_TYPE).getValue();
		
		if(inputs != null && inputs.size() > 0) {
			
			if(operator instanceof CrossBlockOp) {
				MemoEntry memoEntry = inputs.get(0);
				RewriteConfigSet inputConfig = memoEntry.getConfig();
				Integer inputBlockSize = inputConfig.getConfigByType(RewriteConfigType.BLOCK_SIZE).getValue();
				Integer inputFormat = inputConfig.getConfigByType(RewriteConfigType.FORMAT_CHANGE).getValue();
				Integer inputLocation = inputConfig.getConfigByType(RewriteConfigType.EXEC_TYPE).getValue();
				if(rootBlockSize != inputBlockSize) {
					return false;
				}
				
				if(rootFormat != inputFormat) {
					return false;
				}
				
				if(rootLocation != inputLocation) {
					return false;
				}
			}
			
			if(operator instanceof DataOp && 
					(((DataOp)operator).get_dataop().equals(DataOpTypes.TRANSIENTWRITE)
					|| ((DataOp)operator).get_dataop().equals(DataOpTypes.PERSISTENTWRITE))
			) {
				MemoEntry memoEntry = inputs.get(0);
				RewriteConfigSet inputConfig = memoEntry.getConfig();
				int inputBlockSize = inputConfig.getConfigByType(RewriteConfigType.BLOCK_SIZE).getValue();
				int inputFormat = inputConfig.getConfigByType(RewriteConfigType.BLOCK_SIZE).getValue();
				
				if(rootFormat==FormatType.BINARY_BLOCK.ordinal() && inputFormat != rootFormat) {
					return false;
				}
				
				if(!rootBlockSize.equals(inputBlockSize) 
						&& rootFormat == FormatType.BINARY_BLOCK.ordinal() && inputFormat==FormatType.BINARY_BLOCK.ordinal()) {
					return false;
				}
				
			}
		}
		return true;
	}

	/** 
	 * Combine every candidate of the existing intermediate candidates with the enumerated optimal sub plans 
	 * (@param children) of the input
	 * @param intermediates
	 * @param children
	 * @param summary
	 * @param record
	 * @return
	 */
	public Map<InterestingPropertySet, Set<OptimizedPlan>> combine(
			Map<InterestingPropertySet, Set<OptimizedPlan>> intermediates,
			Map<InterestingPropertySet, MemoEntry> children, Summary summary, SummaryEntry record) {
		
		long combineCounter = 0;
		Map<InterestingPropertySet, Set<OptimizedPlan>> combinations = 
			new HashMap<InterestingPropertySet, Set<OptimizedPlan>>();
		
		if(children.isEmpty()) {
			combinations.putAll(intermediates);
		} else {
			
			for(Entry<InterestingPropertySet, Set<OptimizedPlan>> entry : intermediates.entrySet()) {
				InterestingPropertySet key = entry.getKey();
				Set<OptimizedPlan> nodePlanSet = entry.getValue();
				
				for(OptimizedPlan plan : nodePlanSet) {
					
					for(Entry<InterestingPropertySet, MemoEntry> childEntry : children.entrySet()){
						OptimizedPlan newPlan = new OptimizedPlan();;
						newPlan.setOperator(plan.getOperator());
						newPlan.setExtractedConfig(plan.getExtractedConfig());
						newPlan.setRuntimeProgram(plan.getRuntimeProgram());
						newPlan.setConfig(plan.getConfig());
						
						if(plan.getInputPlans() != null) {
							//TODO: test that the order is preserved here!!!
							newPlan.getInputPlans().addAll(plan.getInputPlans());
						} else {
							newPlan.setInputPlans(new ArrayList<MemoEntry>());
						}
						newPlan.addInputLop(childEntry.getValue());
						if(combinations.get(key) == null)
						{
							combinations.put(key, new HashSet<OptimizedPlan>());
						}
						
						combinations.get(key).add(newPlan);
						summary.incrementGeneratedPlan();
						combineCounter++;
					}
					
				}
			}
		}
		
		if(LOG.isInfoEnabled()) {
			LOG.info("Generated " + combineCounter + " combinations.");
		}
		record.addAction("combine");
		record.addEntryVal(combineCounter);
		return combinations;
	}

	/**
	 * Checks if the combination of configuration, inputplans and interesting properties 
	 * requires a (reblock) rewrite or not. 
	 * TODO: This needs to be consolidated with the apply-methos of {@link Rewrite}s since there is also 
	 * logic applied.  
	 * 
	 * @param newPlan
	 */
	private void checkRewrites(Collection<Set<OptimizedPlan>> planSets) {
		
		for(Set<OptimizedPlan> plans : planSets) {
			for(OptimizedPlan plan : plans) {
				
				RewriteConfigSet config = plan.getConfig();
				RewriteConfig param = config.getConfigByType(RewriteConfigType.BLOCK_SIZE);
				int nodeBlockSize = param.getValue();
				
				Hop operator = plan.getOperator();
				
				if (operator instanceof DataOp
						&& (((DataOp) operator).get_dataop().equals(DataOpTypes.PERSISTENTWRITE)
								|| ((DataOp) operator).get_dataop().equals(DataOpTypes.TRANSIENTWRITE))
					) {
					return;
				}				
				
				checkReblockRewriteForMR(plan, nodeBlockSize, operator);
				
				if(operator instanceof DataOp && 
						operator.get_dataType().equals(DataType.MATRIX) &&
						(((DataOp)operator).get_dataop().equals(DataOpTypes.TRANSIENTREAD) 
								|| ((DataOp)operator).get_dataop().equals(DataOpTypes.PERSISTENTREAD)
					    )	
				  ) {
					checkRewriteForReadOperators(plan, config, operator);
				} else {
					checkRewriteFromFirstChild(plan, config);
				}
			}
		}
		
	}

	/**
	 * Examines the interestiong properties from the dirst input matrix 
	 * @param plan
	 * @param config
	 */
	private void checkRewriteFromFirstChild(OptimizedPlan plan,
			RewriteConfigSet config) {
		List<MemoEntry> inputPlans = plan.getInputPlans();
		if(inputPlans != null && inputPlans.size() > 0) {
			MemoEntry firstEntry = inputPlans.get(0);
			InterestingPropertySet inputInterestingProperties = firstEntry.getInterestingProperties();
			InterestingProperty inputFormatProperty = inputInterestingProperties.getPropertyByType(InterestingPropertyType.FORMAT);
			
			RewriteConfig formatParam = config.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
			
			if(!(formatParam.getValue()==inputFormatProperty.getValue())) {
				ReblockRewrite rewrite = new ReblockRewrite();
				rewrite.setFormat(FormatType.values()[formatParam.getValue()]);
					InterestingProperty blockSizeProperty = inputInterestingProperties.getPropertyByType(InterestingPropertyType.BLOCK_SIZE);
					RewriteConfig blockSizeParam = config.getConfigByType(RewriteConfigType.BLOCK_SIZE);
					rewrite.setFromBlockSize((int)blockSizeProperty.getValue());
					rewrite.setToBlockSize(blockSizeParam.getValue());
					
					Hop operator = plan.getOperator();
					operator.set_cols_in_block(blockSizeProperty.getValue());
					operator.set_rows_in_block(blockSizeProperty.getValue());
//				}
				plan.addRewrite(RewriteConfigType.FORMAT_CHANGE, rewrite);
			}
		}
	}

	/**
	 * TODO: redundant -> refactor
	 * @param plan
	 * @param config
	 * @param operator
	 */
	private void checkRewriteForReadOperators(OptimizedPlan plan,
			RewriteConfigSet config, Hop operator) {
		DataOp dataOp = (DataOp)operator;
		FileFormatTypes currentFormat = dataOp.getFormatType();
		RewriteConfig formatParam = config.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
		if(!formatParam.equals(currentFormat)) {
			ReblockRewrite rewrite = new ReblockRewrite();
			rewrite.setFormat(FormatType.values()[formatParam.getValue()]);
			if(formatParam.getValue()==FormatType.BINARY_BLOCK.ordinal()) {
				RewriteConfig blockSizeParam = config.getConfigByType(RewriteConfigType.BLOCK_SIZE);
				rewrite.setFromBlockSize((int)dataOp.get_cols_in_block());
				rewrite.setToBlockSize(blockSizeParam.getValue());
				plan.addRewrite(RewriteConfigType.BLOCK_SIZE, rewrite);
			}
			plan.addRewrite(RewriteConfigType.FORMAT_CHANGE, rewrite);
		}
	}

	/**
	 * @param plan
	 * @param nodeBlockSize
	 * @param operator
	 */
	private void checkReblockRewriteForMR(OptimizedPlan plan,
			int nodeBlockSize, Hop operator) {
		if(nodeBlockSize != -1 || operator instanceof CrossBlockOp)	{
			List<MemoEntry> inputs = plan.getInputPlans();
			if(inputs != null && inputs.size() > 0) {
				for(MemoEntry e : inputs) {
					InterestingPropertySet interestingProperties = e.getInterestingProperties();
					InterestingProperty prop = interestingProperties.getPropertyByType(InterestingPropertyType.BLOCK_SIZE);
					if(prop.getValue() != -1) {
						int outgoingBlockSize = (int)prop.getValue();
						if(nodeBlockSize != outgoingBlockSize) {
							if(LOG.isInfoEnabled()) {
								LOG.info("adding reblock to " + operator);
							}
							ReblockRewrite rewrite = new ReblockRewrite();
							rewrite.setFromBlockSize(outgoingBlockSize);
							rewrite.setToBlockSize(nodeBlockSize);
							//RewriteConfigFormat param = new RewriteConfigFormat(FormatType.BINARY_BLOCK.ordinal());
							rewrite.setFormat(FormatType.BINARY_BLOCK);
							plan.addRewrite(RewriteConfigType.BLOCK_SIZE, rewrite);
							plan.addRewrite(RewriteConfigType.FORMAT_CHANGE, rewrite);
						}
					}
				}
			}
		}
	}

	/**
	 * Depending on the type of operator and the number of interesting properties 
	 * all the meta data combinations will be generated.
	 * 
	 * TODO: 
	 * 	- requires reblock should be a very concise, type&rule specific method => refactor to ConfigParam, Configuration or ...
	 * 
	 * 
	 * @param out
	 * @param record 
	 * @return
	 */
	public Map<InterestingPropertySet, Set<OptimizedPlan>> enumConfig(Hop out, Summary summary, SummaryEntry record) {
		long planCounter = 0;
		
		Map<InterestingPropertySet, Set<OptimizedPlan>> retVal = new HashMap<InterestingPropertySet, Set<OptimizedPlan>>();
		boolean toConfigure = isToConfigure(out);
		if(toConfigure) {
			for (RewriteConfigSet c : generatedConfigs) {
				if(out.get_dataType().equals(DataType.SCALAR)
					&& (c.getConfigByType(RewriteConfigType.EXEC_TYPE).isValue(ExecType.CP.ordinal()))) 
				{
					//don't generate MR configs for scalars
				}else {
					InterestingPropertySet combination = this.configToProperties.get(c);
					//TODO: enable and complete for external function usecase
//					if(out instanceof FunctionOp) {
//						FunctionOp functionOp = (FunctionOp)out;
//						InterestingProperty formatProperty = combination.getPropertyByName(FormatProperty.NAME);
//						
//						planCounter = enumerateFunctionOps(out, summary,
//								planCounter, retVal, c, combination,
//								functionOp, formatProperty);
//						
//					}else {
						OptimizedPlan p = new OptimizedPlan();
						p.setConfig(c);
						p.setOperator(out);
						
						RewriteConfigSet extractedConfig = RewriteConfigUtils.extractConfigurationFromHop(out, c);
						p.setExtractedConfig(extractedConfig);
						p.setRuntimeProgram(this.currentProgram);
						Set<OptimizedPlan> plans = new HashSet<OptimizedPlan>();
						plans.add(p);
						retVal.put(combination, plans);
						summary.incrementGeneratedPlan();
						planCounter++;
//					}
				}
			}
		}
		record.addAction("config");
		record.addEntryVal(planCounter);
		return retVal;
	}

	/**
	 * @param out
	 * @param summary
	 * @param planCounter
	 * @param retVal
	 * @param c
	 * @param combination
	 * @param functionOp
	 * @param formatProperty
	 * @return
	 */
	
	private long enumerateFunctionOps(Hop out, Summary summary,
			long planCounter,
			Map<InterestingPropertySet, Set<OptimizedPlan>> retVal,
			RewriteConfigSet c, InterestingPropertySet combination,
			FunctionOp functionOp, InterestingProperty formatProperty) {
		Set<OptimizedPlan> plans = new HashSet<OptimizedPlan>();
//		if(formatProperty.getValue().getAsLong() == FormatProperty.TEXT) {
		    String[] outputs = functionOp.getOutputVariableNames(); //MB
		    //String[] outputs = functionOp.getOutputs().toArray(new String[0]);
			for(String output : outputs) {
				FunctionOptimizedPlan p = new FunctionOptimizedPlan();
				p.setConfig(c);
				p.setOperator(out);
				p.setOutput(output);
				RewriteConfigSet extractedConfig = RewriteConfigUtils.extractConfigurationFromHop(out, c);
				p.setExtractedConfig(extractedConfig);
				p.setRuntimeProgram(this.currentProgram);
				plans.add(p);
				
				summary.incrementGeneratedPlan();
				planCounter++;
			}
		retVal.put(combination, plans);
		
		return planCounter;
	}

	private boolean isToConfigure(Hop out) {
		if(out instanceof LiteralOp) {
			return false;
		}
		
		if(out instanceof DataOp && out.get_dataType().equals(DataType.SCALAR)) {
			return false;
		}
		return true;
	}

	OptimizedPlan pickOptimal(Map<InterestingPropertySet, MemoEntry> plans) {
		OptimizedPlan optimal = null;
		double optimalCost = Double.MAX_VALUE;
		if(plans != null && plans.size() > 0) {
			for(MemoEntry e : plans.values()) {
				double currCost = e.getCost();
				if(LOG.isDebugEnabled()) {
					LOG.debug("curreCost: " + currCost + ", " + e.getOptPlan());
				}
				if(currCost <= optimalCost) {
					optimalCost = currCost;
					optimal = e.getOptPlan();
				}
			}
		}
		return optimal;
	}

	//TODO: reason about incompatible combinations
	private Set<InterestingPropertySet> generatePropertyCombinations(
			Collection<RewriteConfigSet> configs) {
		Set<InterestingPropertySet> retVal = new HashSet<InterestingPropertySet>();
		
		for(RewriteConfigSet c : configs) {
			InterestingPropertySet combination = new InterestingPropertySet();
			for(RewriteConfig p : c.getConfigs()) {
				combination.addProperty(p.getInterestingProperty());
			}
			retVal.add(combination);
			this.configToProperties.put(c, combination);
		}
		
		return retVal;
	}
	
	/**
	 * This has been necessary for a while to convert into a map of strings and sets. That ist not 
	 * necessary anymore. However, is this method need as a copy? 
	 * @param dataConfigs
	 */
	@Deprecated
	public Map<InterestingPropertySet, Set<OptimizedPlan>> createIntermediatesSet(
			Map<InterestingPropertySet, OptimizedPlan> dataConfigs) {
		Map<InterestingPropertySet, Set<OptimizedPlan>> intermediates = new HashMap<InterestingPropertySet, Set<OptimizedPlan>>();
		for(Entry<InterestingPropertySet, OptimizedPlan> e : dataConfigs.entrySet()) {
			Set<OptimizedPlan> set = new HashSet<OptimizedPlan>();
			set.add(e.getValue());
			intermediates.put(e.getKey(), set);
		}
		return intermediates;
	}

	public Set<InterestingPropertySet> getInterestingPropertyCombinations() {
		return interestingPropertyCombinations;
	}

	public Program getCurrentProgram() {
		return currentProgram;
	}

	public void setCurrentProgram(Program currentProgram) {
		this.currentProgram = currentProgram;
	}
	
}
