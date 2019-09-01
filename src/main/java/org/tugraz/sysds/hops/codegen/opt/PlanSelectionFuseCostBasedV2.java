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

package org.tugraz.sysds.hops.codegen.opt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.AggBinaryOp;
import org.tugraz.sysds.hops.AggUnaryOp;
import org.tugraz.sysds.hops.BinaryOp;
import org.tugraz.sysds.hops.DnnOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.IndexingOp;
import org.tugraz.sysds.hops.LiteralOp;
import org.tugraz.sysds.hops.NaryOp;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.hops.ParameterizedBuiltinOp;
import org.tugraz.sysds.hops.ReorgOp;
import org.tugraz.sysds.hops.TernaryOp;
import org.tugraz.sysds.hops.UnaryOp;
import org.tugraz.sysds.hops.Hop.AggOp;
import org.tugraz.sysds.hops.Hop.DataGenMethod;
import org.tugraz.sysds.hops.Hop.DataOpTypes;
import org.tugraz.sysds.hops.Hop.Direction;
import org.tugraz.sysds.hops.Hop.OpOp2;
import org.tugraz.sysds.hops.Hop.OpOpN;
import org.tugraz.sysds.hops.codegen.opt.ReachabilityGraph.SubProblem;
import org.tugraz.sysds.hops.codegen.template.CPlanMemoTable;
import org.tugraz.sysds.hops.codegen.template.TemplateOuterProduct;
import org.tugraz.sysds.hops.codegen.template.TemplateRow;
import org.tugraz.sysds.hops.codegen.template.TemplateUtils;
import org.tugraz.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.tugraz.sysds.hops.codegen.template.TemplateBase.TemplateType;
import org.tugraz.sysds.hops.rewrite.HopRewriteUtils;
import org.tugraz.sysds.runtime.codegen.LibSpoofPrimitives;
import org.tugraz.sysds.runtime.controlprogram.caching.LazyWriteBuffer;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import org.tugraz.sysds.utils.Statistics;

/**
 * This cost-based plan selection algorithm chooses fused operators
 * based on the DAG structure and resulting overall costs. This includes
 * holistic decisions on 
 * <ul>
 *   <li>Materialization points per consumer</li>
 *   <li>Sparsity exploitation and operator ordering</li>
 *   <li>Decisions on overlapping template types</li>
 *   <li>Decisions on multi-aggregates with shared reads</li>
 *   <li>Constraints (e.g., memory budgets and block sizes)</li>  
 * </ul>
 * 
 */
public class PlanSelectionFuseCostBasedV2 extends PlanSelection
{
	private static final Log LOG = LogFactory.getLog(PlanSelectionFuseCostBasedV2.class.getName());
	
	//common bandwidth characteristics, with a conservative write bandwidth in order 
	//to cover result allocation, write into main memory, and potential evictions
	private static final double WRITE_BANDWIDTH_IO  =      512*1024*1024;  //512MB/s
	private static final double WRITE_BANDWIDTH_MEM =  2d*1024*1024*1024;  //2GB/s
	private static final double READ_BANDWIDTH_MEM  = 32d*1024*1024*1024;  //32GB/s
	private static final double READ_BANDWIDTH_BROADCAST = WRITE_BANDWIDTH_IO/4;
	private static final double COMPUTE_BANDWIDTH  =   2d*1024*1024*1024   //1GFLOPs/core
		* InfrastructureAnalyzer.getLocalParallelism();
	
	//sparsity estimate for unknown sparsity to prefer sparse-safe fusion plans
	private static final double SPARSE_SAFE_SPARSITY_EST = 0.1;
	
	//after evaluating the costs of the opening heuristics fuse-all and fuse-no-redundancy,
	//remaining candidate plans of large partitions (w/ >= COST_MIN_EPS_NUM_POINTS) are
	//only evaluated if the current costs are > (1+COST_MIN_EPS) * static (i.e., minimal) costs.
	public static final double COST_MIN_EPS = 0.01; //1%
	public static final int COST_MIN_EPS_NUM_POINTS = 20; //2^20 = 1M plans
	
	//In order to avoid unnecessary repeated reoptimization we use a plan cache for
	//mapping partition signatures (including input sizes) to optimal plans. However,
	//since hop ids change during dynamic recompilation, we use an approximate signature
	//that is cheap to compute and therefore only use this for large partitions.
	private static final int PLAN_CACHE_NUM_POINTS = 10; //2^10 = 1024
	private static final int PLAN_CACHE_SIZE = 1024;
	private static final LinkedHashMap<PartitionSignature, boolean[]> _planCache = new LinkedHashMap<>();
	
	//optimizer configuration
	public static boolean COST_PRUNING = true;
	public static boolean STRUCTURAL_PRUNING = true;
	public static boolean PLAN_CACHING = true;
	private static final TemplateRow ROW_TPL = new TemplateRow();
	
	//cost vector id generator, whose ids are only used for memoization per call to getPlanCost;
	//hence, we use a sequence generator per optimizer instance to avoid thread contention in 
	//multi-threaded parfor scenarios with concurrent dynamic recompilation and thus optimization.
	private final IDSequence COST_ID = new IDSequence();
	
	@Override
	public void selectPlans(CPlanMemoTable memo, ArrayList<Hop> roots) 
	{
		//step 1: analyze connected partitions (nodes, roots, mat points)
		Collection<PlanPartition> parts = PlanAnalyzer.analyzePlanPartitions(memo, roots, true);
		
		//step 2: optimize individual plan partitions
		int sumMatPoints = 0;
		for( PlanPartition part : parts ) {
			//create composite templates (within the partition)
			createAndAddMultiAggPlans(memo, part.getPartition(), part.getRoots());
			
			//plan enumeration and plan selection
			selectPlans(memo, part);
			sumMatPoints += part.getMatPointsExt().length;
		}
		
		//step 3: add composite templates (across partitions)
		createAndAddMultiAggPlans(memo, roots);
		
		//take all distinct best plans
		for( Entry<Long, List<MemoTableEntry>> e : getBestPlans().entrySet() )
			memo.setDistinct(e.getKey(), e.getValue());
		
		//maintain statistics
		if( DMLScript.STATISTICS ) {
			if( sumMatPoints >= 63 )
				LOG.warn("Long overflow on maintaining codegen statistics "
					+ "for a DAG with "+sumMatPoints+" interesting points.");
			Statistics.incrementCodegenEnumAll(UtilFunctions.pow(2, sumMatPoints));
		}
	}
	
	private void selectPlans(CPlanMemoTable memo, PlanPartition part) 
	{
		//prune special case patterns and invalid plans (e.g., blocksize)
		pruneInvalidAndSpecialCasePlans(memo, part);
		
		//if no materialization points, use basic fuse-all w/ partition awareness
		if( part.getMatPointsExt() == null || part.getMatPointsExt().length==0 ) {
			for( Long hopID : part.getRoots() )
				rSelectPlansFuseAll(memo, 
					memo.getHopRefs().get(hopID), null, part.getPartition());
		}
		else {
			//obtain hop compute costs per cell once
			HashMap<Long, Double> computeCosts = new HashMap<>();
			for( Long hopID : part.getPartition() )
				getComputeCosts(memo.getHopRefs().get(hopID), computeCosts);
			
			//prepare pruning helpers and prune memo table w/ determined mat points
			StaticCosts costs = new StaticCosts(computeCosts, sumComputeCost(computeCosts),
				getReadCost(part, memo), getWriteCost(part.getRoots(), memo), minOuterSparsity(part, memo));
			ReachabilityGraph rgraph = STRUCTURAL_PRUNING ? new ReachabilityGraph(part, memo) : null;
			if( STRUCTURAL_PRUNING ) {
				part.setMatPointsExt(rgraph.getSortedSearchSpace());
				for( Long hopID : part.getPartition() )
					memo.pruneRedundant(hopID, true, part.getMatPointsExt());
			}
			
			//enumerate and cost plans, returns optional plan
			boolean[] bestPlan = enumPlans(memo, part,
				costs, rgraph, part.getMatPointsExt(), 0);
			
			//prune memo table wrt best plan and select plans
			HashSet<Long> visited = new HashSet<>();
			for( Long hopID : part.getRoots() )
				rPruneSuboptimalPlans(memo, memo.getHopRefs().get(hopID), 
					visited, part, part.getMatPointsExt(), bestPlan);
			HashSet<Long> visited2 = new HashSet<>();
			for( Long hopID : part.getRoots() )
				rPruneInvalidPlans(memo, memo.getHopRefs().get(hopID), 
					visited2, part, bestPlan);
			
			for( Long hopID : part.getRoots() )
				rSelectPlansFuseAll(memo, 
					memo.getHopRefs().get(hopID), null, part.getPartition());
		}
	}
	
	/**
	 * Core plan enumeration algorithm, invoked recursively for conditionally independent
	 * subproblems. This algorithm fully explores the exponential search space of 2^m,
	 * where m is the number of interesting materialization points. We iterate over
	 * a linearized search space without every instantiating the search tree. Furthermore,
	 * in order to reduce the enumeration overhead, we apply two high-impact pruning
	 * techniques (1) pruning by evolving lower/upper cost bounds, and (2) pruning by
	 * conditional structural properties (so-called cutsets of interesting points). 
	 * 
	 * @param memo memoization table of partial fusion plans
	 * @param part connected component (partition) of partial fusion plans with all necessary meta data
	 * @param costs summary of static costs (e.g., partition reads, writes, and compute costs per operator)
	 * @param rgraph reachability graph of interesting materialization points
	 * @param matPoints sorted materialization points (defined the search space)
	 * @param off offset for recursive invocation, indicating the fixed plan part
	 * @return optimal assignment of materialization points
	 */
	private boolean[] enumPlans(CPlanMemoTable memo, PlanPartition part, StaticCosts costs, 
		ReachabilityGraph rgraph, InterestingPoint[] matPoints, int off)
	{
		//scan linearized search space, w/ skips for branch and bound pruning
		//and structural pruning (where we solve conditionally independent problems)
		//bestC is monotonically non-increasing and serves as the upper bound
		final int Mlen = matPoints.length-off;
		final long len = UtilFunctions.pow(2, Mlen);
		long numEvalPlans = 2, numEvalPartPlans = 0;
		
		//evaluate heuristics fuse-all and fuse-no-redundancy to quickly obtain a good lower bound
		final boolean[] plan0 = createAssignment(Mlen, off, 0); // fuse-all
		final boolean[] planN = createAssignment(Mlen, off, len-1); //fuse-no-redundancy
		final double C0 = getPlanCost(memo, part, matPoints, plan0, costs._computeCosts, Double.MAX_VALUE);
		final double CN = getPlanCost(memo, part, matPoints, planN, costs._computeCosts, Double.MAX_VALUE);
		boolean[] bestPlan = (C0 <= CN) ? plan0 : planN;
		double bestC = Math.min(C0, CN);
		final boolean evalRemain = (Mlen < COST_MIN_EPS_NUM_POINTS 
			|| !COST_PRUNING || bestC > (1+COST_MIN_EPS) * costs.getMinCosts());
		if( LOG.isTraceEnabled() )
			LOG.trace("Enum opening: " + Arrays.toString(bestPlan) + " -> " + bestC);
		if( !evalRemain )
			LOG.warn("Skip enum for |M|="+Mlen+", C="+bestC+", Cmin="+costs.getMinCosts());
		
		//probe plan cache for existing optimized plan
		PartitionSignature pKey = null;
		if( probePlanCache(matPoints) ) {
			pKey = new PartitionSignature(part, matPoints.length, costs, C0, CN);
			boolean[] plan = getPlan(pKey);
			if( plan != null ) {
				Statistics.incrementCodegenEnumAllP((rgraph!=null||!STRUCTURAL_PRUNING)?len:0);
				return plan;
			}
		}
		
		//evaluate remaining plans, except already evaluated heuristics
		for( long i=1; i<len-1 & evalRemain; i++ ) {
			//construct assignment
			boolean[] plan = createAssignment(Mlen, off, i);
			long pskip = 0; //skip after costing
			
			//skip plans with structural pruning
			if( STRUCTURAL_PRUNING && (rgraph!=null) && rgraph.isCutSet(plan) ) {
				//compute skip (which also acts as boundary for subproblems)
				pskip = rgraph.getNumSkipPlans(plan);
				if( LOG.isTraceEnabled() )
					LOG.trace("Enum: Structural pruning for cut set: "+rgraph.getCutSet(plan));
				
				//start increment rgraph get subproblems
				SubProblem[] prob = rgraph.getSubproblems(plan);
				
				//solve subproblems independently and combine into best plan
				for( int j=0; j<prob.length; j++ ) {
					if( LOG.isTraceEnabled() )
						LOG.trace("Enum: Subproblem "+(j+1)+"/"+prob.length+": "+prob[j]);
					boolean[] bestTmp = enumPlans(memo, part, 
						costs, null, prob[j].freeMat, prob[j].offset);
					LibSpoofPrimitives.vectWrite(bestTmp, plan, prob[j].freePos);
				}
				
				//note: the overall plan costs are evaluated in full, which reused
				//the default code path; hence we postpone the skip after costing
			}
			//skip plans with branch and bound pruning (cost)
			else if( COST_PRUNING ) {
				double lbC = getLowerBoundCosts(part, matPoints, memo, costs, plan);
				if( lbC >= bestC ) {
					long skip = getNumSkipPlans(plan);
					if( LOG.isTraceEnabled() )
						LOG.trace("Enum: Skip "+skip+" plans (by cost).");
					i += skip - 1;
					continue;
				}
			}
			
			//cost assignment on hops. Stop early if exceeds bestC.
			double pCBound = COST_PRUNING ? bestC : Double.MAX_VALUE;
			double C = getPlanCost(memo, part, matPoints, plan, costs._computeCosts, pCBound);
			if (LOG.isTraceEnabled())
				LOG.trace("Enum: " + Arrays.toString(plan) + " -> " + C);
			numEvalPartPlans += (C==Double.POSITIVE_INFINITY) ? 1 : 0;
			numEvalPlans++;
			
			//cost comparisons
			if( bestPlan == null || C < bestC ) {
				bestC = C;
				bestPlan = plan;
				if( LOG.isTraceEnabled() )
					LOG.trace("Enum: Found new best plan.");
			}
			
			//post skipping
			i += pskip;
			if( pskip !=0 && LOG.isTraceEnabled() )
				LOG.trace("Enum: Skip "+pskip+" plans (by structure).");
		}
		
		if( DMLScript.STATISTICS ) {
			Statistics.incrementCodegenEnumAllP((rgraph!=null||!STRUCTURAL_PRUNING)?len:0);
			Statistics.incrementCodegenEnumEval(numEvalPlans);
			Statistics.incrementCodegenEnumEvalP(numEvalPartPlans);
		}
		if( LOG.isTraceEnabled() )
			LOG.trace("Enum: Optimal plan: "+Arrays.toString(bestPlan));
		
		//keep large plans 
		if( probePlanCache(matPoints) )
			putPlan(pKey, bestPlan);
		
		//copy best plan w/o fixed offset plan
		return (bestPlan==null) ? new boolean[Mlen] :
			Arrays.copyOfRange(bestPlan, off, bestPlan.length);
	}
	
	private static boolean[] createAssignment(int len, int off, long pos) {
		boolean[] ret = new boolean[off+len];
		Arrays.fill(ret, 0, off, true);
		long tmp = pos;
		for( int i=0; i<len; i++ ) {
			long mask = UtilFunctions.pow(2, len-i-1);
			ret[off+i] = tmp >= mask;
			tmp %= mask;
		}
		return ret;	
	}
	
	private static long getNumSkipPlans(boolean[] plan) {
		int pos = ArrayUtils.lastIndexOf(plan, true);
		return UtilFunctions.pow(2, plan.length-pos-1);
	}
	
	private static double getLowerBoundCosts(PlanPartition part, InterestingPoint[] M, CPlanMemoTable memo, StaticCosts costs, boolean[] plan) {
		//compute the lower bound from static and plan-dependent costs
		double lb = Math.max(costs._read, costs._compute) + costs._write
			+ getMaterializationCost(part, M, memo, plan);
		
		//if the partition contains outer templates, we need to correct the lower bound
		if( part.hasOuter() )
			lb *= costs._minSparsity;
		
		return lb;
	}
	
	private static double getMaterializationCost(PlanPartition part, InterestingPoint[] M, CPlanMemoTable memo, boolean[] plan) {
		double costs = 0;
		//currently active materialization points
		HashSet<Long> matTargets = new HashSet<>();
		for( int i=0; i<plan.length; i++ ) {
			long hopID = M[i].getToHopID();
			if( plan[i] && !matTargets.contains(hopID) ) {
				matTargets.add(hopID);
				Hop hop = memo.getHopRefs().get(hopID);
				long size = getSize(hop);
				costs += size * 8 / WRITE_BANDWIDTH_MEM + 
						size * 8 / READ_BANDWIDTH_MEM;
			}
		}
		//points with non-partition consumers
		for( Long hopID : part.getExtConsumed() )
			if( !matTargets.contains(hopID) ) {
				matTargets.add(hopID);
				Hop hop = memo.getHopRefs().get(hopID);
				costs += getSize(hop) * 8 / WRITE_BANDWIDTH_MEM;
			}
		
		return costs;
	}
	
	private static double getReadCost(PlanPartition part, CPlanMemoTable memo) {
		double costs = 0;
		//get partition input reads (at least read once)
		for( Long hopID : part.getInputs() ) {
			Hop hop = memo.getHopRefs().get(hopID);
			costs += getSafeMemEst(hop) / READ_BANDWIDTH_MEM;
		}
		return costs;
	}
	
	private static double getWriteCost(Collection<Long> R, CPlanMemoTable memo) {
		double costs = 0;
		for( Long hopID : R ) {
			Hop hop = memo.getHopRefs().get(hopID);
			costs += getSize(hop) * 8 / WRITE_BANDWIDTH_MEM;
		}
		return costs;
	}
	
	private static double sumComputeCost(HashMap<Long, Double> computeCosts) {
		return computeCosts.values().stream()
			.mapToDouble(d -> d/COMPUTE_BANDWIDTH).sum();
	}
	
	private static double minOuterSparsity(PlanPartition part, CPlanMemoTable memo) {
		return !part.hasOuter() ? 1.0 : part.getPartition().stream()
			.map(k -> HopRewriteUtils.getLargestInput(memo.getHopRefs().get(k)))
			.mapToDouble(h -> h.dimsKnown(true) ? h.getSparsity() : SPARSE_SAFE_SPARSITY_EST)
			.min().orElse(SPARSE_SAFE_SPARSITY_EST);
	}
	
	private static double sumTmpInputOutputSize(CPlanMemoTable memo, CostVector vect) {
		//size of intermediate inputs and outputs, i.e., output and inputs other than treads
		return vect.outSize + vect.inSizes.entrySet().stream()
			.filter(e -> !HopRewriteUtils.isData(memo.getHopRefs().get(e.getKey()), DataOpTypes.TRANSIENTREAD))
			.mapToDouble(e -> e.getValue()).sum();
	}
	
	private static double sumInputMemoryEstimates(CPlanMemoTable memo, CostVector vect) {
		return vect.inSizes.keySet().stream()
			.mapToDouble(e -> getSafeMemEst(memo.getHopRefs().get(e))).sum();
	}
	
	private static double getSafeMemEst(Hop hop) {
		return !hop.dimsKnown() ? getSize(hop) * 8
			: hop.getOutputMemEstimate();
	}
	
	private static long getSize(Hop hop) {
		return Math.max(hop.getDim1(),1) 
			* Math.max(hop.getDim2(),1);
	}
	
	//within-partition multi-agg templates
	private static void createAndAddMultiAggPlans(CPlanMemoTable memo, HashSet<Long> partition, HashSet<Long> R)
	{
		//create index of plans that reference full aggregates to avoid circular dependencies
		HashSet<Long> refHops = new HashSet<>();
		for( Entry<Long, List<MemoTableEntry>> e : memo.getPlans().entrySet() )
			if( !e.getValue().isEmpty() ) {
				Hop hop = memo.getHopRefs().get(e.getKey());
				for( Hop c : hop.getInput() )
					refHops.add(c.getHopID());
			}
		
		//find all full aggregations (the fact that they are in the same partition guarantees 
		//that they also have common subexpressions, also full aggregations are by def root nodes)
		ArrayList<Long> fullAggs = new ArrayList<>();
		for( Long hopID : R ) {
			Hop root = memo.getHopRefs().get(hopID);
			if( !refHops.contains(hopID) && isMultiAggregateRoot(root) )
				fullAggs.add(hopID);
		}
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Found within-partition ua(RC) aggregations: " +
				Arrays.toString(fullAggs.toArray(new Long[0])));
		}
		
		//construct and add multiagg template plans (w/ max 3 aggregations)
		for( int i=0; i<fullAggs.size(); i+=3 ) {
			int ito = Math.min(i+3, fullAggs.size());
			if( ito-i >= 2 ) {
				MemoTableEntry me = new MemoTableEntry(TemplateType.MAGG,
					fullAggs.get(i), fullAggs.get(i+1), ((ito-i)==3)?fullAggs.get(i+2):-1, ito-i);
				if( isValidMultiAggregate(memo, me) ) {
					for( int j=i; j<ito; j++ ) {
						memo.add(memo.getHopRefs().get(fullAggs.get(j)), me);
						if( LOG.isTraceEnabled() )
							LOG.trace("Added multiagg plan: "+fullAggs.get(j)+" "+me);
					}
				}
				else if( LOG.isTraceEnabled() ) {
					LOG.trace("Removed invalid multiagg plan: "+me);
				}
			}
		}
	}
	
	//across-partition multi-agg templates with shared reads
	private void createAndAddMultiAggPlans(CPlanMemoTable memo, ArrayList<Hop> roots)
	{
		//collect full aggregations as initial set of candidates
		HashSet<Long> fullAggs = new HashSet<>();
		Hop.resetVisitStatus(roots);
		for( Hop hop : roots )
			rCollectFullAggregates(hop, fullAggs);
		Hop.resetVisitStatus(roots);

		//remove operators with assigned multi-agg plans
		fullAggs.removeIf(p -> memo.contains(p, TemplateType.MAGG));
	
		//check applicability for further analysis
		if( fullAggs.size() <= 1 )
			return;
	
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Found across-partition ua(RC) aggregations: " +
				Arrays.toString(fullAggs.toArray(new Long[0])));
		}
		
		//collect information for all candidates 
		//(subsumed aggregations, and inputs to fused operators) 
		List<AggregateInfo> aggInfos = new ArrayList<>();
		for( Long hopID : fullAggs ) {
			Hop aggHop = memo.getHopRefs().get(hopID);
			AggregateInfo tmp = new AggregateInfo(aggHop);
			for( int i=0; i<aggHop.getInput().size(); i++ ) {
				Hop c = HopRewriteUtils.isMatrixMultiply(aggHop) && i==0 ? 
					aggHop.getInput().get(0).getInput().get(0) : aggHop.getInput().get(i);
				rExtractAggregateInfo(memo, c, tmp, TemplateType.CELL);
			}
			if( tmp._fusedInputs.isEmpty() ) {
				if( HopRewriteUtils.isMatrixMultiply(aggHop) ) {
					tmp.addFusedInput(aggHop.getInput().get(0).getInput().get(0).getHopID());
					tmp.addFusedInput(aggHop.getInput().get(1).getHopID());
				}
				else	
					tmp.addFusedInput(aggHop.getInput().get(0).getHopID());
			}
			aggInfos.add(tmp);	
		}
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Extracted across-partition ua(RC) aggregation info: ");
			for( AggregateInfo info : aggInfos )
				LOG.trace(info);
		}
		
		//sort aggregations by num dependencies to simplify merging
		//clusters of aggregations with parallel dependencies
		aggInfos = aggInfos.stream()
			.sorted(Comparator.comparing(a -> a._inputAggs.size()))
			.collect(Collectors.toList());
		
		//greedy grouping of multi-agg candidates
		boolean converged = false;
		while( !converged ) {
			AggregateInfo merged = null;
			for( int i=0; i<aggInfos.size(); i++ ) {
				AggregateInfo current = aggInfos.get(i);
				for( int j=i+1; j<aggInfos.size(); j++ ) {
					AggregateInfo that = aggInfos.get(j);
					if( current.isMergable(that) ) {
						merged = current.merge(that);
						aggInfos.remove(j); j--;
					}
				}
			}
			converged = (merged == null);
		}
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Merged across-partition ua(RC) aggregation info: ");
			for( AggregateInfo info : aggInfos )
				LOG.trace(info);
		}
		
		//construct and add multiagg template plans (w/ max 3 aggregations)
		for( AggregateInfo info : aggInfos ) {
			if( info._aggregates.size()<=1 )
				continue;
			Long[] aggs = info._aggregates.keySet().toArray(new Long[0]);
			MemoTableEntry me = new MemoTableEntry(TemplateType.MAGG,
				aggs[0], aggs[1], (aggs.length>2)?aggs[2]:-1, aggs.length);
			for( int i=0; i<aggs.length; i++ ) {
				memo.add(memo.getHopRefs().get(aggs[i]), me);
				addBestPlan(aggs[i], me);
				if( LOG.isTraceEnabled() )
					LOG.trace("Added multiagg* plan: "+aggs[i]+" "+me);
				
			}
		}
	}
	
	private static boolean isMultiAggregateRoot(Hop root) {
		return (HopRewriteUtils.isAggUnaryOp(root, AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX) 
				&& ((AggUnaryOp)root).getDirection()==Direction.RowCol)
			|| (root instanceof AggBinaryOp && root.getDim1()==1 && root.getDim2()==1
				&& HopRewriteUtils.isTransposeOperation(root.getInput().get(0)));
	}
	
	private static boolean isValidMultiAggregate(CPlanMemoTable memo, MemoTableEntry me) {
		//ensure input consistent sizes (otherwise potential for incorrect results)
		boolean ret = true;
		Hop refSize = memo.getHopRefs().get(me.input1).getInput().get(0);
		for( int i=1; ret && i<3; i++ ) {
			if( me.isPlanRef(i) )
				ret &= HopRewriteUtils.isEqualSize(refSize, 
					memo.getHopRefs().get(me.input(i)).getInput().get(0));
		}
		
		//ensure that aggregates are independent of each other, i.e.,
		//they to not have potentially transitive parent child references
		for( int i=0; ret && i<3; i++ ) 
			if( me.isPlanRef(i) ) {
				HashSet<Long> probe = new HashSet<>();
				for( int j=0; j<3; j++ )
					if( i != j )
						probe.add(me.input(j));
				ret &= rCheckMultiAggregate(memo.getHopRefs().get(me.input(i)), probe);
			}
		return ret;
	}
	
	private static boolean rCheckMultiAggregate(Hop current, HashSet<Long> probe) {
		boolean ret = true;
		for( Hop c : current.getInput() )
			ret &= rCheckMultiAggregate(c, probe);
		ret &= !probe.contains(current.getHopID());
		return ret;
	}
	
	private static void rCollectFullAggregates(Hop current, HashSet<Long> aggs) {
		if( current.isVisited() )
			return;
		
		//collect all applicable full aggregations per read
		if( isMultiAggregateRoot(current) )
			aggs.add(current.getHopID());
		
		//recursively process children
		for( Hop c : current.getInput() )
			rCollectFullAggregates(c, aggs);
		
		current.setVisited();
	}
	
	private static void rExtractAggregateInfo(CPlanMemoTable memo, Hop current, AggregateInfo aggInfo, TemplateType type) {
		//collect input aggregates (dependents)
		if( isMultiAggregateRoot(current) )
			aggInfo.addInputAggregate(current.getHopID());
		
		//recursively process children
		MemoTableEntry me = (type!=null) ? memo.getBest(current.getHopID()) : null;
		for( int i=0; i<current.getInput().size(); i++ ) {
			Hop c = current.getInput().get(i);
			if( me != null && me.isPlanRef(i) )
				rExtractAggregateInfo(memo, c, aggInfo, type);
			else {
				if( type != null && c.getDataType().isMatrix()  ) //add fused input
					aggInfo.addFusedInput(c.getHopID());
				rExtractAggregateInfo(memo, c, aggInfo, null);
			}
		}
	}
	
	private static HashSet<Long> collectIrreplaceableRowOps(CPlanMemoTable memo, PlanPartition part) {
		//get row entries that are (a) reachable from rowwise ops (top down) other than
		//operator root nodes, or dependent upon row-wise ops (bottom up)
		HashSet<Long> blacklist = new HashSet<>();
		HashSet<Pair<Long, Integer>> visited = new HashSet<>();
		for( Long hopID : part.getRoots() ) {
			rCollectDependentRowOps(memo.getHopRefs().get(hopID),
				memo, part, blacklist, visited, null, false);
		}
		return blacklist;
	}
	
	private static void rCollectDependentRowOps(Hop hop, CPlanMemoTable memo, PlanPartition part,
		HashSet<Long> blacklist, HashSet<Pair<Long, Integer>> visited, TemplateType type, boolean foundRowOp) 
	{
		//avoid redundant evaluation of processed and non-partition nodes
		Pair<Long, Integer> key = Pair.of(hop.getHopID(),
			(foundRowOp?Short.MAX_VALUE:0) + ((type!=null)?type.ordinal()+1:0));
		if( visited.contains(key) || !part.getPartition().contains(hop.getHopID()) ) {
			return;
		}
		
		//process node itself (top-down)
		MemoTableEntry me = (type == null) ? memo.getBest(hop.getHopID()) :
			memo.getBest(hop.getHopID(), type);
		boolean inRow = (me != null && me.type == TemplateType.ROW && type == TemplateType.ROW);
		boolean diffPlans = part.getMatPointsExt().length > 0 //guard against plan differences
			&& memo.contains(hop.getHopID(), TemplateType.ROW)
			&& !memo.hasOnlyExactMatches(hop.getHopID(), TemplateType.ROW, TemplateType.CELL);
		if( inRow && foundRowOp )
			blacklist.add(hop.getHopID());
		if( isRowAggOp(hop, inRow) || diffPlans ) { 
			blacklist.add(hop.getHopID());
			foundRowOp = true;
		}
		
		//process children recursively
		for( int i=0; i<hop.getInput().size(); i++ ) {
			boolean lfoundRowOp = foundRowOp && me != null 
				&& (me.isPlanRef(i) || isImplicitlyFused(hop, i, me.type));
			rCollectDependentRowOps(hop.getInput().get(i), memo,
				part, blacklist, visited, me!=null?me.type:null, lfoundRowOp);
		}
		
		//process node itself (bottom-up)
		if( !blacklist.contains(hop.getHopID()) ) {
			for( int i=0; i<hop.getInput().size(); i++ )
				if( me != null && me.type == TemplateType.ROW
					&& (me.isPlanRef(i) || isImplicitlyFused(hop, i, me.type))
					&& blacklist.contains(hop.getInput().get(i).getHopID()) ) {
					blacklist.add(hop.getHopID());
				}
		}
		
		visited.add(key);
	}
	
	private static boolean isRowAggOp(Hop hop, boolean inRow) {
		return HopRewriteUtils.isBinary(hop, OpOp2.CBIND)
			|| HopRewriteUtils.isNary(hop, OpOpN.CBIND)
			|| (hop instanceof AggBinaryOp && (inRow || !hop.dimsKnown()
				|| (hop.getDim1()!=1 && hop.getDim2()!=1)))
			|| (HopRewriteUtils.isTransposeOperation(hop)
				&& (hop.getDim1()!=1 && hop.getDim2()!=1)
				&& !HopRewriteUtils.isDataGenOp(hop.getInput().get(0),DataGenMethod.SEQ))
			|| (hop instanceof AggUnaryOp && inRow);
	}
	
	private static boolean isValidRow2CellOp(Hop hop) {
		return !(HopRewriteUtils.isBinary(hop, OpOp2.CBIND)
			|| (hop instanceof AggBinaryOp && hop.getDim1()!=1 && hop.getDim2()!=1));
	}
	
	private static void pruneInvalidAndSpecialCasePlans(CPlanMemoTable memo, PlanPartition part) 
	{	
		//prune invalid row entries w/ violated blocksize constraint
		if( OptimizerUtils.isSparkExecutionMode() ) {
			for( Long hopID : part.getPartition() ) {
				if( !memo.contains(hopID, TemplateType.ROW) )
					continue;
				Hop hop = memo.getHopRefs().get(hopID);
				boolean isSpark = DMLScript.getGlobalExecMode() == ExecMode.SPARK
					|| OptimizerUtils.getTotalMemEstimate(hop.getInput().toArray(new Hop[0]), hop, true)
						> OptimizerUtils.getLocalMemBudget();
				boolean validNcol = hop.getDataType().isScalar() || (HopRewriteUtils.isTransposeOperation(hop) ? 
					hop.getDim1() <= hop.getBlocksize() : hop.getDim2() <= hop.getBlocksize());
				for( Hop in : hop.getInput() )
					validNcol &= in.getDataType().isScalar()
						|| (in.getDim2() <= in.getBlocksize())
						|| (hop instanceof AggBinaryOp && in.getDim1() <= in.getBlocksize()
							&& HopRewriteUtils.isTransposeOperation(in));
				if( isSpark && !validNcol ) {
					List<MemoTableEntry> blacklist = memo.get(hopID, TemplateType.ROW);
					memo.remove(memo.getHopRefs().get(hopID), TemplateType.ROW);
					memo.removeAllRefTo(hopID, TemplateType.ROW);
					if( LOG.isTraceEnabled() ) {
						LOG.trace("Removed row memo table entries w/ violated blocksize constraint ("+hopID+"): "
							+ Arrays.toString(blacklist.toArray(new MemoTableEntry[0])));
					}
				}
			}
		}
		
		//prune row aggregates with pure cellwise operations
		//(we determine a blacklist of all operators in a partition that either
		//depend upon row aggregates or on which row aggregates depend)
		HashSet<Long> blacklist = collectIrreplaceableRowOps(memo, part);
		for( Long hopID : part.getPartition() ) {
			if( blacklist.contains(hopID) ) continue;
			MemoTableEntry me = memo.getBest(hopID, TemplateType.ROW);
			if( me != null && me.type == TemplateType.ROW
				&& memo.hasOnlyExactMatches(hopID, TemplateType.ROW, TemplateType.CELL) ) {
				List<MemoTableEntry> rmList = memo.get(hopID, TemplateType.ROW); 
				memo.remove(memo.getHopRefs().get(hopID), new HashSet<>(rmList));
				if( LOG.isTraceEnabled() ) {
					LOG.trace("Removed row memo table entries w/o aggregation: "
						+ Arrays.toString(rmList.toArray(new MemoTableEntry[0])));
				}
			}
		}
		
		//prune suboptimal outer product plans that are dominated by outer product plans w/ same number of 
		//references but better fusion properties (e.g., for the patterns Y=X*(U%*%t(V)) and sum(Y*(U2%*%t(V2))), 
		//we'd prune sum(X*(U%*%t(V))*Z), Z=U2%*%t(V2) because this would unnecessarily destroy a fusion pattern.
		for( Long hopID : part.getPartition() ) {
			if( memo.countEntries(hopID, TemplateType.OUTER) == 2 ) {
				List<MemoTableEntry> entries = memo.get(hopID, TemplateType.OUTER);
				MemoTableEntry me1 = entries.get(0);
				MemoTableEntry me2 = entries.get(1);
				MemoTableEntry rmEntry = TemplateOuterProduct.dropAlternativePlan(memo, me1, me2);
				if( rmEntry != null ) {
					memo.remove(memo.getHopRefs().get(hopID), Collections.singleton(rmEntry));
					memo.getPlansBlacklisted().remove(rmEntry.input(rmEntry.getPlanRefIndex()));
					if( LOG.isTraceEnabled() )
						LOG.trace("Removed dominated outer product memo table entry: " + rmEntry);
				}
			}
		}
	}
	
	private static void rPruneSuboptimalPlans(CPlanMemoTable memo, Hop current, HashSet<Long> visited, 
		PlanPartition part, InterestingPoint[] matPoints, boolean[] plan) 
	{
		//memoization (not via hops because in middle of dag)
		if( visited.contains(current.getHopID()) )
			return;
		
		//remove memo table entries if necessary
		long hopID = current.getHopID();
		if( part.getPartition().contains(hopID) && memo.contains(hopID) ) {
			Iterator<MemoTableEntry> iter = memo.get(hopID).iterator();
			while( iter.hasNext() ) {
				MemoTableEntry me = iter.next();
				if( !hasNoRefToMatPoint(hopID, me, matPoints, plan) && me.type!=TemplateType.OUTER ) {
					iter.remove();
					if( LOG.isTraceEnabled() )
						LOG.trace("Removed memo table entry: "+me);
				}
			}
		}
		
		//process children recursively
		for( Hop c : current.getInput() )
			rPruneSuboptimalPlans(memo, c, visited, part, matPoints, plan);
		
		visited.add(current.getHopID());
	}
	
	private static void rPruneInvalidPlans(CPlanMemoTable memo, Hop current, HashSet<Long> visited, PlanPartition part, boolean[] plan) {
		//memoization (not via hops because in middle of dag)
		if( visited.contains(current.getHopID()) )
			return;
		
		//process children recursively
		for( Hop c : current.getInput() )
			rPruneInvalidPlans(memo, c, visited, part, plan);
		
		//find invalid row aggregate leaf nodes (see TemplateRow.open) w/o matrix inputs, 
		//i.e., plans that become invalid after the previous pruning step
		long hopID = current.getHopID();
		if( part.getPartition().contains(hopID) && memo.contains(hopID, TemplateType.ROW) ) {
			Iterator<MemoTableEntry> iter = memo.get(hopID, TemplateType.ROW).iterator();
			while( iter.hasNext() ) {
				MemoTableEntry me = iter.next();
				//convert leaf node with pure vector inputs
				boolean applyLeaf = (!me.hasPlanRef() 
					&& !TemplateUtils.hasMatrixInput(current));
				
				//convert inner node without row template input
				boolean applyInner = !applyLeaf && !ROW_TPL.open(current);
				for( int i=0; i<3 & applyInner; i++ )
					if( me.isPlanRef(i) )
						applyInner &= !memo.contains(me.input(i), TemplateType.ROW);
				
				if( applyLeaf || applyInner ) {
					String type = applyLeaf ? "leaf" : "inner";
					if( isValidRow2CellOp(current) ) {
						me.type = TemplateType.CELL;
						if( LOG.isTraceEnabled() )
							LOG.trace("Converted "+type+" memo table entry from row to cell: "+me);
					}
					else {
						if( LOG.isTraceEnabled() )
							LOG.trace("Removed "+type+" memo table entry row (unsupported cell): "+me);
						iter.remove();
					}
				}
			}
		}
		
		visited.add(current.getHopID());
	}
	
	/////////////////////////////////////////////////////////
	// Cost model fused operators w/ materialization points
	//////////
	
	private double getPlanCost(CPlanMemoTable memo, PlanPartition part, 
			InterestingPoint[] matPoints,boolean[] plan, HashMap<Long, Double> computeCosts,
			final double costBound)
	{
		//high level heuristic: every hop or fused operator has the following cost: 
		//WRITE + max(COMPUTE, READ), where WRITE costs are given by the output size, 
		//READ costs by the input sizes, and COMPUTE by operation specific FLOP
		//counts times number of cells of main input, disregarding sparsity for now.
		
		HashSet<VisitMarkCost> visited = new HashSet<>();
		double costs = 0;
		int rem = part.getRoots().size();
		for( Long hopID : part.getRoots() ) {
			costs += rGetPlanCosts(memo, memo.getHopRefs().get(hopID), 
				visited, part, matPoints, plan, computeCosts, null, null, costBound-costs);
			if( costs >= costBound && --rem > 0 ) //stop early
				return Double.POSITIVE_INFINITY;
		}
		return costs;
	}
	
	private double rGetPlanCosts(CPlanMemoTable memo, final Hop current, HashSet<VisitMarkCost> visited,
			PlanPartition part, InterestingPoint[] matPoints, boolean[] plan, HashMap<Long, Double> computeCosts,
			CostVector costsCurrent, TemplateType currentType, final double costBound)
	{
		final long currentHopId = current.getHopID();
		//memoization per hop id and cost vector to account for redundant
		//computation without double counting materialized results or compute
		//costs of complex operation DAGs within a single fused operator
		if( !visited.add(new VisitMarkCost(currentHopId, 
			(costsCurrent==null || currentType==TemplateType.MAGG)?-1:costsCurrent.ID)) )
			return 0; //already existing 
		
		//open template if necessary, including memoization
		//under awareness of current plan choice
		MemoTableEntry best = null;
		boolean opened = (currentType == null);
		if( memo.contains(currentHopId) ) {
			//note: this is the inner loop of plan enumeration and hence, we do not 
			//use streams, lambda expressions, etc to avoid unnecessary overhead
			if( currentType == null ) {
				for( MemoTableEntry me : memo.get(currentHopId) )
					best = me.isValid() 
						&& hasNoRefToMatPoint(currentHopId, me, matPoints, plan)
						&& BasicPlanComparator.icompare(me, best)<0 ? me : best;
				opened = true;
			}
			else {
				for( MemoTableEntry me : memo.get(currentHopId) )
					best = (me.type == currentType || me.type==TemplateType.CELL)
						&& hasNoRefToMatPoint(currentHopId, me, matPoints, plan)
						&& TypedPlanComparator.icompare(me, best, currentType)<0 ? me : best;
			}
		}
		
		//create new cost vector if opened, initialized with write costs
		CostVector costVect = !opened ? costsCurrent : new CostVector(getSize(current));
		double costs = 0;
		
		//add other roots for multi-agg template to account for shared costs
		if( opened && best != null && best.type == TemplateType.MAGG ) {
			//account costs to first multi-agg root 
			if( best.input1 == currentHopId )
				for( int i=1; i<3; i++ ) {
					if( !best.isPlanRef(i) ) continue;
					costs += rGetPlanCosts(memo, memo.getHopRefs().get(best.input(i)), visited, 
						part, matPoints, plan, computeCosts, costVect, TemplateType.MAGG, costBound-costs);
					if( costs >= costBound )
						return Double.POSITIVE_INFINITY;
				}
			//skip other multi-agg roots
			else
				return 0;
		}
		
		//add compute costs of current operator to costs vector
		if( computeCosts.containsKey(currentHopId) )
			costVect.computeCosts += computeCosts.get(currentHopId);
		
		//process children recursively
		for( int i=0; i< current.getInput().size(); i++ ) {
			Hop c = current.getInput().get(i);
			if( best!=null && best.isPlanRef(i) )
				costs += rGetPlanCosts(memo, c, visited, part, matPoints,
						plan, computeCosts, costVect, best.type, costBound-costs);
			else if( best!=null && isImplicitlyFused(current, i, best.type) )
				costVect.addInputSize(c.getInput().get(0).getHopID(), getSize(c));
			else { //include children and I/O costs
				if( part.getPartition().contains(c.getHopID()) )
					costs += rGetPlanCosts(memo, c, visited, part, matPoints,
						plan, computeCosts, null, null, costBound-costs);
				if( costVect != null && c.getDataType().isMatrix() )
					costVect.addInputSize(c.getHopID(), getSize(c));
			}
			if( costs >= costBound )
				return Double.POSITIVE_INFINITY;
		}
		
		//add costs for opened fused operator
		if( opened ) {
			double memInputs = sumInputMemoryEstimates(memo, costVect);
			double tmpCosts = costVect.outSize * 8 / WRITE_BANDWIDTH_MEM
				+ Math.max(memInputs / READ_BANDWIDTH_MEM,
				costVect.computeCosts/ COMPUTE_BANDWIDTH);
			//read correction for distributed computation
			if( memInputs > OptimizerUtils.getLocalMemBudget() )
				tmpCosts += costVect.getSideInputSize() * 8 / READ_BANDWIDTH_BROADCAST;
			//sparsity correction for outer-product template (and sparse-safe cell)
			Hop driver = memo.getHopRefs().get(costVect.getMaxInputSizeHopID());
			if( best != null && best.type == TemplateType.OUTER )
				tmpCosts *= driver.dimsKnown(true) ? driver.getSparsity() : SPARSE_SAFE_SPARSITY_EST;
			//write correction for known evictions in CP
			else if( memInputs <= OptimizerUtils.getLocalMemBudget()
				&& sumTmpInputOutputSize(memo, costVect)*8 > LazyWriteBuffer.getWriteBufferLimit() )
				tmpCosts += costVect.outSize * 8 / WRITE_BANDWIDTH_IO;
			costs += tmpCosts;
			if( LOG.isTraceEnabled() ) {
				String type = (best !=null) ? best.type.name() : "HOP";
				LOG.trace("Cost vector ("+type+" "+currentHopId+"): "+costVect+" -> "+tmpCosts);
			}
		}
		//add costs for non-partition read in the middle of fused operator
		else if( part.getExtConsumed().contains(current.getHopID()) ) {
			costs += rGetPlanCosts(memo, current, visited, part, matPoints, plan,
				computeCosts, null, null, costBound - costs);
		}
		
		//sanity check non-negative costs
		if( costs < 0 || Double.isNaN(costs) || Double.isInfinite(costs) )
			throw new RuntimeException("Wrong cost estimate: "+costs);
		
		return costs;
	}
	
	private static void getComputeCosts(Hop current, HashMap<Long, Double> computeCosts) 
	{
		//get costs for given hop
		double costs = 1;
		if( current instanceof UnaryOp ) {
			switch( ((UnaryOp)current).getOp() ) {
				case ABS:
				case ROUND:
				case CEIL:
				case FLOOR:
				case SIGN:    costs = 1; break; 
				case SPROP:
				case SQRT:    costs = 2; break;
				case EXP:     costs = 18; break;
				case SIGMOID: costs = 21; break;
				case LOG:
				case LOG_NZ:  costs = 32; break;
				case NCOL:
				case NROW:
				case PRINT:
				case ASSERT:
				case CAST_AS_BOOLEAN:
				case CAST_AS_DOUBLE:
				case CAST_AS_INT:
				case CAST_AS_MATRIX:
				case CAST_AS_SCALAR: costs = 1; break;
				case SIN:     costs = 18; break;
				case COS:     costs = 22; break;
				case TAN:     costs = 42; break;
				case ASIN:    costs = 93; break;
				case ACOS:    costs = 103; break;
				case ATAN:    costs = 40; break;
				case SINH:    costs = 93; break; // TODO:
				case COSH:    costs = 103; break;
				case TANH:    costs = 40; break;
				case CUMSUM:
				case CUMMIN:
				case CUMMAX:
				case CUMPROD: costs = 1; break;
				case CUMSUMPROD: costs = 2; break;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((UnaryOp)current).getOp());
			}
		}
		else if( current instanceof BinaryOp ) {
			switch( ((BinaryOp)current).getOp() ) {
				case MULT: 
				case PLUS:
				case MINUS:
				case MIN:
				case MAX: 
				case AND:
				case OR:
				case EQUAL:
				case NOTEQUAL:
				case LESS:
				case LESSEQUAL:
				case GREATER:
				case GREATEREQUAL: 
				case CBIND:
				case RBIND:   costs = 1; break;
				case INTDIV:  costs = 6; break;
				case MODULUS: costs = 8; break;
				case DIV:     costs = 22; break;
				case LOG:
				case LOG_NZ:  costs = 32; break;
				case POW:     costs = (HopRewriteUtils.isLiteralOfValue(
						current.getInput().get(1), 2) ? 1 : 16); break;
				case MINUS_NZ:
				case MINUS1_MULT: costs = 2; break;
				case MOMENT:
					int type = (int) (current.getInput().get(1) instanceof LiteralOp ? 
						HopRewriteUtils.getIntValueSafe((LiteralOp)current.getInput().get(1)) : 2);
					switch( type ) {
						case 0: costs = 1; break; //count
						case 1: costs = 8; break; //mean
						case 2: costs = 16; break; //cm2
						case 3: costs = 31; break; //cm3
						case 4: costs = 51; break; //cm4
						case 5: costs = 16; break; //variance
					}
					break;
				case COV: costs = 23; break;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((BinaryOp)current).getOp());
			}
		}
		else if( current instanceof TernaryOp ) {
			switch( ((TernaryOp)current).getOp() ) {
				case IFELSE:
				case PLUS_MULT: 
				case MINUS_MULT: costs = 2; break;
				case CTABLE:     costs = 3; break;
				case MOMENT:
					int type = (int) (current.getInput().get(1) instanceof LiteralOp ? 
						HopRewriteUtils.getIntValueSafe((LiteralOp)current.getInput().get(1)) : 2);
					switch( type ) {
						case 0: costs = 2; break; //count
						case 1: costs = 9; break; //mean
						case 2: costs = 17; break; //cm2
						case 3: costs = 32; break; //cm3
						case 4: costs = 52; break; //cm4
						case 5: costs = 17; break; //variance
					}
					break;
				case COV: costs = 23; break;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((TernaryOp)current).getOp());
			}
		}
		else if( current instanceof NaryOp ) {
			costs = HopRewriteUtils.isNary(current, OpOpN.MIN, OpOpN.MAX) ?
				current.getInput().size() : 1;
		}
		else if( current instanceof ParameterizedBuiltinOp ) {
			costs = 1;
		}
		else if( current instanceof IndexingOp ) {
			costs = 1;
		}
		else if( current instanceof ReorgOp ) {
			costs = 1;
		}
		else if( current instanceof DnnOp ) {
			switch( ((DnnOp)current).getOp() ) {
				case BIASADD:
				case BIASMULT:
					costs = 2;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((DnnOp)current).getOp());
			}
		}
		else if( current instanceof AggBinaryOp ) {
			//outer product template w/ matrix-matrix 
			//or row template w/ matrix-vector or matrix-matrix
			costs = 2 * current.getInput().get(0).getDim2();
			if( current.getInput().get(0).dimsKnown(true) )
				costs *= current.getInput().get(0).getSparsity();
		}
		else if( current instanceof AggUnaryOp) {
			switch(((AggUnaryOp)current).getOp()) {
				case SUM:    costs = 4; break; 
				case SUM_SQ: costs = 5; break;
				case MIN:
				case MAX:    costs = 1; break;
				default:
					LOG.warn("Cost model not "
						+ "implemented yet for: "+((AggUnaryOp)current).getOp());
			}
			switch(((AggUnaryOp)current).getDirection()) {
				case Col: costs *= Math.max(current.getInput().get(0).getDim1(),1); break;
				case Row: costs *= Math.max(current.getInput().get(0).getDim2(),1); break;
				case RowCol: costs *= getSize(current.getInput().get(0)); break;
			}
		}
		
		//scale by current output size in order to correctly reflect
		//a mix of row and cell operations in the same fused operator
		//(e.g., row template with fused column vector operations)
		costs *= getSize(current);
		
		computeCosts.put(current.getHopID(), costs);
	}
	
	private static boolean hasNoRefToMatPoint(long hopID, 
			MemoTableEntry me, InterestingPoint[] M, boolean[] plan) {
		return !InterestingPoint.isMatPoint(M, hopID, me, plan);
	}
	
	private static boolean isImplicitlyFused(Hop hop, int index, TemplateType type) {
		return type == TemplateType.ROW
			&& HopRewriteUtils.isMatrixMultiply(hop) && index==0 
			&& HopRewriteUtils.isTransposeOperation(hop.getInput().get(index)); 
	}
	
	private static boolean probePlanCache(InterestingPoint[] matPoints) {
		return matPoints.length >= PLAN_CACHE_NUM_POINTS;
	}
	
	private static boolean[] getPlan(PartitionSignature pKey) {
		boolean[] plan = null;
		synchronized( _planCache ) {
			plan = _planCache.get(pKey);
		}
		if( DMLScript.STATISTICS ) {
			if( plan != null )
				Statistics.incrementCodegenPlanCacheHits();
			Statistics.incrementCodegenPlanCacheTotal();
		}
		return plan;
	}
	
	private static void putPlan(PartitionSignature pKey, boolean[] plan) {
		synchronized( _planCache ) {
			//maintain size of plan cache (remove first)
			if( _planCache.size() >= PLAN_CACHE_SIZE ) {
				Iterator<Entry<PartitionSignature, boolean[]>> iter =
					_planCache.entrySet().iterator();
				iter.next();
				iter.remove();
			}
			
			//add last entry 
			_planCache.put(pKey, plan);
		}
	}
	
	private class CostVector {
		public final long ID;
		public final double outSize; 
		public double computeCosts = 0;
		public final HashMap<Long, Double> inSizes = new HashMap<>();
		
		public CostVector(double outputSize) {
			ID = COST_ID.getNextID();
			outSize = outputSize;
		}
		public void addInputSize(long hopID, double inputSize) {
			//ensures that input sizes are not double counted
			inSizes.put(hopID, inputSize);
		}
		@SuppressWarnings("unused")
		public double getInputSize() {
			return inSizes.values().stream()
				.mapToDouble(d -> d.doubleValue()).sum();
		}
		public double getSideInputSize() {
			double max = getMaxInputSize();
			return inSizes.values().stream()
				.filter(d -> d < max)
				.mapToDouble(d -> d.doubleValue()).sum();
		}
		public double getMaxInputSize() {
			return inSizes.values().stream()
				.mapToDouble(d -> d.doubleValue()).max().orElse(0);
		}
		public long getMaxInputSizeHopID() {
			long id = -1; double max = 0;
			for( Entry<Long,Double> e : inSizes.entrySet() )
				if( max < e.getValue() ) {
					id = e.getKey();
					max = e.getValue();
				}
			return id;
		}
		@Override
		public String toString() {
			return "["+outSize+", "+computeCosts+", {"
				+Arrays.toString(inSizes.keySet().toArray(new Long[0]))+", "
				+Arrays.toString(inSizes.values().toArray(new Double[0]))+"}]";
		}
	}
	
	private static class StaticCosts {
		public final HashMap<Long, Double> _computeCosts;
		public final double _compute;
		public final double _read;
		public final double _write;
		public final double _minSparsity;
		public StaticCosts(HashMap<Long,Double> allComputeCosts, double computeCost, double readCost, double writeCost, double minSparsity) {
			_computeCosts = allComputeCosts;
			_compute = computeCost;
			_read = readCost;
			_write = writeCost;
			_minSparsity = minSparsity;
		}
		public double getMinCosts() {
			return Math.max(_read, _compute) + _write;
		}
	}
	
	private static class AggregateInfo {
		public final HashMap<Long,Hop> _aggregates;
		public final HashSet<Long> _inputAggs = new HashSet<>();
		public final HashSet<Long> _fusedInputs = new HashSet<>();
		public AggregateInfo(Hop aggregate) {
			_aggregates = new HashMap<>();
			_aggregates.put(aggregate.getHopID(), aggregate);
		}
		public void addInputAggregate(long hopID) {
			_inputAggs.add(hopID);
		}
		public void addFusedInput(long hopID) {
			_fusedInputs.add(hopID);
		}
		public boolean isMergable(AggregateInfo that) {
			//check independence
			boolean ret = _aggregates.size()<3 
				&& _aggregates.size()+that._aggregates.size()<=3;
			for( Long hopID : that._aggregates.keySet() )
				ret &= !_inputAggs.contains(hopID);
			for( Long hopID : _aggregates.keySet() )
				ret &= !that._inputAggs.contains(hopID);
			//check partial shared reads
			ret &= !CollectionUtils.intersection(
				_fusedInputs, that._fusedInputs).isEmpty();
			//check consistent sizes (result correctness)
			Hop in1 = _aggregates.values().iterator().next();
			Hop in2 = that._aggregates.values().iterator().next();
			return ret && HopRewriteUtils.isEqualSize(
				in1.getInput().get(HopRewriteUtils.isMatrixMultiply(in1)?1:0),
				in2.getInput().get(HopRewriteUtils.isMatrixMultiply(in2)?1:0));
		}
		public AggregateInfo merge(AggregateInfo that) {
			_aggregates.putAll(that._aggregates);
			_inputAggs.addAll(that._inputAggs);
			_fusedInputs.addAll(that._fusedInputs);
			return this;
		}
		@Override
		public String toString() {
			return "["+Arrays.toString(_aggregates.keySet().toArray(new Long[0]))+": "
				+"{"+Arrays.toString(_inputAggs.toArray(new Long[0]))+"}," 
				+"{"+Arrays.toString(_fusedInputs.toArray(new Long[0]))+"}]"; 
		}
	}
	
	private class PartitionSignature {
		private final int partNodes, inputNodes, rootNodes, matPoints;
		private final double cCompute, cRead, cWrite, cPlan0, cPlanN;
		
		public PartitionSignature(PlanPartition part, int M, StaticCosts costs, double cP0, double cPN) {
			partNodes = part.getPartition().size();
			inputNodes = part.getInputs().size();
			rootNodes = part.getRoots().size();
			matPoints = M;
			cCompute = costs._compute;
			cRead = costs._read;
			cWrite = costs._write;
			cPlan0 = cP0;
			cPlanN = cPN;
		}
		@Override
		public int hashCode() {
			return UtilFunctions.intHashCode(
				Arrays.hashCode(new int[]{partNodes, inputNodes, rootNodes, matPoints}),
				Arrays.hashCode(new double[]{cCompute, cRead, cWrite, cPlan0, cPlanN}));
		}
		@Override 
		public boolean equals(Object o) {
			if( !(o instanceof PartitionSignature) )
				return false;
			PartitionSignature that = (PartitionSignature) o;
			return partNodes == that.partNodes
				&& inputNodes == that.inputNodes
				&& rootNodes == that.rootNodes
				&& matPoints == that.matPoints
				&& cCompute == that.cCompute
				&& cRead == that.cRead
				&& cWrite == that.cWrite
				&& cPlan0 == that.cPlan0
				&& cPlanN == that.cPlanN;
		}
	}
}
