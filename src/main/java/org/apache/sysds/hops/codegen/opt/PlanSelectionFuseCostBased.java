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

package org.apache.sysds.hops.codegen.opt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable;
import org.apache.sysds.hops.codegen.template.TemplateOuterProduct;
import org.apache.sysds.hops.codegen.template.TemplateRow;
import org.apache.sysds.hops.codegen.template.TemplateUtils;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysds.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.util.CollectionUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.Statistics;

/**
 * This cost-based plan selection algorithm chooses fused operators
 * based on the DAG structure and resulting overall costs. This primarily
 * includes decisions on materialization points, but also heuristics for 
 * template types, and composed multi output templates. 
 * 
 */
public class PlanSelectionFuseCostBased extends PlanSelection
{	
	private static final Log LOG = LogFactory.getLog(PlanSelectionFuseCostBased.class.getName());
	
	//common bandwidth characteristics, with a conservative write bandwidth in order 
	//to cover result allocation, write into main memory, and potential evictions
	private static final double WRITE_BANDWIDTH = 2d*1024*1024*1024;  //2GB/s
	private static final double READ_BANDWIDTH = 32d*1024*1024*1024;  //32GB/s
	private static final double COMPUTE_BANDWIDTH = 2d*1024*1024*1024 //2GFLOPs/core
		* InfrastructureAnalyzer.getLocalParallelism();
	
	private static final IDSequence COST_ID = new IDSequence();
	private static final TemplateRow ROW_TPL = new TemplateRow();
	
	@Override
	public void selectPlans(CPlanMemoTable memo, ArrayList<Hop> roots) 
	{
		//step 1: analyze connected partitions (nodes, roots, mat points)
		Collection<PlanPartition> parts = PlanAnalyzer.analyzePlanPartitions(memo, roots, false);
		
		//step 2: optimize individual plan partitions
		int sumMatPoints = 0;
		for( PlanPartition part : parts ) {
			//create composite templates (within the partition)
			createAndAddMultiAggPlans(memo, part.getPartition(), part.getRoots());
			
			//plan enumeration and plan selection
			selectPlans(memo, part.getPartition(), part.getRoots(), part.getMatPoints());
			sumMatPoints += part.getMatPoints().size();
		}
		
		//step 3: add composite templates (across partitions)
		createAndAddMultiAggPlans(memo, roots);
	
		//take all distinct best plans
		for( Entry<Long, List<MemoTableEntry>> e : getBestPlans().entrySet() )
			memo.setDistinct(e.getKey(), e.getValue());
		
		//maintain statistics
		if( DMLScript.STATISTICS )
			Statistics.incrementCodegenEnumAll(UtilFunctions.pow(2, sumMatPoints));
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
	
	private void selectPlans(CPlanMemoTable memo, HashSet<Long> partition, HashSet<Long> R, ArrayList<Long> M) 
	{
		//prune row aggregates with pure cellwise operations
		for( Long hopID : R ) {
			MemoTableEntry me = memo.getBest(hopID, TemplateType.ROW);
			if( me.type == TemplateType.ROW && memo.contains(hopID, TemplateType.CELL)
				&& isRowTemplateWithoutAgg(memo, memo.getHopRefs().get(hopID), new HashSet<Long>())) {
				List<MemoTableEntry> excludeList = memo.get(hopID, TemplateType.ROW); 
				memo.remove(memo.getHopRefs().get(hopID), new HashSet<>(excludeList));
				if( LOG.isTraceEnabled() ) {
					LOG.trace("Removed row memo table entries w/o aggregation: "
						+ Arrays.toString(excludeList.toArray(new MemoTableEntry[0])));
				}
			}
		}
		
		//prune suboptimal outer product plans that are dominated by outer product plans w/ same number of 
		//references but better fusion properties (e.g., for the patterns Y=X*(U%*%t(V)) and sum(Y*(U2%*%t(V2))), 
		//we'd prune sum(X*(U%*%t(V))*Z), Z=U2%*%t(V2) because this would unnecessarily destroy a fusion pattern.
		for( Long hopID : partition ) {
			if( memo.countEntries(hopID, TemplateType.OUTER) == 2 ) {
				List<MemoTableEntry> entries = memo.get(hopID, TemplateType.OUTER);
				MemoTableEntry me1 = entries.get(0);
				MemoTableEntry me2 = entries.get(1);
				MemoTableEntry rmEntry = TemplateOuterProduct.dropAlternativePlan(memo, me1, me2);
				if( rmEntry != null ) {
					memo.remove(memo.getHopRefs().get(hopID), Collections.singleton(rmEntry));
					memo.getPlansExcludeListed().remove(rmEntry.input(rmEntry.getPlanRefIndex()));
					if( LOG.isTraceEnabled() )
						LOG.trace("Removed dominated outer product memo table entry: " + rmEntry);
				}
			}
		}
		
		//if no materialization points, use basic fuse-all w/ partition awareness
		if( M == null || M.isEmpty() ) {
			for( Long hopID : R )
				rSelectPlansFuseAll(memo, 
					memo.getHopRefs().get(hopID), null, partition);
		}
		else {
			//TODO branch and bound pruning, right now we use exhaustive enum for early experiments
			//via skip ahead in below enumeration algorithm
			
			//obtain hop compute costs per cell once
			HashMap<Long, Double> computeCosts = new HashMap<>();
			for( Long hopID : R )
				rGetComputeCosts(memo.getHopRefs().get(hopID), partition, computeCosts);
			
			//scan linearized search space, w/ skips for branch and bound pruning
			int len = (int)Math.pow(2, M.size());
			boolean[] bestPlan = null;
			double bestC = Double.MAX_VALUE;
			
			for( int i=0; i<len; i++ ) {
				//construct assignment
				boolean[] plan = createAssignment(M.size(), i);
				
				//cost assignment on hops
				double C = getPlanCost(memo, partition, R, M, plan, computeCosts);
				if( LOG.isTraceEnabled() )
					LOG.trace("Enum: "+Arrays.toString(plan)+" -> "+C);
				
				//cost comparisons
				if( bestPlan == null || C < bestC ) {
					bestC = C;
					bestPlan = plan;
					if( LOG.isTraceEnabled() )
						LOG.trace("Enum: Found new best plan.");
				}
			}
			
			if( DMLScript.STATISTICS ) {
				Statistics.incrementCodegenEnumAllP(len);
				Statistics.incrementCodegenEnumEval(len);
			}
			
			//prune memo table wrt best plan and select plans
			HashSet<Long> visited = new HashSet<>();
			for( Long hopID : R )
				rPruneSuboptimalPlans(memo, memo.getHopRefs().get(hopID), 
					visited, partition, M, bestPlan);
			HashSet<Long> visited2 = new HashSet<>();
			for( Long hopID : R )
				rPruneInvalidPlans(memo, memo.getHopRefs().get(hopID), 
					visited2, partition, M, bestPlan);
			
			for( Long hopID : R )
				rSelectPlansFuseAll(memo, 
					memo.getHopRefs().get(hopID), null, partition);
		}
	}
	
	private static boolean isRowTemplateWithoutAgg(CPlanMemoTable memo, Hop current, HashSet<Long> visited) {
		//consider all aggregations other than root operation
		MemoTableEntry me = memo.getBest(current.getHopID(), TemplateType.ROW);
		boolean ret = true;
		for(int i=0; i<3; i++)
			if( me.isPlanRef(i) )
				ret &= rIsRowTemplateWithoutAgg(memo, 
					current.getInput().get(i), visited);
		return ret;
	}
	
	private static boolean rIsRowTemplateWithoutAgg(CPlanMemoTable memo, Hop current, HashSet<Long> visited) {
		if( visited.contains(current.getHopID()) )
			return true;
		
		boolean ret = true;
		MemoTableEntry me = memo.getBest(current.getHopID(), TemplateType.ROW);
		for(int i=0; i<3; i++)
			if( me.isPlanRef(i) )
				ret &= rIsRowTemplateWithoutAgg(memo, current.getInput().get(i), visited);
		ret &= !(current instanceof AggUnaryOp || current instanceof AggBinaryOp);
		
		visited.add(current.getHopID());
		return ret;
	}
	
	private static void rPruneSuboptimalPlans(CPlanMemoTable memo, Hop current, HashSet<Long> visited, HashSet<Long> partition, ArrayList<Long> M, boolean[] plan) {
		//memoization (not via hops because in middle of dag)
		if( visited.contains(current.getHopID()) )
			return;
		
		//remove memo table entries if necessary
		long hopID = current.getHopID();
		if( partition.contains(hopID) && memo.contains(hopID) ) {
			Iterator<MemoTableEntry> iter = memo.get(hopID).iterator();
			while( iter.hasNext() ) {
				MemoTableEntry me = iter.next();
				if( !hasNoRefToMaterialization(me, M, plan) && me.type!=TemplateType.OUTER ){
					iter.remove();
					if( LOG.isTraceEnabled() )
						LOG.trace("Removed memo table entry: "+me);
				}
			}
		}
		
		//process children recursively
		for( Hop c : current.getInput() )
			rPruneSuboptimalPlans(memo, c, visited, partition, M, plan);
		
		visited.add(current.getHopID());		
	}
	
	private static void rPruneInvalidPlans(CPlanMemoTable memo, Hop current, HashSet<Long> visited, HashSet<Long> partition, ArrayList<Long> M, boolean[] plan) {
		//memoization (not via hops because in middle of dag)
		if( visited.contains(current.getHopID()) )
			return;
		
		//process children recursively
		for( Hop c : current.getInput() )
			rPruneInvalidPlans(memo, c, visited, partition, M, plan);
		
		//find invalid row aggregate leaf nodes (see TemplateRow.open) w/o matrix inputs, 
		//i.e., plans that become invalid after the previous pruning step
		long hopID = current.getHopID();
		if( partition.contains(hopID) && memo.contains(hopID, TemplateType.ROW) ) {
			for( MemoTableEntry me : memo.get(hopID) ) {
				if( me.type==TemplateType.ROW ) {
					//convert leaf node with pure vector inputs
					if( !me.hasPlanRef() && !TemplateUtils.hasMatrixInput(current) ) {
						me.type = TemplateType.CELL;
						if( LOG.isTraceEnabled() )
							LOG.trace("Converted leaf memo table entry from row to cell: "+me);
					}
					
					//convert inner node without row template input
					if( me.hasPlanRef() && !ROW_TPL.open(current) ) {
						boolean hasRowInput = false;
						for( int i=0; i<3; i++ )
							if( me.isPlanRef(i) )
								hasRowInput |= memo.contains(me.input(i), TemplateType.ROW);
						if( !hasRowInput ) {
							me.type = TemplateType.CELL;
							if( LOG.isTraceEnabled() )
								LOG.trace("Converted inner memo table entry from row to cell: "+me);	
						}
					}
					
				}
			}
		}
		
		visited.add(current.getHopID());
	}
	
	private static boolean[] createAssignment(int len, int pos) {
		boolean[] ret = new boolean[len]; 
		int tmp = pos;
		for( int i=0; i<len; i++ ) {
			ret[i] = (tmp < (int)Math.pow(2, len-i-1));
			tmp %= Math.pow(2, len-i-1);
		}
		return ret;	
	}
	
	/////////////////////////////////////////////////////////
	// Cost model fused operators w/ materialization points
	//////////
	
	private static double getPlanCost(CPlanMemoTable memo, HashSet<Long> partition, HashSet<Long> R, 
			ArrayList<Long> M, boolean[] plan, HashMap<Long, Double> computeCosts) 
	{
		//high level heuristic: every hop or fused operator has the following cost: 
		//WRITE + max(COMPUTE, READ), where WRITE costs are given by the output size, 
		//READ costs by the input sizes, and COMPUTE by operation specific FLOP
		//counts times number of cells of main input, disregarding sparsity for now.
		
		HashSet<Pair<Long,Long>> visited = new HashSet<>();
		double costs = 0;
		for( Long hopID : R )
			costs += rGetPlanCosts(memo, memo.getHopRefs().get(hopID), 
					visited, partition, M, plan, computeCosts, null, null);		
		return costs;
	}
	
	private static double rGetPlanCosts(CPlanMemoTable memo, Hop current, HashSet<Pair<Long,Long>> visited, HashSet<Long> partition, 
			ArrayList<Long> M, boolean[] plan, HashMap<Long, Double> computeCosts, CostVector costsCurrent, TemplateType currentType) 
	{
		//memoization per hop id and cost vector to account for redundant
		//computation without double counting materialized results or compute
		//costs of complex operation DAGs within a single fused operator
		Pair<Long,Long> tag = Pair.of(current.getHopID(), 
			(costsCurrent==null)?0:costsCurrent.ID);
		if( visited.contains(tag) )
			return 0; 
		visited.add(tag);	
		
		//open template if necessary, including memoization
		//under awareness of current plan choice
		MemoTableEntry best = null;
		boolean opened = false;
		if( memo.contains(current.getHopID()) ) {
			if( currentType == null ) {
				best = memo.get(current.getHopID()).stream()
					.filter(p -> p.isValid())
					.filter(p -> hasNoRefToMaterialization(p, M, plan))
					.min(new BasicPlanComparator()).orElse(null);
				opened = true;
			}
			else {
				best = memo.get(current.getHopID()).stream()
					.filter(p -> p.type==currentType || p.type==TemplateType.CELL)
					.filter(p -> hasNoRefToMaterialization(p, M, plan))
					.min(Comparator.comparing(p -> 7-((p.type==currentType)?4:0)-p.countPlanRefs()))
					.orElse(null);
			}
		}
		
		//create new cost vector if opened, initialized with write costs
		CostVector costVect = !opened ? costsCurrent : 
			new CostVector(Math.max(current.getDim1(),1)*Math.max(current.getDim2(),1));
		
		//add compute costs of current operator to costs vector 
		if( partition.contains(current.getHopID()) )
			costVect.computeCosts += computeCosts.get(current.getHopID());
		
		//process children recursively
		double costs = 0;
		for( int i=0; i< current.getInput().size(); i++ ) {
			Hop c = current.getInput().get(i);
			if( best!=null && best.isPlanRef(i) )
				costs += rGetPlanCosts(memo, c, visited, partition, M, plan, computeCosts, costVect, best.type);
			else if( best!=null && isImplicitlyFused(current, i, best.type) )
				costVect.addInputSize(c.getInput().get(0).getHopID(), Math.max(c.getDim1(),1)*Math.max(c.getDim2(),1));
			else { //include children and I/O costs
				costs += rGetPlanCosts(memo, c, visited, partition, M, plan, computeCosts, null, null);
				if( costVect != null && c.getDataType().isMatrix() )
					costVect.addInputSize(c.getHopID(), Math.max(c.getDim1(),1)*Math.max(c.getDim2(),1));
			}				
		}	
		
		//add costs for opened fused operator
		if( partition.contains(current.getHopID()) ) {
			if( opened ) {
				if( LOG.isTraceEnabled() )
					LOG.trace("Cost vector for fused operator (hop "+current.getHopID()+"): "+costVect);
				costs += costVect.outSize * 8 / WRITE_BANDWIDTH; //time for output write
				costs += Math.max(
						costVect.computeCosts*costVect.getMaxInputSize()/ COMPUTE_BANDWIDTH, 
						costVect.getSumInputSizes() * 8 / READ_BANDWIDTH); 
			}
			//add costs for non-partition read in the middle of fused operator
			else if( hasNonPartitionConsumer(current, partition) ) {
				costs += rGetPlanCosts(memo, current, visited, partition, M, plan, computeCosts, null, null);
			}
		}
		
		//sanity check non-negative costs
		if( costs < 0 || Double.isNaN(costs) || Double.isInfinite(costs) )
			throw new RuntimeException("Wrong cost estimate: "+costs);
		
		return costs;
	}
	
	private static void rGetComputeCosts(Hop current, HashSet<Long> partition, HashMap<Long, Double> computeCosts) 
	{
		if( computeCosts.containsKey(current.getHopID()) )
			return;
		
		//recursively process children
		for( Hop c : current.getInput() )
			rGetComputeCosts(c, partition, computeCosts);
		
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
		else if( current instanceof ParameterizedBuiltinOp ) {
			costs = 1;
		}
		else if( current instanceof IndexingOp ) {
			costs = 1;
		}
		else if( current instanceof ReorgOp ) {
			costs = 1;
		}
		else if( current instanceof AggBinaryOp ) {
			costs = 2; //matrix vector
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
		}
		
		computeCosts.put(current.getHopID(), costs);
	}
	
	private static boolean hasNoRefToMaterialization(MemoTableEntry me, ArrayList<Long> M, boolean[] plan) {
		boolean ret = true;
		for( int i=0; ret && i<3; i++ )
			ret &= (!M.contains(me.input(i)) || !plan[M.indexOf(me.input(i))]);
		return ret;
	}
	
	private static boolean hasNonPartitionConsumer(Hop hop, HashSet<Long> partition) {
		boolean ret = false;
		for( Hop p : hop.getParent() )
			ret |= !partition.contains(p.getHopID());
		return ret;
	}
	
	private static boolean isImplicitlyFused(Hop hop, int index, TemplateType type) {
		return type == TemplateType.ROW
			&& HopRewriteUtils.isMatrixMultiply(hop) && index==0 
			&& HopRewriteUtils.isTransposeOperation(hop.getInput().get(index)); 
	}
	
	private static class CostVector {
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
		public double getSumInputSizes() {
			return inSizes.values().stream()
				.mapToDouble(d -> d.doubleValue()).sum();
		}
		public double getMaxInputSize() {
			return inSizes.values().stream()
				.mapToDouble(d -> d.doubleValue()).max().orElse(0);
		}
		@Override
		public String toString() {
			return "["+outSize+", "+computeCosts+", {"
				+Arrays.toString(inSizes.keySet().toArray(new Long[0]))+", "
				+Arrays.toString(inSizes.values().toArray(new Double[0]))+"}]";
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
			ret &= CollectionUtils.containsAny(_fusedInputs, that._fusedInputs);
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
}
