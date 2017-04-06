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

package org.apache.sysml.hops.codegen.template;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.ParameterizedBuiltinOp;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.TernaryOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;


/**
 * This cost-based plan selection algorithm chooses fused operators
 * based on the DAG structure and resulting overall costs. This includes
 * decisions on materialization points, template types, and composed
 * multi output templates. 
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
	
	@Override
	public void selectPlans(CPlanMemoTable memo, ArrayList<Hop> roots) 
	{
		//step 1: determine connected sub graphs of plans
		Collection<HashSet<Long>> parts = getConnectedSubGraphs(memo, roots);
		if( LOG.isTraceEnabled() )
			LOG.trace("Connected sub graphs: "+parts.size());
		
		for( HashSet<Long> partition : parts ) {
			//step 2: determine materialization points
			HashSet<Long> R = getPartitionRootNodes(memo, partition);
			if( LOG.isTraceEnabled() )
				LOG.trace("Partition root points: "+Arrays.toString(R.toArray(new Long[0])));
			ArrayList<Long> M = getMaterializationPoints(R, partition, memo);
			if( LOG.isTraceEnabled() )
				LOG.trace("Partition materialization points: "+Arrays.toString(M.toArray(new Long[0])));
			
			//step 3: create composite templates entries
			createAndAddMultiAggPlans(memo, partition, R);
			
			//step 4: plan enumeration and plan selection
			selectPlans(memo, partition, R, M);
		}
	
		//take all distinct best plans
		for( Entry<Long, List<MemoTableEntry>> e : getBestPlans().entrySet() )
			memo.setDistinct(e.getKey(), e.getValue());
	}
	
	private static Collection<HashSet<Long>> getConnectedSubGraphs(CPlanMemoTable memo, ArrayList<Hop> roots) 
	{
		//build inverted index for 'referenced by' relationship 
		HashMap<Long, HashSet<Long>> refBy = new HashMap<Long, HashSet<Long>>();
		for( Entry<Long, List<MemoTableEntry>> e : memo._plans.entrySet() )
			for( MemoTableEntry me : e.getValue() ) 
				for( int i=0; i<3; i++ )
					if( me.isPlanRef(i) ) {
						if( !refBy.containsKey(me.input(i)) )
							refBy.put(me.input(i), new HashSet<Long>());
						refBy.get(me.input(i)).add(e.getKey());
					}
		
		//create a single partition per root node, if reachable over refBy of 
		//other root node the resulting partition is empty and can be discarded
		ArrayList<HashSet<Long>> parts = new ArrayList<HashSet<Long>>();
		HashSet<Long> visited = new HashSet<Long>();
		for( Entry<Long, List<MemoTableEntry>> e : memo._plans.entrySet() )
			if( !refBy.containsKey(e.getKey()) ) { //root node
				HashSet<Long> part = rGetConnectedSubGraphs(e.getKey(), 
						memo, refBy, visited, new HashSet<Long>());
				if( !part.isEmpty() )
					parts.add(part);
			}
		
		return parts;
	}
	
	private static HashSet<Long> rGetConnectedSubGraphs(long hopID, CPlanMemoTable memo, 
			HashMap<Long, HashSet<Long>> refBy, HashSet<Long> visited, HashSet<Long> partition) 
	{
		if( visited.contains(hopID) )
			return partition;
		
		//process node itself w/ memoization
		if( memo.contains(hopID) ) {
			partition.add(hopID);
			visited.add(hopID);	
		}
		
		//recursively process parents
		if( refBy.containsKey(hopID) )
			for( Long ref : refBy.get(hopID) )
				rGetConnectedSubGraphs(ref, memo, refBy, visited, partition);
		
		//recursively process children
		if( memo.contains(hopID) ) {
			long[] refs = memo.getAllRefs(hopID);
			for( int i=0; i<3; i++ )
				if( refs[i] != -1 )
					rGetConnectedSubGraphs(refs[i], memo, refBy, visited, partition);
		}
		
		return partition;
	}
	
	private static HashSet<Long> getPartitionRootNodes(CPlanMemoTable memo, HashSet<Long> partition) 
	{
		//build inverted index of references entries 
		HashSet<Long> ix = new HashSet<Long>();
		for( Long hopID : partition )
			if( memo.contains(hopID) )
				for( MemoTableEntry me : memo.get(hopID) ) {
					ix.add(me.input1); 
					ix.add(me.input2); 
					ix.add(me.input3);
				}
		
		HashSet<Long> roots = new HashSet<Long>();
		for( Long hopID : partition )
			if( !ix.contains(hopID) )
				roots.add(hopID);
		return roots;
	}
	
	private static ArrayList<Long> getMaterializationPoints(HashSet<Long> roots, 
			HashSet<Long> partition, CPlanMemoTable memo) 
	{
		//collect materialization points bottom-up
		ArrayList<Long> ret = new ArrayList<Long>();
		HashSet<Long> visited = new HashSet<Long>();
		for( Long hopID : roots )
			rCollectMaterializationPoints(memo._hopRefs.get(hopID), 
					visited, partition, ret);
		
		//remove special-case materialization points
		Iterator<Long> iter = ret.iterator();
		while(iter.hasNext()) {
			Long hopID = iter.next();
			//remove root nodes w/ multiple consumers
			if( roots.contains(hopID) )
				iter.remove();
			//remove tsmm input if consumed in partition
			else if( HopRewriteUtils.isTsmmInput(memo._hopRefs.get(hopID)))
				iter.remove();
		}
		
		return ret;
	}
	
	private static void rCollectMaterializationPoints(Hop current, HashSet<Long> visited, 
			HashSet<Long> partition, ArrayList<Long> M) 
	{
		//memoization (not via hops because in middle of dag)
		if( visited.contains(current.getHopID()) )
			return;
		
		//process children recursively
		for( Hop c : current.getInput() )
			rCollectMaterializationPoints(c, visited, partition, M);
		
		//collect materialization point
		if( isMaterializationPointCandidate(current, partition) )
			M.add(current.getHopID());
		
		visited.add(current.getHopID());
	}
	
	private static boolean isMaterializationPointCandidate(Hop hop, HashSet<Long> partition) {
		return hop.getParent().size()>=2 
			&& partition.contains(hop.getHopID());
	}
	
	private static void createAndAddMultiAggPlans(CPlanMemoTable memo, HashSet<Long> partition, HashSet<Long> R)
	{
		//create index of plans that reference full aggregates to avoid circular dependencies
		HashSet<Long> refHops = new HashSet<Long>();
		for( Entry<Long, List<MemoTableEntry>> e : memo._plans.entrySet() )
			if( !e.getValue().isEmpty() ) {
				Hop hop = memo._hopRefs.get(e.getKey());
				for( Hop c : hop.getInput() )
					refHops.add(c.getHopID());
			}
		
		//find all full aggregations (the fact that they are in the same partition guarantees 
		//that they also have common subexpressions, also full aggregations are by def root nodes)
		ArrayList<Long> fullAggs = new ArrayList<Long>();
		for( Long hopID : R ) {
			Hop root = memo._hopRefs.get(hopID);
			if( !refHops.contains(hopID) && root instanceof AggUnaryOp 
				&& ((AggUnaryOp)root).getDirection()==Direction.RowCol)
				fullAggs.add(hopID);
		}
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Found ua(RC) aggregations: " +
				Arrays.toString(fullAggs.toArray(new Long[0])));
		}
		
		//construct and add multiagg template plans (w/ max 3 aggregations)
		for( int i=0; i<fullAggs.size(); i+=3 ) {
			int ito = Math.min(i+3, fullAggs.size());
			if( ito-i >= 2 ) {
				MemoTableEntry me = new MemoTableEntry(TemplateType.MultiAggTpl,
					fullAggs.get(i), fullAggs.get(i+1), ((ito-i)==3)?fullAggs.get(i+2):-1);
				if( isValidMultiAggregate(memo, me) ) {
					for( int j=i; j<ito; j++ ) {
						memo.add(memo._hopRefs.get(fullAggs.get(j)), me);
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
	
	private static boolean isValidMultiAggregate(CPlanMemoTable memo, MemoTableEntry me) {
		//ensure that aggregates are independent of each other, i.e.,
		//they to not have potentially transitive parent child references
		boolean ret = true;
		for( int i=0; i<3; i++ ) 
			if( me.isPlanRef(i) ) {
				HashSet<Long> probe = new HashSet<Long>();
				for( int j=0; j<3; j++ )
					if( i != j )
						probe.add(me.input(j));
				ret &= rCheckMultiAggregate(memo._hopRefs.get(me.input(i)), probe);
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
	
	private void selectPlans(CPlanMemoTable memo, HashSet<Long> partition, HashSet<Long> R, ArrayList<Long> M) 
	{
		//if no materialization points, use basic fuse-all w/ partition awareness
		if( M == null || M.isEmpty() ) {
			for( Long hopID : R )
				rSelectPlansFuseAll(memo, 
					memo._hopRefs.get(hopID), null, partition);
		}
		else {
			//TODO branch and bound pruning, right now we use exhaustive enum for early experiments
			//via skip ahead in below enumeration algorithm
			
			//obtain hop compute costs per cell once
			HashMap<Long, Double> computeCosts = new HashMap<Long, Double>();
			for( Long hopID : R )
				rGetComputeCosts(memo._hopRefs.get(hopID), partition, computeCosts);
			
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
			
			//prune memo table wrt best plan and select plans
			HashSet<Long> visited = new HashSet<Long>();
			for( Long hopID : R )
				rPruneSuboptimalPlans(memo, memo._hopRefs.get(hopID), 
					visited, partition, M, bestPlan);
			for( Long hopID : R )
				rSelectPlansFuseAll(memo, 
					memo._hopRefs.get(hopID), null, partition);
		}
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
				if( !hasNoRefToMaterialization(me, M, plan) && me.type!=TemplateType.OuterProdTpl ){
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
	
	private void rSelectPlansFuseAll(CPlanMemoTable memo, Hop current, TemplateType currentType, HashSet<Long> partition) 
	{	
		if( isVisited(current.getHopID(), currentType) 
			|| !partition.contains(current.getHopID()) )
			return;
		
		//step 1: prune subsumed plans of same type
		if( memo.contains(current.getHopID()) ) {
			HashSet<MemoTableEntry> rmSet = new HashSet<MemoTableEntry>();
			List<MemoTableEntry> hopP = memo.get(current.getHopID());
			for( MemoTableEntry e1 : hopP )
				for( MemoTableEntry e2 : hopP )
					if( e1 != e2 && e1.subsumes(e2) )
						rmSet.add(e2);
			memo.remove(current, rmSet);
		}
		
		//step 2: select plan for current path
		MemoTableEntry best = null;
		if( memo.contains(current.getHopID()) ) {
			if( currentType == null ) {
				best = memo.get(current.getHopID()).stream()
					.filter(p -> isValid(p, current))
					.min(new BasicPlanComparator()).orElse(null);
			}
			else {
				best = memo.get(current.getHopID()).stream()
					.filter(p -> p.type==currentType || p.type==TemplateType.CellTpl)
					.min(Comparator.comparing(p -> 7-((p.type==currentType)?4:0)-p.countPlanRefs()))
					.orElse(null);
			}
			addBestPlan(current.getHopID(), best);
		}
		
		//step 3: recursively process children
		for( int i=0; i< current.getInput().size(); i++ ) {
			TemplateType pref = (best!=null && best.isPlanRef(i))? best.type : null;
			rSelectPlansFuseAll(memo, current.getInput().get(i), pref, partition);
		}
		
		setVisited(current.getHopID(), currentType);
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
		//WRITE + min(COMPUTE, READ), where WRITE costs are given by the output size, 
		//READ costs by the input sizes, and COMPUTE by operation specific FLOP
		//counts times number of cells of main input, disregarding sparsity for now.
		
		HashSet<Long> visited = new HashSet<Long>();
		double costs = 0;
		for( Long hopID : R )
			costs += rGetPlanCosts(memo, memo._hopRefs.get(hopID), 
					visited, partition, M, plan, computeCosts, null, null);		
		return costs;
	}
	
	private static double rGetPlanCosts(CPlanMemoTable memo, Hop current, HashSet<Long> visited, HashSet<Long> partition, 
			ArrayList<Long> M, boolean[] plan, HashMap<Long, Double> computeCosts, OperatorStats costsCurrent, TemplateType currentType) 
	{
		if( visited.contains(current.getHopID()) )
			return 0; //dont double count 
		
		//open template if necessary, including memoization
		//under awareness of current plan choice
		MemoTableEntry best = null;
		boolean opened = false;
		if( memo.contains(current.getHopID()) ) {
			if( currentType == null ) {
				best = memo.get(current.getHopID()).stream()
					.filter(p -> isValid(p, current))
					.filter(p -> hasNoRefToMaterialization(p, M, plan))
					.min(new BasicPlanComparator()).orElse(null);
				opened = true;
				visited.add(current.getHopID());
			}
			else {
				best = memo.get(current.getHopID()).stream()
					.filter(p -> p.type==currentType || p.type==TemplateType.CellTpl)
					.filter(p -> hasNoRefToMaterialization(p, M, plan))
					.min(Comparator.comparing(p -> 7-((p.type==currentType)?4:0)-p.countPlanRefs()))
					.orElse(null);
			}
		}
		
		//create new cost vector if opened, initialized with write costs
		OperatorStats costVect = !opened ? costsCurrent : 
			new OperatorStats(Math.max(current.getDim1(),1)*Math.max(current.getDim2(),1));
		
		//add compute costs of current operator to costs vector 
		if( partition.contains(current.getHopID()) )
			costVect.computeCosts += computeCosts.get(current.getHopID());
		
		//process children recursively
		double costs = 0;
		for( int i=0; i< current.getInput().size(); i++ ) {
			Hop c = current.getInput().get(i);
			if( best!=null && best.isPlanRef(i) )
				costs += rGetPlanCosts(memo, c, visited, partition, M, plan, computeCosts, costVect, best.type);
			else { //include children and I/O costs
				costs += rGetPlanCosts(memo, c, visited, partition, M, plan, computeCosts, null, null);
				if( costVect != null )
					costVect.addInputSize( c.getHopID(), Math.max(current.getDim1(),1)*Math.max(current.getDim2(),1));
			}				
		}	
		
		//add costs for opened fused operator
		if( partition.contains(current.getHopID()) ) {
			if( opened ) {
				if( LOG.isTraceEnabled() )
					LOG.trace("Cost vector for fused operator: "+costVect);
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
		double costs = 0;
		if( current instanceof UnaryOp ) {
			switch( ((UnaryOp)current).getOp() ) {
				case ABS:   
				case ROUND:
				case CEIL:
				case FLOOR:
				case SIGN:
				case SELP:    costs = 1; break; 
				case SPROP:
				case SQRT:    costs = 2; break;
				case EXP:     costs = 18; break;
				case SIGMOID: costs = 21; break;
				case LOG:    
				case LOG_NZ:  costs = 32; break;
				case NCOL:
				case NROW:
				case PRINT:
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
				case CUMSUM:
				case CUMMIN:
				case CUMMAX:
				case CUMPROD: costs = 1; break;
				default:
					throw new RuntimeException("Cost model not "
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
				case DIV:    costs = 22; break;
				case LOG:
				case LOG_NZ: costs = 32; break;
				case POW:    costs = (HopRewriteUtils.isLiteralOfValue(
						current.getInput().get(1), 2) ? 1 : 16); break;
				case MINUS_NZ:
				case MINUS1_MULT: costs = 2; break;
				default:
					throw new RuntimeException("Cost model not "
						+ "implemented yet for: "+((BinaryOp)current).getOp());
			}
		}
		else if( current instanceof TernaryOp ) {
			switch( ((TernaryOp)current).getOp() ) {
				case PLUS_MULT: 
				case MINUS_MULT: costs = 2; break;
				default:
					throw new RuntimeException("Cost model not "
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
				throw new RuntimeException("Cost model not "
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
	
	private static class OperatorStats {
		public final double outSize; 
		public double computeCosts = 0;
		public final HashMap<Long, Double> inSizes = new HashMap<Long, Double>();
		
		public OperatorStats(double outputSize) {
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
				+Arrays.toString(inSizes.values().toArray(new Double[0]))+"}]";
		}
	}
}
