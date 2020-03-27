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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.codegen.opt.InterestingPoint.DecisionType;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysds.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;

/**
 * Utility functions to extract structural information from the memo table,
 * including connected components (aka partitions) of partial fusion plans, 
 * materialization points of partitions, and root nodes of partitions.
 * 
 */
public class PlanAnalyzer 
{
	private static final Log LOG = LogFactory.getLog(PlanAnalyzer.class.getName());
	
	public static Collection<PlanPartition> analyzePlanPartitions(CPlanMemoTable memo, ArrayList<Hop> roots, boolean ext) {
		//determine connected sub graphs of plans
		Collection<HashSet<Long>> parts = getConnectedSubGraphs(memo, roots);
		
		//determine roots and materialization points
		Collection<PlanPartition> ret = new ArrayList<>();
		for( HashSet<Long> partition : parts ) {
			HashSet<Long> R = getPartitionRootNodes(memo, partition);
			HashSet<Long> I = getPartitionInputNodes(R, partition, memo);
			ArrayList<Long> M = getMaterializationPoints(R, partition, memo);
			HashSet<Long> Pnpc = getNodesWithNonPartitionConsumers(R, partition, memo);
			InterestingPoint[] Mext = !ext ? null : 
				getMaterializationPointsExt(R, partition, M, memo);
			boolean hasOuter = partition.stream()
				.anyMatch(k -> memo.contains(k, TemplateType.OUTER));
			ret.add(new PlanPartition(partition, R, I, Pnpc, M, Mext, hasOuter));
		}
		
		return ret;
	}
	
	private static Collection<HashSet<Long>> getConnectedSubGraphs(CPlanMemoTable memo, ArrayList<Hop> roots) 
	{
		//build inverted index for 'referenced by' relationship 
		HashMap<Long, HashSet<Long>> refBy = new HashMap<>();
		for( Entry<Long, List<MemoTableEntry>> e : memo.getPlans().entrySet() )
			for( MemoTableEntry me : e.getValue() ) 
				for( int i=0; i<3; i++ )
					if( me.isPlanRef(i) ) {
						if( !refBy.containsKey(me.input(i)) )
							refBy.put(me.input(i), new HashSet<Long>());
						refBy.get(me.input(i)).add(e.getKey());
					}
		
		//create a single partition per root node, if reachable over refBy of 
		//other root node the resulting partition is empty and can be discarded
		ArrayList<HashSet<Long>> parts = new ArrayList<>();
		HashSet<Long> visited = new HashSet<>();
		for( Entry<Long, List<MemoTableEntry>> e : memo.getPlans().entrySet() )
			if( !refBy.containsKey(e.getKey()) ) { //root node
				HashSet<Long> part = rGetConnectedSubGraphs(e.getKey(), 
						memo, refBy, visited, new HashSet<Long>());
				if( !part.isEmpty() )
					parts.add(part);
			}
		
		if( LOG.isTraceEnabled() )
			LOG.trace("Connected sub graphs: "+parts.size());
		
		return parts;
	}
	
	private static HashSet<Long> getPartitionRootNodes(CPlanMemoTable memo, HashSet<Long> partition) 
	{
		//build inverted index of references entries 
		HashSet<Long> ix = new HashSet<>();
		for( Long hopID : partition )
			if( memo.contains(hopID) )
				for( MemoTableEntry me : memo.get(hopID) ) {
					ix.add(me.input1); 
					ix.add(me.input2); 
					ix.add(me.input3);
				}
		
		HashSet<Long> roots = new HashSet<>();
		for( Long hopID : partition )
			if( !ix.contains(hopID) )
				roots.add(hopID);
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Partition root points: "
				+ Arrays.toString(roots.toArray(new Long[0])));
		}
		
		return roots;
	}
	
	private static ArrayList<Long> getMaterializationPoints(HashSet<Long> roots, 
			HashSet<Long> partition, CPlanMemoTable memo) 
	{
		//collect materialization points bottom-up
		ArrayList<Long> ret = new ArrayList<>();
		HashSet<Long> visited = new HashSet<>();
		for( Long hopID : roots )
			rCollectMaterializationPoints(memo.getHopRefs().get(hopID), 
					visited, partition, roots, ret);
		
		//remove special-case materialization points
		//(root nodes w/ multiple consumers, tsmm input if consumed in partition)
		ret.removeIf(hopID -> roots.contains(hopID)
			|| HopRewriteUtils.isTsmmInput(memo.getHopRefs().get(hopID)));
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Partition materialization points: "
				+ Arrays.toString(ret.toArray(new Long[0])));
		}
		
		return ret;
	}
	
	private static void rCollectMaterializationPoints(Hop current, HashSet<Long> visited, 
			HashSet<Long> partition, HashSet<Long> R, ArrayList<Long> M) 
	{
		//memoization (not via hops because in middle of dag)
		if( visited.contains(current.getHopID()) )
			return;
		
		//process children recursively
		for( Hop c : current.getInput() )
			rCollectMaterializationPoints(c, visited, partition, R, M);
		
		//collect materialization point
		if( isMaterializationPointCandidate(current, partition, R) )
			M.add(current.getHopID());
		
		visited.add(current.getHopID());
	}
	
	private static boolean isMaterializationPointCandidate(Hop hop, HashSet<Long> partition, HashSet<Long> R) {
		return hop.getParent().size()>=2 
			&& hop.getDataType().isMatrix()
			&& partition.contains(hop.getHopID())
			&& !R.contains(hop.getHopID());
	}
	
	private static HashSet<Long> getPartitionInputNodes(HashSet<Long> roots, 
			HashSet<Long> partition, CPlanMemoTable memo)
	{
		HashSet<Long> ret = new HashSet<>();
		HashSet<Long> visited = new HashSet<>();
		for( Long hopID : roots ) {
			Hop current = memo.getHopRefs().get(hopID);
			rCollectPartitionInputNodes(current, visited, partition, ret);
		}
		return ret;
	}
	
	private static void rCollectPartitionInputNodes(Hop current, HashSet<Long> visited, 
		HashSet<Long> partition, HashSet<Long> I) 
	{
		//memoization (not via hops because in middle of dag)
		if( visited.contains(current.getHopID()) )
			return;
		
		//process children recursively
		for( Hop c : current.getInput() )
			if( partition.contains(c.getHopID()) )
				rCollectPartitionInputNodes(c, visited, partition, I);
			else
				I.add(c.getHopID());
		
		visited.add(current.getHopID());
	}
	
	private static HashSet<Long> getNodesWithNonPartitionConsumers(
		HashSet<Long> roots, HashSet<Long> partition, CPlanMemoTable memo)
	{
		HashSet<Long> ret = new HashSet<>();
		for( Long hopID : partition ) {
			Hop hop = memo.getHopRefs().get(hopID);
			if( hasNonPartitionConsumer(hop, partition) 
				&& !roots.contains(hopID))
				ret.add(hopID);
		}
		return ret;
	}
	
	private static boolean hasNonPartitionConsumer(Hop hop, HashSet<Long> partition) {
		boolean ret = false;
		for( Hop p : hop.getParent() )
			ret |= !partition.contains(p.getHopID());
		return ret;
	}
	
	private static InterestingPoint[] getMaterializationPointsExt(HashSet<Long> roots, 
			HashSet<Long> partition, ArrayList<Long> M, CPlanMemoTable memo) 
	{
		//collect categories of interesting points
		ArrayList<InterestingPoint> tmp = new ArrayList<>();
		tmp.addAll(getMaterializationPointConsumers(M, partition, memo));
		tmp.addAll(getTemplateChangePoints(partition, memo));
		
		//reduce to distinct hop->hop pairs (see equals of interesting points)
		InterestingPoint[] ret = tmp.stream().distinct()
			.toArray(InterestingPoint[]::new);
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Partition materialization points (extended): "
				+ Arrays.toString(ret));
		}
		
		return ret;
	}
	
	private static ArrayList<InterestingPoint> getMaterializationPointConsumers(
		ArrayList<Long> M, HashSet<Long> partition, CPlanMemoTable memo) 
	{
		//collect all materialization point consumers
		ArrayList<InterestingPoint> ret = new ArrayList<>();
		for( Long hopID : M )
			for( Hop parent : memo.getHopRefs().get(hopID).getParent() )
				if( partition.contains(parent.getHopID()) )
					ret.add(new InterestingPoint(
						DecisionType.MULTI_CONSUMER, parent.getHopID(), hopID));

		if( LOG.isTraceEnabled() ) {
			LOG.trace("Partition materialization point consumers: "
				+ Arrays.toString(ret.toArray(new InterestingPoint[0])));
		}
		
		return ret;
	}
	
	private static ArrayList<InterestingPoint> getTemplateChangePoints (
		HashSet<Long> partition, CPlanMemoTable memo) 
	{
		//collect all template change points 
		ArrayList<InterestingPoint> ret = new ArrayList<>();
		for( Long hopID : partition ) {
			long[] refs = memo.getAllRefs(hopID);
			for( int i=0; i<3; i++ ) {
				if( refs[i] < 0 ) continue;
				List<TemplateType> tmp = memo.getDistinctTemplateTypes(hopID, i, true);
				if( memo.containsNotIn(refs[i], tmp, true) )
					ret.add(new InterestingPoint(DecisionType.TEMPLATE_CHANGE, hopID, refs[i]));
			}
		}
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Partition template change points: "
				+ Arrays.toString(ret.toArray(new InterestingPoint[0])));
		}
		
		return ret;
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
}
