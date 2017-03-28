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
import java.util.Comparator;
import java.util.Map.Entry;
import java.util.HashSet;
import java.util.List;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.codegen.template.TemplateBase.TemplateType;

/**
 * This plan selection heuristic aims for fusion without any redundant 
 * computation, which, however, potentially leads to more materialized 
 * intermediates than the fuse all heuristic.
 * <p>
 * NOTE: This heuristic is essentially the same as FuseAll, except that 
 * any plans that refer to a hop with multiple consumers are removed in 
 * a pre-processing step.
 * 
 */
public class PlanSelectionFuseNoRedundancy extends PlanSelection
{	
	@Override
	public void selectPlans(CPlanMemoTable memo, ArrayList<Hop> roots) {
		//pruning and collection pass
		for( Hop hop : roots )
			rSelectPlans(memo, hop, null);
		
		//take all distinct best plans
		for( Entry<Long, List<MemoTableEntry>> e : getBestPlans().entrySet() )
			memo.setDistinct(e.getKey(), e.getValue());
	}
	
	private void rSelectPlans(CPlanMemoTable memo, Hop current, TemplateType currentType) 
	{	
		if( isVisited(current.getHopID(), currentType) )
			return;
		
		//step 0: remove plans that refer to a common partial plan
		if( memo.contains(current.getHopID()) ) {
			HashSet<MemoTableEntry> rmSet = new HashSet<MemoTableEntry>();
			List<MemoTableEntry> hopP = memo.get(current.getHopID());
			for( MemoTableEntry e1 : hopP )
				for( int i=0; i<3; i++ )
					if( e1.isPlanRef(i) && current.getInput().get(i).getParent().size()>1 )
						rmSet.add(e1); //remove references to hops w/ multiple consumers
			memo.remove(current, rmSet);
		}
		
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
			rSelectPlans(memo, current.getInput().get(i), pref);
		}
		
		setVisited(current.getHopID(), currentType);
	}	
}
