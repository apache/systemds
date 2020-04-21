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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysds.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class PlanSelection 
{
	private static final BasicPlanComparator BASE_COMPARE = new BasicPlanComparator();
	private final TypedPlanComparator _typedCompare = new TypedPlanComparator();
	
	private final HashMap<Long, List<MemoTableEntry>> _bestPlans = new HashMap<>();
	private final HashSet<VisitMark> _visited = new HashSet<>();
	
	/**
	 * Given a HOP DAG G, and a set of partial fusions plans P, find the set of optimal, 
	 * non-conflicting fusion plans P' that applied to G minimizes costs C with
	 * P' = \argmin_{p \subseteq P} C(G, p) s.t. Z \vDash p, where Z is a set of 
	 * constraints such as memory budgets and block size restrictions per fused operator.
	 * 
	 * @param memo partial fusion plans P
	 * @param roots entry points of HOP DAG G
	 */
	public abstract void selectPlans(CPlanMemoTable memo, ArrayList<Hop> roots);
	
	protected void addBestPlan(long hopID, MemoTableEntry me) {
		if( me == null ) return;
		if( !_bestPlans.containsKey(hopID) )
			_bestPlans.put(hopID, new ArrayList<MemoTableEntry>());
		_bestPlans.get(hopID).add(me);
	}
	
	protected HashMap<Long, List<MemoTableEntry>> getBestPlans() {
		return _bestPlans;
	}
	
	protected boolean isVisited(long hopID, TemplateType type) {
		return _visited.contains(new VisitMark(hopID, type));
	}
	
	protected void setVisited(long hopID, TemplateType type) {
		_visited.add(new VisitMark(hopID, type));
	}
	
	protected void rSelectPlansFuseAll(CPlanMemoTable memo, Hop current, TemplateType currentType, HashSet<Long> partition) 
	{
		if( isVisited(current.getHopID(), currentType) 
			|| (partition!=null && !partition.contains(current.getHopID())) )
			return;
		
		//step 1: prune subsumed plans of same type
		if( memo.contains(current.getHopID()) ) {
			HashSet<MemoTableEntry> rmSet = new HashSet<>();
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
					.filter(p -> p.isValid())
					.min(BASE_COMPARE).orElse(null);
			}
			else {
				_typedCompare.setType(currentType);
				best = memo.get(current.getHopID()).stream()
					.filter(p -> p.type==currentType || p.type==TemplateType.CELL)
					.min(_typedCompare).orElse(null);
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
	
	/**
	 * Basic plan comparator to compare memo table entries with regard to
	 * a pre-defined template preference order and the number of references.
	 */
	protected static class BasicPlanComparator implements Comparator<MemoTableEntry> {
		@Override
		public int compare(MemoTableEntry o1, MemoTableEntry o2) {
			return icompare(o1, o2);
		}
		
		public static int icompare(MemoTableEntry o1, MemoTableEntry o2) {
			if( o2 == null ) return -1;
			
			//for different types, select preferred type
			if( o1.type != o2.type )
				return Integer.compare(o1.type.getRank(), o2.type.getRank());
			
			//for same type, prefer plan with more refs
			return Integer.compare(-o1.countPlanRefs(), -o2.countPlanRefs());
		}
	}
	
	protected static class TypedPlanComparator implements Comparator<MemoTableEntry> {
		private TemplateType _type;
		
		public void setType(TemplateType type) {
			_type = type;
		}
		
		@Override
		public int compare(MemoTableEntry o1, MemoTableEntry o2) {
			return icompare(o1, o2, _type);
		}
		
		public static int icompare(MemoTableEntry o1, MemoTableEntry o2, TemplateType type) {
			if( o2 == null ) return -1;
			int score1 = 7 - ((o1.type==type)?4:0) - o1.countPlanRefs();
			int score2 = 7 - ((o2.type==type)?4:0) - o2.countPlanRefs();
			return Integer.compare(score1, score2);
		}
	}
	
	protected static class VisitMark {
		private final long _hopID;
		private final TemplateType _type;
		
		public VisitMark(long hopID, TemplateType type) {
			_hopID = hopID;
			_type = type;
		}
		@Override
		public int hashCode() {
			return UtilFunctions.longHashCode(
				_hopID, (_type!=null)?_type.hashCode():0);
		}
		@Override 
		public boolean equals(Object o) {
			return (o instanceof VisitMark
				&& _hopID == ((VisitMark)o)._hopID
				&& _type == ((VisitMark)o)._type);
		}
	}
	
	public static class VisitMarkCost {
		private final long _hopID;
		private final long _costID;
		
		public VisitMarkCost(long hopID, long costID) {
			_hopID = hopID;
			_costID = costID;
		}
		@Override
		public int hashCode() {
			return UtilFunctions.longHashCode(
				_hopID, _costID);
		}
		@Override 
		public boolean equals(Object o) {
			return (o instanceof VisitMarkCost
				&& _hopID == ((VisitMarkCost)o)._hopID
				&& _costID == ((VisitMarkCost)o)._costID);
		}
	}
}
