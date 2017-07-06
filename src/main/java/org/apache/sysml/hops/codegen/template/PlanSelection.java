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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.runtime.util.UtilFunctions;

public abstract class PlanSelection 
{
	private final HashMap<Long, List<MemoTableEntry>> _bestPlans = 
			new HashMap<Long, List<MemoTableEntry>>();
	private final HashSet<VisitMark> _visited = new HashSet<VisitMark>();
	
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
	
	/**
	 * Determines if the given partial fusion plan is valid.
	 * 
	 * @param me memo table entry
	 * @param hop current hop
	 * @return true if entry is valid as top-level plan
	 */
	protected static boolean isValid(MemoTableEntry me, Hop hop) {
		return (me.type == TemplateType.OuterProdTpl 
				&& (me.closed || HopRewriteUtils.isBinaryMatrixMatrixOperation(hop)))
			|| (me.type == TemplateType.RowTpl)	
			|| (me.type == TemplateType.CellTpl)
			|| (me.type == TemplateType.MultiAggTpl);
	}
	
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
	
	/**
	 * Basic plan comparator to compare memo table entries with regard to
	 * a pre-defined template preference order and the number of references.
	 */
	protected static class BasicPlanComparator implements Comparator<MemoTableEntry> {
		@Override
		public int compare(MemoTableEntry o1, MemoTableEntry o2) {
			//for different types, select preferred type
			if( o1.type != o2.type )
				return Integer.compare(o1.type.getRank(), o2.type.getRank());
			
			//for same type, prefer plan with more refs
			return Integer.compare(
				3-o1.countPlanRefs(), 3-o2.countPlanRefs());
		}
	}
	
	private static class VisitMark {
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
}
