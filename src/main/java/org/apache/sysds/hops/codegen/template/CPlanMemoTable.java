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

package org.apache.sysds.hops.codegen.template;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.opt.InterestingPoint;
import org.apache.sysds.hops.codegen.opt.PlanSelection;
import org.apache.sysds.hops.codegen.template.TemplateBase.CloseType;
import org.apache.sysds.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysds.runtime.util.CollectionUtils;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CPlanMemoTable 
{
	private static final Log LOG = LogFactory.getLog(CPlanMemoTable.class.getName());
	
	protected HashMap<Long, List<MemoTableEntry>> _plans;
	protected HashMap<Long, Hop> _hopRefs;
	protected HashSet<Long> _plansExcludeList;
	
	public CPlanMemoTable() {
		_plans = new HashMap<>();
		_hopRefs = new HashMap<>();
		_plansExcludeList = new HashSet<>();
	}
	
	public HashMap<Long, List<MemoTableEntry>> getPlans() {
		return _plans;
	}
	
	public HashSet<Long> getPlansExcludeListed() {
		return _plansExcludeList;
	}
	
	public HashMap<Long, Hop> getHopRefs() {
		return _hopRefs;
	}
	
	public void addHop(Hop hop) {
		_hopRefs.put(hop.getHopID(), hop);
	}
	
	public boolean containsHop(Hop hop) {
		return _hopRefs.containsKey(hop.getHopID());
	}
	
	public boolean contains(long hopID) {
		return _plans.containsKey(hopID)
			&& !_plans.get(hopID).isEmpty();
	}
	
	public boolean contains(long hopID, TemplateType type) {
		return contains(hopID) && get(hopID).stream()
			.anyMatch(p -> p.type==type);
	}
	
	public boolean contains(long hopID, MemoTableEntry me, TemplateType type) {
		return contains(hopID) && get(hopID).stream()
			.anyMatch(p -> p.type==type && p.equalPlanRefs(me));
	}
	
	public boolean contains(long hopID, boolean checkClose, TemplateType... type) {
		if( !checkClose && type.length==1 )
			return contains(hopID, type[0]);
		Set<TemplateType> probe = CollectionUtils.asSet(type);
		return contains(hopID) && get(hopID).stream()
			.anyMatch(p -> (!checkClose||!p.isClosed()) && probe.contains(p.type));
	}
	
	public boolean containsNotIn(long hopID, 
		Collection<TemplateType> types, boolean checkChildRefs) {
		return contains(hopID) && get(hopID).stream()
			.anyMatch(p -> (!checkChildRefs || p.hasPlanRef())
				&& p.isValid() && !types.contains(p.type));
	}
	
	public boolean hasOnlyExactMatches(long hopID, TemplateType type1, TemplateType type2) {
		List<MemoTableEntry> l1 = get(hopID, type1);
		List<MemoTableEntry> l2 = get(hopID, type2);
		boolean ret = l1.size() == l2.size();
		for( MemoTableEntry me : l1 )
			ret &= l2.stream().anyMatch(p -> p.equalPlanRefs(me));
		return ret;
	}
	
	public int countEntries(long hopID) {
		return get(hopID).size();
	}
	
	public int countEntries(long hopID, TemplateType type) {
		return (int) get(hopID).stream()
			.filter(p -> p.type==type).count();
	}
	
	public boolean containsTopLevel(long hopID) {
		return !_plansExcludeList.contains(hopID)
			&& getBest(hopID) != null;
	}
	
	public void add(Hop hop, TemplateType type) {
		add(hop, type, -1, -1, -1);
	}
	
	public void add(Hop hop, TemplateType type, long in1) {
		add(hop, type, in1, -1, -1);
	}
	
	public void add(Hop hop, TemplateType type, long in1, long in2) {
		add(hop, type, in1, in2, -1);
	}
	
	public void add(Hop hop, TemplateType type, long in1, long in2, long in3) {
		int size = (hop instanceof IndexingOp) ? 1 : hop.getInput().size();
		add(hop, new MemoTableEntry(type, in1, in2, in3, size));
	}
	
	public void add(Hop hop, MemoTableEntry me) {
		_hopRefs.put(hop.getHopID(), hop);
		if( !_plans.containsKey(hop.getHopID()) )
			_plans.put(hop.getHopID(), new ArrayList<MemoTableEntry>());
		_plans.get(hop.getHopID()).add(me);
	}
	
	public void addAll(Hop hop, MemoTableEntrySet P) {
		_hopRefs.put(hop.getHopID(), hop);
		if( !_plans.containsKey(hop.getHopID()) )
			_plans.put(hop.getHopID(), new ArrayList<MemoTableEntry>());
		_plans.get(hop.getHopID()).addAll(P.plans);
	}
	
	public void remove(Hop hop, Set<MemoTableEntry> excludeList) {
		_plans.get(hop.getHopID())
			.removeIf(p -> excludeList.contains(p));
	}
	
	public void remove(Hop hop, TemplateType type) {
		_plans.get(hop.getHopID())
			.removeIf(p -> p.type == type);
	}
	
	public void removeAllRefTo(long hopID) {
		removeAllRefTo(hopID, null);
	}
	
	public void removeAllRefTo(long hopID, TemplateType type) {
		//recursive removal of references
		for( Entry<Long, List<MemoTableEntry>> e : _plans.entrySet() ) {
			if( e.getValue().isEmpty() || e.getKey()==hopID ) 
				continue;
			e.getValue().removeIf(p -> p.hasPlanRefTo(hopID)
				&& (type==null || p.type==type));
		}
	}
	
	public void setDistinct(long hopID, List<MemoTableEntry> plans) {
		_plans.put(hopID, plans.stream()
			.distinct().collect(Collectors.toList()));
	}

	public void pruneRedundant(long hopID, boolean pruneDominated, InterestingPoint[] matPoints) {
		if( !contains(hopID) )
			return;
		
		//prune redundant plans (i.e., equivalent) 
		setDistinct(hopID, _plans.get(hopID));
		
		//prune closed templates without group references
		_plans.get(hopID).removeIf(p -> p.isClosed() && !p.hasPlanRef());
		
		//prune dominated plans (e.g., opened plan subsumed by fused plan 
		//if single consumer of input; however this only applies to fusion
		//heuristic that only consider materialization points)
		if( pruneDominated ) {
			HashSet<MemoTableEntry> rmList = new HashSet<>();
			List<MemoTableEntry> list = _plans.get(hopID);
			Hop hop = _hopRefs.get(hopID);
			for( MemoTableEntry e1 : list )
				for( MemoTableEntry e2 : list )
					if( e1 != e2 && e1.subsumes(e2) ) {
						//check that childs don't have multiple consumers
						boolean rmSafe = true; 
						for( int i=0; i<=2; i++ ) {
							rmSafe &= (e1.isPlanRef(i) && !e2.isPlanRef(i)) ?
								(matPoints!=null && !InterestingPoint.isMatPoint(
									matPoints, hopID, e1.input(i)))
								|| hop.getInput().get(i).getParent().size()==1 : true;
						}
						if( rmSafe )
							rmList.add(e2);
					}
			
			//update current entry list, by removing rmList
			remove(hop, rmList);
		}
	}

	public void pruneSuboptimal(ArrayList<Hop> roots) {
		if( LOG.isTraceEnabled() )
			LOG.trace("#1: Memo before plan selection ("+size()+" plans)\n"+this);
		
		//build index of referenced entries
		HashSet<Long> ix = new HashSet<>();
		for( Entry<Long, List<MemoTableEntry>> e : _plans.entrySet() )
			for( MemoTableEntry me : e.getValue() ) {
				ix.add(me.input1); 
				ix.add(me.input2); 
				ix.add(me.input3);
			}
		
		//prune single-operation (not referenced, and no child references)
		Iterator<Entry<Long, List<MemoTableEntry>>> iter = _plans.entrySet().iterator();
		while( iter.hasNext() ) {
			Entry<Long, List<MemoTableEntry>> e = iter.next();
			if( !(ix.contains(e.getKey()) || TemplateUtils.isValidSingleOperation(_hopRefs.get(e.getKey()))) ) {
				e.getValue().removeIf(p -> !p.hasPlanRef());
				if( e.getValue().isEmpty() )
					iter.remove();
			}
		}
		
		//prune dominated plans (e.g., plan referenced by other plan and this
		//other plan is single consumer) by marking it as exclude-listed because
		//the chain of entries is still required for cplan construction
		if( SpoofCompiler.PLAN_SEL_POLICY.isHeuristic() ) {
			for( Entry<Long, List<MemoTableEntry>> e : _plans.entrySet() )
				for( MemoTableEntry me : e.getValue() ) {
					for( int i=0; i<=2; i++ )
						if( me.isPlanRef(i) && _hopRefs.get(me.input(i)).getParent().size()==1 )
							_plansExcludeList.add(me.input(i));
				}
		}
		
		//core plan selection
		PlanSelection selector = SpoofCompiler.createPlanSelector();
		selector.selectPlans(this, roots);
		
		if( LOG.isTraceEnabled() )
			LOG.trace("#2: Memo after plan selection ("+size()+" plans)\n"+this);
	}
	
	public List<MemoTableEntry> get(long hopID) {
		return _plans.get(hopID);
	}
	
	public List<MemoTableEntry> get(long hopID, TemplateType type) {
		return _plans.get(hopID).stream()
			.filter(p -> p.type==type).collect(Collectors.toList());
	}
	
	public List<MemoTableEntry> getDistinct(long hopID) {
		return _plans.get(hopID).stream()
			.distinct().collect(Collectors.toList());
	}
	
	public List<TemplateBase> getDistinctTemplates(long hopID) {
		if(!contains(hopID))
			return Collections.emptyList();
		//return distinct entries wrt type and closed attributes
		return _plans.get(hopID).stream()
			.map(p -> TemplateUtils.createTemplate(p.type, p.ctype))
			.distinct().collect(Collectors.toList());
	}
	
	public List<TemplateType> getDistinctTemplateTypes(long hopID, int refAt) {
		return getDistinctTemplateTypes(hopID, refAt, false);
	}
	
	public List<TemplateType> getDistinctTemplateTypes(long hopID, int refAt, boolean exclInvalOuter) {
		if(!contains(hopID))
			return Collections.emptyList();
		//return distinct template types with reference at given position
		return _plans.get(hopID).stream()
			.filter(p -> p.isPlanRef(refAt) && (!exclInvalOuter 
				|| p.type!=TemplateType.OUTER || p.isValid()) )
			.map(p -> p.type) //extract type
			.distinct().collect(Collectors.toList());
	}
	
	public MemoTableEntry getBest(long hopID) {
		List<MemoTableEntry> tmp = get(hopID);
		if( tmp == null || tmp.isEmpty() )
			return null;

		//single plan per type, get plan w/ best rank in preferred order
		//but ensure that the plans valid as a top-level plan
		return tmp.stream().filter(p -> p.isValid())
			.min(Comparator.comparing(p -> p.type.getRank())).orElse(null);
	}
	
	public MemoTableEntry getBest(long hopID, TemplateType pref) {
		List<MemoTableEntry> tmp = get(hopID);
		if( tmp == null || tmp.isEmpty() )
			return null;

		//single plan per type, get plan w/ best rank in preferred order
		return Collections.min(tmp, Comparator.comparing(
			p -> (p.type==pref) ? -p.countPlanRefs() : p.type.getRank()+1));
	}
	
	public MemoTableEntry getBest(long hopID, TemplateType pref1, TemplateType pref2) {
		List<MemoTableEntry> tmp = get(hopID);
		if( tmp == null || tmp.isEmpty() )
			return null;

		//single plan per type, get plan w/ best rank in preferred order
		return Collections.min(tmp, Comparator.comparing(
			p -> (p.type==pref1) ? -p.countPlanRefs()-4 :
				(p.type==pref2) ? -p.countPlanRefs() : p.type.getRank()+1));
	}
	
	public long[] getAllRefs(long hopID) {
		long[] refs = new long[3];
		for( MemoTableEntry me : get(hopID) )
			for( int i=0; i<3; i++ )
				if( me.isPlanRef(i) )
					refs[i] = me.input(i);
		return refs;
	}
	
	public int size() {
		return _plans.values().stream()
			.map(list -> list.size())
			.mapToInt(x -> x.intValue()).sum();
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("----------------------------------\n");
		sb.append("MEMO TABLE: \n");
		sb.append("----------------------------------\n");
		for( Entry<Long, List<MemoTableEntry>> e : _plans.entrySet() ) {
			sb.append(e.getKey() + " "+_hopRefs.get(e.getKey()).getOpString()+": ");
			sb.append(Arrays.toString(e.getValue().toArray(new MemoTableEntry[0]))+"\n");
		}
		sb.append("----------------------------------\n");
		sb.append("ExcludeListed Plans: ");
		sb.append(Arrays.toString(_plansExcludeList.toArray(new Long[0]))+"\n");
		sb.append("----------------------------------\n");
		return sb.toString();	
	}

	////////////////////////////////////////
	// Memo table entry abstractions
	//////
	
	public static class MemoTableEntry 
	{
		public TemplateType type;
		public final long input1; 
		public final long input2;
		public final long input3;
		public final int size;
		public CloseType ctype;
		public MemoTableEntry(TemplateType t, long in1, long in2, long in3, int inlen) {
			this(t, in1, in2, in3, inlen, CloseType.OPEN_VALID);
		}
		public MemoTableEntry(TemplateType t, long in1, long in2, long in3, int inlen, CloseType close) {
			type = t;
			input1 = in1;
			input2 = in2;
			input3 = in3;
			size = inlen;
			ctype = close;
		}
		public boolean isClosed() {
			return ctype.isClosed();
		}
		public boolean isValid() {
			return ctype.isValid();
		}
		public boolean isPlanRef(int index) {
			return (index==0 && input1 >=0)
				|| (index==1 && input2 >=0)
				|| (index==2 && input3 >=0);
		}
		public boolean hasPlanRef() {
			return isPlanRef(0) || isPlanRef(1) || isPlanRef(2);
		}
		public boolean hasPlanRefTo(long hopID) {
			return (input1==hopID || input2==hopID || input3==hopID); 
		}
		public int countPlanRefs() {
			return ((input1 >= 0) ? 1 : 0)
				+  ((input2 >= 0) ? 1 : 0)
				+  ((input3 >= 0) ? 1 : 0);
		}
		public int getPlanRefIndex() {
			return (input1>=0) ? 0 : (input2>=0) ? 
				1 : (input3>=0) ? 2 : -1;
		}
		public boolean equalPlanRefs(MemoTableEntry that) {
			return (input1 == that.input1
				&& input2 == that.input2
				&& input3 == that.input3);
		}
		public long input(int index) {
			return (index==0) ? input1 : (index==1) ? input2 : input3;
		}
		public boolean subsumes(MemoTableEntry that) {
			return (type == that.type 
				&& !(!isPlanRef(0) && that.isPlanRef(0))
				&& !(!isPlanRef(1) && that.isPlanRef(1))
				&& !(!isPlanRef(2) && that.isPlanRef(2)));
		}
		@Override
		public int hashCode() {
			int h = UtilFunctions.intHashCode(type.ordinal(), Long.hashCode(input1));
			h = UtilFunctions.intHashCode(h, Long.hashCode(input2));
			h = UtilFunctions.intHashCode(h, Long.hashCode(input3));
			h = UtilFunctions.intHashCode(h, size);
			h = UtilFunctions.intHashCode(h, ctype.ordinal());
			return h;
		}
		@Override
		public boolean equals(Object obj) {
			if( !(obj instanceof MemoTableEntry) )
				return false;
			MemoTableEntry that = (MemoTableEntry)obj;
			return type == that.type && input1 == that.input1
				&& input2 == that.input2 && input3 == that.input3
				&& size == that.size && ctype == that.ctype;
		}
		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append(type.name());
			sb.append("(");
			for( int i=0; i<size; i++ ) {
				if( i > 0 )
					sb.append(",");
				sb.append(input(i));
			}
			if( !isValid() )
				sb.append("|x");
			sb.append(")");
			return sb.toString();
		}
	}
	
	public static class MemoTableEntrySet 
	{
		public ArrayList<MemoTableEntry> plans = new ArrayList<>();
		
		public MemoTableEntrySet(Hop hop, Hop c, TemplateBase tpl) {
			int pos = (c != null) ? hop.getInput().indexOf(c) : -1;
			int size = (hop instanceof IndexingOp) ? 1 : hop.getInput().size();
			plans.add(new MemoTableEntry(tpl.getType(), (pos==0)?c.getHopID():-1,
				(pos==1)?c.getHopID():-1, (pos==2)?c.getHopID():-1, size, tpl.getCType()));
		}
		
		public void crossProduct(int pos, Long... refs) {
			if( refs.length==1 && refs[0] == -1 )
				return; //unmodified plan set
			ArrayList<MemoTableEntry> tmp = new ArrayList<>();
			for( MemoTableEntry me : plans )
				for( Long ref : refs )
					tmp.add(new MemoTableEntry(me.type, (pos==0)?ref:me.input1, 
						(pos==1)?ref:me.input2, (pos==2)?ref:me.input3, me.size));
			plans = tmp;
		}
		
		@Override
		public String toString() {
			return Arrays.toString(plans.toArray(new MemoTableEntry[0]));
		}
	}
}
