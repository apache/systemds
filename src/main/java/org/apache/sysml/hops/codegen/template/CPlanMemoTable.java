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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.codegen.SpoofCompiler;
import org.apache.sysml.hops.codegen.template.TemplateBase.TemplateType;

public class CPlanMemoTable 
{
	private static final Log LOG = LogFactory.getLog(CPlanMemoTable.class.getName());
	
	protected HashMap<Long, List<MemoTableEntry>> _plans;
	protected HashMap<Long, Hop> _hopRefs;
	protected HashSet<Long> _plansBlacklist;
	
	public CPlanMemoTable() {
		_plans = new HashMap<Long, List<MemoTableEntry>>();
		_hopRefs = new HashMap<Long, Hop>();
		_plansBlacklist = new HashSet<Long>();
	}
	
	public void addHop(Hop hop) {
		_hopRefs.put(hop.getHopID(), hop);
	}
	
	public boolean containsHop(Hop hop) {
		return _hopRefs.containsKey(hop.getHopID());
	}
	
	public boolean contains(long hopID) {
		return _plans.containsKey(hopID);
	}
	
	public boolean contains(long hopID, TemplateType type) {
		return contains(hopID) && get(hopID).stream()
			.filter(p -> p.type==type).findAny().isPresent();
	}
	
	public int countEntries(long hopID) {
		return get(hopID).size();
	}
	
	public int countEntries(long hopID, TemplateType type) {
		return (int) get(hopID).stream()
			.filter(p -> p.type==type).count();
	} 
	
	public boolean containsTopLevel(long hopID) {
		return !_plansBlacklist.contains(hopID)
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
		add(hop, new MemoTableEntry(type, in1, in2, in3));
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
	
	public void remove(Hop hop, HashSet<MemoTableEntry> blackList) {
		_plans.put(hop.getHopID(), _plans.get(hop.getHopID()).stream()
			.filter(p -> !blackList.contains(p)).collect(Collectors.toList()));
	}
	
	public void setDistinct(long hopID, List<MemoTableEntry> plans) {
		_plans.put(hopID, plans.stream()
			.distinct().collect(Collectors.toList()));
	}

	public void pruneRedundant(long hopID) {
		if( !contains(hopID) )
			return;
		
		//prune redundant plans (i.e., equivalent) 
		setDistinct(hopID, _plans.get(hopID));
		
		//prune dominated plans (e.g., opened plan subsumed
		//by fused plan if single consumer of input)
		HashSet<MemoTableEntry> rmList = new HashSet<MemoTableEntry>();
		List<MemoTableEntry> list = _plans.get(hopID);
		Hop hop = _hopRefs.get(hopID);
		for( MemoTableEntry e1 : list )
			for( MemoTableEntry e2 : list )
				if( e1 != e2 && e1.subsumes(e2) ) {
					//check that childs don't have multiple consumers
					boolean rmSafe = true; 
					for( int i=0; i<=2; i++ )
						rmSafe &= (e1.isPlanRef(i) && !e2.isPlanRef(i)) ?
							hop.getInput().get(i).getParent().size()==1 : true;
					if( rmSafe )
						rmList.add(e2);
				}
		
		//update current entry list, by removing rmList
		remove(hop, rmList);
	}

	public void pruneSuboptimal(ArrayList<Hop> roots) {
		if( LOG.isTraceEnabled() )
			LOG.trace("#1: Memo before plan selection ("+size()+" plans)\n"+this);
		
		//build index of referenced entries
		HashSet<Long> ix = new HashSet<Long>();
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
			if( !ix.contains(e.getKey()) ) {
				e.setValue(e.getValue().stream().filter(
					p -> p.hasPlanRef()).collect(Collectors.toList()));
				if( e.getValue().isEmpty() )
					iter.remove();
			}
		}
		
		//prune dominated plans (e.g., plan referenced by other plan and this
		//other plan is single consumer) by marking it as blacklisted because
		//the chain of entries is still required for cplan construction
		for( Entry<Long, List<MemoTableEntry>> e : _plans.entrySet() )
			for( MemoTableEntry me : e.getValue() ) {
				for( int i=0; i<=2; i++ )
					if( me.isPlanRef(i) && _hopRefs.get(me.input(i)).getParent().size()==1 )
						_plansBlacklist.add(me.input(i));
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
		//return distinct entries wrt type and closed attributes
		return _plans.get(hopID).stream()
			.map(p -> new MemoTableEntry(p.type,-1,-1,-1,p.closed))
			.distinct().collect(Collectors.toList());
	}
	
	public MemoTableEntry getBest(long hopID) {
		List<MemoTableEntry> tmp = get(hopID);
		if( tmp == null || tmp.isEmpty() )
			return null;

		//single plan per type, get plan w/ best rank in preferred order
		//but ensure that the plans valid as a top-level plan
		return tmp.stream().filter(p -> PlanSelection.isValid(p, _hopRefs.get(hopID)))
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
	
	public long[] getAllRefs(long hopID) {
		long[] refs = new long[3];
		for( MemoTableEntry me : get(hopID) )
			for( int i=0; i<3; i++ )
				if( me.isPlanRef(i) )
					refs[i] |= me.input(i);
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
		sb.append("Blacklisted Plans: ");
		sb.append(Arrays.toString(_plansBlacklist.toArray(new Long[0]))+"\n");
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
		public boolean closed = false;
		public MemoTableEntry(TemplateType t, long in1, long in2, long in3) {
			this(t, in1, in2, in3, false);
		}
		public MemoTableEntry(TemplateType t, long in1, long in2, long in3, boolean close) {
			type = t;
			input1 = in1;
			input2 = in2;
			input3 = in3;
			closed = close;
		}
		public boolean isPlanRef(int index) {
			return (index==0 && input1 >=0)
				|| (index==1 && input2 >=0)
				|| (index==2 && input3 >=0);
		}
		public boolean hasPlanRef() {
			return isPlanRef(0) || isPlanRef(1) || isPlanRef(2);
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
			return Arrays.hashCode(
				new long[]{(long)type.ordinal(), input1, input2, input3});
		}
		@Override
		public boolean equals(Object obj) {
			if( !(obj instanceof MemoTableEntry) )
				return false;
			MemoTableEntry that = (MemoTableEntry)obj;
			return type == that.type && input1 == that.input1
				&& input2 == that.input2 && input3 == that.input3;
		}
		@Override
		public String toString() {
			return type.name()+"("+input1+","+input2+","+input3+")";
		}
	}
	
	public static class MemoTableEntrySet 
	{
		public ArrayList<MemoTableEntry> plans = new ArrayList<MemoTableEntry>();
		
		public MemoTableEntrySet(TemplateType type, boolean close) {
			plans.add(new MemoTableEntry(type, -1, -1, -1, close));
		}
		
		public MemoTableEntrySet(TemplateType type, int pos, long hopID, boolean close) {
			plans.add(new MemoTableEntry(type, (pos==0)?hopID:-1, 
					(pos==1)?hopID:-1, (pos==2)?hopID:-1));
		}
		
		public void crossProduct(int pos, Long... refs) {
			ArrayList<MemoTableEntry> tmp = new ArrayList<MemoTableEntry>();
			for( MemoTableEntry me : plans )
				for( Long ref : refs )
					tmp.add(new MemoTableEntry(me.type, (pos==0)?ref:me.input1, 
						(pos==1)?ref:me.input2, (pos==2)?ref:me.input3));
			plans = tmp;
		}
		
		@Override
		public String toString() {
			return Arrays.toString(plans.toArray(new MemoTableEntry[0]));
		}
	}
}
