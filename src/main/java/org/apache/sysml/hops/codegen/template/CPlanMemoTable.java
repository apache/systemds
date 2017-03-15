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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.commons.collections.CollectionUtils;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.codegen.template.BaseTpl.TemplateType;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;

import scala.tools.jline_embedded.internal.Log;


public class CPlanMemoTable 
{
	public enum PlanSelection {
		FUSE_ALL,             //maximal fusion, possible w/ redundant compute
		FUSE_NO_REDUNDANCY,   //fusion without redundant compute 
		FUSE_COST_BASED,      //cost-based decision on materialization points
	}
	
	private HashMap<Long, ArrayList<MemoTableEntry>> _plans;
	private HashMap<Long, Hop> _hopRefs;
	private HashSet<Long> _plansBlacklist;
	
	public CPlanMemoTable() {
		_plans = new HashMap<Long, ArrayList<MemoTableEntry>>();
		_hopRefs = new HashMap<Long, Hop>();
		_plansBlacklist = new HashSet<Long>();
	}
	
	public boolean contains(long hopID) {
		return _plans.containsKey(hopID);
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
		_hopRefs.put(hop.getHopID(), hop);
		if( !_plans.containsKey(hop.getHopID()) )
			_plans.put(hop.getHopID(), new ArrayList<MemoTableEntry>());
		_plans.get(hop.getHopID()).add(new MemoTableEntry(type, in1, in2, in3));
	}

	@SuppressWarnings("unchecked")
	public void pruneRedundant(long hopID) {
		if( !contains(hopID) )
			return;
		
		//prune redundant plans (i.e., equivalent) 
		HashSet<MemoTableEntry> set = new HashSet<MemoTableEntry>();
		ArrayList<MemoTableEntry> list = _plans.get(hopID);
		for( MemoTableEntry me : list )
			set.add(me);
		
		//prune dominated plans (e.g., opened plan subsumed
		//by fused plan if single consumer of input)
		ArrayList<MemoTableEntry> rmList = new ArrayList<MemoTableEntry>();
		Hop hop = _hopRefs.get(hopID);
		for( MemoTableEntry e1 : set )
			for( MemoTableEntry e2 : set )
				if( e1 != e2 && e1.subsumes(e2) ) {
					//check that childs don't have multiple consumers
					boolean rmSafe = true; 
					for( int i=0; i<=2; i++ )
						rmSafe &= (e1.isPlanRef(i) && !e2.isPlanRef(i)) ?
							hop.getInput().get(i).getParent().size()==1 : true;
					if( rmSafe )
						rmList.add(e2);
				}
		
		//update current entry list
		list.clear();
		list.addAll(CollectionUtils.subtract(set, rmList));
	}

	public void pruneSuboptimal() {
		//build index of referenced entries
		HashSet<Long> ix = new HashSet<Long>();
		for( Entry<Long, ArrayList<MemoTableEntry>> e : _plans.entrySet() )
			for( MemoTableEntry me : e.getValue() ) {
				ix.add(me.input1); 
				ix.add(me.input2); 
				ix.add(me.input3);
			}
		
		//prune single-operation (not referenced, and no child references)
		Iterator<Entry<Long, ArrayList<MemoTableEntry>>> iter = _plans.entrySet().iterator();
		while( iter.hasNext() ) {
			Entry<Long, ArrayList<MemoTableEntry>> e = iter.next();
			if( !ix.contains(e.getKey()) ) {
				ArrayList<MemoTableEntry> list = e.getValue();
				for( int i=0; i<list.size(); i++ )
					if( !list.get(i).hasPlanRef() )
						list.remove(i--);
				if( list.isEmpty() )
					iter.remove();
			}
		}
		
		//prune dominated plans (e.g., plan referenced by other plan and this
		//other plan is single consumer) by marking it as blacklisted because
		//the chain of entries is still required for cplan construction
		for( Entry<Long, ArrayList<MemoTableEntry>> e : _plans.entrySet() )
			for( MemoTableEntry me : e.getValue() ) {
				for( int i=0; i<=2; i++ )
					if( me.isPlanRef(i) && _hopRefs.get(me.intput(i)).getParent().size()==1 )
						_plansBlacklist.add(me.intput(i));
			}
	}

	public ArrayList<MemoTableEntry> get(long hopID) {
		return _plans.get(hopID);
	}
	
	public MemoTableEntry getBest(long hopID) {
		ArrayList<MemoTableEntry> tmp = get(hopID);
		if( tmp == null || tmp.isEmpty() )
			return null;
		
		//get first plan by type preference
		TemplateType[] ttPrefOrder = new TemplateType[]{TemplateType.RowAggTpl, 
				TemplateType.OuterProdTpl, TemplateType.CellTpl}; 
		for( TemplateType pref : ttPrefOrder )
			for( MemoTableEntry me : tmp )
				if( me.type == pref && isValid(me, _hopRefs.get(hopID)) )
					return me;
		return null;
	}
	
	//TODO revisit requirement for preference once cost-based pruning (pruneSuboptimal) ready
	public MemoTableEntry getBest(long hopID, TemplateType pref) {
		ArrayList<MemoTableEntry> tmp = get(hopID);
		if( tmp.size()==1 ) //single plan available
			return tmp.get(0);
		
		//try to find plan with preferred type
		Log.warn("Multiple memo table entries available, searching for preferred type.");
		ArrayList<MemoTableEntry> tmp2 = new ArrayList<MemoTableEntry>();
		for( MemoTableEntry me : tmp )
			if( me.type == pref )
				tmp2.add(me);
		if( !tmp2.isEmpty() ) {
			if( tmp2.size() > 1 )
				Log.warn("Multiple memo table entries w/ preferred type available, return max refs entry.");
			return getMaxRefsEntry(tmp2);
		}
		else {
			Log.warn("Multiple memo table entries available but none with preferred type, return max refs entry.");
			return getMaxRefsEntry(tmp);
		}
	}
	
	private static MemoTableEntry getMaxRefsEntry(ArrayList<MemoTableEntry> tmp) {
		int maxPos = 0;
		int maxRefs = 0;
		for( int i=0; i<tmp.size(); i++ ) {
			int cntRefs = tmp.get(i).countPlanRefs();
			if( cntRefs > maxRefs ) {
				maxRefs = cntRefs;
				maxPos = i;
			}
		}
		return tmp.get(maxPos);
	}
	
	private static boolean isValid(MemoTableEntry me, Hop hop) {
		return (me.type == TemplateType.OuterProdTpl 
				&& (me.closed || HopRewriteUtils.isBinaryMatrixMatrixOperation(hop)))
			|| (me.type == TemplateType.RowAggTpl && me.closed)	
			|| (me.type == TemplateType.CellTpl);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("----------------------------------\n");
		sb.append("MEMO TABLE: \n");
		sb.append("----------------------------------\n");
		for( Entry<Long, ArrayList<MemoTableEntry>> e : _plans.entrySet() ) {
			sb.append(e.getKey() + " "+_hopRefs.get(e.getKey()).getOpString()+": ");
			sb.append(Arrays.toString(e.getValue().toArray(new MemoTableEntry[0]))+"\n");
		}
		sb.append("----------------------------------\n");
		return sb.toString();	
	}
	
	public static class MemoTableEntry 
	{
		public final TemplateType type;
		public final long input1; 
		public final long input2;
		public final long input3;
		public boolean closed = false;
		public MemoTableEntry(TemplateType t, long in1, long in2, long in3) {
			type = t;
			input1 = in1;
			input2 = in2;
			input3 = in3;
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
		public long intput(int index) {
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
}
