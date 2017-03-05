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
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map.Entry;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeCell;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeRowAggVector;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.BaseTpl.TemplateType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.utils.Statistics;

public class CplanRegister {
	
	//HashMap: key: TemplateType - Value: List of all the patterns fused by that template 
	//LinkedHashMap: key: HopID of the original hop to be fused , Value: Input hops to the fused operation 
	  	//Note: LinkedHashMap holds intermediate cplans as well (e.g, log(exp(round(X))) ) We store in the LinkedHashMao three keys 
			    //for the three hops (log, exp and round). The key that was inserted last is the key of the hop to be fused
		
	private HashMap<TemplateType, ArrayList<LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>>>  _cplans;
	
	public CplanRegister() {
		_cplans = new HashMap<TemplateType, ArrayList<LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>>>();
	}
	
	public void insertCpplans(TemplateType type, LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> cplans) {
		if( !_cplans.containsKey(type) )
			_cplans.put(type, new ArrayList<LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>>());
		
		_cplans.get(type).add(cplans);
		
		if( DMLScript.STATISTICS )
			Statistics.incrementCodegenCPlanCompile(1); 
		//note: cplans.size() would also contain all subsets of cpplans
	}

	public boolean containsHop(TemplateType type, long hopID) {
		if(!_cplans.containsKey(type))
			return false;
		for (LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> cpplans : _cplans.get(type) )
			if(cpplans.containsKey(hopID))
				return true;
		
		return false;
	}
	
	public LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> getTopLevelCplans()
	{
		if( _cplans.isEmpty() )
			return new LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>();
			
		//resolve conflicts, i.e., overlap, between template types 
		resolvePlanConflicts(); 
		
		//extract top level (subsuming) cplans per type and operator chain
		LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> ret = new LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>();
		for (TemplateType key : _cplans.keySet()) {
			for (LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> intermediateCplans : _cplans.get(key)) {
				Entry<Long, Pair<Hop[],CNodeTpl>> cplan = TemplateUtils.getTopLevelCpplan(intermediateCplans);
				if(cplan !=null)
					ret.put(cplan.getKey(), cplan.getValue());			
			}
		}
		
		//merge top level plans if possible //TODO move to rowagg template
		ret = mergeRowAggregateCellwisePlans(ret);
		
		return ret;
	}
	
	/**
	 * Resolves conflicts between overlapping cplans of different types.
	 * 
	 */
	private void resolvePlanConflicts()
	{
		//get different plan categories
		ArrayList<LinkedHashMap<Long, Pair<Hop[], CNodeTpl>>> cellwisePlans = _cplans.get(TemplateType.CellTpl);
		ArrayList<LinkedHashMap<Long, Pair<Hop[], CNodeTpl>>> outerprodPlans = _cplans.get(TemplateType.OuterProductTpl);
		ArrayList<LinkedHashMap<Long, Pair<Hop[], CNodeTpl>>> rowaggPlans = _cplans.get(TemplateType.RowAggTpl);
		
		//prefer outer product plans over cellwise plans -> remove overlap
		if( cellwisePlans != null && outerprodPlans != null ) {
			for( LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> outerprodCplan : outerprodPlans ) {
				for( LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> map : cellwisePlans )
					for( Long key : outerprodCplan.keySet() )
						map.remove(key);
			}		
		}
		
		//prefer row aggregate plans over cellwise plans -> remove overlap
		if( cellwisePlans != null && rowaggPlans != null ) {
			for( LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> rowaggCplan : rowaggPlans ) {
				for( LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> map : cellwisePlans )
					for( Long key : rowaggCplan.keySet() )
						map.remove(key);
			}	
		}
	}
	
	private static LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> mergeRowAggregateCellwisePlans(LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> plans)
	{
		LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> ret = new LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>(plans);
		
		//extract row aggregate templates
		HashMap<Long, Pair<Hop[],CNodeTpl>> rowaggPlans = new HashMap<Long, Pair<Hop[],CNodeTpl>>();
		for( Entry<Long, Pair<Hop[],CNodeTpl>> e : plans.entrySet() )
			if( e.getValue().getValue() instanceof CNodeRowAggVector )
				rowaggPlans.put(e.getKey(), e.getValue());
		
		//probe and merge row aggregate secondary inputs (by definition vectors)
		for( Entry<Long, Pair<Hop[],CNodeTpl>> e : rowaggPlans.entrySet() ) {
			//check all inputs for existing cell plans
			Hop[] inputs = e.getValue().getKey();
			for( int i=1; i<inputs.length; i++ ) {
				long inhopID = inputs[i].getHopID();
				if( ret.containsKey(inhopID) && ret.get(inhopID).getValue() instanceof CNodeCell
					&& !((CNodeCell)ret.get(inhopID).getValue()).hasMultipleConsumers() ) 
				{
					//merge row agg template
					CNodeRowAggVector rowaggtpl = (CNodeRowAggVector) e.getValue().getValue();
					CNodeCell celltpl = (CNodeCell)ret.get(inhopID).getValue();
					celltpl.getInput().get(0).setDataType(DataType.MATRIX);
					rowaggtpl.rReplaceDataNode(rowaggtpl.getOutput(), inhopID, celltpl.getOutput());
					rowaggtpl.rInsertLookupNode(rowaggtpl.getOutput(), ((CNodeData)celltpl.getInput().get(0)).getHopID(), 
							new HashMap<Long, CNode>(), UnaryType.LOOKUP_R);
					for( CNode input : celltpl.getInput() )
						rowaggtpl.addInput(input);
					HashSet<Long> inputIDs = TemplateUtils.rGetInputHopIDs(rowaggtpl.getOutput(), new HashSet<Long>());
					Hop[] hops = TemplateUtils.mergeDistinct(inputIDs, inputs, ret.get(inhopID).getKey());
					e.getValue().setKey(hops);
					
					//remove cell template 
					ret.remove(inhopID);
				}
			}
		}
		
		return ret;
	}
}
