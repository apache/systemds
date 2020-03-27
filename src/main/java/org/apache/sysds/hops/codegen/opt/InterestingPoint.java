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

import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Interesting decision point with regard to materialization of intermediates.
 * These points are defined by a type, as well as hop ID for consumer-producer
 * relationships. Equivalence is defined solely on the hop IDs, to simplify
 * their processing and avoid redundant enumeration.
 *  
 */
public class InterestingPoint 
{
	public enum DecisionType {
		MULTI_CONSUMER,
		TEMPLATE_CHANGE,
	}
	
	private final DecisionType _type;
	public final long _fromHopID; //consumers
	public final long _toHopID; //producers
	
	public InterestingPoint(DecisionType type, long fromHopID, long toHopID) {
		_type = type;
		_fromHopID = fromHopID;
		_toHopID = toHopID;
	}
	
	public DecisionType getType() {
		return _type;
	}
	
	public long getFromHopID() {
		return _fromHopID;
	}
	
	public long getToHopID() {
		return _toHopID;
	}
	
	public static boolean isMatPoint(InterestingPoint[] list, long from, MemoTableEntry me, boolean[] plan) {
		for(int i=0; i<plan.length; i++) {
			if( !plan[i] ) continue;
			InterestingPoint p = list[i];
			if( p._fromHopID!=from ) continue;
			for( int j=0; j<3; j++ )
				if( p._toHopID==me.input(j) )
					return true;
		}
		return false;
	}
	
	public static boolean isMatPoint(InterestingPoint[] list, long from, long to) {
		for(int i=0; i<list.length; i++) {
			InterestingPoint p = list[i];
			if( p._fromHopID==from && p._toHopID==to )
				return true;
		}
		return false;
	}
	
	@Override
	public int hashCode() {
		return UtilFunctions.longHashCode(_fromHopID, _toHopID);
	}
	
	@Override
	public boolean equals(Object o) {
		if( !(o instanceof InterestingPoint) )
			return false;
		InterestingPoint that = (InterestingPoint) o;
		return _fromHopID == that._fromHopID
			&& _toHopID == that._toHopID;
	}
	
	@Override
	public String toString() {
		String stype = (_type==DecisionType.MULTI_CONSUMER) ? "M" : "T";
		return "(" + stype+ ":" + _fromHopID + "->" + _toHopID + ")"; 
	}
}
