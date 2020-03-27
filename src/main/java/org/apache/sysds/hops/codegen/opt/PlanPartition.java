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
import java.util.HashSet;

public class PlanPartition 
{
	//nodes of partition (hop IDs)
	private final HashSet<Long> _nodes;
	
	//root nodes of partition (hop IDs)
	private final HashSet<Long> _roots;
	
	//partition inputs 
	private final HashSet<Long> _inputs;
	
	//nodes with non-partition consumers 
	private final HashSet<Long> _nodesNpc;
	
	//materialization points (hop IDs)
	private final ArrayList<Long> _matPoints;
	
	//interesting operator dependencies
	private InterestingPoint[] _matPointsExt;
	
	//indicator if the partitions contains outer templates
	private final boolean _hasOuter;
	
	public PlanPartition(HashSet<Long> P, HashSet<Long> R, HashSet<Long> I, HashSet<Long> Pnpc, ArrayList<Long> M, InterestingPoint[] Mext, boolean hasOuter) {
		_nodes = P;
		_roots = R;
		_inputs = I;
		_nodesNpc = Pnpc;
		_matPoints = M;
		_matPointsExt = Mext;
		_hasOuter = hasOuter;
	}
	
	public HashSet<Long> getPartition() {
		return _nodes;
	}
	
	public HashSet<Long> getRoots() {
		return _roots;
	}
	
	public HashSet<Long> getInputs() {
		return _inputs;
	}
	
	public HashSet<Long> getExtConsumed() {
		return _nodesNpc;
	} 
	
	public ArrayList<Long> getMatPoints() {
		return _matPoints;
	}
	
	public InterestingPoint[] getMatPointsExt() {
		return _matPointsExt;
	}

	public void setMatPointsExt(InterestingPoint[] points) {
		_matPointsExt = points;
	}
	
	public boolean hasOuter() {
		return _hasOuter;
	}
}
