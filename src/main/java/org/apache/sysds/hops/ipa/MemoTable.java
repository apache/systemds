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

package org.apache.sysds.hops.ipa;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.cost.HopRel;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Memoization of federated execution alternatives.
 * This memoization data structure is used when generating optimal federated execution plans.
 * The alternative executions are stored as HopRels and the methods of this class are used to
 * add, update, and retrieve the alternatives.
 */
public class MemoTable {
	//TODO refactoring: could we generalize the privacy and codegen memo tables into 
	// a generic implementation (e.g., MemoTable<HopRel>) that can be reused in both? 
	
	/**
	 * Map holding the relation between Hop IDs and execution plan alternatives.
	 */
	private final static Map<Long, List<HopRel>> hopRelMemo = new HashMap<>();

	/**
	 * Get the HopRel with minimum cost for given root hop
	 * @param root hop for which minimum cost HopRel is found
	 * @return HopRel with minimum cost for given hop
	 */
	public HopRel getMinCostAlternative(Hop root){
		return hopRelMemo.get(root.getHopID()).stream()
			.min(Comparator.comparingDouble(HopRel::getCost))
			.orElseThrow(() -> new DMLException("Hop root " + root + " has no feasible federated output alternatives"));
	}

	/**
	 * Checks if any of the federated execution alternatives for the given root hop has federated output.
	 * @param root hop for which execution alternatives are checked
	 * @return true if root has federated output as an execution alternative
	 */
	public boolean hasFederatedOutputAlternative(Hop root){
		return hopRelMemo.get(root.getHopID()).stream().anyMatch(HopRel::hasFederatedOutput);
	}

	/**
	 * Get the federated output alternative for given root hop or throw exception if not found.
	 * @param root hop for which federated output HopRel is returned
	 * @return federated output HopRel for given root hop
	 */
	public HopRel getFederatedOutputAlternative(Hop root){
		return getFederatedOutputAlternativeOptional(root).orElseThrow(
			() -> new DMLException("Hop root " + root + " has no FOUT alternative"));
	}

	/**
	 * Get the federated output alternative for given root hop or null if not found.
	 * @param root hop for which federated output HopRel is returned
	 * @return federated output HopRel for given root hop
	 */
	public HopRel getFederatedOutputAlternativeOrNull(Hop root){
		return getFederatedOutputAlternativeOptional(root).orElse(null);
	}

	private Optional<HopRel> getFederatedOutputAlternativeOptional(Hop root){
		return hopRelMemo.get(root.getHopID()).stream().filter(HopRel::hasFederatedOutput).findFirst();
	}

	/**
	 * Memoize hopRels related to given root.
	 * @param root for which hopRels are added
	 * @param hopRels execution alternatives related to the given root
	 */
	public void put(Hop root, List<HopRel> hopRels){
		hopRelMemo.put(root.getHopID(), hopRels);
	}

	/**
	 * Checks if root hop has been added to memo.
	 * @param root hop
	 * @return true if root has been added to memo.
	 */
	public boolean containsHop(Hop root){
		return hopRelMemo.containsKey(root.getHopID());
	}

	/**
	 * Checks if given HopRel has been added to memo.
	 * @param root HopRel
	 * @return true if root HopRel has been added to memo.
	 */
	public boolean containsHopRel(HopRel root){
		return containsHop(root.getHopRef())
			&& hopRelMemo.get(root.getHopRef().getHopID()).stream()
			.anyMatch(h -> h.getFederatedOutput() == root.getFederatedOutput());
	}
}
