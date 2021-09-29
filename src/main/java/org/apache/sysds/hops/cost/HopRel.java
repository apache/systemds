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

package org.apache.sysds.hops.cost;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class HopRel {
	protected Hop hopRef;
	protected FEDInstruction.FederatedOutput fedOut;
	protected FederatedCost cost;
	protected Set<Long> costPointerSet = new HashSet<>();
	protected List<HopRel> inputDependency = new ArrayList<>();

	public HopRel(Hop associatedHop, FEDInstruction.FederatedOutput fedOut, Map<Long, List<HopRel>> hopRelMemo){
		hopRef = associatedHop;
		this.fedOut = fedOut;
		setInputDependency(hopRelMemo);
		cost = new FederatedCostEstimator().costEstimate(this, hopRelMemo);
	}

	public void addCostPointer(long hopID){
		costPointerSet.add(hopID);
	}

	/**
	 * Checks if another Hop is refering to this HopRel in memo table.
	 * A reference from a HopRel with same Hop ID is allowed, so this
	 * ID is ignored when checking references.
	 * @param currentHopID to ignore when checking references
	 * @return true if another Hop refers to this HopRel in memo table
	 */
	public boolean existingCostPointer(long currentHopID){
		if ( costPointerSet.contains(currentHopID) )
			return costPointerSet.size() > 1;
		else return costPointerSet.size() > 0;
	}

	public boolean hasLocalOutput(){
		return fedOut == FederatedOutput.LOUT;
	}

	public boolean hasFederatedOutput(){
		return fedOut == FederatedOutput.FOUT;
	}

	public FederatedOutput getFederatedOutput(){
		return fedOut;
	}

	public List<HopRel> getInputDependency(){
		return inputDependency;
	}

	public Hop getHopRef(){
		return hopRef;
	}

	private HopRel getFOUTHopRel(Hop hop, Map<Long, List<HopRel>> hopRelMemo){
		return hopRelMemo.get(hop.getHopID()).stream().filter(in->in.fedOut==FederatedOutput.FOUT).findFirst().orElse(null);
	}

	private HopRel getMinOfInput(Map<Long, List<HopRel>> hopRelMemo, Hop input){
		return hopRelMemo.get(input.getHopID()).stream()
			.min(Comparator.comparingDouble(a -> a.cost.getTotal()))
			.orElseThrow(() -> new DMLException("No element in Memo Table found for input"));
	}

	private void setInputDependency(Map<Long, List<HopRel>> hopRelMemo){
		//TODO: Set inputDependency depending on which inputs are valid and optimal.
		//TODO: isFederatedDataOp may break cost estimation. The cost estimation needs the inputs in the memo table.
		// How can we add the inputs to the memo table for federated dataops?
		// Perhaps we need a completely different cost estimation for this type?
		//!hopRef.isFederatedDataOp() &&
		if (hopRef.getInput() != null && hopRef.getInput().size() > 0) {
			if ( fedOut == FederatedOutput.FOUT && !hopRef.isFederatedDataOp() ) {
				int lowestFOUTIndex = 0;
				HopRel lowestFOUTHopRel = getFOUTHopRel(hopRef.getInput().get(0), hopRelMemo);
				for(int i = 1; i < hopRef.getInput().size(); i++) {
					Hop input = hopRef.getInput(i);
					HopRel foutHopRel = getFOUTHopRel(input, hopRelMemo);
					if(lowestFOUTHopRel == null) {
						lowestFOUTHopRel = foutHopRel;
						lowestFOUTIndex = i;
					}
					else if(foutHopRel != null) {
						if(foutHopRel.getCost() < lowestFOUTHopRel.getCost()) {
							lowestFOUTHopRel = foutHopRel;
							lowestFOUTIndex = i;
						}
					}
				}

				HopRel[] inputHopRels = new HopRel[hopRef.getInput().size()];
				for(int i = 0; i < hopRef.getInput().size(); i++) {
					if(i != lowestFOUTIndex) {
						Hop input = hopRef.getInput(i);
						inputHopRels[i] = getMinOfInput(hopRelMemo, input);
					}
					else {
						inputHopRels[i] = lowestFOUTHopRel;
					}
				}
				inputDependency.addAll(Arrays.asList(inputHopRels));
			} else {
				inputDependency.addAll(
					hopRef.getInput().stream()
						.map(input -> getMinOfInput(hopRelMemo, input))
						.collect(Collectors.toList()));
			}
		}
		validateInputDependency();
	}

	private void validateInputDependency(){
		for ( int i = 0; i < inputDependency.size(); i++){
			if ( inputDependency.get(i) == null)
				throw new DMLException("HopRel input number " + i + " (" + hopRef.getInput(i) + ")"
					+ " is null for root: \n" + this);
		}
	}

	public double getCost(){
		return cost.getTotal();
	}

	public FederatedCost getCostObject(){
		return cost;
	}

	@Override
	public String toString(){
		StringBuilder strB = new StringBuilder();
		strB.append(this.getClass().getSimpleName());
		strB.append(" {HopID: ");
		strB.append(hopRef.getHopID());
		strB.append(", Opcode: ");
		strB.append(hopRef.getOpString());
		strB.append(", FedOut: ");
		strB.append(fedOut);
		strB.append(", Cost: ");
		strB.append(cost);
		strB.append(", Number of inputs: ");
		strB.append(inputDependency.size());
		strB.append("}");
		return strB.toString();
	}
}
