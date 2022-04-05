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
import org.apache.sysds.hops.fedplanner.FTypes;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.hops.fedplanner.MemoTable;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * HopRel provides a representation of the relation between a hop, the cost of setting a given FederatedOutput value,
 * and the input dependency with the given FederatedOutput value.
 * The HopRel class is used when building and selecting an optimal federated execution plan in IPAPassRewriteFederatedPlan.
 * The input dependency is needed to hold the valid and optimal FederatedOutput values for the inputs.
 */
public class HopRel {
	protected final Hop hopRef;
	protected final FEDInstruction.FederatedOutput fedOut;
	protected FTypes.FType fType;
	protected final FederatedCost cost;
	protected final Set<Long> costPointerSet = new HashSet<>();
	protected List<Hop> inputHops;
	protected List<HopRel> inputDependency = new ArrayList<>();

	/**
	 * Constructs a HopRel with input dependency and cost estimate based on entries in hopRelMemo.
	 * @param associatedHop hop associated with this HopRel
	 * @param fedOut FederatedOutput value assigned to this HopRel
	 * @param hopRelMemo memo table storing other HopRels including the inputs of associatedHop
	 */
	public HopRel(Hop associatedHop, FEDInstruction.FederatedOutput fedOut, MemoTable hopRelMemo){
		this(associatedHop, fedOut, null, hopRelMemo,associatedHop.getInput());
	}

	/**
	 * Constructs a HopRel with input dependency and cost estimate based on entries in hopRelMemo.
	 * @param associatedHop hop associated with this HopRel
	 * @param fedOut FederatedOutput value assigned to this HopRel
	 * @param hopRelMemo memo table storing other HopRels including the inputs of associatedHop
	 * @param inputs hop inputs which input dependencies and cost is based on
	 */
	public HopRel(Hop associatedHop, FEDInstruction.FederatedOutput fedOut, MemoTable hopRelMemo, ArrayList<Hop> inputs){
		this(associatedHop, fedOut, null, hopRelMemo, inputs);
	}

	/**
	 * Constructs a HopRel with input dependency and cost estimate based on entries in hopRelMemo.
	 * @param associatedHop hop associated with this HopRel
	 * @param fedOut FederatedOutput value assigned to this HopRel
	 * @param fType Federated Type of the output of this hopRel
	 * @param hopRelMemo memo table storing other HopRels including the inputs of associatedHop
	 * @param inputs hop inputs which input dependencies and cost is based on
	 */
	public HopRel(Hop associatedHop, FEDInstruction.FederatedOutput fedOut, FType fType, MemoTable hopRelMemo, ArrayList<Hop> inputs){
		hopRef = associatedHop;
		this.fedOut = fedOut;
		this.fType = fType;
		this.inputHops = inputs;
		setInputDependency(hopRelMemo);
		cost = FederatedCostEstimator.costEstimate(this, hopRelMemo);
	}

	public HopRel(Hop associatedHop, FEDInstruction.FederatedOutput fedOut, FType fType, MemoTable hopRelMemo, List<Hop> inputs, List<FType> inputDependency){
		hopRef = associatedHop;
		this.fedOut = fedOut;
		this.inputHops = inputs;
		this.fType = fType;
		setInputFTypeDependency(inputs, inputDependency, hopRelMemo);
		cost = FederatedCostEstimator.costEstimate(this, hopRelMemo);
	}

	private void setInputFTypeDependency(List<Hop> inputs, List<FType> inputDependency, MemoTable hopRelMemo){
		for ( int i = 0; i < inputs.size(); i++ ){
			this.inputDependency.add(hopRelMemo.getHopRel(inputs.get(i), inputDependency.get(i)));
		}
		validateInputDependency();
	}

	/**
	 * Adds hopID to set of hops pointing to this HopRel.
	 * By storing the hopID it can later be determined if the cost
	 * stored in this HopRel is already used as input cost in another HopRel.
	 * @param hopID added to set of stored cost pointers
	 */
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

	public FType getFType(){
		return fType;
	}

	/**
	 * Returns FOUT HopRel for given hop found in hopRelMemo or returns null if HopRel not found.
	 * @param hop to look for in hopRelMemo
	 * @param hopRelMemo memo table storing HopRels
	 * @return FOUT HopRel found in hopRelMemo
	 */
	private HopRel getFOUTHopRel(Hop hop, MemoTable hopRelMemo){
		return hopRelMemo.getFederatedOutputAlternativeOrNull(hop);
	}

	/**
	 * Set valid and optimal input dependency for this HopRel as a field.
	 * @param hopRelMemo memo table storing input HopRels
	 */
	private void setInputDependency(MemoTable hopRelMemo){
		if (inputHops != null && inputHops.size() > 0) {
			if ( fedOut == FederatedOutput.FOUT && !hopRef.isFederatedDataOp() ) {
				int lowestFOUTIndex = 0;
				HopRel lowestFOUTHopRel = getFOUTHopRel(inputHops.get(0), hopRelMemo);
				for(int i = 1; i < inputHops.size(); i++) {
					Hop input = inputHops.get(i);
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

				HopRel[] inputHopRels = new HopRel[inputHops.size()];
				for(int i = 0; i < inputHops.size(); i++) {
					if(i != lowestFOUTIndex) {
						Hop input = inputHops.get(i);
						inputHopRels[i] = hopRelMemo.getMinCostAlternative(input);
					}
					else {
						inputHopRels[i] = lowestFOUTHopRel;
					}
				}
				inputDependency.addAll(Arrays.asList(inputHopRels));
			} else {
				inputDependency.addAll(
					inputHops.stream()
						.map(hopRelMemo::getMinCostAlternative)
						.collect(Collectors.toList()));
			}
		}
		validateInputDependency();
	}

	/**
	 * Throws exception if any input dependency is null.
	 * If any of the input dependencies are null, it is not possible to build a federated execution plan.
	 * If this null-state is not found here, an exception will be thrown at another difficult-to-debug place.
	 */
	private void validateInputDependency(){
		for ( int i = 0; i < inputDependency.size(); i++){
			if ( inputDependency.get(i) == null)
				throw new DMLException("HopRel input number " + i + " (" + hopRef.getInput(i) + ")"
					+ " is null for root: \n" + this);
		}
	}

	/**
	 * Get total cost as double
	 * @return cost as double
	 */
	public double getCost(){
		return cost.getTotal();
	}

	/**
	 * Get cost object
	 * @return cost object
	 */
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
		strB.append(cost.getTotal());
		strB.append(", Inputs: ");
		strB.append(inputDependency.stream().map(i -> "{" + i.getHopRef().getHopID() +
			", " + i.getFederatedOutput() + "}").collect(Collectors.toList()));
		strB.append("}");
		return strB.toString();
	}
}
