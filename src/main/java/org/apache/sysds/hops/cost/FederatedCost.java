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

/**
 * Class storing execution cost estimates for federated executions with cost estimates split into different categories
 * such as compute, read, and transfer cost.
 */
public class FederatedCost {
	protected double _computeCost = 0;
	protected double _readCost = 0;
	protected double _inputTransferCost = 0;
	protected double _outputTransferCost = 0;
	protected double _inputTotalCost = 0;

	public FederatedCost(){}

	public FederatedCost(double readCost, double inputTransferCost, double outputTransferCost,
		double computeCost, double inputTotalCost){
		_readCost = readCost;
		_inputTransferCost = inputTransferCost;
		_outputTransferCost = outputTransferCost;
		_computeCost = computeCost;
		_inputTotalCost = inputTotalCost;
	}

	/**
	 * Get the total sum of costs stored in this object.
	 * @return total cost
	 */
	public double getTotal(){
		return _computeCost + _readCost + _inputTransferCost + _outputTransferCost + _inputTotalCost;
	}

	/**
	 * Multiply the input costs by the number of times the costs are repeated.
	 * @param repetitionNumber number of repetitions of the costs
	 */
	public void addRepetitionCost(int repetitionNumber){
		_inputTotalCost *= repetitionNumber;
	}

	/**
	 * Get summed input costs.
	 * @return summed input costs
	 */
	public double getInputTotalCost(){
		return _inputTotalCost;
	}

	public void setInputTotalCost(double inputTotalCost){
		_inputTotalCost = inputTotalCost;
	}

	/**
	 * Add cost to the stored input cost.
	 * @param additionalCost to add to total input cost
	 */
	public void addInputTotalCost(double additionalCost){
		_inputTotalCost += additionalCost;
	}

	/**
	 * Add total of federatedCost to stored inputTotalCost.
	 * @param federatedCost input cost from which the total is retrieved
	 */
	public void addInputTotalCost(FederatedCost federatedCost){
		_inputTotalCost += federatedCost.getTotal();
	}

	/**
	 * Add costs of FederatedCost object to this object's current costs.
	 * @param additionalCost object to add to this object
	 */
	public void addFederatedCost(FederatedCost additionalCost){
		_readCost += additionalCost._readCost;
		_inputTransferCost += additionalCost._inputTransferCost;
		_outputTransferCost += additionalCost._outputTransferCost;
		_computeCost += additionalCost._computeCost;
		_inputTotalCost += additionalCost._inputTotalCost;
	}

	@Override
	public String toString(){
		StringBuilder builder = new StringBuilder();
		builder.append(" computeCost: ");
		builder.append(_computeCost);
		builder.append("\n readCost: ");
		builder.append(_readCost);
		builder.append("\n inputTransferCost: ");
		builder.append(_inputTransferCost);
		builder.append("\n outputTransferCost: ");
		builder.append(_outputTransferCost);
		builder.append("\n inputTotalCost: ");
		builder.append(_inputTotalCost);
		builder.append("\n total cost: ");
		builder.append(getTotal());
		return builder.toString();
	}
}
