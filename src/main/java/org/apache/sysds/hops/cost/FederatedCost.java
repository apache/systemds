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

public class FederatedCost {
	protected double computeCost = 0;
	protected double readCost = 0;
	protected double inputTransferCost = 0;
	protected double outputTransferCost = 0;
	protected double inputTotalCost = 0;

	public FederatedCost(){

	}

	public FederatedCost(double readCost, double inputTransferCost, double outputTransferCost,
		double computeCost, double inputTotalCost){
		this.readCost = readCost;
		this.inputTransferCost = inputTransferCost;
		this.outputTransferCost = outputTransferCost;
		this.computeCost = computeCost;
		this.inputTotalCost = inputTotalCost;
	}

	public double getTotal(){
		return computeCost + readCost + inputTransferCost + outputTransferCost + inputTotalCost;
	}

	public void addRepititionCost(int repititionNumber){
		this.inputTotalCost = inputTotalCost * repititionNumber;
	}

	public double getInputTotalCost(){
		return inputTotalCost;
	}

	public void setInputTotalCost(double inputTotalCost){
		this.inputTotalCost = inputTotalCost;
	}

	public void addInputTotalCost(double additionalCost){
		this.inputTotalCost += additionalCost;
	}

	public void addFederatedCost(FederatedCost additionalCost){
		this.readCost += additionalCost.readCost;
		this.inputTransferCost += additionalCost.inputTransferCost;
		this.outputTransferCost += additionalCost.outputTransferCost;
		this.computeCost += additionalCost.computeCost;
		this.inputTotalCost += additionalCost.inputTotalCost;
	}

	@Override
	public String toString(){
		StringBuilder builder = new StringBuilder();
		builder.append("computeCost: ");
		builder.append(computeCost);
		builder.append("\n readCost: ");
		builder.append(readCost);
		builder.append("\n inputTransferCost: ");
		builder.append(inputTransferCost);
		builder.append("\n outputTransferCost: ");
		builder.append(outputTransferCost);
		builder.append("\n inputTotalCost: ");
		builder.append(inputTotalCost);
		builder.append("\n total cost: ");
		builder.append(getTotal());
		return builder.toString();
	}
}
