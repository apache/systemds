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

package org.apache.sysds.resource.enumeration;

import java.util.*;

public class GridBasedEnumerator extends Enumerator {
	// marks if the number of executors should
	// be increased by a given step
	private final int stepSizeExecutors;
	// marks if the number of executors should
	// be increased exponentially
	// (single node execution mode is not excluded)
	// -1 marks no exp. increasing
	private final int expBaseExecutors;
	public GridBasedEnumerator(Builder builder, int stepSizeExecutors, int expBaseExecutors) {
		super(builder);
		this.stepSizeExecutors = stepSizeExecutors;
		this.expBaseExecutors = expBaseExecutors;
	}

	/**
	 * Initializes the pool for driver and executor
	 * instances parsed at processing with all the
	 * available instances
	 */
	@Override
	public void preprocessing() {
		driverSpace.initSpace(instances);
		executorSpace.initSpace(instances);
	}

	@Override
	public boolean evaluateSingleNodeExecution(long driverMemory) {
		return minExecutors == 0;
	}

	@Override
	public ArrayList<Integer> estimateRangeExecutors(long executorMemory, int executorCores) {
		// consider the maximum level of parallelism and
		// based on the initiated flags decides for the following methods
		// for enumeration of the number of executors:
		// 1. Increasing the number of executor with given step size (default 1)
		// 2. Exponentially increasing number of executors based on
		//	a given exponent base - with additional option for 0 executors
		int currentMax = Math.min(maxExecutors, MAX_LEVEL_PARALLELISM / executorCores);
		ArrayList<Integer> result;
		if (expBaseExecutors > 1) {
			int maxCapacity = (int) Math.floor(Math.log(currentMax) / Math.log(2));
			result = new ArrayList<>(maxCapacity);
			int exponent = 0;
			int numExecutors;
			while ((numExecutors = (int) Math.pow(expBaseExecutors, exponent)) <= currentMax) {
				if (numExecutors >= minExecutors) {
					result.add(numExecutors);
				}
				exponent++;
			}
		} else {
			int capacity = (int) Math.floor((double) (currentMax - minExecutors + 1) / stepSizeExecutors);
			result = new ArrayList<>(capacity);
			// exclude the 0 from the iteration while keeping it as starting point to ensure predictable steps
			int numExecutors = minExecutors == 0? minExecutors + stepSizeExecutors : minExecutors;
			while (numExecutors <= currentMax) {
				result.add(numExecutors);
				numExecutors += stepSizeExecutors;
			}
		}

		return result;
	}
}
