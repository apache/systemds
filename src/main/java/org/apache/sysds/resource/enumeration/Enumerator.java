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

import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.CloudUtils;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.resource.cost.CostEstimationException;
import org.apache.sysds.resource.cost.CostEstimator;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.resource.enumeration.EnumerationUtils.InstanceSearchSpace;
import org.apache.sysds.resource.enumeration.EnumerationUtils.ConfigurationPoint;
import org.apache.sysds.resource.enumeration.EnumerationUtils.SolutionPoint;

import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

public abstract class Enumerator {

	public enum EnumerationStrategy {
		GridBased, // considering all combinations within a given range of configuration
		InterestBased, // considering only combinations of configurations with memory budge close to memory estimates
		PruneBased // considering potentially all combinations within a given range of configuration but decides for pruning following pre-defined rules
	}

	public enum OptimizationStrategy {
		MinCosts, // use linearized scoring function to minimize both time and price, no constrains apply
		MinTime, // minimize execution time constrained to a given price limit
		MinPrice, // minimize  time constrained to a given price limit
	}

	// Static variables ------------------------------------------------------------------------------------------------
	private static final double LINEAR_OBJECTIVE_RATIO = 0.01; // time/price ratio
	public static final int DEFAULT_MIN_EXECUTORS = 0; // Single Node execution allowed
	/**
	 * A reasonable upper bound for the possible number of executors
	 * is required to set limits for the search space and to avoid
	 * evaluating cluster configurations that most probably would
	 * have too high distribution overhead
	 */
	public static final int DEFAULT_MAX_EXECUTORS = 200;

	/**
	 * A generally applied services quotes in AWS - 1152:
	 * number of vCPUs running at the same time for the account in a single region.
	 * Set 1000 to account for the vCPUs of the driver node
	 */
	public static final int DEFAULT_MAX_LEVEL_PARALLELISM = 1000;

	/** Time/Monetary delta for considering optimal solutions as fraction */
	public static final double COST_DELTA_FRACTION = 0.02;

	// Instance variables ----------------------------------------------------------------------------------------------
	HashMap<String, CloudInstance> instances;
	Program program;
	EnumerationStrategy enumStrategy;
	OptimizationStrategy optStrategy;
	private final double maxTime;
	private final double maxPrice;
	protected final int minExecutors;
	protected final int maxExecutors;
	protected final Set<CloudUtils.InstanceFamily> instanceTypesRange;
	protected final Set<CloudUtils.InstanceSize> instanceSizeRange;

	protected final InstanceSearchSpace driverSpace = new InstanceSearchSpace();
	protected final InstanceSearchSpace executorSpace = new InstanceSearchSpace();
	protected AtomicReference<SolutionPoint> optimalSolution = new AtomicReference<>(null);

	// Initialization functionality ------------------------------------------------------------------------------------

	public Enumerator(Builder builder) {
		this.program = builder.program;
		this.instances = builder.instances;
		this.enumStrategy = builder.enumStrategy;
		this.optStrategy = builder.optStrategy;
		this.maxTime = builder.maxTime;
		this.maxPrice = builder.maxPrice;
		this.minExecutors = builder.minExecutors;
		this.maxExecutors = builder.maxExecutors;
		this.instanceTypesRange = builder.instanceFamiliesRange;
		this.instanceSizeRange = builder.instanceSizeRange;
	}

	// Main functionality ----------------------------------------------------------------------------------------------

	/**
	 * Called once to enumerate the search space for
	 * VM instances for driver or executor nodes.
	 * These instances are being represented as
	 */
	public abstract void preprocessing();

	/**
	 * Called once after preprocessing to fill the
	 * pool with optimal solutions by parsing
	 * the enumerated search space.
	 * Within its execution the number of potential
	 * executor nodes is being estimated (enumerated)
	 * dynamically for each parsed executor instance.
	 */
	public void processing() {
		ConfigurationPoint configurationPoint;
		SolutionPoint initSolutionPoint = new SolutionPoint(
				new ConfigurationPoint(null, null, -1),
				Double.MAX_VALUE,
				Double.MAX_VALUE
		);
		optimalSolution.set(initSolutionPoint);
		for (Map.Entry<Long, TreeMap<Integer, LinkedList<CloudInstance>>> dMemoryEntry: driverSpace.entrySet()) {
			// loop over the search space to enumerate the driver configurations
			for (Map.Entry<Integer, LinkedList<CloudInstance>> dCoresEntry: dMemoryEntry.getValue().entrySet()) {
				// single node execution mode
				if (evaluateSingleNodeExecution(dMemoryEntry.getKey())) {
					program = ResourceCompiler.doFullRecompilation(
							program,
							dMemoryEntry.getKey(),
							dCoresEntry.getKey()
					);
					// no need of recompilation for single nodes with identical memory budget and #v. cores
					for (CloudInstance dInstance: dCoresEntry.getValue()) {
						configurationPoint = new ConfigurationPoint(dInstance);
						updateOptimalSolution(configurationPoint);
					}
				}
				// enumeration for distributed execution
				for (Map.Entry<Long, TreeMap<Integer, LinkedList<CloudInstance>>> eMemoryEntry: executorSpace.entrySet()) {
					// loop over the search space to enumerate the executor configurations
					for (Map.Entry<Integer, LinkedList<CloudInstance>> eCoresEntry: eMemoryEntry.getValue().entrySet()) {
						List<Integer> numberExecutorsSet = estimateRangeExecutors(eMemoryEntry.getKey(), eCoresEntry.getKey());
						// Spark execution mode
						for (int numberExecutors: numberExecutorsSet) {
							program = ResourceCompiler.doFullRecompilation(
									program,
									dMemoryEntry.getKey(),
									dCoresEntry.getKey(),
									numberExecutors,
									eMemoryEntry.getKey(),
									eCoresEntry.getKey()
							);
							// no need of recompilation for a cluster with identical #executors and
							// with identical memory and #v. cores for driver and executor nodes
							for (CloudInstance dInstance: dCoresEntry.getValue()) {
								for (CloudInstance eInstance: eCoresEntry.getValue()) {
									configurationPoint = new ConfigurationPoint(dInstance, eInstance, numberExecutors);
									updateOptimalSolution(configurationPoint);
								}
							}
						}
					}
				}
			}
		}
	}

	/**
	 * Retrieving the estimated optimal configurations after processing.
	 *
	 * @return optimal cluster configuration and corresponding costs
	 */
	public SolutionPoint postprocessing() {
		if (optimalSolution.get() == null) {
			throw new RuntimeException("No solution have met the constrains. Try adjusting the time/price constrain or switch to 'MinCosts' optimization strategy");
		}
		return optimalSolution.get();
	}

	// Helper methods --------------------------------------------------------------------------------------------------

	public abstract boolean evaluateSingleNodeExecution(long driverMemory);

	/**
	 * Estimates the minimum and maximum number of
	 * executors based on given VM instance characteristics
	 * and on the enumeration strategy
	 *
	 * @param executorMemory memory of currently considered executor instance
	 * @param executorCores  CPU of cores of currently considered executor instance
	 * @return - [min, max]
	 */
	public abstract ArrayList<Integer> estimateRangeExecutors(long executorMemory, int executorCores);

	/**
	 * Estimates the time cost for the current program based on the
	 * given cluster configurations and following this estimation
	 * it calculates the corresponding monetary cost.
	 * @param point - cluster configuration used for (re)compiling the current program
	 * @return - [time cost, monetary cost]
	 */
	private double[] getCostEstimate(ConfigurationPoint point) {
		// get the estimated time cost
		double timeCost;
		double monetaryCost;
		try {
			// estimate execution time of the current program
			timeCost = CostEstimator.estimateExecutionTime(program, point.driverInstance, point.executorInstance)
					+ CloudUtils.DEFAULT_CLUSTER_LAUNCH_TIME;
			monetaryCost = CloudUtils.calculateClusterPrice(point, timeCost, CloudUtils.CloudProvider.AWS);
		} catch (CostEstimationException e) {
			timeCost = Double.MAX_VALUE;
			monetaryCost = Double.MAX_VALUE;
		}
		// calculate monetary cost
		return new double[] {timeCost, monetaryCost}; // time cost, monetary cost
	}

	/**
	 * Invokes the estimation of the time and monetary cost
	 * based on the compiled program and the given cluster configurations.
	 * Following the optimization strategy, the given current optimal solution
	 * and the new cost estimation, it decides if the given cluster configuration
	 * can be potential optimal solution having lower cost or such a cost
	 * that is negligibly higher than the current lowest one.
	 *
	 * @param newPoint new cluster configuration for estimation
	 */
	private void updateOptimalSolution(ConfigurationPoint newPoint) {
		SolutionPoint currentOptimal = optimalSolution.get();
		double[] newCost = getCostEstimate(newPoint);
		if (optStrategy == OptimizationStrategy.MinCosts) {
			double optimalScore = linearScoringFunction(currentOptimal.timeCost, currentOptimal.monetaryCost);
			double newScore = linearScoringFunction(newCost[0], newCost[1]);
			if (newScore > optimalScore) {
				return;
			}
			if (newScore == optimalScore && newCost[1] > currentOptimal.monetaryCost) {
				// prioritize cost for the edge case
				return;
			}
		} else if (optStrategy == OptimizationStrategy.MinTime) {
			if (newCost[1] > maxPrice || newCost[0] > currentOptimal.timeCost) {
				return;
			}
			if (newCost[0] == currentOptimal.timeCost && newCost[1] > currentOptimal.monetaryCost) {
				return;
			}
		} else if (optStrategy == OptimizationStrategy.MinPrice) {
			if (newCost[0] > maxTime || newCost[1] > currentOptimal.monetaryCost) {
				return;
			}
			if (newCost[1] == currentOptimal.monetaryCost && newCost[0] > currentOptimal.timeCost) {
				return;
			}
		}
		SolutionPoint newSolution = new SolutionPoint(newPoint, newCost[0], newCost[1]);
		optimalSolution.set(newSolution);
	}

	private static double linearScoringFunction(double time, double price) {
		return LINEAR_OBJECTIVE_RATIO * time + (1 - LINEAR_OBJECTIVE_RATIO) * price;
	}

	// Class builder ---------------------------------------------------------------------------------------------------

	public static class Builder {
		private Program program = null;
		private HashMap<String, CloudInstance> instances = null;
		private EnumerationStrategy enumStrategy = null;
		private OptimizationStrategy optStrategy = null;
		private double maxTime = -1d;
		private double maxPrice = -1d;
		private int minExecutors = DEFAULT_MIN_EXECUTORS;
		private int maxExecutors = DEFAULT_MAX_EXECUTORS;
		private Set<CloudUtils.InstanceFamily> instanceFamiliesRange = null;
		private Set<CloudUtils.InstanceSize> instanceSizeRange = null;

		// GridBased specific ------------------------------------------------------------------------------------------
		private int stepSizeExecutors = 1;
		private int expBaseExecutors = -1; // flag for exp. increasing number of executors if -1
		// InterestBased specific --------------------------------------------------------------------------------------
		private boolean interestLargestEstimate = true;
		private boolean interestEstimatesInCP = true;
		private boolean interestBroadcastVars = true;
		private boolean interestOutputCaching = false; // caching not fully considered by the cost estimator
		public Builder() {}

		public Builder withRuntimeProgram(Program program) {
			this.program = program;
			return this;
		}

		public Builder withAvailableInstances(HashMap<String, CloudInstance> instances) {
			this.instances = instances;
			return this;
		}

		public Builder withEnumerationStrategy(EnumerationStrategy strategy) {
			this.enumStrategy = strategy;
			return this;
		}

		public Builder withOptimizationStrategy(OptimizationStrategy strategy) {
			this.optStrategy = strategy;
			return this;
		}

		public Builder withTimeLimit(double time) {
			if (time < CloudUtils.MINIMAL_EXECUTION_TIME) {
				throw new IllegalArgumentException(CloudUtils.MINIMAL_EXECUTION_TIME +
						"s is the minimum target execution time.");
			}
			this.maxTime = time;
			return this;
		}

		public Builder withBudget(double price) {
			if (price <= 0) {
				throw new IllegalArgumentException("The given budget (target price) should be positive");
			}
			this.maxPrice = price;
			return this;
		}

		public Builder withNumberExecutorsRange(int min, int max) {
			this.minExecutors = min < 0? DEFAULT_MIN_EXECUTORS : min;
			this.maxExecutors = max < 0? DEFAULT_MAX_EXECUTORS : max;
			return this;
		}

		public Builder withInstanceFamilyRange(String[] instanceFamilies) {
			this.instanceFamiliesRange = typeRangeFromStrings(instanceFamilies);
			return this;
		}

		public Builder withInstanceSizeRange(String[] instanceSizes) {
			this.instanceSizeRange = sizeRangeFromStrings(instanceSizes);
			return this;
		}

		public Builder withStepSizeExecutor(int stepSize) {
			this.stepSizeExecutors = stepSize;
			return this;
		}

		public Builder withInterestLargestEstimate(boolean fitSingleNodeMemory) {
			this.interestLargestEstimate = fitSingleNodeMemory;
			return this;
		}

		public Builder withInterestEstimatesInCP(boolean fitDriverMemory) {
			this.interestEstimatesInCP = fitDriverMemory;
			return this;
		}

		public Builder withInterestBroadcastVars(boolean fitExecutorMemory) {
			this.interestBroadcastVars = fitExecutorMemory;
			return this;
		}

		public Builder withInterestOutputCaching(boolean fitCheckpointMemory) {
			this.interestOutputCaching = fitCheckpointMemory;
			return this;
		}

		public Builder withExpBaseExecutors(int expBaseExecutors) {
			if (expBaseExecutors != -1 && expBaseExecutors < 2) {
				throw new IllegalArgumentException("Given exponent base for number of executors should be -1 or bigger than 1.");
			}
			this.expBaseExecutors = expBaseExecutors;
			return this;
		}

		public Enumerator build() {
			if (program == null) {
				throw new IllegalArgumentException("Providing runtime program is required");
			}

			if (instances == null) {
				throw new IllegalArgumentException("Providing available instances is required");
			}

			if (instanceFamiliesRange == null) {
				instanceFamiliesRange = EnumSet.allOf(CloudUtils.InstanceFamily.class);
			}
			if (instanceSizeRange == null) {
				instanceSizeRange = EnumSet.allOf(CloudUtils.InstanceSize.class);
			}
			// filter instances that are not supported or not of the desired type/size
			HashMap<String, CloudInstance> instancesWithinRange = new HashMap<>();
			for (String key: instances.keySet()) {
				if (instanceFamiliesRange.contains(CloudUtils.getInstanceFamily(key))
						&& instanceSizeRange.contains(CloudUtils.getInstanceSize(key))) {
					instancesWithinRange.put(key, instances.get(key));
				}
			}
			instances = instancesWithinRange;

			switch (optStrategy) {
				case MinCosts:
					// no constraints apply
					break;
				case MinTime:
					if (this.maxPrice < 0) {
						throw new IllegalArgumentException("Budget not specified but required " +
								"for the chosen optimization strategy: " + optStrategy);
					}
					break;
				case MinPrice:
					if (this.maxTime < 0) {
						throw new IllegalArgumentException("Time limit not specified but required " +
								"for the chosen optimization strategy: " + optStrategy);
					}
					break;
				default: // in case optimization strategy was not configured
					throw new IllegalArgumentException("Setting an optimization strategy is required.");
			}

			switch (enumStrategy) {
				case GridBased:
					return new GridBasedEnumerator(this, stepSizeExecutors, expBaseExecutors);
				case InterestBased:
					return new InterestBasedEnumerator(this,
							interestLargestEstimate,
							interestEstimatesInCP,
							interestBroadcastVars,
							interestOutputCaching
					);
				default:
					throw new IllegalArgumentException("Setting an enumeration strategy is required.");
			}
		}

		protected static Set<CloudUtils.InstanceFamily> typeRangeFromStrings(String[] types) throws IllegalArgumentException {
			Set<CloudUtils.InstanceFamily> result = EnumSet.noneOf(CloudUtils.InstanceFamily.class);
			for (String typeAsString: types) {
				CloudUtils.InstanceFamily type = CloudUtils.InstanceFamily.customValueOf(typeAsString);
				result.add(type);
			}
			return result;
		}

		protected static Set<CloudUtils.InstanceSize> sizeRangeFromStrings(String[] sizes) throws IllegalArgumentException {
			Set<CloudUtils.InstanceSize> result = EnumSet.noneOf(CloudUtils.InstanceSize.class);
			for (String sizeAsString: sizes) {
				CloudUtils.InstanceSize size = CloudUtils.InstanceSize.customValueOf(sizeAsString);
				result.add(size);
			}
			return result;
		}
	}

	// Public Getters and Setter meant for testing purposes only -------------------------------------------------------

	/**
	 * Meant to be used for testing purposes
	 * @return the available instances for enumeration
	 */
	public HashMap<String, CloudInstance> getInstances() {
		return instances;
	}

	/**
	 * Meant to be used for testing purposes
	 * @return the object representing the driver search space
	 */
	public InstanceSearchSpace getDriverSpace() {
		return driverSpace;
	}

	/**
	 * Meant to be used for testing purposes
	 * @param inputSpace the object representing the driver search space
	 */
	public void setDriverSpace(InstanceSearchSpace inputSpace) {
		driverSpace.putAll(inputSpace);
	}

	/**
	 * Meant to be used for testing purposes
	 * @return the object representing the executor search space
	 */
	public InstanceSearchSpace getExecutorSpace() {
		return executorSpace;
	}

	/**
	 * Meant to be used for testing purposes
	 * @param inputSpace the object representing the executor search space
	 */
	public void setExecutorSpace(InstanceSearchSpace inputSpace) {
		executorSpace.putAll(inputSpace);
	}

	/**
	 * Meant to be used for testing purposes
	 * @return applied enumeration strategy
	 */
	public EnumerationStrategy getEnumStrategy() { return enumStrategy; }

	/**
	 * Meant to be used for testing purposes
	 * @return applied optimization strategy
	 */
	public OptimizationStrategy getOptStrategy() { return optStrategy; }

	/**
	 * Meant to be used for testing purposes
	 * @return configured max time for consideration (seconds)
	 */
	public double getMaxTime() {
		return maxTime;
	}

	/**
	 * Meant to be used for testing purposes
	 * @return configured max price for consideration (dollars)
	 */
	public double getMaxPrice() {
		return maxPrice;
	}
}
