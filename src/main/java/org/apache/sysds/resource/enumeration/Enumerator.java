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
import org.apache.sysds.resource.CloudUtils.InstanceFamily;
import org.apache.sysds.resource.CloudUtils.InstanceSize;
import org.apache.sysds.resource.CloudUtils.CloudProvider;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.resource.cost.CostEstimationException;
import org.apache.sysds.resource.cost.CostEstimator;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.resource.enumeration.EnumerationUtils.InstanceSearchSpace;
import org.apache.sysds.resource.enumeration.EnumerationUtils.ConfigurationPoint;
import org.apache.sysds.resource.enumeration.EnumerationUtils.SolutionPoint;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicReference;

public abstract class Enumerator {

	public enum EnumerationStrategy {
		// considering all combinations within a given range of configurations
		GridBased,
		// considering only combinations of configurations with memory budge close to program memory estimates
		InterestBased,
		// considering potentially all combinations within a given range of configurations
		// but decides for pruning following pre-defined rules
		PruneBased
	}

	public enum OptimizationStrategy {
		MinCosts, // use linearized scoring function to minimize both time and price, no constrains apply
		MinTime, // minimize execution time constrained to a given price limit
		MinPrice, // minimize  time constrained to a given price limit
	}

	// Static variables ------------------------------------------------------------------------------------------------
	public static final int DEFAULT_MIN_EXECUTORS = 0; // Single Node execution allowed
	/**
	 * A reasonable upper bound for the possible number of executors
	 * is required to set limits for the search space and to avoid
	 * evaluating cluster configurations that most probably would
	 * have too high distribution overhead
	 */
	public static final int DEFAULT_MAX_EXECUTORS = 200;
	/** Time/Monetary delta for considering optimal solutions as fraction */
	public static final double COST_DELTA_FRACTION = 0.02;
	/**
	 * A generally applied services quotes in AWS - 1152:
	 * number of vCPUs running at the same time for the account in a single region.
	 */
	// Static configurations -------------------------------------------------------------------------------------------
	static double LINEAR_OBJECTIVE_RATIO = 0.01; // time/price ratio
	static double MAX_TIME = Double.MAX_VALUE; // no limit by default
	static double MAX_PRICE = Double.MAX_VALUE; // no limit by default
	static int CPU_QUOTA = 1152;
	// allow changing the default quota value
	public static void setCostsWeightFactor(double newFactor) { LINEAR_OBJECTIVE_RATIO = newFactor; }
	public static void setMinTime(double maxTime) { MAX_TIME = maxTime; }
	public static void setMinPrice(double maxPrice) { MAX_PRICE = maxPrice; }
	public static void setCpuQuota(int newQuotaValue) { CPU_QUOTA = newQuotaValue; }

	// Instance variables ----------------------------------------------------------------------------------------------
	HashMap<String, CloudInstance> instances;
	Program program;
	EnumerationStrategy enumStrategy;
	OptimizationStrategy optStrategy;
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
		this.minExecutors = builder.minExecutors;
		this.maxExecutors = builder.maxExecutors;
		this.instanceTypesRange = builder.instanceFamiliesRange;
		this.instanceSizeRange = builder.instanceSizeRange;
		// init optimal solution here to allow errors at comparing before the first update
		SolutionPoint initSolutionPoint = new SolutionPoint(
				new ConfigurationPoint(null, null, -1),
				Double.MAX_VALUE,
				Double.MAX_VALUE
		);
		optimalSolution.set(initSolutionPoint);
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
		long driverMemory, executorMemory;
		int driverCores, executorCores;
		ConfigurationPoint configurationPoint;

		for (Entry<Long, TreeMap<Integer, LinkedList<CloudInstance>>> dMemoryEntry: driverSpace.entrySet()) {
			driverMemory = dMemoryEntry.getKey();
			// loop over the search space to enumerate the driver configurations
			for (Entry<Integer, LinkedList<CloudInstance>> dCoresEntry: dMemoryEntry.getValue().entrySet()) {
				driverCores = dCoresEntry.getKey();
				// single node execution mode
				if (evaluateSingleNodeExecution(driverMemory, driverCores)) {
					ResourceCompiler.setSingleNodeResourceConfigs(driverMemory, driverCores);
					program = ResourceCompiler.doFullRecompilation(program);
					// no need of recompilation for single nodes with identical memory budget and #v. cores
					for (CloudInstance dInstance: dCoresEntry.getValue()) {
						// iterate over all driver nodes with the currently evaluated memory and #cores values
						configurationPoint = new ConfigurationPoint(dInstance);
						double[] newEstimates = getCostEstimate(configurationPoint);
						updateOptimalSolution(newEstimates[0], newEstimates[1], configurationPoint);
					}
				}
				// enumeration for distributed execution
				for (Entry<Long, TreeMap<Integer, LinkedList<CloudInstance>>> eMemoryEntry: executorSpace.entrySet()) {
					executorMemory = eMemoryEntry.getKey();
					// loop over the search space to enumerate the executor configurations
					for (Entry<Integer, LinkedList<CloudInstance>> eCoresEntry: eMemoryEntry.getValue().entrySet()) {
						executorCores = eCoresEntry.getKey();
						List<Integer> numberExecutorsSet =
								estimateRangeExecutors(driverCores, executorMemory, executorCores);
						// for Spark execution mode
						for (int numberExecutors: numberExecutorsSet) {
							try {
								ResourceCompiler.setSparkClusterResourceConfigs(
										driverMemory,
										driverCores,
										numberExecutors,
										executorMemory,
										executorCores
								);
							} catch (IllegalArgumentException e) {
								// insufficient driver memory detected
								break;
							}
							program = ResourceCompiler.doFullRecompilation(program);
//							System.out.println(Explain.explain(program));
							// no need of recompilation for a cluster with identical #executors and
							// with identical memory and #v. cores for driver and executor nodes
							for (CloudInstance dInstance: dCoresEntry.getValue()) {
								// iterate over all driver nodes with the currently evaluated memory and #cores values
								for (CloudInstance eInstance: eCoresEntry.getValue()) {
									// iterate over all executor nodes for the evaluated cluster size
									// with the currently evaluated memory and #cores values
									configurationPoint = new ConfigurationPoint(dInstance, eInstance, numberExecutors);
									double[] newEstimates = getCostEstimate(configurationPoint);
									updateOptimalSolution(newEstimates[0], newEstimates[1], configurationPoint);
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
			throw new RuntimeException("No solution have met the constrains. " +
					"Try adjusting the time/price constrain or switch to 'MinCosts' optimization strategy");
		}
		return optimalSolution.get();
	}

	// Helper methods --------------------------------------------------------------------------------------------------

	public abstract boolean evaluateSingleNodeExecution(long driverMemory, int cores);

	/**
	 * Estimates the minimum and maximum number of
	 * executors based on given VM instance characteristics,
	 * the enumeration strategy and the user-defined configurations
	 *
	 * @param driverCores CPU cores for the currently evaluated driver node
	 * @param executorMemory memory of currently evaluated executor node
	 * @param executorCores  CPU cores of currently evaluated executor node
	 * @return - [min, max]
	 */
	public abstract ArrayList<Integer> estimateRangeExecutors(int driverCores, long executorMemory, int executorCores);

	/**
	 * Estimates the time cost for the current program based on the
	 * given cluster configurations and following this estimation
	 * it calculates the corresponding monetary cost.
	 * @param point cluster configuration used for (re)compiling the current program
	 * @return - [time cost, monetary cost]
	 */
	protected double[] getCostEstimate(ConfigurationPoint point) {
		// get the estimated time cost
		double timeCost;
		double monetaryCost;
		try {
			// estimate execution time of the current program
			timeCost = CostEstimator.estimateExecutionTime(program, point.driverInstance, point.executorInstance)
					+ CloudUtils.DEFAULT_CLUSTER_LAUNCH_TIME;
			monetaryCost = CloudUtils.calculateClusterPrice(point, timeCost, CloudProvider.AWS);
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
	 * @param newTimeEstimate     estimated time cost for the given configurations
	 * @param newMonetaryEstimate estimated monetary cost for the given configurations
	 * @param newPoint            new cluster configuration for estimation
	 */
	public void updateOptimalSolution(double newTimeEstimate, double newMonetaryEstimate, ConfigurationPoint newPoint) {
		SolutionPoint currentOptimal = optimalSolution.get();
		if (optStrategy == OptimizationStrategy.MinCosts) {
			double optimalScore = linearScoringFunction(currentOptimal.timeCost, currentOptimal.monetaryCost);
			double newScore = linearScoringFunction(newTimeEstimate, newMonetaryEstimate);
			if (newScore > optimalScore) {
				return;
			}
			if (newScore == optimalScore && newMonetaryEstimate >= currentOptimal.monetaryCost) {
				// prioritize cost for the edge case
				return;
			}
		} else if (optStrategy == OptimizationStrategy.MinTime) {
			if (newMonetaryEstimate > MAX_PRICE || newTimeEstimate > currentOptimal.timeCost) {
				return;
			}
			if (newTimeEstimate == currentOptimal.timeCost && newMonetaryEstimate > currentOptimal.monetaryCost) {
				return;
			}
		} else if (optStrategy == OptimizationStrategy.MinPrice) {
			if (newTimeEstimate > MAX_TIME || newMonetaryEstimate > currentOptimal.monetaryCost) {
				return;
			}
			if (newMonetaryEstimate == currentOptimal.monetaryCost && newTimeEstimate > currentOptimal.timeCost) {
				return;
			}
		}
		SolutionPoint newSolution = new SolutionPoint(newPoint, newTimeEstimate, newMonetaryEstimate);
		optimalSolution.set(newSolution);
	}

	static double linearScoringFunction(double time, double price) {
		return LINEAR_OBJECTIVE_RATIO * time + (1 - LINEAR_OBJECTIVE_RATIO) * price;
	}

	// Class builder ---------------------------------------------------------------------------------------------------

	public static class Builder {
		private Program program = null;
		private HashMap<String, CloudInstance> instances = null;
		private EnumerationStrategy enumStrategy = null;
		private OptimizationStrategy optStrategy = null;
		private int minExecutors = DEFAULT_MIN_EXECUTORS;
		private int maxExecutors = DEFAULT_MAX_EXECUTORS;
		private Set<InstanceFamily> instanceFamiliesRange = null;
		private Set<InstanceSize> instanceSizeRange = null;

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
				throw new IllegalArgumentException(
						"Given exponent base for number of executors should be -1 or bigger than 1."
				);
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
				instanceFamiliesRange = EnumSet.allOf(InstanceFamily.class);
			}
			if (instanceSizeRange == null) {
				instanceSizeRange = EnumSet.allOf(InstanceSize.class);
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
				case PruneBased:
					return new PruneBasedEnumerator(this);
				default:
					throw new IllegalArgumentException("Setting an enumeration strategy is required.");
			}
		}

		protected static Set<InstanceFamily> typeRangeFromStrings(String[] types) throws IllegalArgumentException {
			Set<InstanceFamily> result = EnumSet.noneOf(InstanceFamily.class);
			for (String typeAsString: types) {
				InstanceFamily type = InstanceFamily.customValueOf(typeAsString);
				result.add(type);
			}
			return result;
		}

		protected static Set<InstanceSize> sizeRangeFromStrings(String[] sizes) throws IllegalArgumentException {
			Set<InstanceSize> result = EnumSet.noneOf(InstanceSize.class);
			for (String sizeAsString: sizes) {
				InstanceSize size = InstanceSize.customValueOf(sizeAsString);
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
	 * @return configured weight factor optimization function 'costs'
	 */
	public double getCostsWeightFactor() {
		return Enumerator.LINEAR_OBJECTIVE_RATIO;
	}

	/**
	 * Meant to be used for testing purposes
	 * @return configured max time for consideration (seconds)
	 */
	public double getMaxTime() {
		return Enumerator.MAX_TIME;
	}

	/**
	 * Meant to be used for testing purposes
	 * @return configured max price for consideration (dollars)
	 */
	public double getMaxPrice() {
		return Enumerator.MAX_PRICE;
	}

	/**
	 * Meant to be used for testing purposes
	 * @return current optimal solution
	 */
	public SolutionPoint getOptimalSolution() {
		return optimalSolution.get();
	}
}
