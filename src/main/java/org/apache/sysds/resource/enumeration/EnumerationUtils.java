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

import java.util.*;

public class EnumerationUtils {
	/**
	 * Data structure representing a projected search space for
	 * VM instances as node's memory mapped to further maps with
	 * the node's numbers of cores for the given memory
	 * mapped to a list of unique object of type {@code CloudInstance}
	 * which have this corresponding characteristics (memory and cores).
	 * The higher layer keep the memory since is more significant
	 * for the program compilation. The lower map level contains
	 * the different options for number of core for the memory that
	 * this map data structure is being mapped to. The last layer
	 * of LinkedLists represents the unique VM instances in lists
	 * since the memory - cores combinations is often not unique.
	 * The {@code CloudInstance} objects are unique over the whole
	 * set of lists within this lowest level of the search space.
	 * <br>
	 * This representation allows compact storing of VM instance
	 * characteristics relevant for program compilation while
	 * still keeping a reference to the object carrying the
	 * whole instance information, relevant for cost estimation.
	 * <br>
	 * {@code TreeMap} data structures are used as building blocks for
	 * the complex search space structure to ensure ascending order
	 * of the instance characteristics - memory and number of cores.
	 */
	public static class InstanceSearchSpace extends TreeMap<Long, TreeMap<Integer, LinkedList<CloudInstance>>> {
		private static final long serialVersionUID = -8855424955793322839L;

		public void initSpace(HashMap<String, CloudInstance> instances) {
			for (CloudInstance instance: instances.values()) {
				long currentMemory = instance.getMemory();

				this.putIfAbsent(currentMemory, new TreeMap<>());
				TreeMap<Integer, LinkedList<CloudInstance>> currentSubTree = this.get(currentMemory);

				currentSubTree.putIfAbsent(instance.getVCPUs(), new LinkedList<>());
				LinkedList<CloudInstance> currentList = currentSubTree.get(instance.getVCPUs());

				currentList.add(instance);
				// ensure total order based on price (ascending)
				currentList.sort(Comparator.comparingDouble(CloudInstance::getPrice));
			}
		}
	}

	/**
	 * Simple data structure to hold cluster configurations
	 */
	public static class ConfigurationPoint {
		public CloudInstance driverInstance;
		public CloudInstance executorInstance;
		public int numberExecutors;

		public ConfigurationPoint(CloudInstance driverInstance) {
			this.driverInstance = driverInstance;
			this.executorInstance = null;
			this.numberExecutors = 0;
		}

		public ConfigurationPoint(CloudInstance driverInstance, CloudInstance executorInstance, int numberExecutors) {
			this.driverInstance = driverInstance;
			this.executorInstance = executorInstance;
			this.numberExecutors = numberExecutors;
		}

		@Override
		public String toString() {
			StringBuilder builder = new StringBuilder();
			builder.append("Driver: ").append(driverInstance.getInstanceName());
			builder.append("\n	mem: ").append((double) driverInstance.getMemory()/(1024*1024*1024));
			builder.append(", v. cores: ").append(driverInstance.getVCPUs());
			builder.append("\nExecutors: ");
			if (numberExecutors > 0) {
				builder.append(numberExecutors).append(" x ").append(executorInstance.getInstanceName());
				builder.append("\n	mem: ").append((double) executorInstance.getMemory()/(1024*1024*1024));
				builder.append(", v. cores: ").append(executorInstance.getVCPUs());
			} else {
				builder.append("-");
			}
			return builder.toString();
		}
	}

	/**
	 * Data structure to hold all data related to cost estimation
	 */
	public static class SolutionPoint extends ConfigurationPoint {
		double timeCost;
		double monetaryCost;

		public SolutionPoint(ConfigurationPoint inputPoint, double timeCost, double monetaryCost) {
			super(inputPoint.driverInstance, inputPoint.executorInstance, inputPoint.numberExecutors);
			this.timeCost = timeCost;
			this.monetaryCost = monetaryCost;
		}

		public void update(ConfigurationPoint point, double timeCost, double monetaryCost) {
			this.driverInstance = point.driverInstance;
			this.executorInstance = point.executorInstance;
			this.numberExecutors = point.numberExecutors;
			this.timeCost = timeCost;
			this.monetaryCost = monetaryCost;
		}

		public double getTimeCost() {
			return timeCost;
		}

		public double getMonetaryCost() {
			return monetaryCost;
		}
	}
}
