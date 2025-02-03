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

package org.apache.sysds.lops.compile.linearization;

import java.util.List;

import org.apache.sysds.lops.Lop;

public class LinearizerResourceAwareOptimal extends IDagLinearizer {

	@Override
	public List<Lop> linearize(List<Lop> v) {
		return v;
	}
	
// FIXME this implementation does not terminate (even for toy examples)
	
//	static class OptimalList {
//		List<Lop> optimalSequence;
//		double maxMemoryUsage;
//
//		public OptimalList() {
//			optimalSequence = new ArrayList<>();
//			maxMemoryUsage = -1;
//		}
//
//		public void setNewValues(List<Lop> optimalSequence, double maxMemoryUsage) {
//			this.optimalSequence = optimalSequence;
//			this.maxMemoryUsage = maxMemoryUsage;
//		}
//
//		public double getMaxMemoryUsage() {
//			return maxMemoryUsage;
//		}
//
//		public List<Lop> getOptimalSequence() {
//			return optimalSequence;
//		}
//	}
//
//	static class Dependency {
//		Lop node;
//		Lop dependsOn;
//
//		public Dependency(Lop node, Lop dependency) {
//			this.node = node;
//			this.dependsOn = dependency;
//		}
//
//		public Lop getDependency() {
//			return dependsOn;
//		}
//
//		public Lop getNode() {
//			return node;
//		}
//	}
//
//	static class MemoryEntry {
//		Set<Long> targets;
//		double requiredMemory;
//
//		public MemoryEntry(Set<Long> targets, double requiredMemory) {
//			this.targets = targets;
//			this.requiredMemory = requiredMemory;
//		}
//
//		public Set<Long> getTargets() {
//			return targets;
//		}
//
//		public void removeTarget(long target) {
//			targets.remove(target);
//		}
//
//		public double getRequiredMemory() {
//			return requiredMemory;
//		}
//	}
//
//	Set<Dependency> dependencies = new HashSet<>();
//	OptimalList optimalList = new OptimalList();
//
//	@Override
//	public List<Lop> linearize(List<Lop> v) {
//		// At first, we want to find all transitive dependencies between all nodes
//		// We will take every node and recursively search for all lops our node depends on.
//		// Then we create a set with all does dependencies.
//		v.forEach(lop -> {
//			Set<Lop> dependencyNodes = getDependenciesOfNode(lop);
//			dependencyNodes.forEach(dependencyNode -> {
//				dependencies.add(new Dependency(lop, dependencyNode));
//			});
//		});
//
//		// The Lops without outputs need to stay in order because they could be f.e. prints.
//		// Therefore, we create a dependency chain between all lops without outputs to let them stay in order.
//		List<Lop> outputLops = v.stream().filter(lop -> lop.getOutputs().isEmpty()).collect(Collectors.toList());
//		for(int i = 0; i < outputLops.size() - 1; i++) {
//			dependencies.add(new Dependency(outputLops.get(i + 1), outputLops.get(i)));
//		}
//
//		// In the next step we want to discover all possible permutations
//		// and find the one with the least memory requirement.
//		findBestSequence(new ArrayList<>(), v);
//
//		return optimalList.getOptimalSequence();
//	}
//
//	Set<Lop> getDependenciesOfNode(Lop node) {
//		Set<Lop> dependencyNodes = new HashSet<>();
//
//		node.getInputs().forEach(input -> {
//			dependencyNodes.addAll(getDependenciesOfNode(input));
//		});
//		dependencyNodes.addAll(node.getInputs());
//
//		return dependencyNodes;
//	}
//
//	boolean isItemInDependencyList(Dependency item) {
//		return dependencies.stream().anyMatch(i -> i.getDependency().getID() == item.getDependency().getID() &&
//			i.getNode().getID() == item.getNode().getID());
//	}
//
//	void findBestSequence(List<Lop> visitedLops, List<Lop> remainingLops) {
//		if(remainingLops.isEmpty()) {
//			double maxMemoryUsage = getMemoryUsage(visitedLops);
//			if(optimalList.getOptimalSequence().isEmpty() || maxMemoryUsage < optimalList.getMaxMemoryUsage()) {
//				optimalList.setNewValues(visitedLops, maxMemoryUsage);
//			}
//		}
//
//		remainingLops.parallelStream().forEach(entry -> {
//			boolean dependencyViolation = false;
//
//			for(Lop remainingLop : remainingLops) {
//				if(entry.getID() != remainingLop.getID() &&
//					isItemInDependencyList(new Dependency(entry, remainingLop))) {
//					dependencyViolation = true;
//					break;
//				}
//			}
//
//			if(!dependencyViolation) {
//				List<Lop> newVisitedLops = new ArrayList<>(visitedLops);
//				List<Lop> newRemainingLops = new ArrayList<>(remainingLops);
//
//				newVisitedLops.add(entry);
//				newRemainingLops.remove(entry);
//
//				findBestSequence(newVisitedLops, newRemainingLops);
//			}
//		});
//	}
//
//	double getMemoryUsage(List<Lop> list) {
//		Set<MemoryEntry> intermediates = new HashSet<>();
//		double memoryEstimate = 0;
//
//		for(Lop lop : list) {
//			Iterator<MemoryEntry> intermediateIter = intermediates.iterator();
//
//			while(intermediateIter.hasNext()) {
//				MemoryEntry entry = intermediateIter.next();
//				entry.removeTarget(lop.getID());
//				if (entry.getTargets().isEmpty()) intermediateIter.remove();
//			}
//
//			Set<Long> outputIDs = lop.getOutputs().stream().map(Lop::getID).collect(Collectors.toSet());
//			intermediates.add(new MemoryEntry(outputIDs, lop.getOutputMemoryEstimate()));
//
//			double requiredMemory = intermediates.stream().map(MemoryEntry::getRequiredMemory)
//				.reduce((double) 0, Double::sum);
//
//			memoryEstimate = Math.max(requiredMemory, memoryEstimate);
//
//			if(optimalList.getMaxMemoryUsage() != -1 && memoryEstimate > optimalList.getMaxMemoryUsage()) {
//				return memoryEstimate;
//			}
//		}
//
//		return memoryEstimate;
//	}
}
