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

import org.apache.sysds.lops.Lop;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class LinearizerResourceAwareFast extends IDagLinearizer {

	static class MemoryUsageLop {
		Lop lop;
		MemoryUsageEntry memoryEntry;

		MemoryUsageLop(Lop lop, MemoryUsageEntry memoryEntry) {
			this.lop = lop;
			this.memoryEntry = memoryEntry;
		}

		public Lop getLop() {
			return lop;
		}

		public MemoryUsageEntry getMemoryEntry() {
			return memoryEntry;
		}
	}

	static class MemoryUsageEntry {
		List<Integer> indecies;
		double estimatedMemoryUsage;

		MemoryUsageEntry(List<Integer> indecies, double estimatedMemoryUsage) {
			this.indecies = indecies;
			this.estimatedMemoryUsage = estimatedMemoryUsage;
		}

		List<Integer> getIndecies() {
			return indecies;
		}

		double getEstimatedMemoryUsage() {
			return estimatedMemoryUsage;
		}

		boolean IndeciesAreTheSame(List<Integer> indecies) {
			for(int i = 0; i < indecies.size(); i++) {
				if(!indecies.get(i).equals(this.getIndecies().get(i)))
					return false;
			}
			return true;
		}
	}

	static class Path {
		List<Lop> sequence;
		double maxMemoryUsage;

		Path(List<Lop> sequence, double maxMemoryUsage) {
			this.sequence = sequence;
			this.maxMemoryUsage = maxMemoryUsage;
		}

		void addLop(Lop lop) {
			sequence.add(lop);
		}

		void resetSequence() {
			sequence.clear();
		}

		List<Lop> getSequence() {
			return sequence;
		}

		double getMaxMemoryUsage() {
			return maxMemoryUsage;
		}
	}

	static class VisitedNode {
		List<Integer> indecies;
		List<Lop> bestPath;
		double maxMemoryUsage;

		VisitedNode(List<Integer> indecies, List<Lop> bestPath, double maxMemoryUsage) {
			this.indecies = indecies;
			this.bestPath = bestPath;
			this.maxMemoryUsage = maxMemoryUsage;
		}

		public List<Integer> getIndecies() {
			return indecies;
		}

		public List<Lop> getBestPath() {
			return bestPath;
		}

		public double getMaxMemoryUsage() {
			return maxMemoryUsage;
		}

		boolean IndeciesAreTheSame(List<Integer> indecies) {
			for(int i = 0; i < indecies.size(); i++) {
				if(!indecies.get(i).equals(this.getIndecies().get(i)))
					return false;
			}
			return true;
		}
	}

	static class Dependency {
		int nodeIndex;
		int sequenceIndex;
		List<Integer> dependencies;

		Dependency(int sequenceIndex, int nodeIndex, List<Integer> dependencies) {
			this.sequenceIndex = sequenceIndex;
			this.nodeIndex = nodeIndex;
			this.dependencies = dependencies;
		}

		public int getSequenceIndex() {
			return sequenceIndex;
		}

		public int getNodeIndex() {
			return nodeIndex;
		}

		public List<Integer> getDependencies() {
			return dependencies;
		}
	}

	List<MemoryUsageEntry> memory = new ArrayList<>();
	List<VisitedNode> visited = new ArrayList<>();

	@Override
	public List<Lop> linearize(List<Lop> dag) {
		ArrayList<List<Lop>> sequences = new ArrayList<>();

		List<Lop> outputNodes = dag.stream().filter(node -> node.getOutputs().isEmpty()).collect(Collectors.toList());

		for(Lop outputNode : outputNodes) {
			sequences.add(findSequence(outputNode, dag));
		}

		while(!dag.isEmpty()) {
			int maxLevel = dag.stream().mapToInt(Lop::getLevel).max().getAsInt();
			Lop node = dag.stream().filter(n -> n.getLevel() == maxLevel).findFirst().orElseThrow();
			sequences.add(findSequence(node, dag));
		}

		return scheduleSequences(sequences);
	}

	void clearMemoryEntries() {
		memory.clear();
		visited.clear();
	}

	List<Lop> findSequence(Lop startNode, List<Lop> remaining) {
		ArrayList<Lop> sequence = new ArrayList<>();

		if(!remaining.contains(startNode)) {
			throw new RuntimeException();
		}

		Lop currentNode = startNode;
		sequence.add(currentNode);
		remaining.remove(currentNode);

		while(currentNode.getInputs().size() == 1) {
			if(remaining.contains(currentNode.getInput(0))) {
				currentNode = currentNode.getInput(0);
				sequence.add(currentNode);
				remaining.remove(currentNode);
			}
			else {
				Collections.reverse(sequence);
				return sequence;
			}
		}

		Collections.reverse(sequence);

		List<Lop> children = currentNode.getInputs();

		if(children.isEmpty()) {
			return sequence;
		}

		List<List<Lop>> childSequences = new ArrayList<>();

		for(Lop child : children) {
			if(remaining.contains(child)) {
				childSequences.add(findSequence(child, remaining));
			}
		}

		List<Lop> finalSequence = scheduleSequences(childSequences);

		return Stream.concat(finalSequence.stream(), sequence.stream()).collect(Collectors.toList());
	}

	List<Lop> scheduleSequences(List<List<Lop>> sequences) {

		List<Dependency> dependencies = checkDependencies(sequences);

		int lastSequenceWithOutputNodeIndex = -1;
		for (int i = 0; i < sequences.size(); i++) {
			List<Lop> sequence = sequences.get(i);
			if (sequence.get(sequence.size() - 1).getOutputs().isEmpty()) {
				if (lastSequenceWithOutputNodeIndex == -1) {
					lastSequenceWithOutputNodeIndex = i;
				}else{
					List<Integer> dependencyList = new ArrayList<>(Collections.nCopies(sequences.size(), -1));
					dependencyList.set(lastSequenceWithOutputNodeIndex, sequences.get(lastSequenceWithOutputNodeIndex).size() - 1);
					dependencies.add(new Dependency(i, sequence.size() - 1, dependencyList));
				}
			}
		}

		List<Integer> indecies = new ArrayList<>(Collections.nCopies(sequences.size(), -1));

		Path bestPath = walkPath(sequences, indecies, dependencies, new ArrayList<>(), 0);

		clearMemoryEntries();

		return bestPath.getSequence();
	}

	MemoryUsageEntry getMemoryEntry(List<List<Lop>> sequences, List<Integer> indecies) {
		Optional<MemoryUsageEntry> optionalMemoryEntry = memory.stream().filter(entry -> entry.IndeciesAreTheSame(indecies))
			.findFirst();

		if(optionalMemoryEntry.isPresent()) {
			return optionalMemoryEntry.get();
		}
		else {
			MemoryUsageEntry newEntry = getMemoryRequirement(sequences, indecies);
			memory.add(newEntry);
			return newEntry;
		}
	}

	MemoryUsageEntry getMemoryRequirement(List<List<Lop>> sequences, List<Integer> indecies) {
		if(sequences.size() != indecies.size())
			throw new RuntimeException();

		double memory = 0;

		for(int i = 0; i < sequences.size(); i++) {
			int sequenceIndex = indecies.get(i);

			if(sequenceIndex >= 0) {
				memory += sequences.get(i).get(sequenceIndex).getOutputMemoryEstimate();
			}
		}

		return new MemoryUsageEntry(indecies, memory);
	}

	Path walkPath(List<List<Lop>> sequences, List<Integer> indecies, List<Dependency> dependencies, List<Lop> sequence,
		double maxMemoryUsage) {

		//checks if an indecies is out of bounce
		if(isAtTheEnd(sequences, indecies)) {
			return new Path(sequence, maxMemoryUsage);
		}

		//checks if this position was already visited. If then return the saved value.
		Optional<VisitedNode> optVisitedPath = visited.parallelStream()
			.filter(entry -> entry.IndeciesAreTheSame(indecies)).findFirst();

		if(optVisitedPath.isPresent()) {
			VisitedNode path = optVisitedPath.get();
			return new Path(path.getBestPath(), path.getMaxMemoryUsage());
		}

		//position was not already visited
		List<MemoryUsageLop> nextSteps = new ArrayList<>();

		//gather next possible steps
		for(int i = 0; i < indecies.size(); i++) {
			if(indecies.get(i) + 1 < sequences.get(i).size()) {
				List<Integer> newIndecies = new ArrayList<>(indecies);
				newIndecies.set(i, indecies.get(i) + 1);

				MemoryUsageEntry entry = getMemoryEntry(sequences, newIndecies);

				nextSteps.add(new MemoryUsageLop(sequences.get(i).get(newIndecies.get(i)), entry));
			}
		}

		List<MemoryUsageLop> newNextSteps = new ArrayList<>();

		for(MemoryUsageLop step : nextSteps) {
			MemoryUsageEntry nextStep = step.getMemoryEntry();
			boolean nextStepIsPossible = true;

			for(Dependency dependency : dependencies) {
				int sequenceIndex = dependency.getSequenceIndex();
				int nodeIndex = dependency.getNodeIndex();
				List<Integer> nextStepIndecies = nextStep.getIndecies();

				if(nextStepIndecies.get(sequenceIndex) == nodeIndex) {
					for(int j = 0; j < nextStepIndecies.size(); j++) {
						if(j != sequenceIndex && nextStepIndecies.get(j) < dependency.getDependencies().get(j)) {
							nextStepIsPossible = false;
							break;
						}
					}
				}

				if(!nextStepIsPossible)
					break;
			}

			if(nextStepIsPossible) {
				newNextSteps.add(step);
			}
		}

		if(newNextSteps.isEmpty()) {
			throw new RuntimeException();
		}

		//sort next possible steps so it starts with the entry with the smallest memory usage
		newNextSteps.sort(Comparator.comparing(item -> item.getMemoryEntry().getEstimatedMemoryUsage()));

		Path bestPath = new Path(new ArrayList<>(), -1);

		for(MemoryUsageLop newNextStep : newNextSteps) {
			MemoryUsageEntry entry = newNextStep.getMemoryEntry();
			Lop entryLop = newNextStep.getLop();

			if(bestPath.getMaxMemoryUsage() == -1 || entry.getEstimatedMemoryUsage() < bestPath.getMaxMemoryUsage()) {
				List<Lop> newSequence = new ArrayList<>(sequence);
				newSequence.add(entryLop);
				double entryMaxMemory = maxMemoryUsage;
				if(entry.getEstimatedMemoryUsage() > entryMaxMemory) {
					entryMaxMemory = entry.getEstimatedMemoryUsage();
				}

				Path test = walkPath(sequences, entry.getIndecies(), dependencies, newSequence, entryMaxMemory);
				if(bestPath.getMaxMemoryUsage() == -1 || test.getMaxMemoryUsage() < bestPath.getMaxMemoryUsage()) {
					bestPath = test;
				}
			}
		}

		visited.add(new VisitedNode(indecies, bestPath.getSequence(), bestPath.getMaxMemoryUsage()));

		return bestPath;
	}

	boolean isAtTheEnd(List<List<Lop>> sequences, List<Integer> indecies) {
		for(int a = 0; a < indecies.size(); a++) {
			if(indecies.get(a) < sequences.get(a).size() - 1) {
				return false;
			}
		}

		return true;
	}

	List<Dependency> checkDependencies(List<List<Lop>> sequences) {
		List<Dependency> dependencies = new ArrayList<>();

		for(int j = 0; j < sequences.size(); j++) {
			List<Lop> sequence = sequences.get(j);

			if(!sequence.get(0).getInputs().isEmpty()) {
				int sequenceIndex = j;

				sequence.get(0).getInputs().forEach(input -> {
					long id = input.getID();
					List<Integer> dependencyIndecies = new ArrayList<>();

					sequences.forEach(seq -> {
						int index = -1;
						for(int i = 0; i < seq.size(); i++) {
							if(seq.get(i).getID() == id) {
								index = i;
								break;
							}
						}
						dependencyIndecies.add(index);
					});

					dependencies.add(new Dependency(sequenceIndex, 0, dependencyIndecies));
				});
			}
		}

		return dependencies;
	}
}
