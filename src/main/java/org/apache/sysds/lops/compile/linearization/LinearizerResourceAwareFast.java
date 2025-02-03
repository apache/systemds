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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class LinearizerResourceAwareFast extends IDagLinearizer {

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

	static class Item {
		List<Integer> steps;
		List<Integer> current;
		Set<Intermediate> intermediates;
		double maxMemoryUsage;

		Item(List<Integer> steps, List<Integer> current, Set<Intermediate> intermediates, double maxMemoryUsage) {
			this.steps = steps;
			this.current = current;
			this.intermediates = intermediates;
			this.maxMemoryUsage = maxMemoryUsage;
		}

		public List<Integer> getSteps() {
			return steps;
		}

		public List<Integer> getCurrent() {
			return current;
		}

		public double getMaxMemoryUsage() {
			return maxMemoryUsage;
		}

		public Set<Intermediate> getIntermediates() {
			return intermediates;
		}
	}

	static class Intermediate {
		List<Long> lopIDs;
		double memoryUsage;

		Intermediate(List<Long> lopIDs, double memoryUsage) {
			this.lopIDs = lopIDs;
			this.memoryUsage = memoryUsage;
		}

		void remove(long ID) {
			lopIDs.remove(ID);
		}

		public List<Long> getLopIDs() {
			return lopIDs;
		}

		public double getMemoryUsage() {
			return memoryUsage;
		}
	}

	List<Lop> remaining;

	@Override
	public List<Lop> linearize(List<Lop> dag) {
		List<List<Lop>> sequences = new ArrayList<>();
		remaining = new ArrayList<>(dag);

		List<Lop> outputNodes = remaining.stream().filter(node -> node.getOutputs().isEmpty())
			.collect(Collectors.toList());

		for(Lop outputNode : outputNodes) {
			sequences.add(findSequence(outputNode));
		}

		while(!remaining.isEmpty()) {
			int maxLevel = remaining.stream().mapToInt(Lop::getLevel).max().getAsInt();
			Lop node = remaining.stream().filter(n -> n.getLevel() == maxLevel).findFirst().orElseThrow();
			sequences.add(findSequence(node));
		}

		return scheduleSequences(sequences);
	}

	List<Lop> scheduleSequences(List<List<Lop>> sequences) {
		Set<List<Integer>> visited = new HashSet<>();
		List<Item> scheduledItems = new ArrayList<>();

		Set<Dependency> dependencies = getDependencies(sequences);
		List<Integer> sequencesMaxIndex = sequences.stream().map(entry -> entry.size() - 1)
			.collect(Collectors.toList());
		
		Item currentItem = new Item(new ArrayList<>(), Collections.nCopies(sequences.size(), -1), new HashSet<>(), 0.0);

		while(!currentItem.getCurrent().equals(sequencesMaxIndex)) {

			for(int i = 0; i < sequences.size(); i++) {

				List<Lop> sequence = sequences.get(i);

				if(currentItem.getCurrent().get(i) + 1 < sequence.size()) {
					List<Integer> newCurrent = new ArrayList<>(currentItem.getCurrent());
					newCurrent.set(i, newCurrent.get(i) + 1);

					if(!visited.contains(newCurrent)) {
						Set<Dependency> filteredDependencies = dependencies.stream()
							.filter(entry -> entry.getNodeIndex() == newCurrent.get(entry.getSequenceIndex()))
							.collect(Collectors.toSet());

						boolean dependencyIssue = filteredDependencies.parallelStream().anyMatch(
							dependency -> IntStream.range(0, newCurrent.size()).anyMatch(
								j -> j != dependency.getSequenceIndex() &&
									newCurrent.get(j) < dependency.getDependencies().get(j)));

						if(!dependencyIssue) {
							Set<Intermediate> newIntermediates = new HashSet<>(currentItem.getIntermediates());

							Lop nextLop = sequence.get(newCurrent.get(i));

							Iterator<Intermediate> intermediateIter = newIntermediates.iterator();

							while(intermediateIter.hasNext()) {
								Intermediate entry = intermediateIter.next();
								entry.remove(nextLop.getID());
								if(entry.getLopIDs().isEmpty())
									intermediateIter.remove();
							}

							newIntermediates.add(new Intermediate(
								nextLop.getOutputs().stream().map(Lop::getID).collect(Collectors.toList()),
								nextLop.getOutputMemoryEstimate()));

							List<Integer> newSteps = new ArrayList<>(currentItem.getSteps());
							newSteps.add(i);

							double mem = newIntermediates.stream().map(Intermediate::getMemoryUsage)
								.reduce((double) 0, Double::sum);

							Item newItem = new Item(newSteps, newCurrent, newIntermediates,
								Math.max(mem, currentItem.getMaxMemoryUsage()));

							int index = Collections.binarySearch(scheduledItems, newItem,
								Comparator.comparing(Item::getMaxMemoryUsage));

							if(index < 0) {
								index = -index - 1;
							}

							scheduledItems.add(index, newItem);
						}
						visited.add(newCurrent);
					}
				}
			}

			currentItem = scheduledItems.remove(0);
		}

		return walkPath(sequences, currentItem.getSteps());
	}

	List<Lop> walkPath(List<List<Lop>> sequences, List<Integer> path) {
		Iterator<Integer> iterator = path.iterator();
		List<Lop> sequence = new ArrayList<>();

		while(iterator.hasNext()) {
			sequence.add(sequences.get(iterator.next()).remove(0));
		}

		return sequence;
	}

	List<Lop> findSequence(Lop startNode) {
		List<Lop> sequence = new ArrayList<>();
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
				childSequences.add(findSequence(child));
			}
		}

		List<Lop> finalSequence = scheduleSequences(childSequences);

		return Stream.concat(finalSequence.stream(), sequence.stream()).collect(Collectors.toList());
	}

	Set<Dependency> getDependencies(List<List<Lop>> sequences) {
		Set<Dependency> dependencies = new HashSet<>();

		// Get IDs of each Lop in each sequence for faster lookup
		List<List<Long>> sequencesLopIDs = sequences.stream()
			.map(sequence -> sequence.stream().map(Lop::getID).collect(Collectors.toList()))
			.collect(Collectors.toList());

		int lastSequenceWithOutput = -1;

		// Go through each sequence and check for dependencies
		for(int j = 0; j < sequences.size(); j++) {
			List<Lop> sequence = sequences.get(j);
			int sequenceSize = sequence.size();
			int sequenceIndex = j;

			// Check if the current sequence depends on other sequences
			sequence.get(0).getInputs().forEach(input -> {
				long inputID = input.getID();
				List<Integer> dependencyIndices = sequencesLopIDs.stream()
					.map(list -> list.contains(inputID) ? list.indexOf(inputID) : -1).collect(Collectors.toList());

				dependencies.add(new Dependency(sequenceIndex, 0, dependencyIndices));
			});

			// Check for Lops that depends on Lops from other sequences
			for(int k = 0; k < sequenceSize; k++) {
				int finalK = k;
				int finalJ = j;
				sequence.get(k).getInputs().forEach(input -> {
					long inputID = input.getID();
					if(!sequencesLopIDs.get(finalJ).contains(inputID)) {
						List<Integer> dependencyIndices = sequencesLopIDs.stream()
							.map(list -> list.contains(inputID) ? list.indexOf(inputID) : -1)
							.collect(Collectors.toList());

						dependencies.add(new Dependency(finalJ, finalK, dependencyIndices));
					}
				});
			}

			// Dependency chain between output Lops so that the outputs are in the correct order
			if(sequence.get(sequenceSize - 1).getOutputs().isEmpty()) {
				if(lastSequenceWithOutput != -1) {
					List<Integer> dependencyList = new ArrayList<>(Collections.nCopies(sequences.size(), -1));
					dependencyList.set(lastSequenceWithOutput,
						sequences.get(lastSequenceWithOutput).size() - 1);
					dependencies.add(new Dependency(j, sequenceSize - 1, dependencyList));
				}
				lastSequenceWithOutput = j;
			}
		}

		return dependencies;
	}
}
