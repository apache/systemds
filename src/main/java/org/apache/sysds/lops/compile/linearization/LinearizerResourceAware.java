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
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class LinearizerResourceAware extends IDagLinearizer {

	@Override
	public List<Lop> linearize(List<Lop> v) {

		ArrayList<Lop> remaining = new ArrayList<>(v);
		ArrayList<List<Lop>> sequences = new ArrayList<>();

		List<Lop> outputNodes = v.stream().filter(node -> node.getOutputs().isEmpty()).collect(Collectors.toList());

		for(Lop outputNode : outputNodes) {
			sequences.add(findSequence(outputNode, remaining));
		}

		while(!remaining.isEmpty()) {
			int maxLevel = remaining.stream().mapToInt(Lop::getLevel).max().getAsInt();
			Lop node = remaining.stream().filter(n -> n.getLevel() == maxLevel).findFirst().get();
			sequences.add(findSequence(node, remaining));
		}

		return scheduleSequences(sequences);
	}

	List<Lop> findSequence(Lop startNode, List<Lop> remaining) {

		ArrayList<Lop> sequence = new ArrayList<Lop>();

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

		if(currentNode.getInputs().isEmpty()) {
			return sequence;
		}
		else {
			ArrayList<Lop> children = currentNode.getInputs();
			ArrayList<List<Lop>> childSequences = new ArrayList<>();

			for(Lop child : children) {
				if(remaining.contains(child)) {
					childSequences.add(findSequence(child, remaining));
				}
			}

			List<Lop> finalSequence = scheduleSequences(childSequences);

			return Stream.concat(finalSequence.stream(), sequence.stream()).collect(Collectors.toList());
		}
	}

	List<Lop> scheduleSequences(List<List<Lop>> sequences) {
		//TODO
		List<Lop> returnList = new ArrayList<>();
		for(List<Lop> sequence : sequences) {
			returnList = Stream.concat(returnList.stream(), sequence.stream()).collect(Collectors.toList());
		}
		return returnList;
	}
}
