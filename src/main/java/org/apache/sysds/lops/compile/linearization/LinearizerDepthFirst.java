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

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class LinearizerDepthFirst extends IDagLinearizer
{
	// previously called doTopologicalSortTwoLevelOrder
	@Override
	public List<Lop> linearize(List<Lop> v) {
		// partition nodes into leaf/inner nodes and dag root nodes,
		// + sort leaf/inner nodes by ID to force depth-first scheduling
		// + append root nodes in order of their original definition
		// (which also preserves the original order of prints)
		List<Lop> nodes = Stream
			.concat(v.stream().filter(l -> !l.getOutputs().isEmpty()).sorted(Comparator.comparing(l -> l.getID())),
				v.stream().filter(l -> l.getOutputs().isEmpty()))
			.collect(Collectors.toList());

		// NOTE: in contrast to hadoop execution modes, we avoid computing the transitive
		// closure here to ensure linear time complexity because its unnecessary for CP and Spark
		return nodes;
	}
}
