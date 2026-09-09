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

package org.apache.sysds.runtime.ooc.planning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.ooc.primitives.MaterializeOOCPrimitive;
import org.apache.sysds.runtime.ooc.primitives.OOCPrimitive;

public final class OOCPlanner {
	public static void compile(OOCPrimitive root) {
		injectMaterializations(root, Collections.newSetFromMap(new IdentityHashMap<>()), new IdentityHashMap<>());
		List<OOCPrimitive> primitives = new ArrayList<>();
		collect(root, Collections.newSetFromMap(new IdentityHashMap<>()), primitives);
		if(primitives.isEmpty())
			return;

		for(int i = primitives.size() - 1; i >= 0; i--)
			if(primitives.get(i).getAccessPattern().isUnset())
				primitives.get(i).inferPatterns();
		if(root.getAccessPattern() == OOCAccessPattern.ANY || root.getAccessPattern().isUnset())
			root.requestPattern(OOCAccessPattern.ROW_MAJOR);

		for(OOCPrimitive primitive : primitives)
			primitive.tryStartExecution();
	}

	@SuppressWarnings("unchecked")
	private static void injectMaterializations(OOCPrimitive primitive, Set<OOCPrimitive> visited,
		IdentityHashMap<OOCStreamable<IndexedMatrixValue>, MaterializeOOCPrimitive> boundaries) {
		if(primitive.hasStartedExecution() || !visited.add(primitive))
			return;
		for(OOCPrimitive.OOCMaterializedInputRequest request : primitive.requiredMaterializedInputs()) {
			OOCStreamable<IndexedMatrixValue> input = (OOCStreamable<IndexedMatrixValue>) primitive
				.getInput(request.inputIndex());
			MaterializeOOCPrimitive boundary = boundaries.get(input);
			if(boundary == null) {
				boundary = new MaterializeOOCPrimitive(input, request.layout(), primitive.getContext());
				boundaries.put(input, boundary);
			}
			primitive.discardInputHandle(request.inputIndex());
			boolean live = boundary.registerRequest(request.expectedReaders(), request.liveConsumer());
			if(request.liveRegistration() != null)
				request.liveRegistration().accept(live);
			primitive.installMaterializedInput(request.inputIndex(), boundary);
		}
		for(OOCPrimitive child : primitive.getChildren())
			injectMaterializations(child, visited, boundaries);
	}

	private static void collect(OOCPrimitive primitive, Set<OOCPrimitive> visited, List<OOCPrimitive> result) {
		if(primitive.hasStartedExecution() || !visited.add(primitive))
			return;
		result.add(primitive);
		for(OOCPrimitive child : primitive.getChildren())
			collect(child, visited, result);
	}
}
