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

package org.apache.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.BinaryOp;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;

import org.apache.sysds.hops.Hop;

/**
 * Rule: RewriteFloorCompress. Detects the sequence `M2 = floor(M * S)` followed by `C = compress(M2)` and prepares for
 * fusion into a single operation. This rewrite improves performance by avoiding intermediate results. Currently, it
 * identifies the pattern without applying fusion.
 */
public class RewriteQuantizationFusedCompression extends HopRewriteRule {
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if(roots == null)
			return null;

		// traverse the HOP DAG
		HashMap<String, Hop> floors = new HashMap<>();
		HashMap<String, Hop> compresses = new HashMap<>();
		for(Hop h : roots)
			collectFloorCompressSequences(h, floors, compresses);

		Hop.resetVisitStatus(roots);

		// check compresses for compress-after-floor pattern
		for(Entry<String, Hop> e : compresses.entrySet()) {
			String inputname = e.getKey();
			Hop compresshop = e.getValue();

			if(floors.containsKey(inputname) // floors same name
				&& ((floors.get(inputname).getBeginLine() < compresshop.getBeginLine()) ||
					(floors.get(inputname).getEndLine() < compresshop.getEndLine()) ||
					(floors.get(inputname).getBeginLine() == compresshop.getBeginLine() &&
					 floors.get(inputname).getEndLine() == compresshop.getBeginLine() &&
					 floors.get(inputname).getBeginColumn() < compresshop.getBeginColumn()))) {

				// retrieve the floor hop and inputs
				Hop floorhop = floors.get(inputname);
				Hop floorInput = floorhop.getInput().get(0);

				// check if the input of the floor operation is a matrix
				if(floorInput.getDataType() == DataType.MATRIX) {

					// Check if the input of the floor operation involves a multiplication operation
					if(floorInput instanceof BinaryOp && ((BinaryOp) floorInput).getOp() == OpOp2.MULT) {
						Hop initialMatrix = floorInput.getInput().get(0);
						Hop sf = floorInput.getInput().get(1);

						// create fused hop
						BinaryOp fusedhop = new BinaryOp("test", DataType.MATRIX, ValueType.FP64,
							OpOp2.QUANTIZE_COMPRESS, initialMatrix, sf);

						// rewire compress consumers to fusedHop
						List<Hop> parents = new ArrayList<>(compresshop.getParent());
						for(Hop p : parents) {
							HopRewriteUtils.replaceChildReference(p, compresshop, fusedhop);
						}
					}
				}
			}
		}
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		// do nothing, floor/compress do not occur in predicates
		return root;
	}

	private void collectFloorCompressSequences(Hop hop, HashMap<String, Hop> floors, HashMap<String, Hop> compresses) {
		if(hop.isVisited())
			return;

		// process childs
		if(!hop.getInput().isEmpty())
			for(Hop c : hop.getInput())
				collectFloorCompressSequences(c, floors, compresses);

		// process current hop
		if(hop instanceof UnaryOp) {
			UnaryOp uop = (UnaryOp) hop;
			if(uop.getOp() == OpOp1.FLOOR) {
				floors.put(uop.getName(), uop);
			}
			else if(uop.getOp() == OpOp1.COMPRESS) {
				compresses.put(uop.getInput(0).getName(), uop);
			}
		}
		hop.setVisited();
	}
}
