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
package org.apache.sysml.hops.estim;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.random.Well1024a;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class EstimatorLayeredGraph extends SparsityEstimator {

	private static final int ROUNDS = 128;
	private final int _rounds;
	
	public EstimatorLayeredGraph() {
		this(ROUNDS);
	}
	
	public EstimatorLayeredGraph(int rounds) {
		_rounds = rounds;
	}
	
	@Override
	public double estim(MMNode root) {
		throw new NotImplementedException();
	}
	
	@Override
	public double estim(MatrixCharacteristics mc1, MatrixCharacteristics mc2) {
		throw new NotImplementedException();
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2){
		int layer = 3;
		LayeredGraph LGraph = new LayeredGraph(m1, m2);
		//lambda is not the mean, if lambda is 2 hand in 1/2
		ExponentialDistribution random = new ExponentialDistribution(new Well1024a(), 1);
		for (int h = 0; h < LGraph.nodes.size(); h++) {
			if (LGraph.nodes.get(h).getY() == 1) {
				double[] doubArray = new double[_rounds];
				for (int g = 0; g < _rounds; g++)
					doubArray[g] = random.sample();
				LGraph.nodes.get(h).setVector(doubArray);
			}
		}
		// get r for nodes of upper layer
		for (int h = 0; h < LGraph.nodes.size(); h++) {
			if (LGraph.nodes.get(h).getY() == layer) {
				double[] ret = recr(_rounds, LGraph.nodes.get(h));
				if(ret != null)
					LGraph.nodes.get(h).setVector(ret);
				LGraph.nodes.get(h).setValue(
					calcNNZ(LGraph.nodes.get(h).getVector(), _rounds));
			}
		}
		//calc final sparsity
		double nnz = LGraph.nodes.stream().filter(n -> n.getY()==layer)
			.mapToDouble(n -> n.getValue()).sum();
		return nnz / m1.getNumRows() / m2.getNumColumns();
	}
	
	
	public double[] recr(int numr, Node tempnode) {
		if (tempnode.getInput().isEmpty())
			return (tempnode.getY() == 1) ? tempnode.getVector() : null;
		else if (tempnode.getInput().size() == 1)
			return recr(numr, tempnode.getInput().get(0));
		else {
			return tempnode.getInput().stream()
				.map(n -> recr(numr, n)).filter(v -> v != null)
				.reduce((v1,v2) -> min(v1,v2)).get();
		}
	}
	
	private double[] min(double[] v1, double[] v2) {
		double[] ret = new double[v1.length];
		for(int i=0; i<v1.length; i++)
			ret[i] = Math.min(v1[i], v2[i]);
		return ret;
	}

	public double calcNNZ(double[] inpvec, int numr) {
		return (inpvec != null && inpvec.length > 0) ?
			(numr - 1) / Arrays.stream(inpvec).sum() : 0;
	}

	private class LayeredGraph {
		List<Node> nodes = new ArrayList<>();

		public LayeredGraph(MatrixBlock m1, MatrixBlock m2) {
			createNodes(m1, 1, nodes);
			createNodes(m2, 2, nodes);
		}
	}

	public void createNodes(MatrixBlock m, int mpos, List<Node> nodes) {
		if( m.isEmpty() )
			return;
		
		Node nodeout = null;
		Node nodein = null;
		//TODO perf: separate handling sparse and dense
		//TODO perf: hash lookups for existing nodes
		for (int i = 0; i < m.getNumRows(); i++) {
			for (int j = 0; j < m.getNumColumns(); j++) {
				if (m.getValue(i, j) == 0) continue;
				boolean alreadyExists = false;
				boolean alreadyExists2 = false;
				for (int k = 0; k < nodes.size(); k++) {
					if (nodes.get(k).getX() == i && nodes.get(k).getY() == mpos) {
						alreadyExists = true;
					}
				}
				if (!alreadyExists) {
					nodein = new Node(i, mpos);
					nodes.add(nodein);
				} else {
					for (int k = 0; k < nodes.size(); k++) {
						if (nodes.get(k).getX() == i && nodes.get(k).getY() == mpos) {
							nodein = nodes.get(k);
						}
					}
				}
				for (int k = 0; k < nodes.size(); k++) {
					if (nodes.get(k).getX() == j && nodes.get(k).getY() == mpos + 1) {
						alreadyExists2 = true;
					}
				}
				if (!alreadyExists2) {
					nodeout = new Node(j, mpos + 1);
					nodes.add(nodeout);

				} else {
					for (int k = 0; k < nodes.size(); k++) {
						if (nodes.get(k).getX() == j && nodes.get(k).getY() == mpos + 1) {
							nodeout = nodes.get(k);
						}
					}
				}
				nodeout.addnz(nodein);
			}
		}
	}

	private static class Node {
		int xpos;
		int ypos;
		double[] r_vector;
		List<Node> input = new ArrayList<>();
		double value;

		public Node(int x, int y) {
			xpos = x;
			ypos = y;
		}

		public void setValue(double inp) {
			value = inp;
		}

		public double getValue() {
			return value;
		}

		public List<Node> getInput() {
			return input;
		}

		public int getX() {
			return xpos;
		}

		public int getY() {
			return ypos;
		}

		public double[] getVector() {
			return r_vector;
		}

		public void setVector(double[] r_input) {
			r_vector = r_input;
		}

		public void addnz(Node dest) {
			input.add(dest);
		}
	}
}
