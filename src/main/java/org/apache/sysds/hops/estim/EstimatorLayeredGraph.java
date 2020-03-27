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

package org.apache.sysds.hops.estim;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.random.Well1024a;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * This estimator implements an approach based on a so-called layered graph,
 * introduced in
 * Edith Cohen. Structure prediction and computation of sparse matrix
 * products. J. Comb. Optim., 2(4):307–332, 1998.
 * 
 */
public class EstimatorLayeredGraph extends SparsityEstimator {

	private static final int ROUNDS = 32;
	private final int _rounds;
	
	public EstimatorLayeredGraph() {
		this(ROUNDS);
	}
	
	public EstimatorLayeredGraph(int rounds) {
		_rounds = rounds;
	}
	
	@Override
	public DataCharacteristics estim(MMNode root) {
		List<MatrixBlock> leafs = getMatrices(root, new ArrayList<>());
		long nnz = new LayeredGraph(leafs, _rounds).estimateNnz();
		return root.setDataCharacteristics(new MatrixCharacteristics(
			leafs.get(0).getNumRows(), leafs.get(leafs.size()-1).getNumColumns(), nnz));
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		if( op == OpCode.MM )
			return estim(m1, m2);
		throw new NotImplementedException();
	}

	@Override
	public double estim(MatrixBlock m, OpCode op) {
		throw new NotImplementedException();
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		LayeredGraph graph = new LayeredGraph(Arrays.asList(m1,m2), _rounds);
		return OptimizerUtils.getSparsity(
			m1.getNumRows(), m2.getNumColumns(), graph.estimateNnz());
	}
	
	private List<MatrixBlock> getMatrices(MMNode node, List<MatrixBlock> leafs) {
		//NOTE: this extraction is only correct and efficient for chains, no DAGs
		if( node.isLeaf() )
			leafs.add(node.getData());
		else {
			getMatrices(node.getLeft(), leafs);
			getMatrices(node.getRight(), leafs);
		}
		return leafs;
	}

	public static class LayeredGraph {
		private final List<Node[]> _nodes; //nodes partitioned by graph level
		private final int _rounds;         //length of propagated r-vectors 
		
		public LayeredGraph(List<MatrixBlock> chain, int r) {
			_nodes = new ArrayList<>();
			_rounds = r;
			chain.forEach(i -> buildNext(i));
		}
		
		public void buildNext(MatrixBlock mb) {
			if( mb.isEmpty() )
				return;
			final int m = mb.getNumRows();
			final int n = mb.getNumColumns();
			
			//step 1: create node arrays for rows/cols
			Node[] rows = null, cols = null;
			if( _nodes.size() == 0 ) {
				rows = new Node[m];
				for(int i=0; i<m; i++)
					rows[i] = new Node();
				_nodes.add(rows);
			}
			else {
				rows = _nodes.get(_nodes.size()-1);
			}
			cols = new Node[n];
			for(int j=0; j<n; j++)
				cols[j] = new Node();
			_nodes.add(cols);
			
			//step 2: create edges for non-zero values
			if( mb.isInSparseFormat() ) {
				SparseBlock a = mb.getSparseBlock();
				for(int i=0; i < m; i++) {
					if( a.isEmpty(i) ) continue;
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					for(int k=apos; k<apos+alen; k++)
						cols[aix[k]].addInput(rows[i]);
				}
			}
			else { //dense
				DenseBlock a = mb.getDenseBlock();
				for (int i=0; i<m; i++) {
					double[] avals = a.values(i);
					int aix = a.pos(i);
					for (int j=0; j<n; j++)
						if( avals[aix+j] != 0 )
							cols[j].addInput(rows[i]);
				}
			}
		}
		
		public long estimateNnz() {
			//step 1: assign random vectors ~exp(lambda=1) to all leaf nodes
			//(lambda is not the mean, if lambda is 2 mean is 1/2)
			ExponentialDistribution random = new ExponentialDistribution(new Well1024a(), 1);
			for( Node n : _nodes.get(0) ) {
				double[] rvect = new double[_rounds];
				for (int g = 0; g < _rounds; g++)
					rvect[g] = random.sample();
				n.setVector(rvect);
			}
			
			//step 2: propagate vectors bottom-up and aggregate nnz
			return Math.round(Arrays.stream(_nodes.get(_nodes.size()-1))
				.mapToDouble(n -> calcNNZ(n.computeVector(_rounds), _rounds)).sum());
		}
		
		private static double calcNNZ(double[] inpvec, int rounds) {
			return (inpvec != null && inpvec.length > 0) ?
				(rounds - 1) / Arrays.stream(inpvec).sum() : 0;
		}
		
		private static class Node {
			private List<Node> _input = new ArrayList<>();
			private double[] _rvect;
			
			public List<Node> getInput() {
				return _input;
			}
			
			@SuppressWarnings("unused")
			public double[] getVector() {
				return _rvect;
			}
		
			public void setVector(double[] rvect) {
				_rvect = rvect;
			}
		
			public void addInput(Node dest) {
				_input.add(dest);
			}
			
			private double[] computeVector(int rounds) {
				if( _rvect != null || getInput().isEmpty() )
					return _rvect;
				//recursively compute input vectors
				List<double[]> ltmp = getInput().stream().map(n -> n.computeVector(rounds))
					.filter(v -> v!=null).collect(Collectors.toList());
				if( ltmp.isEmpty() )
					return null;
				else if( ltmp.size() == 1 )
					return _rvect = ltmp.get(0);
				else {
					double[] tmp = ltmp.get(0).clone();
					for(int i=1; i<ltmp.size(); i++) {
						double[] v2 = ltmp.get(i);
						for(int j=0; j<rounds; j++)
							tmp[j] = Math.min(tmp[j], v2[j]);
					}
					return _rvect = tmp;
				}
			}
		}
	}
}
