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

import org.apache.commons.lang3.NotImplementedException;
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
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * This estimator implements an approach based on a so-called layered graph,
 * introduced in
 * Edith Cohen. Structure prediction and computation of sparse matrix
 * products. J. Comb. Optim., 2(4):307–332, 1998.
 * 
 */
public class EstimatorLayeredGraph extends SparsityEstimator {

	public static final int ROUNDS = 512;
	private final int _rounds;
	private final Random _seeds;
	
	public EstimatorLayeredGraph() {
		this(ROUNDS);
	}
	
	public EstimatorLayeredGraph(int rounds) {
		this(rounds, (int)System.currentTimeMillis());
	}
	
	public EstimatorLayeredGraph(int rounds, int seed) {
		_rounds = rounds;
		_seeds = new Random(seed);
	}
	
	@Override
	public DataCharacteristics estim(MMNode root) {
		//List<MatrixBlock> leafs = getMatrices(root, new ArrayList<>());
		//List<OpCode> ops = getOps(root, new ArrayList<>());
		//List<LayeredGraph> LGs = new ArrayList<>();
		LayeredGraph ret = traverse(root);
		long nnz = ret.estimateNnz();
		return root.setDataCharacteristics(new MatrixCharacteristics(
			ret._nodes.get(0).length, ret._nodes.get(ret._nodes.size() - 1).length, nnz));
	}

	public LayeredGraph traverse(MMNode node) {
		if(node.getLeft() == null || node.getRight() == null) return null;
		LayeredGraph retL = traverse(node.getLeft());
		LayeredGraph retR = traverse(node.getRight());
		LayeredGraph ret, left, right;

		left = (node.getLeft().getData() == null)
			? retL : new LayeredGraph(node.getLeft().getData(), _rounds, _seeds.nextInt());
		right = (node.getRight().getData() == null)
			? retR : new LayeredGraph(node.getRight().getData(), _rounds, _seeds.nextInt());

		ret = estimInternal(left, right, node.getOp());

		return ret;
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		if( op == OpCode.MM )
			return estim(m1, m2);
		LayeredGraph lg1 = new LayeredGraph(m1, _rounds, _seeds.nextInt());
		LayeredGraph lg2 = new LayeredGraph(m2, _rounds, _seeds.nextInt());
		LayeredGraph output = estimInternal(lg1, lg2, op);
		return OptimizerUtils.getSparsity(
			output._nodes.get(0).length, output._nodes.get(output._nodes.size() - 1).length, output.estimateNnz());
	}

	@Override
	public double estim(MatrixBlock m, OpCode op) {
		LayeredGraph lg1 = new LayeredGraph(m, _rounds, _seeds.nextInt());
		LayeredGraph output = estimInternal(lg1, null, op);
		return OptimizerUtils.getSparsity(
			output._nodes.get(0).length, output._nodes.get(output._nodes.size() - 1).length, output.estimateNnz());
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		LayeredGraph graph = new LayeredGraph(Arrays.asList(m1,m2), _rounds, _seeds.nextInt());
		return OptimizerUtils.getSparsity(
			m1.getNumRows(), m2.getNumColumns(), graph.estimateNnz());
	}

	private static LayeredGraph estimInternal(LayeredGraph lg1, LayeredGraph lg2, OpCode op) {
		switch(op) {
			case MM:	  return lg1.matMult(lg2);
			case MULT:	  return lg1.and(lg2);
			case PLUS:	  return lg1.or(lg2);
			case RBIND:	  return lg1.rbind(lg2);
			case CBIND:	  return lg1.cbind(lg2);
//			case NEQZERO:
//			case EQZERO:
			case TRANS:   return lg1.transpose();
			case DIAG:	  return lg1.diag();
//			case RESHAPE:
			default:
				throw new NotImplementedException();
		}
	}
	
	@SuppressWarnings("unused")
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

	@SuppressWarnings("unused")
	private List<OpCode> getOps(MMNode node, List<OpCode> ops) {
		//NOTE: this extraction is only correct and efficient for chains, no DAGs
		if(node.isLeaf()) {
		}
		else {
			getOps(node.getLeft(), ops);
			getOps(node.getRight(), ops);
			ops.add(node.getOp());
		}
		return ops;
	}

	public static class LayeredGraph {
		private final List<Node[]> _nodes; //nodes partitioned by graph level
		private final int _rounds;         //length of propagated r-vectors 
		private final Random _seeds;
		
		public LayeredGraph(int r, int seed) {
			_nodes = new ArrayList<>();
			_rounds = r;
			_seeds = new Random(seed);
		}
		
		public LayeredGraph(List<MatrixBlock> chain, int r, int seed) {
			this(r, seed);
			chain.forEach(i -> buildNext(i));
		}

		public LayeredGraph(MatrixBlock m, int r, int seed) {
			this(r, seed);
			buildNext(m);
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
			ExponentialDistribution random = new ExponentialDistribution(
				new Well1024a(_seeds.nextInt()), 1);
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

		public LayeredGraph rbind(LayeredGraph lg) {
			LayeredGraph ret = new LayeredGraph(List.of(), _rounds, _seeds.nextInt());

			Node[] rows = new Node[_nodes.get(0).length + lg._nodes.get(0).length];
			Node[] columns = _nodes.get(1).clone();

			System.arraycopy(_nodes.get(0), 0, rows, 0, _nodes.get(0).length);

			for (int i = _nodes.get(0).length; i < rows.length; i++)
				rows[i] = new Node();

			for(int i = 0; i < lg._nodes.get(0).length; i++) {
				for(int j = 0; j < columns.length; j++) {
					List<Node> edges = lg._nodes.get(1)[j].getInput();
					if(edges.contains(lg._nodes.get(0)[i])) {
						columns[j].addInput(rows[i + _nodes.get(0).length]);
					}
				}
			}
			ret._nodes.add(rows);
			ret._nodes.add(columns);
			return ret;
		}

		public LayeredGraph cbind(LayeredGraph lg) {
			LayeredGraph ret = new LayeredGraph(List.of(), _rounds, _seeds.nextInt());
			int colLength = _nodes.get(1).length + lg._nodes.get(1).length;

			Node[] rows = _nodes.get(0).clone();
			Node[] columns = new Node[colLength];

			System.arraycopy(_nodes.get(1), 0, columns, 0, _nodes.get(1).length);

			for (int i = _nodes.get(1).length; i < columns.length; i++)
				columns[i] = new Node();

			for(int i = 0; i < rows.length; i++) {
				for(int j = 0; j < lg._nodes.get(1).length; j++) {
					List<Node> edges = lg._nodes.get(1)[j].getInput();
					if(edges.contains(lg._nodes.get(0)[i])) {
						columns[j + _nodes.get(1).length].addInput(rows[i]);
					}
				}
			}
			ret._nodes.add(rows);
			ret._nodes.add(columns);
			return ret;
		}

		public LayeredGraph matMult(LayeredGraph lg) {
			List<MatrixBlock> m = Stream.concat(
				this.toMatrixBlockList().stream(), lg.toMatrixBlockList().stream())
				.collect(Collectors.toList());
			return new LayeredGraph(m, _rounds, _seeds.nextInt());
		}

		public LayeredGraph or(LayeredGraph lg) {
			LayeredGraph ret = new LayeredGraph(List.of(), _rounds, _seeds.nextInt());
			Node[] rows = new Node[_nodes.get(0).length];
			for (int i = 0; i < _nodes.get(0).length; i++)
				rows[i] = new Node();
			ret._nodes.add(rows);

			for(int x = 0; x < _nodes.size() - 1; x++) {
				int y = x + 1;
				rows = ret._nodes.get(x);
				Node[] columns = new Node[_nodes.get(y).length];
				for (int i = 0; i < _nodes.get(y).length; i++)
					columns[i] = new Node();

				for(int i = 0; i < _nodes.get(x).length; i++) {
					for(int j = 0; j < _nodes.get(y).length; j++) {
						List<Node> edges1 = _nodes.get(y)[j].getInput();
						List<Node> edges2 = lg._nodes.get(y)[j].getInput();
						if(edges1.contains(_nodes.get(x)[i]) || edges2.contains(lg._nodes.get(x)[i]))
						{
							columns[j].addInput(rows[i]);
						}
					}
				}
				ret._nodes.add(columns);
			}
			return ret;
		}

		public LayeredGraph and(LayeredGraph lg) {
			LayeredGraph ret = new LayeredGraph(List.of(), _rounds, _seeds.nextInt());
			Node[] rows = new Node[_nodes.get(0).length];
			for (int i = 0; i < _nodes.get(0).length; i++)
				rows[i] = new Node();
			ret._nodes.add(rows);

			for(int x = 0; x < _nodes.size() - 1; x++) {
				int y = x + 1;
				rows = ret._nodes.get(x);
				Node[] columns = new Node[_nodes.get(y).length];
				for (int i = 0; i < _nodes.get(y).length; i++)
					columns[i] = new Node();

				for(int i = 0; i < _nodes.get(x).length; i++) {
					for(int j = 0; j < _nodes.get(y).length; j++) {
						List<Node> edges1 = _nodes.get(y)[j].getInput();
						List<Node> edges2 = lg._nodes.get(y)[j].getInput();
						if(edges1.contains(_nodes.get(x)[i]) && edges2.contains(lg._nodes.get(x)[i]))
						{
							columns[j].addInput(rows[i]);
						}
					}
				}
				ret._nodes.add(columns);
			}
			return ret;
		}

		public LayeredGraph transpose() {
			LayeredGraph ret = new LayeredGraph(List.of(), _rounds, _seeds.nextInt());
			Node[] rows = new Node[_nodes.get(_nodes.size() - 1).length];
			for (int i = 0; i < rows.length; i++)
				rows[i] = new Node();
			ret._nodes.add(rows);

			for(int x = _nodes.size() - 1; x > 0; x--) {
				rows = ret._nodes.get(ret._nodes.size() - 1);
				Node[] columnsOld = _nodes.get(x);
				Node[] rowsOld = _nodes.get(x - 1);
				Node[] columns = new Node[rowsOld.length];

				for (int i = 0; i < rowsOld.length; i++)
					columns[i] = new Node();

				for(int i = 0; i < rowsOld.length; i++) {
					for(int j = 0; j < columnsOld.length; j++) {
						List<Node> edges = columnsOld[j].getInput();
						if(edges.contains(rowsOld[i])) {
							columns[i].addInput(rows[j]);
						}
					}
				}
				ret._nodes.add(columns);
			}
			return ret;
		}

		public LayeredGraph diag() {
			LayeredGraph ret = new LayeredGraph(List.of(), _rounds, _seeds.nextInt());
			Node[] rowsOld = _nodes.get(0);
			Node[] columnsOld = _nodes.get(1);

			if(_nodes.get(1).length == 1) {
				Node[] rows = new Node[rowsOld.length];
				Node[] columns = new Node[rowsOld.length];

				for (int i = 0; i < rowsOld.length; i++)
					rows[i] = new Node();
				for (int i = 0; i < rowsOld.length; i++)
					columns[i] = new Node();

				List<Node> edges = columnsOld[0].getInput();
				for(int i = 0; i < rowsOld.length; i++) {
					for(int j = 0; j < rowsOld.length; j++) {
						if(edges.contains(rowsOld[i]) && i == j) {
							columns[j].addInput(rows[i]);
						}
					}
				}
				ret._nodes.add(rows);
				ret._nodes.add(columns);
				return ret;
			}
			else if(_nodes.get(0).length == 1){
				Node[] rows = new Node[columnsOld.length];
				Node[] columns = new Node[columnsOld.length];

				for (int i = 0; i < columnsOld.length; i++)
					rows[i] = new Node();
				for (int i = 0; i < columnsOld.length; i++)
					columns[i] = new Node();

				for(int i = 0; i < columnsOld.length; i++) {
					for(int j = 0; j < columnsOld.length; j++) {
						List<Node> edges = columnsOld[j].getInput();
						if(edges.contains(rowsOld[0]) && i == j) {
							columns[j].addInput(rows[i]);
						}
					}
				}
				ret._nodes.add(rows);
				ret._nodes.add(columns);
				return ret;
			}
			else {
				Node[] rows = new Node[rowsOld.length];
				Node[] columns = new Node[1];
				for (int i = 0; i < rowsOld.length; i++)
					rows[i] = new Node();
				for (int i = 0; i < 1; i++)
					columns[i] = new Node();
				for(int i = 0; i < rowsOld.length; i++) {
					for(int j = 0; j < columnsOld.length; j++) {
						List<Node> edges = columnsOld[j].getInput();
						if(edges.contains(rowsOld[i]) && i == j) {
							columns[0].addInput(rows[i]);
						}
					}
				}
				ret._nodes.add(rows);
				ret._nodes.add(columns);
				return ret;
			}
		}

		public MatrixBlock toMatrixBlock() {
			List<Double> a = new ArrayList<>();
			int rows = _nodes.get(0).length;
			int cols = _nodes.get(1).length;
			for(int i = 0; i < rows * cols; i++) {
				a.add(0.);
			}
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					List<Node> edges = _nodes.get(1)[j].getInput();
					if(edges.contains(_nodes.get(0)[i])) {
						a.set(i * cols + j, 1. + a.get(i * cols + j));
					}
					else {
						a.set(i * cols + j, 0.);
					}
				}
			}
			double[] arr = a.stream().mapToDouble(d -> d).toArray();
			return new MatrixBlock(rows, cols, arr);
		}

		public List<MatrixBlock> toMatrixBlockList() {
			List<MatrixBlock> m = new ArrayList<>();
			for(int x = 0; x < _nodes.size() - 1; x++) {
				int y = x + 1;
				List<Double> a = new ArrayList<>();
				int rows = _nodes.get(x).length;
				int cols = _nodes.get(y).length;
				for(int i = 0; i < rows * cols; i++) {
					a.add(0.);
				}
				for(int i = 0; i < rows; i++) {
					for(int j = 0; j < cols; j++) {
						List<Node> edges = _nodes.get(y)[j].getInput();
						if(edges.contains(_nodes.get(x)[i])) {
							a.set(i * cols + j, 1. + a.get(i * cols + j));
						}
						else {
							a.set(i * cols + j, 0.);
						}
					}
				}
				double[] arr = a.stream().mapToDouble(d -> d).toArray();
				m.add(new MatrixBlock(rows, cols, arr));
			}
			return m;
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
