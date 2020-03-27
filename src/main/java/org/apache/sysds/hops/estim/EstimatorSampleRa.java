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
import org.apache.commons.math3.random.Well1024a;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

/**
 * This estimator implements an approach based on row/column sampling
 * 
 * Rasmus Resen Amossen, Andrea Campagna, Rasmus Pagh:
 * Better Size Estimation for Sparse Matrix Products. Algorithmica 69(3): 741-757 (2014)
 * 
 * Credit: This code is based on the original implementation provided by the authors,
 * modified to fit the SparsityEstimator API, support binary matrix products, avoid 
 * unnecessary file access, use Well1024a for seeding local RNGs, and generally 
 * improve performance.
 */
public class EstimatorSampleRa extends SparsityEstimator 
{
	private static final int RUNS = -1;
	private static final double SAMPLE_FRACTION = 0.1; //10%
	private static final double EPSILON = 0.05; // Multiplicative error
	private static final double DELTA = 0.1; // Probability of error
	private static final int K = -1;
	
	private final int _runs;
	private final double _sampleFrac; //sample fraction (0,1]
	private final double _eps; //target error
	private final double _delta; //probability of error
	private final int _k; //k-minimum hash values
	
	private final Well1024a _bigrand;
	
	private double[] h1; // hash "function" rows A
	private double[] h2; // hash "function" cols B
	private double[] h3; // hash "function" cols A
	private double[] h4; // hash "function" rows B
	
	public EstimatorSampleRa() {
		this(RUNS, SAMPLE_FRACTION, EPSILON, DELTA, K);
	}
	
	public EstimatorSampleRa(double sampleFrac) {
		this(RUNS, sampleFrac, EPSILON, DELTA, K);
	}
	
	public EstimatorSampleRa(int runs, double sampleFrac, double eps, double delta, int k) {
		if( sampleFrac <= 0 || sampleFrac > 1.0 )
			throw new DMLRuntimeException("Invalid sample fraction: "+sampleFrac);
		_sampleFrac = sampleFrac;
		_eps = eps;
		_delta = delta;
		
		//if runs/k not specified compute from epsilon and delta
		_runs = (runs < 0) ? (int) (Math.log(1/_delta) / Math.log(2)) : runs;
		_k = (k < 0) ? (int) Math.ceil(1 / (_eps * _eps)) : k;
		
		//construct Well1024a generator for good random numbers
		_bigrand = LibMatrixDatagen.setupSeedsForRand(_k);
	}
	
	@Override
	public DataCharacteristics estim(MMNode root) {
		LOG.warn("Recursive estimates not supported by EstimatorSampleRa,"
			+ " falling back to EstimatorBasicAvg.");
		return new EstimatorBasicAvg().estim(root);
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
		// perform runs to obtain desired precision (Chernoﬀ bound)
		double[] results = new double[_runs];
		for(int i=0; i<_runs; i++) {
			initHashArrays(m1.getNumRows(),
				m1.getNumColumns(), m2.getNumColumns());
			results[i] = estimateSize(m1, m2);
		}
		
		//compute estimate as median of all results
		//error bound: nnz*(1-10/sqrt(k)), nnz*(1+10/sqrt(k)));
		Arrays.sort(results);
		long nnz = (long) results[_runs/2];
		
		//convert from nnz to sparsity
		return OptimizerUtils.getSparsity(
			m1.getNumRows(), m2.getNumColumns(), nnz);
	}
	
	private void initHashArrays(int m, int n, int l) {
		if( h1 == null ) {
			h1 = new double[m];
			h2 = new double[l];
			h3 = new double[l];
			h4 = new double[m];
		}
		
		//create local random number generator
		Random rand = new Random(_bigrand.nextLong());
		for(int t=0; t < h1.length; t++)
			h1[t] = rand.nextDouble();
		for(int t=0; t < h2.length; t++)
			h2[t] = rand.nextDouble();
		for(int t=0; t < h3.length; t++)
			h3[t] = rand.nextDouble();
		for(int t=0; t < h4.length; t++)
			h4[t] = rand.nextDouble();
	}
	
	private double estimateSize(MatrixBlock mb1, MatrixBlock mb2) {
		AdjacencyLists A = new AdjacencyLists(mb1, false);
		AdjacencyLists C = new AdjacencyLists(mb2, true);
		ArrayList<Double> sketch = new ArrayList<>();
		
		//pick a large p, it will soon be decreased anyway
		double p = 1;
		int bufferSize = 0;
		
		for( int i=0; i<mb1.getNumColumns(); i++ ) {
			ArrayList<Integer> Ai = A.getList(i);
			ArrayList<Integer> Ci = C.getList(i);
			if( Ai.isEmpty() || Ci.isEmpty() )
				continue;
			
			//get Ai and Ci sorted by hash values h1, h2
			Integer[] x = Ai.stream().sorted(Comparator.comparing(a -> h1[a])).toArray(Integer[]::new);
			Integer[] y = Ci.stream().sorted(Comparator.comparing(a -> h2[a])).toArray(Integer[]::new);

			int s = 0;
			int sHat = 0;
			for(int t=0; t<y.length; t++) {
				int xIdx = (sHat > 0) ? sHat-1 : x.length-1;
				while( h(x[sHat], y[t]) > h(x[xIdx], y[t]))
					sHat = (sHat + 1) % x.length;
				s = sHat;
				//column sampling
				if(h3[y[t]] > _sampleFrac)
					continue;
				int num = 0;
				while(h(x[s], y[t]) < p && num < x.length) {
					//row sampling
					if(h4[x[s]] > _sampleFrac) {
						s = (s + 1) % x.length;
						num++;
						continue;
					}
					//add hash to sketch
					sketch.add(h(x[s], y[t]));
					bufferSize++;
					//truncate to size k if necessary
					if(bufferSize > _k) {
						sortAndTruncate(sketch);
						if (sketch.size()==_k)
							p = sketch.get(sketch.size()-1);
						bufferSize = 0;
					}
					s = (s + 1) % x.length;
					num++;
				}
			}
		}

		//all pairs generated, truncate and finally estimate size
		sortAndTruncate(sketch);
		if(sketch.size() == _k) {
			//k'th smallest elements are at the top in the sketch
			double v = sketch.get(sketch.size()-1);
			return _k/(v*_sampleFrac*_sampleFrac);
		}
		else {
			return sketch.size()/(_sampleFrac*_sampleFrac);
		}
	}
	
	public void sortAndTruncate(ArrayList<Double> sketch) {
		Collections.sort(sketch);
		//remove duplicates (within some epsilon precision)
		for(int t=1; t < sketch.size(); t++) {
			//sketch.get(t) is always larger than sketch.get(t-1)
			if(sketch.get(t)/sketch.get(t-1) < (1+1.0E-10)) {
				sketch.remove(t); t--;
			}
		}
		//truncate after the first k elements
		sketch.subList(Math.min(sketch.size(),_k), sketch.size()).clear();
	}
	
	public double h(int x, int y) {
		//h(x,y) hash function
		double a = (h1[x] - h2[y]);
		return (a < 0) ? a + 1 : a;
	}
	
	private static class AdjacencyLists {
		private ArrayList<Integer>[] indexes;
		
		@SuppressWarnings("unchecked")
		public AdjacencyLists(MatrixBlock mb, boolean row) {
			int len = row ? mb.getNumRows() : mb.getNumColumns();
			indexes = new ArrayList[len];
			for(int i=0; i<len; i++)
				indexes[i] = new ArrayList<>();
			if( mb.isEmptyBlock(false) )
				return; //early abort
			if( mb.isInSparseFormat() ) {
				SparseBlock sblock = mb.getSparseBlock();
				for(int i=0; i<sblock.numRows(); i++)  {
					if( sblock.isEmpty(i) ) continue;
					int apos = sblock.pos(i);
					int alen = sblock.size(i);
					int[] aix = sblock.indexes(i);
					for(int k=apos; k<apos+alen; k++)
						indexes[row?i:aix[k]].add(row?aix[k]:i);
				}
			}
			else {
				for(int i=0; i<mb.getNumRows(); i++)
					for(int j=0; j<mb.getNumColumns(); j++)
						if( mb.quickGetValue(i, j) != 0 )
							indexes[row?i:j].add(row?j:i);
			}
		}
		
		public ArrayList<Integer> getList(int i) {
			return indexes[i];
		}
	}
}
