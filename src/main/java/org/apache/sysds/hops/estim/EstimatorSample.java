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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * This estimator implements an approach based on row/column sampling
 * Yongyang Yu, MingJie Tang, Walid G. Aref, Qutaibah M. Malluhi, Mostafa M. Abbas, Mourad Ouzzani:
 * In-Memory Distributed Matrix Computation Processing and Optimization. ICDE 2017: 1047-1058
 * 
 * The basic idea is to draw random samples of aligned columns SA and rows SB,
 * and compute the output nnz as max(nnz(SA_i)*nnz(SB_i)). However, this estimator is
 * biased toward underestimation as the maximum is unlikely sampled and collisions are
 * not accounted for. Accordingly, we also support an extended estimator that relies
 * on similar ideas for element-wise addition as the other estimators.
 */
public class EstimatorSample extends SparsityEstimator
{
	private static final double SAMPLE_FRACTION = 0.1; //10%
	
	private final double _frac;
	private final boolean _extended;
	
	public EstimatorSample() {
		this(SAMPLE_FRACTION, false);
	}
	
	public EstimatorSample(double sampleFrac) {
		this(sampleFrac, false);
	}
	
	public EstimatorSample(double sampleFrac, boolean extended) {
		if( sampleFrac <= 0 || sampleFrac > 1.0 )
			throw new DMLRuntimeException("Invalid sample fraction: "+sampleFrac);
		_frac = sampleFrac;
		_extended = extended;
	}
	
	@Override
	public DataCharacteristics estim(MMNode root) {
		LOG.warn("Recursive estimates not supported by EstimatorSample, falling back to EstimatorBasicAvg.");
		return new EstimatorBasicAvg().estim(root);
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		switch(op) {
			case MM: {
				int k =  m1.getNumColumns();
				int[] ix = UtilFunctions.getSortedSampleIndexes(
					k, (int)Math.max(k*_frac, 1));
				int p = ix.length;
				int[] cnnz = computeColumnNnz(m1, ix);
				if( _extended ) {
					double ml = (long)m1.getNumRows()*m2.getNumColumns();
					double sumS = 0, prodS = 1;
					for(int i=0; i<ix.length; i++) {
						long rnnz = m2.recomputeNonZeros(ix[i], ix[i]);
						double v = (double)cnnz[i] * rnnz /ml;
						sumS += v;
						prodS *= 1-v;
					}
					return 1-Math.pow(1-1d/p * sumS, k - p) * prodS;
				}
				else {
					//biased sampling-based estimator
					long nnzOut = 0;
					for(int i=0; i<p; i++)
						nnzOut = Math.max(nnzOut, cnnz[i] * m2.recomputeNonZeros(ix[i], ix[i]));
					return OptimizerUtils.getSparsity( 
						m1.getNumRows(), m2.getNumColumns(), nnzOut);
				}
			}
			case MULT: {
				int k = Math.max(m1.getNumColumns(), m1.getNumRows());
				int[] ix = UtilFunctions.getSortedSampleIndexes(
					k, (int)Math.max(k*_frac, 1));
				double spOut = 0;
				if( m1.getNumColumns() > m1.getNumRows() ) {
					int[] cnnz1 = computeColumnNnz(m1, ix);
					int[] cnnz2 = computeColumnNnz(m2, ix);
					for(int i=0; i<ix.length; i++)
						spOut += ((double)cnnz1[i]/m1.getNumRows()) 
							* ((double)cnnz2[i]/m1.getNumRows());
				}
				else {
					int[] rnnz1 = computeRowNnz(m1, ix);
					int[] rnnz2 = computeRowNnz(m2, ix);
					for(int i=0; i<ix.length; i++)
						spOut += ((double)rnnz1[i]/m1.getNumColumns()) 
							* ((double)rnnz2[i]/m1.getNumColumns());
				}
				return spOut/ix.length;
			}
			case PLUS: {
				int k = Math.max(m1.getNumColumns(), m1.getNumRows());
				int[] ix = UtilFunctions.getSortedSampleIndexes(
					k, (int)Math.max(k*_frac, 1));
				double spOut = 0;
				if( m1.getNumColumns() > m1.getNumRows() ) {
					int[] cnnz1 = computeColumnNnz(m1, ix);
					int[] cnnz2 = computeColumnNnz(m2, ix);
					for(int i=0; i<ix.length; i++) {
						spOut += ((double)cnnz1[i]/m1.getNumRows()) 
							+ ((double)cnnz2[i]/m1.getNumRows())
							- ((double)cnnz1[i]/m1.getNumRows())
							* ((double)cnnz2[i]/m1.getNumRows());
					}
				}
				else {
					int[] rnnz1 = computeRowNnz(m1, ix);
					int[] rnnz2 = computeRowNnz(m2, ix);
					for(int i=0; i<ix.length; i++) {
						spOut += ((double)rnnz1[i]/m1.getNumColumns()) 
							+ ((double)rnnz2[i]/m1.getNumColumns())
							- ((double)rnnz1[i]/m1.getNumColumns()) 
							* ((double)rnnz2[i]/m1.getNumColumns());
					}
				}
				return spOut/ix.length;
			}
			case RBIND:
			case CBIND:
			case EQZERO:
			case NEQZERO:
			case TRANS:
			case DIAG:
			case RESHAPE:
				DataCharacteristics mc1 = m1.getDataCharacteristics();
				DataCharacteristics mc2 = m2.getDataCharacteristics();
				return OptimizerUtils.getSparsity(estimExactMetaData(mc1, mc2, op));
			default:
				throw new NotImplementedException();
		}
	}
	
	@Override
	public double estim(MatrixBlock m, OpCode op) {
		return estim(m, null, op);
	}
	
	private static int[] computeColumnNnz(MatrixBlock in, int[] ix) {
		int[] nnz = new int[in.getNumColumns()];
		//count column nnz brute force or selective
		if( in.isInSparseFormat() ) {
			SparseBlock sblock = in.getSparseBlock();
			for( int i=0; i<in.getNumRows(); i++ ) {
				if( sblock.isEmpty(i) ) continue;
				LibMatrixAgg.countAgg(sblock.values(i), nnz,
					sblock.indexes(i), sblock.pos(i), sblock.size(i));
			}
		}
		else {
			DenseBlock dblock = in.getDenseBlock();
			for( int i=0; i<in.getNumRows(); i++ ) {
				double[] avals = dblock.values(i);
				int aix = dblock.pos(i);
				for( int j=0; j<in.getNumColumns(); j++ )
					nnz[j] += (avals[aix+j] != 0) ? 1 : 0;
			}
		}
		
		//copy nnz into reduced vector
		int[] ret = new int[ix.length];
		for(int i=0; i<ix.length; i++)
			ret[i] = nnz[ix[i]];
		return ret;
	}
	
	private static int[] computeRowNnz(MatrixBlock in, int[] ix) {
		//copy nnz into reduced vector
		int[] ret = new int[ix.length];
		for(int i=0; i<ix.length; i++)
			ret[i] = (int) in.recomputeNonZeros(ix[i], ix[i]);
		return ret;
	}
}
