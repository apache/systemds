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

import java.util.Arrays;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class EstimationUtils 
{
	/**
	 * This utility function computes the exact output nnz
	 * of a self matrix product without need to materialize
	 * the output.
	 * 
	 * @param m1 dense or sparse input matrix
	 * @return exact output number of non-zeros.
	 */
	public static long getSelfProductOutputNnz(MatrixBlock m1) {
		final int m = m1.getNumRows();
		final int n = m1.getNumColumns();
		long retNnz = 0;
		
		if( m1.isInSparseFormat() ) {
			SparseBlock a = m1.getSparseBlock();
			SparseRowVector tmpS = new SparseRowVector(1024);
			double[] tmpD = null;
			
			for( int i=0; i<m; i++ ) {
				if( a.isEmpty(i) ) continue;
				int alen = a.size(i);
				int apos = a.pos(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				
				//compute number of aggregated non-zeros for input row
				int nnz1 = (int) Math.min(UtilFunctions.computeNnz(a, aix, apos, alen), n);
				boolean ldense = nnz1 > n / 128;
				
				//perform vector-matrix multiply w/ dense or sparse output
				if( ldense ) { //init dense tmp row
					tmpD = (tmpD == null) ? new double[n] : tmpD;
					Arrays.fill(tmpD, 0);
				}
				else {
					tmpS.setSize(0);
				}
				for( int k=apos; k<apos+alen; k++ ) {
					if( a.isEmpty(aix[k]) ) continue;
					int blen = a.size(aix[k]);
					int bpos = a.pos(aix[k]);
					int[] bix = a.indexes(aix[k]);
					double aval = avals[k];
					double[] bvals = a.values(aix[k]);
					if( ldense ) { //dense aggregation
						for( int j=bpos; j<bpos+blen; j++ )
							tmpD[bix[j]] += aval * bvals[j];
					}
					else { //sparse aggregation
						for( int j=bpos; j<bpos+blen; j++ )
							tmpS.add(bix[j], aval * bvals[j]);
					}
				}
				retNnz += !ldense ? tmpS.size() :
					UtilFunctions.computeNnz(tmpD, 0, n);
			}
		}
		else { //dense
			DenseBlock a = m1.getDenseBlock();
			double[] tmp = new double[n];
			for( int i=0; i<m; i++ ) {
				double[] avals = a.values(i);
				int aix = a.pos(i);
				Arrays.fill(tmp, 0); //reset
				for( int k=0; k<n; k++ ) {
					double aval = avals[aix+k];
					if( aval == 0 ) continue;
					double[] bvals = a.values(k);
					int bix = a.pos(k);
					for( int j=0; j<n; j++ )
						tmp[j] += aval * bvals[bix+j];
				}
				retNnz += UtilFunctions.computeNnz(tmp, 0, n);
			}
		}
		return retNnz;
	}
	
	public static long getSparseProductOutputNnz(MatrixBlock m1, MatrixBlock m2) {
		if( !m1.isInSparseFormat() || !m2.isInSparseFormat() )
			throw new DMLRuntimeException("Invalid call to sparse output nnz estimation.");
		
		final int m = m1.getNumRows();
		final int n2 = m2.getNumColumns();
		long retNnz = 0;
		
		SparseBlock a = m1.getSparseBlock();
		SparseBlock b = m2.getSparseBlock();
		
		SparseRowVector tmpS = new SparseRowVector(1024);
		double[] tmpD = null;
			
		for( int i=0; i<m; i++ ) {
			if( a.isEmpty(i) ) continue;
			int alen = a.size(i);
			int apos = a.pos(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			
			//compute number of aggregated non-zeros for input row
			int nnz1 = (int) Math.min(UtilFunctions.computeNnz(b, aix, apos, alen), n2);
			boolean ldense = nnz1 > n2 / 128;
			
			//perform vector-matrix multiply w/ dense or sparse output
			if( ldense ) { //init dense tmp row
				tmpD = (tmpD == null) ? new double[n2] : tmpD;
				Arrays.fill(tmpD, 0);
			}
			else {
				tmpS.setSize(0);
			}
			for( int k=apos; k<apos+alen; k++ ) {
				if( b.isEmpty(aix[k]) ) continue;
				int blen = b.size(aix[k]);
				int bpos = b.pos(aix[k]);
				int[] bix = b.indexes(aix[k]);
				double aval = avals[k];
				double[] bvals = b.values(aix[k]);
				if( ldense ) { //dense aggregation
					for( int j=bpos; j<bpos+blen; j++ )
						tmpD[bix[j]] += aval * bvals[j];
				}
				else { //sparse aggregation
					for( int j=bpos; j<bpos+blen; j++ )
						tmpS.add(bix[j], aval * bvals[j]);
				}
			}
			retNnz += !ldense ? tmpS.size() :
				UtilFunctions.computeNnz(tmpD, 0, n2);
		}
		return retNnz;
	}
}
