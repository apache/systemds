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
package org.apache.sysds.runtime.matrix.data;


import org.apache.sysds.utils.NativeHelper;


public class LibMatrixDNNHelper 
{
	protected static class CellIndex3 {
		public int ix1;
		public int ix2;
		public int ix3;
		@Override
		public String toString() {
			return "("+ix1+", "+ix2+", "+ix3+")";
		}
	}
	
	protected static CellIndex3 computeTensorIndexes(int j, int H, int W) {
		return computeTensorIndexes(j, H, W, new CellIndex3());
	}
	
	/**
	 * Computes tensor indexes from a linearized column index such that
	 * the column index is equal to ix1*NM + ix2*M + ix3
	 * 
	 * @param j column index
	 * @param N second last dimension
	 * @param M last dimension
	 * @param ret output object for reuse
	 * @return tensor indexes
	 */
	protected static CellIndex3 computeTensorIndexes(int j, int N, int M, CellIndex3 ret) {
		int tmp = j / M;
		ret.ix1 = tmp / N;
		ret.ix2 = tmp % N;
		ret.ix3 = j % M;
		return ret;
	}
	
	protected static void singleThreadedMatMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
		boolean recomputeNNZM1, boolean recomputeNNZM2, DnnParameters params) {
		if( !params.enableNative || m1.sparse || m2.sparse ) {
			prepNonZerosForMatrixMult(m1, recomputeNNZM1);
			prepNonZerosForMatrixMult(m2, recomputeNNZM2);
			LibMatrixMult.matrixMult(m1, m2, ret, true);
		}
		else {
			ret.sparse = false;
			if(ret.getDenseBlock() == null)
				ret.allocateDenseBlock();
			NativeHelper.dmmdd(m1.getDenseBlockValues(), m2.getDenseBlockValues(),
				ret.getDenseBlockValues(), m1.rlen, m1.clen, m2.clen, 1);
		}
		
		//no need to maintain nnz exactly, as consumed by other operations
		ret.setNonZeros((long)ret.rlen*ret.clen);
	}
	
	private static void prepNonZerosForMatrixMult(MatrixBlock mb, boolean update) {
		if( !update )
			return;
		//non-zeros are not evaluated for dense matrix multiplies
		//so we simply need to ensure the block is not marked empty 
		if( !mb.isInSparseFormat() )
			mb.setNonZeros((long)mb.getNumRows() * mb.getNumColumns());
		else
			mb.recomputeNonZeros();
	}
}
