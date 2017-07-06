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

package org.apache.sysml.runtime.compress.utils;

import org.apache.sysml.runtime.matrix.data.LibMatrixMult;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * This library contains all vector primitives that are used compressed 
 * linear algebra. For primitives that exist in LibMatrixMult, these 
 * calls are simply forwarded to ensure consistency in performance and 
 * result correctness. 
 */
public class LinearAlgebraUtils 
{
	//forwarded calls to LibMatrixMult
	
	public static double dotProduct(double[] a, double[] b, final int len) {
		return LibMatrixMult.dotProduct(a, b, 0, 0, len);
	}

	public static double dotProduct( double[] a, double[] b, int ai, int bi, final int len ) {
		return LibMatrixMult.dotProduct(a, b, ai, bi, len);
	}
		
	public static void vectMultiplyAdd( final double aval, double[] b, double[] c, int bi, int ci, final int len ) {
		LibMatrixMult.vectMultiplyAdd(aval, b, c, bi, ci, len);
	}
	
	public static void vectMultiplyAdd( final double aval, double[] b, double[] c, int[] bix, final int bi, final int ci, final int len ) {
		LibMatrixMult.vectMultiplyAdd(aval, b, c, bix, bi, ci, len);
	}

	public static void vectAdd( double[] a, double[] c, int ai, int ci, final int len ) {
		LibMatrixMult.vectAdd(a, c, ai, ci, len);
	}

	public static void vectAdd( final double aval, double[] c, char[] bix, final int bi, final int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = bi; j < bi+bn; j++ )
			c[ ci + bix[j] ] += aval;
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bi+bn; j < bi+len; j+=8 )
		{
			c[ ci+bix[j+0] ] += aval;
			c[ ci+bix[j+1] ] += aval;
			c[ ci+bix[j+2] ] += aval;
			c[ ci+bix[j+3] ] += aval;
			c[ ci+bix[j+4] ] += aval;
			c[ ci+bix[j+5] ] += aval;
			c[ ci+bix[j+6] ] += aval;
			c[ ci+bix[j+7] ] += aval;
		}
	}

	public static void vectAdd( final double aval, double[] c, final int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++ )
			c[ ci + j ] += aval;
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8 )
		{
			c[ ci+j+0 ] += aval;
			c[ ci+j+1 ] += aval;
			c[ ci+j+2 ] += aval;
			c[ ci+j+3 ] += aval;
			c[ ci+j+4 ] += aval;
			c[ ci+j+5 ] += aval;
			c[ ci+j+6 ] += aval;
			c[ ci+j+7 ] += aval;
		}
	}


	public static double vectSum( double[] a, char[] bix, final int ai, final int bi, final int len )
	{
		double val = 0;
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = bi; j < bi+bn; j++ )
			val += a[ ai + bix[j] ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bi+bn; j < bi+len; j+=8 )
		{
			val += a[ ai+bix[j+0] ]
			     + a[ ai+bix[j+1] ]
			     + a[ ai+bix[j+2] ]
			     + a[ ai+bix[j+3] ]
			     + a[ ai+bix[j+4] ]
			     + a[ ai+bix[j+5] ]
			     + a[ ai+bix[j+6] ]
			     + a[ ai+bix[j+7] ];
		}
		
		return val;
	}

	public static double vectSum( double[] a, int ai, final int len )
	{
		double val = 0;
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = ai; j < ai+bn; j++ )
			val += a[ j ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = ai+bn; j < ai+len; j+=8 ) {
			val += a[ j+0 ] + a[ j+1 ] + a[ j+2 ] + a[ j+3 ]
			     + a[ j+4 ] + a[ j+5 ] + a[ j+6 ] + a[ j+7 ];
		}
		
		return val;
	}

	public static long copyUpperToLowerTriangle( MatrixBlock ret ) {
		return LibMatrixMult.copyUpperToLowerTriangle(ret);
	}
	
	public static void copyNonZerosToUpperTriangle( MatrixBlock ret, MatrixBlock tmp, int ix ) {
		double[] a = tmp.getDenseBlock();
		for(int i=0; i<tmp.getNumColumns(); i++) {
			if( a[i] != 0 ) {
				ret.setValueDenseUnsafe(
					(ix<i)?ix:i, (ix<i)?i:ix, a[i]);
			}
		}
	}
	
	/**
	 * Obtain the index of the closest element in a to the value x.
	 * 
	 * @param a array of ints
	 * @param x value
	 * @return the index of the closest element in a to the value x
	 */
	public static int getClosestK(int[] a, int x) {

		int low = 0;
		int high = a.length - 1;

		while (low < high) {
			int mid = (low + high) / 2;
			int d1 = Math.abs(a[mid] - x);
			int d2 = Math.abs(a[mid + 1] - x);
			if (d2 <= d1) {
				low = mid + 1;
			} else {
				high = mid;
			}
		}
		return high;
	}
}
