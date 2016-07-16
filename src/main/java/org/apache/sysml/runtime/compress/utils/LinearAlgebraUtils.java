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

import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Various low-level primitives for compressed matrix blocks, some of which
 * were copied from LibMatrixMult.
 * 
 */
public class LinearAlgebraUtils {

	/**
	 * 
	 * @param a
	 * @param b
	 * @param len
	 * @return
	 */
	public static double dotProduct(double[] a, double[] b, final int len) 
	{
		double val = 0;
		final int bn = len % 8;

		// compute rest
		for (int i = 0; i < bn; i++)
			val += a[i] * b[i];

		// unrolled 8-block (for better instruction-level parallelism)
		for (int i = bn; i < len; i += 8) {
			// read 64B cachelines of a and b
			// compute cval' = sum(a * b) + cval
			val += a[i + 0] * b[i + 0] 
				 + a[i + 1] * b[i + 1] 
				 + a[i + 2] * b[i + 2] 
				 + a[i + 3] * b[i + 3] 
				 + a[i + 4] * b[i + 4]
				 + a[i + 5] * b[i + 5] 
				 + a[i + 6] * b[i + 6] 
				 + a[i + 7] * b[i + 7];
		}

		// scalar result
		return val;
	}
	
	/**
	 * 
	 * @param a
	 * @param b
	 * @param ai
	 * @param bi
	 * @param len
	 * @return
	 */
	public static double dotProduct( double[] a, double[] b, int ai, int bi, final int len )
	{
		double val = 0;
		final int bn = len%8;
				
		//compute rest
		for( int i = 0; i < bn; i++, ai++, bi++ )
			val += a[ ai ] * b[ bi ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int i = bn; i < len; i+=8, ai+=8, bi+=8 )
		{
			//read 64B cachelines of a and b
			//compute cval' = sum(a * b) + cval
			val += a[ ai+0 ] * b[ bi+0 ]
			     + a[ ai+1 ] * b[ bi+1 ]
			     + a[ ai+2 ] * b[ bi+2 ]
			     + a[ ai+3 ] * b[ bi+3 ]
			     + a[ ai+4 ] * b[ bi+4 ]
			     + a[ ai+5 ] * b[ bi+5 ]
			     + a[ ai+6 ] * b[ bi+6 ]
			     + a[ ai+7 ] * b[ bi+7 ];
		}
		
		//scalar result
		return val; 
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param ci
	 * @param len
	 */
	public static void vectAdd( double[] a, double[] c, int ai, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] += a[ ai ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8, ci+=8) 
		{
			//read 64B cachelines of a and c
			//compute c' = c * a
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += a[ ai+0 ];
			c[ ci+1 ] += a[ ai+1 ];
			c[ ci+2 ] += a[ ai+2 ];
			c[ ci+3 ] += a[ ai+3 ];
			c[ ci+4 ] += a[ ai+4 ];
			c[ ci+5 ] += a[ ai+5 ];
			c[ ci+6 ] += a[ ai+6 ];
			c[ ci+7 ] += a[ ai+7 ];
		}
	}
	
	/**
	 * 
	 * @param aval
	 * @param b
	 * @param c
	 * @param bix
	 * @param ci
	 * @param len
	 */
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
	
	/**
	 * 
	 * @param aval
	 * @param c
	 * @param ci
	 * @param len
	 */
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
	
	/**
	 * 
	 * @param aval
	 * @param b
	 * @param c
	 * @param bix
	 * @param bi
	 * @param ci
	 * @param len
	 */
	public static void vectMultiplyAdd( final double aval, double[] b, double[] c, int[] bix, final int bi, final int ci, final int len )
	{
		final int bn = (len-bi)%8;
		
		//rest, not aligned to 8-blocks
		for( int j = bi; j < bi+bn; j++ )
			c[ ci + bix[j] ] += aval * b[ j ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bi+bn; j < len; j+=8 )
		{
			c[ ci+bix[j+0] ] += aval * b[ j+0 ];
			c[ ci+bix[j+1] ] += aval * b[ j+1 ];
			c[ ci+bix[j+2] ] += aval * b[ j+2 ];
			c[ ci+bix[j+3] ] += aval * b[ j+3 ];
			c[ ci+bix[j+4] ] += aval * b[ j+4 ];
			c[ ci+bix[j+5] ] += aval * b[ j+5 ];
			c[ ci+bix[j+6] ] += aval * b[ j+6 ];
			c[ ci+bix[j+7] ] += aval * b[ j+7 ];
		}
	}
	
	/**
	 * 
	 * @param aval
	 * @param b
	 * @param c
	 * @param bi
	 * @param ci
	 * @param len
	 */
	public static void vectMultiplyAdd( final double aval, double[] b, double[] c, int bi, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, bi++, ci++)
			c[ ci ] += aval * b[ bi ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, bi+=8, ci+=8) 
		{
			//read 64B cachelines of b and c
			//compute c' = aval * b + c
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += aval * b[ bi+0 ];
			c[ ci+1 ] += aval * b[ bi+1 ];
			c[ ci+2 ] += aval * b[ bi+2 ];
			c[ ci+3 ] += aval * b[ bi+3 ];
			c[ ci+4 ] += aval * b[ bi+4 ];
			c[ ci+5 ] += aval * b[ bi+5 ];
			c[ ci+6 ] += aval * b[ bi+6 ];
			c[ ci+7 ] += aval * b[ bi+7 ];
		}
	}

	/**
	 * 
	 * @param a
	 * @param aix
	 * @param ai
	 * @param ai2
	 * @param len
	 * @return
	 */
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
	
	/**
	 * 
	 * @param a
	 * @param ai
	 * @param len
	 * @return
	 */
	public static double vectSum( double[] a, int ai, final int len )
	{
		double val = 0;
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++ )
			val += a[ ai ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8 )
		{
			val += a[ ai+0 ]
			     + a[ ai+1 ]
			     + a[ ai+2 ]
			     + a[ ai+3 ]
			     + a[ ai+4 ]
			     + a[ ai+5 ]
			     + a[ ai+6 ]
			     + a[ ai+7 ];
		}
		
		return val;
	}
	
	/**
	 * 
	 * @param ret
	 */
	public static void copyUpperToLowerTriangle( MatrixBlock ret )
	{
		double[] c = ret.getDenseBlock();
		final int m = ret.getNumRows();
		final int n = ret.getNumColumns();
		
		//copy symmetric values
		for( int i=0, uix=0; i<m; i++, uix+=n )
			for( int j=i+1, lix=j*n+i; j<n; j++, lix+=n )
				c[ lix ] = c[ uix+j ];
	}
	
	/**
	 * 
	 * @param ret
	 * @param tmp
	 * @param ix
	 */
	public static void copyNonZerosToRowCol( MatrixBlock ret, MatrixBlock tmp, int ix )
	{
		for(int i=0; i<tmp.getNumColumns(); i++) {
			double val = tmp.quickGetValue(0, i);
			if( val != 0 ) {
				ret.setValueDenseUnsafe(ix, i, val);
				ret.setValueDenseUnsafe(i, ix, val);
			}
		}
	}
	
	/**
	 * 
	 * @param a
	 * @param x
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
