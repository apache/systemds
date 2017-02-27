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

package org.apache.sysml.runtime.codegen;

import org.apache.sysml.runtime.matrix.data.LibMatrixMult;

/**
 * This library contains all vector primitives that are used in 
 * generated source code for fused operators. For primitives that
 * exist in LibMatrixMult, these calls are simply forwarded to
 * ensure consistency in performance and result correctness. 
 *
 */
public class LibSpoofPrimitives 
{
	// forwarded calls to LibMatrixMult
	
	public static double dotProduct( double[] a, double[] b, int ai, int bi, final int len ) {
		return LibMatrixMult.dotProduct(a, b, ai, bi, len);
	}
	
	public static double dotProduct( double[] a, double[] b, int[] aix, int ai, final int bi, final int len ) {
		return LibMatrixMult.dotProduct(a, b, aix, ai, bi, len);
	}
	
	public static void vectMultiplyAdd( final double aval, double[] b, double[] c, int bi, int ci, final int len ) {
		LibMatrixMult.vectMultiplyAdd(aval, b, c, bi, ci, len);
	}
	
	public static void vectMultiplyAdd( final double aval, double[] b, double[] c, int[] bix, final int bi, final int ci, final int len ) {
		LibMatrixMult.vectMultiplyAdd(aval, b, c, bix, bi, ci, len);
	}
	
	public static void vectMultiplyWrite( final double aval, double[] b, double[] c, int bi, int ci, final int len ) {
		LibMatrixMult.vectMultiplyWrite(aval, b, c, bi, ci, len);
	}

	// custom methods
	
	/**
	 * Computes c = sum(A), where A is a dense vectors. 
	 * 
	 * @param a dense input vector A
	 * @param ai start position in A
	 * @param len number of processed elements
	 * @return sum value
	 */
	public static double vectSum( double[] a, int ai, final int len ) { 
		double val = 0;
		final int bn = len%8;
				
		//compute rest
		for( int i = 0; i < bn; i++, ai++ )
			val += a[ ai ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int i = bn; i < len; i+=8, ai+=8 ) {
			//read 64B cacheline of a, compute cval' = sum(a) + cval
			val += a[ ai+0 ] + a[ ai+1 ] + a[ ai+2 ] + a[ ai+3 ]
			     + a[ ai+4 ] + a[ ai+5 ] + a[ ai+6 ] + a[ ai+7 ];
		}
		
		//scalar result
		return val; 
	} 
	
	/**
	 * Computes c = sum(A), where A is a sparse vector. 
	 * 
	 * @param avals sparse input vector A values A
	 * @param aix sparse input vector A column indexes
	 * @param ai start position in A
	 * @param len number of processed elements
	 * @return sum value
	 */
	public static double vectSum( double[] avals, int[] aix, int ai, int len) {
		double val = 0;
		final int bn = len%8;
				
		//compute rest
		for( int i = ai; i < ai+bn; i++ )
			val += avals[ ai+aix[i] ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int i = ai+bn; i < ai+len; i+=8 )
		{
			//read 64B of a via 'gather'
			//compute cval' = sum(a) + cval
			val += avals[ ai+aix[i+0] ] + avals[ ai+aix[i+1] ]
			     + avals[ ai+aix[i+2] ] + avals[ ai+aix[i+3] ]
			     + avals[ ai+aix[i+4] ] + avals[ ai+aix[i+5] ]
			     + avals[ ai+aix[i+6] ] + avals[ ai+aix[i+7] ];
		}
		
		//scalar result
		return val; 
	} 
	
	/**
	 * Computes C += A / b, where C and A are dense vectors and b is a scalar. 
	 * 
	 * @param a dense input vector A
	 * @param bval input scalar b 
	 * @param c dense input-output vector C
	 * @param ai start position in A
	 * @param ci start position in C
	 * @param len number of processed elements.
	 */
	public static void vectDivAdd( double[] a, final double bval, double[] c, int ai, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] +=  a[ ai ] / bval;
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8, ci+=8) 
		{
			//read 64B cachelines of b and c
			//compute c' = aval * b + c
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += a[ ai+0 ] / bval;
			c[ ci+1 ] += a[ ai+1 ] / bval;
			c[ ci+2 ] += a[ ai+2 ] / bval;
			c[ ci+3 ] += a[ ai+3 ] / bval;
			c[ ci+4 ] += a[ ai+4 ] / bval;
			c[ ci+5 ] += a[ ai+5 ] / bval;
			c[ ci+6 ] += a[ ai+6 ] / bval;
			c[ ci+7 ] += a[ ai+7 ] / bval;
		}
	} 
	
	/**
	 * Computes C += A / b, where C is a dense vector, A is a sparse vector, and b is a scalar. 
	 * 
	 * @param a sparse input vector A values
	 * @param bval input scalar b 
	 * @param c dense input-output vector C
	 * @param aix sparse input vector A column indexes
	 * @param ai start position in A
	 * @param ci start position in C
	 * @param len number of processed elements.
	 */
	public static void vectDivAdd( double[] a, final double bval, double[] c, int[] aix, final int ai, final int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = ai; j < ai+bn; j++ )
			c[ ci + aix[j] ] += a[ j ] / bval;
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = ai+bn; j < ai+len; j+=8 )
		{
			//read 64B cacheline of b
			//read 64B of c via 'gather'
			//compute c' = aval * b + c
			//write back 64B of c = c' via 'scatter'
			c[ ci+aix[j+0] ] += a[ j+0 ] / bval;
			c[ ci+aix[j+1] ] += a[ j+1 ] / bval;
			c[ ci+aix[j+2] ] += a[ j+2 ] / bval;
			c[ ci+aix[j+3] ] += a[ j+3 ] / bval;
			c[ ci+aix[j+4] ] += a[ j+4 ] / bval;
			c[ ci+aix[j+5] ] += a[ j+5 ] / bval;
			c[ ci+aix[j+6] ] += a[ j+6 ] / bval;
			c[ ci+aix[j+7] ] += a[ j+7 ] / bval;
		}
	}
	
	/**
	 * Computes C = A / b, where C and A are dense vectors, and b is a scalar. 
	 * 
	 * @param a dense input vector A
	 * @param bval input scalar b 
	 * @param c dense input-output vector C
	 * @param ai start position in A
	 * @param ci start position in C
	 * @param len number of processed elements.
	 */
	public static void vectDivWrite( double[] a, final double bval, double[] c, int ai, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] = a[ ai ] / bval;
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8, ci+=8) 
		{
			//read 64B cachelines of a and c
			//compute c' = a / bval + c
			//write back 64B cacheline of c = c'
			c[ ci+0 ] = a[ ai+0 ] / bval;
			c[ ci+1 ] = a[ ai+1 ] / bval;
			c[ ci+2 ] = a[ ai+2 ] / bval;
			c[ ci+3 ] = a[ ai+3 ] / bval;
			c[ ci+4 ] = a[ ai+4 ] / bval;
			c[ ci+5 ] = a[ ai+5 ] / bval;
			c[ ci+6 ] = a[ ai+6 ] / bval;
			c[ ci+7 ] = a[ ai+7 ] / bval;
		}
	}
	
	/**
	 * Computes C = A / b, where C is a dense vector and A is a sparse vector, and b is a scalar. 
	 * 
	 * @param a sparse input vector A values
	 * @param aix sparse input vector A column indexes
	 * @param bval input scalar b 
	 * @param c dense input-output vector C
	 * @param ai start position in A
	 * @param ci start position in C
	 * @param len number of processed elements.
	 */
	public static void vectDivWrite( double[] a, int[] aix, final double bval, double[] c, final int ai, final int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = ai; j < ai+bn; j++ )
			c[ ci + aix[j] ] += a[ j ] / bval;
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = ai+bn; j < ai+len; j+=8 )
		{
			//read 64B cachelines of a, compute c = a/b
			//and write back c via 'scatter'
			c[ ci+aix[j+0] ] = a[ j+0 ] / bval;
			c[ ci+aix[j+1] ] = a[ j+1 ] / bval;
			c[ ci+aix[j+2] ] = a[ j+2 ] / bval;
			c[ ci+aix[j+3] ] = a[ j+3 ] / bval;
			c[ ci+aix[j+4] ] = a[ j+4 ] / bval;
			c[ ci+aix[j+5] ] = a[ j+5 ] / bval;
			c[ ci+aix[j+6] ] = a[ j+6 ] / bval;
			c[ ci+aix[j+7] ] = a[ j+7 ] / bval;
		}
	}
}
