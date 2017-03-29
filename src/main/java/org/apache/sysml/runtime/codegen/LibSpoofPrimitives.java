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
	
	public static double dotProduct(double[] a, double[] b, int ai, int bi, int len) {
		return LibMatrixMult.dotProduct(a, b, ai, bi, len);
	}
	
	public static double dotProduct(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		return LibMatrixMult.dotProduct(a, b, aix, ai, bi, len);
	}
	
	public static void vectMultAdd(double aval, double[] b, double[] c, int bi, int ci, int len) {
		LibMatrixMult.vectMultiplyAdd(aval, b, c, bi, ci, len);
	}
	
	public static void vectMultAdd(double aval, double[] b, double[] c, int[] bix, int bi, int ci, int len) {
		LibMatrixMult.vectMultiplyAdd(aval, b, c, bix, bi, ci, len);
	}
	
	public static void vectMultWrite(double aval, double[] b, double[] c, int bi, int ci, int len) {
		LibMatrixMult.vectMultiplyWrite(aval, b, c, bi, ci, len);
	}
	
	public static void vectMultWrite(double aval, double[] b, double[] c, int[] bix, int bi, int ci, int len) {
		LibMatrixMult.vectMultiplyAdd(aval, b, c, bix, bi, ci, len);
	}

	// custom vector sums
	
	/**
	 * Computes c = sum(A), where A is a dense vectors. 
	 * 
	 * @param a dense input vector A
	 * @param ai start position in A
	 * @param len number of processed elements
	 * @return sum value
	 */
	public static double vectSum(double[] a, int ai, int len) { 
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
	public static double vectSum(double[] avals, int[] aix, int ai, int len) {
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
	
	//custom vector div
	
	public static void vectDivAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] / bval;
	} 

	public static void vectDivAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += a[j] / bval;
	}
	
	public static void vectDivWrite(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] = a[j] / bval;
	}

	public static void vectDivWrite(double bval, double[] a, int[] aix, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] = a[j] / bval;
	}
	
	//custom vector equal
	
	public static void vectEqualAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] == bval) ? 1 : 0;
	} 

	public static void vectEqualAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] == bval) ? 1 : 0;
	}
	
	public static void vectEqualWrite(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] = (a[j] == bval) ? 1 : 0;
	}

	public static void vectEqualWrite(double bval, double[] a, int[] aix, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] = (a[j] == bval) ? 1 : 0;
	}
	
	//custom vector notequal
	
	public static void vectNotequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] != bval) ? 1 : 0;
	} 

	public static void vectNotequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] != bval) ? 1 : 0;
	}
	
	public static void vectNotequalWrite(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] = (a[j] != bval) ? 1 : 0;
	}

	public static void vectNotequalWrite(double bval, double[] a, int[] aix, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] = (a[j] != bval) ? 1 : 0;
	}
	
	//custom vector less
	
	public static void vectLessAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] < bval) ? 1 : 0;
	} 

	public static void vectLessAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] < bval) ? 1 : 0;
	}
	
	public static void vectLessWrite(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] = (a[j] < bval) ? 1 : 0;
	}

	public static void vectLessWrite(double bval, double[] a, int[] aix, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] = (a[j] < bval) ? 1 : 0;
	}
	
	//custom vector lessequal
	
	public static void vectLessequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] <= bval) ? 1 : 0;
	} 

	public static void vectLessequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] <= bval) ? 1 : 0;
	}
	
	public static void vectLessequalWrite(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] = (a[j] <= bval) ? 1 : 0;
	}

	public static void vectLessequalWrite(double bval, double[] a, int[] aix, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] = (a[j] <= bval) ? 1 : 0;
	}

	//custom vector greater
	
	public static void vectGreaterAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] > bval) ? 1 : 0;
	} 

	public static void vectGreaterAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] > bval) ? 1 : 0;
	}
	
	public static void vectGreaterWrite(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] = (a[j] > bval) ? 1 : 0;
	}

	public static void vectGreaterWrite(double bval, double[] a, int[] aix, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] = (a[j] > bval) ? 1 : 0;
	}
	
	//custom vector greaterequal
	
	public static void vectGreaterequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] >= bval) ? 1 : 0;
	} 

	public static void vectGreaterequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] >= bval) ? 1 : 0;
	}
	
	public static void vectGreaterequalWrite(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] = (a[j] >= bval) ? 1 : 0;
	}

	public static void vectGreaterequalWrite(double bval, double[] a, int[] aix, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] = (a[j] >= bval) ? 1 : 0;
	}
}
