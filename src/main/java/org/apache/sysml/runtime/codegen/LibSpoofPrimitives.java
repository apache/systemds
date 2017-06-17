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

import java.util.Arrays;
import java.util.LinkedList;

import org.apache.commons.math3.util.FastMath;
import org.apache.sysml.runtime.functionobjects.IntegerDivide;
import org.apache.sysml.runtime.functionobjects.Modulus;
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
	private static IntegerDivide intDiv = IntegerDivide.getFnObject();
	private static Modulus mod = Modulus.getFnObject();
	
	//global pool of reusable vectors, individual operations set up their own thread-local
	//ring buffers of reusable vectors with specific number of vectors and vector sizes 
	private static ThreadLocal<LinkedList<double[]>> memPool = new ThreadLocal<LinkedList<double[]>>() {
		@Override protected LinkedList<double[]> initialValue() { return new LinkedList<double[]>(); }
	};
	
	// forwarded calls to LibMatrixMult
	
	public static double dotProduct(double[] a, double[] b, int ai, int bi, int len) {
		if( a == null || b == null ) return 0;
		return LibMatrixMult.dotProduct(a, b, ai, bi, len);
	}
	
	public static double dotProduct(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		if( a == null || b == null ) return 0;
		return LibMatrixMult.dotProduct(a, b, aix, ai, bi, len);
	}
	
	public static void vectMultAdd(double[] a, double bval, double[] c, int bi, int ci, int len) {
		if( a == null ) return;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, bi, ci, len);
	}
	
	public static void vectMultAdd(double bval, double[] a, double[] c, int bi, int ci, int len) {
		if( a == null ) return;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, bi, ci, len);
	}
	
	public static void vectMultAdd(double[] a, double bval, double[] c, int[] bix, int bi, int ci, int len) {
		if( a == null ) return;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, bix, bi, ci, len);
	}
	
	public static void vectMultAdd(double bval, double[] a, double[] c, int[] bix, int bi, int ci, int len) {
		if( a == null ) return;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, bix, bi, ci, len);
	}
	
	public static void vectMultAdd(double[] a, double[] b, double[] c, int bi, int ci, int len) {
		if( a == null || b == null ) return;
		double[] tmp = vectMultWrite(a, b, 0, bi, len);
		LibMatrixMult.vectAdd(tmp, c, 0, ci, len);
	}
	
	public static double[] vectMultWrite(double[] a, double bval, int bi, int len) {
		double[] c = allocVector(len, false);
		if( a == null ) return c;
		LibMatrixMult.vectMultiplyWrite(bval, a, c, bi, 0, len);
		return c;
	}
	
	public static double[] vectMultWrite(double bval, double[] a, int bi, int len) {
		return vectMultWrite(a, bval, bi, len);
	}
	
	public static double[] vectMultWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		if( a == null || b == null ) return c;
		LibMatrixMult.vectMultiplyWrite(a, b, c, ai, bi, 0, len);
		return c;
	}
	
	public static double[] vectMultWrite(double[] a, double bval, int[] bix, int bi, int len) {
		double[] c = allocVector(len, true);
		if( a == null ) return c;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, bix, bi, 0, len);
		return c;
	}
	
	public static double[] vectMultWrite(double bval, double[] a, int[] bix, int bi, int len) {
		return vectMultWrite(a, bval, bix, bi, len);
	}
	
	public static double[] vectMultWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		if( a == null || b == null ) return c;
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] * b[bi+aix[j]];
		return c;
	}
	
	public static void vectWrite(double[] a, double[] c, int ci, int len) {
		if( a == null ) return;
		System.arraycopy(a, 0, c, ci, len);
	}

	// custom vector sums, mins, maxs
	
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
		for( int i = ai; i < ai+bn; i++ )
			val += a[ i ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int i = ai+bn; i < ai+len; i+=8 ) {
			//read 64B cacheline of a, compute cval' = sum(a) + cval
			val += a[ i+0 ] + a[ i+1 ] + a[ i+2 ] + a[ i+3 ]
			     + a[ i+4 ] + a[ i+5 ] + a[ i+6 ] + a[ i+7 ];
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
		//forward to dense as column indexes not required here
		return vectSum(avals, ai, len);
	}
	
	public static double vectMin(double[] a, int ai, int len) { 
		double val = Double.MAX_VALUE;
		for( int i = ai; i < ai+len; i++ )
			val = Math.min(a[i], val);
		return val; 
	}
	
	public static double vectMin(double[] avals, int[] aix, int ai, int len) {
		double val = Double.MAX_VALUE;
		for( int i = ai; i < ai+len; i++ )
			val = Math.min(avals[aix[i]], val);
		return val;
	}
	
	public static double vectMax(double[] a, int ai, int len) { 
		double val = -Double.MAX_VALUE;
		for( int i = ai; i < ai+len; i++ )
			val = Math.max(a[i], val);
		return val; 
	} 
	
	public static double vectMax(double[] avals, int[] aix, int ai, int len) {
		double val = -Double.MAX_VALUE;
		for( int i = ai; i < ai+len; i++ )
			val = Math.max(avals[aix[i]], val);
		return val;
	}
	
	//custom vector div
	
	public static void vectDivAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] / bval;
	}
	
	public static void vectDivAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  bval / a[j];
	}

	public static void vectDivAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += a[j] / bval;
	}
	
	public static void vectDivAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += bval / a[j];
	}
	
	public static double[] vectDivWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[ai] / bval;
		return c;
	}
	
	public static double[] vectDivWrite(double bval, double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = bval / a[ai] / bval;
		return c;
	}
	
	public static double[] vectDivWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = a[ai] / b[bi];
		return c;
	}

	public static double[] vectDivWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] / bval;
		return c;
	}
	
	public static double[] vectDivWrite(double bval, double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = bval / a[j];
		return c;
	}
	
	public static double[] vectDivWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] / b[bi+aix[j]];
		return c;
	}
	
	//custom vector minus
	
	public static void vectMinusAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] - bval;
	}
	
	public static void vectMinusAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  bval - a[j];
	}

	public static void vectMinusAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += a[j] - bval;
	}
	
	public static void vectMinusAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += bval - a[j];
	}
	
	public static double[] vectMinusWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[ai] - bval;
		return c;
	}
	
	public static double[] vectMinusWrite(double bval, double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = bval - a[ai];
		return c;
	}
	
	public static double[] vectMinusWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = a[ai] - b[bi];
		return c;
	}
	
	public static double[] vectMinusWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] - bval;
		return c;
	}
	
	public static double[] vectMinusWrite(double bval, double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = bval - a[j];
		return c;
	}
	
	public static double[] vectMinusWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] - b[bi+aix[j]];
		return c;
	}
	
	//custom vector plus
	
	public static void vectPlusAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] + bval;
	}
	
	public static void vectPlusAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectPlusAdd(a, bval, c, ai, ci, len);
	}

	public static void vectPlusAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += a[j] + bval;
	}
	
	public static void vectPlusAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectPlusAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectPlusWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[ai] + bval;
		return c;
	}
	
	public static double[] vectPlusWrite(double bval, double[] a, int ai, int len) {
		return vectPlusWrite(bval, a, ai, len);
	}
	
	public static double[] vectPlusWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[ai] + b[bi];
		return c;
	}

	public static double[] vectPlusWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] + bval;
		return c;
	}
	
	public static double[] vectPlusWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectPlusWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectPlusWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] + b[bi+aix[j]];
		return c;
	}
	
	//custom vector pow
	
	public static void vectPowAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += Math.pow(a[j], bval);
	}
	
	public static void vectPowAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += Math.pow(bval, a[j]);
	}

	public static void vectPowAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += Math.pow(a[j], bval);
	}
	
	public static void vectPowAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += Math.pow(bval, a[j]);
	}
	
	public static double[] vectPowWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.pow(a[ai], bval);
		return c;
	}
	
	public static double[] vectPowWrite(double bval, double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.pow(bval, a[ai]);
		return c;
	}
	
	public static double[] vectPowWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = Math.pow(a[ai], b[bi]);
		return c;
	}

	public static double[] vectPowWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.pow(a[j], bval);
		return c;
	}
	
	public static double[] vectPowWrite(double bval, double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.pow(bval, a[j]);
		return c;
	}
	
	//custom vector min
	
	public static void vectMinAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += Math.min(a[j], bval);
	}
	
	public static void vectMinAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectMinAdd(a, bval, c, ai, ci, len);
	}

	public static void vectMinAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += Math.min(a[j], bval);
	}
	
	public static void vectMinAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectMinAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectMinWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.min(a[ai], bval);
		return c;
	}
	
	public static double[] vectMinWrite(double bval, double[] a, int ai, int len) {
		return vectMinWrite(a, bval, ai, len);
	}
	
	public static double[] vectMinWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = Math.min(a[ai], b[bi]);
		return c;
	}

	public static double[] vectMinWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.min(a[j], bval);
		return c;
	}
	
	public static double[] vectMinWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectMinWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectMinWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.min(a[j], b[bi+aix[j]]);
		return c;
	}
	
	//custom vector max
	
	public static void vectMaxAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += Math.max(a[j], bval);
	}
	
	public static void vectMaxAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectMaxAdd(a, bval, c, ai, ci, len);
	}

	public static void vectMaxAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += Math.max(a[j], bval);
	}
	
	public static void vectMaxAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectMaxAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectMaxWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.max(a[ai], bval);
		return c;
	}
	
	public static double[] vectMaxWrite(double bval, double[] a, int ai, int len) {
		return vectMaxWrite(a, bval, ai, len);
	}
	
	public static double[] vectMaxWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = Math.max(a[ai], b[bi]);
		return c;
	}

	public static double[] vectMaxWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.max(a[j], bval);
		return c;
	}
	
	public static double[] vectMaxWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectMaxWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectMaxWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.max(a[j], b[bi+aix[j]]);
		return c;
	}

	//custom exp
	
	public static void vectExpAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.exp(a[j]);
	}

	public static void vectExpAdd(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += FastMath.exp(a[j]);
	}
	
	public static double[] vectExpWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.exp(a[ai]);
		return c;
	}

	public static double[] vectExpWrite(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = FastMath.exp(a[j]);
		return c;
	}

	//custom log
	
	public static void vectLogAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.log(a[j]);
	}

	public static void vectLogAdd(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += FastMath.log(a[j]);
	}
	
	public static double[] vectLogWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.log(a[ai]);
		return c;
	}

	public static double[] vectLogWrite(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = FastMath.log(a[j]);
		return c;
	}
	
	//custom abs
	
	public static void vectAbsAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  Math.abs(a[j]);
	}

	public static void vectAbsAdd(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += Math.log(a[j]);
	}
	
	public static double[] vectAbsWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.log(a[ai]);
		return c;
	}

	public static double[] vectAbsWrite(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.log(a[j]);
		return c;
	}
	
	//custom round
	
	public static void vectRoundAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  Math.round(a[j]);
	}

	public static void vectRoundAdd(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += Math.round(a[j]);
	}
	
	public static double[] vectRoundWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.round(a[ai]);
		return c;
	}

	public static double[] vectRoundWrite(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.round(a[j]);
		return c;
	}
	
	//custom ceil
	
	public static void vectCeilAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.ceil(a[j]);
	}

	public static void vectCeilAdd(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += FastMath.ceil(a[j]);
	}
	
	public static double[] vectCeilWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.ceil(a[ai]);
		return c;
	}

	public static double[] vectCeilWrite(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = FastMath.ceil(a[j]);
		return c;
	}
	
	//custom floor
	
	public static void vectFloorAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.floor(a[j]);
	}

	public static void vectFloorAdd(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += FastMath.floor(a[j]);
	}
	
	public static double[] vectFloorWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.floor(a[ai]);
		return c;
	}

	public static double[] vectFloorWrite(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = FastMath.floor(a[j]);
		return c;
	}
	
	//custom sign
	
	public static void vectSignAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.signum(a[j]);
	}

	public static void vectSignAdd(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += FastMath.signum(a[j]);
	}
	
	public static double[] vectSignWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.signum(a[ai]);
		return c;
	}

	public static double[] vectSignWrite(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = FastMath.signum(a[j]);
		return c;
	}
	
	//custom pow2
	
	public static void vectPow2Add(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] * a[j];
	}

	public static void vectPow2Add(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += a[j] * a[j];
	}
	
	public static double[] vectPow2Write(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[ai] * a[ai];
		return c;
	}

	public static double[] vectPow2Write(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] * a[j];
		return c;
	}
	
	//custom mult2
	
	public static void vectMult2Add(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] + a[j];
	}

	public static void vectMult2Add(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += a[j] + a[j];
	}
	
	public static double[] vectMult2Write(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[ai] + a[ai];
		return c;
	}

	public static double[] vectMult2Write(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = a[j] + a[j];
		return c;
	}
	
	//custom sqrt
	
	public static void vectSqrtAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  Math.sqrt(a[j]);
	}

	public static void vectSqrtAdd(double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += Math.sqrt(a[j]);
	}
	
	public static double[] vectSqrtWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.sqrt(a[ai]);
		return c;
	}

	public static double[] vectSqrtWrite(double[] a, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = Math.sqrt(a[j]);
		return c;
	}
	
	//custom vector equal
	
	public static void vectEqualAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] == bval) ? 1 : 0;
	}
	
	public static void vectEqualAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectEqualAdd(a, bval, c, ai, ci, len);
	}

	public static void vectEqualAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] == bval) ? 1 : 0;
	}
	
	public static void vectEqualAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectEqualAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectEqualWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = (a[ai] == bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectEqualWrite(double bval, double[] a, int ai, int len) {
		return vectEqualWrite(a, bval, ai, len);
	}
	
	public static double[] vectEqualWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = (a[ai] == b[bi]) ? 1 : 0;
		return c;
	}

	public static double[] vectEqualWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] == bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectEqualWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectEqualWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectEqualWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] == b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	//custom vector not equal
	
	public static void vectNotequalAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] != bval) ? 1 : 0;
	}
	
	public static void vectNotequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectNotequalAdd(a, bval, c, ai, ci, len);
	}

	public static void vectNotequalAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] != bval) ? 1 : 0;
	}
	
	public static void vectNotequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectNotequalAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectNotequalWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = (a[ai] != bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectNotequalWrite(double bval, double[] a, int ai, int len) {
		return vectNotequalWrite(a, bval, ai, len);
	}
	
	public static double[] vectNotequalWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = (a[ai] != b[bi]) ? 1 : 0;
		return c;
	}

	public static double[] vectNotequalWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] != bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectNotequalWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectNotequalWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectNotequalWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] != b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	//custom vector less
	
	public static void vectLessAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] < bval) ? 1 : 0;
	}
	
	public static void vectLessAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectGreaterequalAdd(a, bval, c, ai, ci, len);
	}

	public static void vectLessAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] < bval) ? 1 : 0;
	}
	
	public static void vectLessAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectGreaterequalAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectLessWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = (a[ai] < bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectLessWrite(double bval, double[] a, int ai, int len) {
		return vectGreaterequalWrite(a, bval, ai, len);
	}
	
	public static double[] vectLessWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = (a[ai] < b[bi]) ? 1 : 0;
		return c;
	}

	public static double[] vectLessWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] < bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectLessWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectGreaterequalWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectLessWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] < b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	//custom vector less equal
	
	public static void vectLessequalAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] <= bval) ? 1 : 0;
	}
	
	public static void vectLessequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectGreaterAdd(a, bval, c, ai, ci, len);
	}

	public static void vectLessequalAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] <= bval) ? 1 : 0;
	}
	
	public static void vectLessequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectGreaterAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectLessequalWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = (a[ai] <= bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectLessequalWrite(double bval, double[] a, int ai, int len) {
		return vectGreaterWrite(a, bval, ai, len);
	}
	
	public static double[] vectLessequalWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = (a[ai] <= b[bi]) ? 1 : 0;
		return c;
	}

	public static double[] vectLessequalWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] <= bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectLessequalWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectGreaterWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectLessequalWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] <= b[bi+aix[j]]) ? 1 : 0;
		return c;
	}

	//custom vector greater
	
	public static void vectGreaterAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] > bval) ? 1 : 0;
	}
	
	public static void vectGreaterAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectLessequalAdd(a, bval, c, ai, ci, len);
	}

	public static void vectGreaterAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] > bval) ? 1 : 0;
	}
	
	public static void vectGreaterAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectLessequalAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectGreaterWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = (a[ai] > bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectGreaterWrite(double bval, double[] a, int ai, int len) {
		return vectGreaterWrite(a, bval, ai, len);
	}
	
	public static double[] vectGreaterWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = (a[ai] > b[bi]) ? 1 : 0;
		return c;
	}

	public static double[] vectGreaterWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] > bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectGreaterWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectLessequalWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectGreaterWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] > b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	//custom vector greater equal
	
	public static void vectGreaterequalAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] >= bval) ? 1 : 0;
	}
	
	public static void vectGreaterequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectLessAdd(a, bval, c, ai, ci, len);
	}

	public static void vectGreaterequalAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++ )
			c[ci + aix[j]] += (a[j] >= bval) ? 1 : 0;
	}
	
	public static void vectGreaterequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int len) {
		vectLessAdd(a, bval, c, aix, ai, ci, len);
	}
	
	public static double[] vectGreaterequalWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = (a[ai] >= bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectGreaterequalWrite(double bval, double[] a, int ai, int len) {
		return vectLessWrite(a, bval, ai, len);
	}
	
	public static double[] vectGreaterequalWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = (a[ai] >= b[bi]) ? 1 : 0;
		return c;
	}

	public static double[] vectGreaterequalWrite(double[] a, double bval, int[] aix, int ai, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] >= bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectGreaterequalWrite(double bval, double[] a, int[] aix, int ai, int len) {
		return vectLessWrite(a, bval, aix, ai, len);
	}
	
	public static double[] vectGreaterequalWrite(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = ai; j < ai+len; j++ )
			c[aix[j]] = (a[j] >= b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	//complex builtin functions that are not directly generated
	//(included here in order to reduce the number of imports)
	
	public static double intDiv(double in1, double in2) {
		return intDiv.execute(in1, in2);
	}
	
	public static double mod(double in1, double in2) {
		return mod.execute(in1, in2);
	}
	
	
	//dynamic memory management
	
	public static void setupThreadLocalMemory(int numVectors, int len) {
		LinkedList<double[]> list = new LinkedList<double[]>();
		for( int i=0; i<numVectors; i++ )
			list.addLast(new double[len]);
		memPool.set(list);
	}
	
	public static void cleanupThreadLocalMemory() {
		memPool.remove();
	}
	
	private static double[] allocVector(int len, boolean reset) {
		LinkedList<double[]> list = memPool.get();
		
		//sanity check for missing setup
		if( list.isEmpty() )
			return new double[len];
		
		//get and re-queue first entry
		double[] tmp = list.removeFirst();
		list.addLast(tmp);
		
		//reset vector if required
		if( reset )
			Arrays.fill(tmp, 0);
		return tmp;
	}
}

