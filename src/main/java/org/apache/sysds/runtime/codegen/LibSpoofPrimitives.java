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

package org.apache.sysds.runtime.codegen;

import java.util.Arrays;

import org.apache.commons.math3.util.FastMath;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.functionobjects.BitwAnd;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNN;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNNIm2Col;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNNPooling;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNN.PoolingType;

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
	private static BitwAnd bwAnd = BitwAnd.getBitwAndFnObject();
	
	//global pool of reusable vectors, individual operations set up their own thread-local
	//ring buffers of reusable vectors with specific number of vectors and vector sizes 
	private static ThreadLocal<VectorBuffer> memPool = new ThreadLocal<>() {
		@Override protected VectorBuffer initialValue() { return new VectorBuffer(0,0,0); }
	};

	private static ThreadLocal<SparseVectorBuffer> sparseMemPool = new ThreadLocal<>() {
		@Override protected SparseVectorBuffer initialValue() { return new SparseVectorBuffer(0,0,0); }
	};

	public static double rowMaxsVectMult(double[] a, double[] b, int ai, int bi, int len) {
		double val = Double.NEGATIVE_INFINITY;
		int j=0;
		for( int i = ai; i < ai+len; i++ )
			val = Math.max(a[i]*b[j++], val);
		return val;
	}

	public static double rowMaxsVectMult(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double val = Double.NEGATIVE_INFINITY;
		for( int i = ai; i < ai+len; i++ )
			val = Math.max(a[i]*b[aix[i]], val);
		return val;
	}

	// forwarded calls to LibMatrixMult
	public static double dotProduct(double[] a, double[] b, int ai, int bi, int len) {
		if( a == null || b == null ) return 0;
		return LibMatrixMult.dotProduct(a, b, ai, bi, len);
	}
	
	public static double dotProduct(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		if( a == null || b == null ) return 0;
		return LibMatrixMult.dotProduct(a, b, aix, ai, bi, len);
	}
	
	public static double[] vectMatrixMult(double[] a, double[] b, int ai, int bi, int len) {
		//note: assumption b is already transposed for efficient dot products
		int m2clen = b.length / len;
		double[] c = allocVector(m2clen, false);
		for( int j = 0, bix = bi; j < m2clen; j++, bix+=len )
			c[j] = LibMatrixMult.dotProduct(a, b, ai, bix, len);
		return c;
	}
	
	public static double[] vectMatrixMult(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		//note: assumption b is already transposed for efficient dot products
		int m2clen = b.length / len;
		double[] c = allocVector(m2clen, false);
		for( int j = 0, bix = bi; j < m2clen; j++, bix+=len )
			c[j] = LibMatrixMult.dotProduct(a, b, aix, ai, bix, alen);
		return c;
	}
	
	public static void vectOuterMultAdd(double[] a, double[] b, double[] c, int ai, int bi, int ci, int len1, int len2) {
		if( isFlipOuter(len1, len2) ) {
			for( int i=0, cix=ci; i < len2; i++, cix+=len1 ) {
				final double val = b[bi+i];
				if( val != 0 )
					LibMatrixMult.vectMultiplyAdd(val, a, c, ai, cix, len1);
			}
		}
		else {
			//rest, not aligned to 4-blocks
			final int bn = len1%4;
			for( int i=0, cix=ci; i < bn; i++, cix+=len2 )
				if( a[ai+i] != 0 )
					LibMatrixMult.vectMultiplyAdd(a[ai+i], b, c, bi, cix, len2);
			
			//unrolled 4-block (for fewer L1-dcache loads)
			for( int i=bn, cix=ci+bn*len2; i < len1; i+=4, cix+=4*len2 ) {
				final int cix1=cix, cix2=cix+len2, cix3=cix+2*len2, cix4=cix+3*len2;
				final double aval1=a[ai+i], aval2=a[ai+i+1], aval3=a[ai+i+2], aval4=a[ai+i+3];
				for( int j=0; j<len2; j++ ) {
					final double bval = b[bi+j];
					c[cix1 + j] += aval1 * bval;
					c[cix2 + j] += aval2 * bval;
					c[cix3 + j] += aval3 * bval;
					c[cix4 + j] += aval4 * bval;
				}
			}	
		}	
	}
	
	public static void vectOuterMultAdd(double[] a, double[] b, double[] c, int[] aix, int ai, int bi, int ci, int alen, int len1, int len2) {
		if( isFlipOuter(len1, len2) ) {
			for( int i=0, cix=ci; i < len2; i++, cix+=len1 ) {
				final double val = b[bi+i];
				if( val != 0 )
					LibMatrixMult.vectMultiplyAdd(val, a, c, aix, ai, cix, alen);
			}
		}
		else {
			for( int i=0; i < alen; i++ )
				LibMatrixMult.vectMultiplyAdd(a[ai+i], b, c, bi, ci+aix[ai+i]*len2, len2);
		}
	}
	
	public static void vectOuterMultAdd(double[] a, double[] b, double[] c, int ai, int[] bix, int bi, int ci, int blen, int len1, int len2) {
		if( isFlipOuter(len1, len2) ) {
			for( int i=bi; i<bi+blen; i++ ) {
				final int cix = ci + bix[i] * len1;
				LibMatrixMult.vectMultiplyAdd(b[i], a, c, ai, cix, len1);
			}
		}
		else {
			for( int i=0, cix=ci; i < len1; i++, cix+=len2 )
				LibMatrixMult.vectMultiplyAdd(a[ai+i], b, c, bix, bi, cix, blen);
		}
	}
	
	public static void vectMultAdd(double[] a, double bval, double[] c, int bi, int ci, int len) {
		if( a == null || bval == 0 ) return;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, bi, ci, len);
	}
	
	public static void vectMultAdd(double bval, double[] a, double[] c, int bi, int ci, int len) {
		if( a == null || bval == 0 ) return;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, bi, ci, len);
	}
	
	public static void vectMultAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( a == null || bval == 0 ) return;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, aix, ai, ci, alen);
	}
	
	public static void vectMultAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( a == null || bval == 0 ) return;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, aix, ai, ci, alen);
	}
	
	public static void vectMultAdd(double[] a, double[] b, double[] c, int bi, int ci, int len) {
		if( a == null || b == null ) return;
		double[] tmp = vectMultWrite(a, b, 0, bi, len);
		LibMatrixMult.vectAdd(tmp, c, 0, ci, len);
	}
	
	public static double[] vectMultWrite(double[] a, double bval, int bi, int len) {
		if( a == null || bval == 0 ) 
			return allocVector(len, true);
		double[] c = allocVector(len, false);
		LibMatrixMult.vectMultiplyWrite(bval, a, c, bi, 0, len);
		return c;
	}
	
	public static double[] vectMultWrite(double bval, double[] a, int bi, int len) {
		return vectMultWrite(a, bval, bi, len);
	}
	
	public static double[] vectMultWrite(double[] a, double[] b, int ai, int bi, int len) {
		if( a == null || b == null )
			return allocVector(len, true);
		double[] c = allocVector(len, false);
		LibMatrixMult.vectMultiplyWrite(a, b, c, ai, bi, 0, len);
		return c;
	}
	
	public static double[] vectMultWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		if( a == null ) return c;
		LibMatrixMult.vectMultiplyAdd(bval, a, c, aix, ai, 0, alen);
		return c;
	}
	
	public static double[] vectMultWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectMultWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectMultWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, true);
		if( a == null || b == null ) return c;
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = a[j] * b[bi+aix[j]];
		return c;
	}
	
	public static double[] vectMultWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectMultWrite(b, a, bix, ai, bi, blen, len);
	}
	
	public static void vectWrite(double[] a, double[] c, int ci, int len) {
		if( a == null ) return;
		System.arraycopy(a, 0, c, ci, len);
	}
	
	public static void vectWrite(double[] a, double[] c, int ai, int ci, int len) {
		if( a == null ) return;
		System.arraycopy(a, ai, c, ci, len);
	}
	
	public static void vectWrite(boolean[] a, boolean[] c, int[] aix) {
		if( a == null ) return;
		for( int i=0; i<aix.length; i++ )
			c[aix[i]] = a[i];
	}
	
	public static void vectWrite(boolean[] a, boolean[] c, int[] aix, int ai, int ci, int alen) {
		if( a == null ) return;
		for( int i=ai; i<ai+alen; i++ )
			c[ci+aix[i]] = a[i];
	}
	
	// cbind handling
	
	public static double[] vectCbindAdd(double[] a, double b, double[] c, int ai, int ci, int len) {
		LibMatrixMult.vectAdd(a, c, ai, ci, len);
		c[ci+len] += b;
		return c;
	}
	
	public static double[] vectCbindAdd(double[] a, double b, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		LibMatrixMult.vectAdd(a, c, aix, ai, ci, alen);
		c[ci+len] += b;
		return c;
	}
	
	public static double[] vectCbindWrite(double a, double b) {
		double[] c = allocVector(2, false);
		c[0] = a;
		c[1] = b;
		return c;
	}
	
	public static double[] vectCbindWrite(double[] a, double b, int aix, int len) {
		double[] c = allocVector(len+1, false);
		System.arraycopy(a, aix, c, 0, len);
		c[len] = b;
		return c;
	}
	
	public static double[] vectCbindWrite(double[] a, double b, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len+1, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = a[j];
		c[len] = b;
		return c;
	}
	
	public static double[] vectCbindWrite(double[] a, double[] b, int ai, int bi, int alen, int blen) {
		double[] c = allocVector(alen+blen, false);
		System.arraycopy(a, ai, c, 0, alen);
		System.arraycopy(b, bi, c, alen, blen);
		return c;
	}
	
	public static double[] vectCbindWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int alen2, int blen) {
		double[] c = allocVector(alen2+blen, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = a[j];
		System.arraycopy(b, bi, c, alen2, blen);
		return c;
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
	
	public static double vectSum(double[] avals, int[] aix, int ai, int alen, int len) {
		//forward to dense as column indexes not required here
		return vectSum(avals, ai, alen);
	}
	
	public static double vectSumsq(double[] a, int ai, int len) { 
		return LibMatrixMult.dotProduct(a, a, ai, ai, len);
	}
	
	public static double vectSumsq(double[] avals, int[] aix, int ai, int alen, int len) {
		return LibMatrixMult.dotProduct(avals, avals, ai, ai, alen);
	}
	
	public static double vectMin(double[] a, int ai, int len) { 
		double val = Double.POSITIVE_INFINITY;
		for( int i = ai; i < ai+len; i++ )
			val = Math.min(a[i], val);
		return val; 
	}
	
	public static double vectMin(double[] avals, int[] aix, int ai, int alen, int len) {
		double val = vectMin(avals, ai, alen);
		return (alen<len) ? Math.min(val, 0) : val;
	}
	
	public static double vectMax(double[] a, int ai, int len) { 
		double val = Double.NEGATIVE_INFINITY;
		for( int i = ai; i < ai+len; i++ )
			val = Math.max(a[i], val);
		return val; 
	} 
	
	public static double vectMax(double[] avals, int[] aix, int ai, int alen, int len) {
		double val = vectMax(avals, ai, alen);
		return (alen<len) ? Math.max(val, 0) : val;
	}
	
	public static double vectCountnnz(double[] a, int ai, int len) { 
		int count = 0;
		for( int i = ai; i < ai+len; i++ )
			count += (a[i] != 0) ? 1 : 0;
		return count;
	} 
	
	public static double vectCountnnz(double[] avals, int[] aix, int ai, int alen, int len) {
		//pure meta data operation
		return alen;
	}
	
	public static double vectMean(double[] a, int ai, int len) {
		return vectSum(a, ai, len) / len;
	} 
	
	public static double vectMean(double[] avals, int[] aix, int ai, int alen, int len) {
		return vectSum(avals, aix, ai, alen, len) / len;
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

	public static void vectDivAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += a[j] / bval;
	}
	
	public static void vectDivAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += bval / a[j];
	}
	
	public static double[] vectDivWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = a[ai+j] / bval;
		return c;
	}
	
	public static double[] vectDivWrite(double bval, double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = bval / a[ai + j];
		return c;
	}
	
	public static double[] vectDivWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = a[ai + j] / b[bi + j];
		return c;
	}

	public static double[] vectDivWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval != 0) ? 0 : Double.NaN;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = a[j] / bval;
		return c;
	}
	
	public static double[] vectDivWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		double init = (bval != 0) ? Double.POSITIVE_INFINITY : Double.NaN;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = bval / a[j];
		return c;
	}
	
	public static double[] vectDivWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++ )
			if( b[bi + j] == 0 ) //prep 0/0=NaN
				c[j] = Double.NaN;
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = a[j] / b[bi+aix[j]];
		return c;
	}
	
	public static double[] vectDivWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++ ) {
			double aval = a[bi + j];
			c[j] = (aval==0) ? Double.NaN : (aval>0) ? 
				Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
		}
		for( int j = bi; j < bi+blen; j++ )
			c[bix[j]] = a[ai+bix[j]] / b[j];
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

	public static void vectMinusAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval != 0 ) //subtract bval if necessary
			for( int j = ci; j < ci+len; j++ )
				c[j] -= bval;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += a[j];
	}
	
	public static void vectMinusAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval != 0 ) //add bval is necessary
			for( int j = ci; j < ci+len; j++ )
				c[j] += bval;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] -= a[j];
	}
	
	public static double[] vectMinusWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = a[ai + j] - bval;
		return c;
	}
	
	public static double[] vectMinusWrite(double bval, double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = bval - a[ai + j];
		return c;
	}
	
	public static double[] vectMinusWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = a[ai + j] - b[bi + j];
		return c;
	}
	
	public static double[] vectMinusWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, -bval);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] += a[j];
		return c;
	}
	
	public static double[] vectMinusWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, bval);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] -= a[j];
		return c;
	}
	
	public static double[] vectMinusWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++ )
			c[j] = -b[bi+j];
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] += a[j];
		return c;
	}
	
	public static double[] vectMinusWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		double[] c = allocVector(len, false);
		System.arraycopy(a, ai, c, 0, len);
		for( int j = bi; j < bi+blen; j++ )
			c[bix[j]] -= b[j];
		return c;
	}
	
	//custom vector plus
	
	public static void vectPlusAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		LibMatrixMult.vectAdd(a, bval, c, ai, ci, len);
	}
	
	public static void vectPlusAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		LibMatrixMult.vectAdd(a, bval, c, ai, ci, len);
	}

	public static void vectPlusAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ci; j < ci+len; j++ )
			c[j] += bval;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += a[j];
	}
	
	public static void vectPlusAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectPlusAdd(a, bval, c, aix, ai, ci, alen, len);
	}
	
	public static double[] vectPlusWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = a[ai + j] + bval;
		return c;
	}
	
	public static double[] vectPlusWrite(double bval, double[] a, int ai, int len) {
		return vectPlusWrite(a, bval, ai, len);
	}
	
	public static double[] vectPlusWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = a[ai+j] + b[bi+j];
		return c;
	}

	public static double[] vectPlusWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, bval);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] += a[j];
		return c;
	}
	
	public static double[] vectPlusWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectPlusWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectPlusWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		System.arraycopy(b, bi, c, 0, len);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] += a[j];
		return c;
	}
	
	public static double[] vectPlusWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectPlusWrite(b, a, bix, bi, ai, blen, len);
	}

	//custom vector xor
	/**
	 * Computes c = xor(A,B)
	 *
	 * @param a dense input vector A
	 * @param bval scalar value
	 * @param c resultant vector
	 * @param ai start position in A
	 * @param ci index of c
	 * @param len number of processed elements
	 */
	public static void vectXorAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  ( (a[j] != 0) != (bval != 0) ) ? 1 : 0;
	}

	public static void vectXorAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectXorAdd(a, bval, c, ai, ci, len);
	}

	public static void vectXorAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += ( (a[j] != 0) != (bval != 0) ) ? 1 : 0;
	}

	public static void vectXorAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectXorAdd(a, bval, c, aix, ai, ci, alen, len);
	}

	//1. scalar vs. dense vector
	public static double[] vectXorWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = ( ( a[ai+j] != 0) != (bval != 0) ) ? 1 : 0;
		return c;
	}

	//2. dense vector vs. scalar
	public static double[] vectXorWrite(double bval, double[] a, int ai, int len) {
		return vectXorWrite(a, bval, ai, len);
	}

	//3. dense vector vs. dense vector
	public static double[] vectXorWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = ( (a[ai + j] != 0) != (b[bi + j] != 0) ) ? 1 : 0;
		return c;
	}

	//4. sparse vector vs scalar
	public static double[] vectXorWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval != 0) ? 1 : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = (a[j] != 0) ? 0 : 1;
		return c;
	}

	//5. scalar vs. sparse vector
	public static double[] vectXorWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectXorWrite(a, bval, aix, ai, alen, len);
	}

	//6. sparse vector vs. dense vector
	public static double[] vectXorWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++ )
			c[j] = (b[bi+j] != 0) ? 1 : 0;
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = ( ( a[j] != 0) != (c[aix[j]] != 0) )? 1 : 0;
		return c;
	}

	//6. sparse vector vs. dense vector
	public static double[] vectXorWrite(double[] a, double[] b, int ai, int[] aix, int bi, int alen, int len) {
		return vectXorWrite(a, b, aix, ai, bi, alen, len);
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

	public static void vectPowAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval == 0 ) //handle 0^0=1 & a^0=1
			for( int j=0; j<len; j++ )
				c[ci + j] += 1;
		else //handle 0^b=0 & a^b
			for( int j = ai; j < ai+alen; j++ )
				c[ci + aix[j]] += Math.pow(a[j], bval);
	}
	
	public static void vectPowAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j=0; j<len; j++ ) //handle 0^0=1 & b^0=1
			c[ci + j] += 1;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += Math.pow(bval, a[j]) - 1;
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

	public static double[] vectPowWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval == 0) ? 1 : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.pow(a[j], bval) - init;
		return c;
	}
	
	public static double[] vectPowWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, 1);
		for( int j = ai; j < ai+alen; j++ )
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

	public static void vectMinAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval < 0 )
			for( int j=0; j<len; j++ )
				c[ci +j] += bval;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += (bval >= 0) ? Math.min(a[j], bval) : 0;
	}
	
	public static void vectMinAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectMinAdd(a, bval, c, aix, ai, ci, alen, len);
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

	public static double[] vectMinWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval < 0) ? bval : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.min(a[j], bval);
		return c;
	}
	
	public static double[] vectMinWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectMinWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectMinWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++ )
			c[j] = Math.min(b[bi + j], 0);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.min(a[j], b[bi + aix[j]]);
		return c;
	}
	
	public static double[] vectMinWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectMinWrite(b, a, bix, bi, ai, blen, len);
	}
	
	//custom vector max
	
	public static void vectMaxAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += Math.max(a[j], bval);
	}
	
	public static void vectMaxAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectMaxAdd(a, bval, c, ai, ci, len);
	}

	public static void vectMaxAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval > 0 )
			for( int j=0; j<len; j++ )
				c[ci + j] += bval;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += (bval <= 0) ? Math.max(a[j], bval) : 0;
	}
	
	public static void vectMaxAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectMaxAdd(a, bval, c, aix, ai, ci, alen, len);
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

	public static double[] vectMaxWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval > 0) ? bval : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.max(a[j], bval);
		return c;
	}
	
	public static double[] vectMaxWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectMaxWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectMaxWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++ )
			c[j] = Math.max(b[bi + j], 0);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.max(a[j], b[bi + aix[j]]);
		return c;
	}
	
	public static double[] vectMaxWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectMaxWrite(b, a, bix, bi, ai, blen, len);
	}

	//custom exp
	
	public static void vectExpAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = 0; j < len; j++)
			c[ci+j] +=  FastMath.exp(a[ai+j]);
	}

	public static void vectExpAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j=ci; j<ci+len; j++ ) //exp(0)=1
			c[j] += 1;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.exp(a[j]) - 1;
	}
	
	public static double[] vectExpWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = FastMath.exp(a[ai+j]);
		return c;
	}

	public static double[] vectExpWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, 1); //exp(0)=1
		for( int j = ai; j < ai+alen; j++ )    //overwrite
			c[aix[j]] = FastMath.exp(a[j]);
		return c;
	}

	//custom cumsum
	
	public static void vectCumsumAdd(double[] a, double[] c, int ai, int ci, int len) {
		double val = 0;
		for( int j = 0; j < len; j++ ) {
			val += a[ai * j]; 
			c[ci+j] += val;
		}
	}

	public static void vectCumsumAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		double val = 0;
		int lastIx = -1;
		for( int j = ai; j < ai+alen; j++ ) {
			//add non-existing indexes
			for( int j2=lastIx+1; j2<aix[j]; j2++ )
				c[j2] += val;
			//update value and add current index
			val += a[j];
			c[aix[j]] = val;
			lastIx = aix[j];
		}
		//add non-existing indexes
		for( int j2=lastIx+1; j2<len; j2++ )
			c[j2] += val;
	}
	
	public static double[] vectCumsumWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		double val = 0;
		for( int j = 0; j < len; j++ ) {
			val += a[ai+j];
			c[j] = val;
		}
		return c;
	}

	public static double[] vectCumsumWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, false);
		double val = 0;
		int lastIx = -1;
		for( int j = ai; j < ai+alen; j++ ) {
			//add non-existing indexes
			Arrays.fill(c, lastIx+1, aix[j], val);
			//update value and add current index
			val += a[j];
			c[aix[j]] = val;
			lastIx = aix[j];
		}
		//add non-existing indexes
		Arrays.fill(c, lastIx+1, len, val);
		return c;
	}

	//custom cummin
	
	public static void vectCumminAdd(double[] a, double[] c, int ai, int ci, int len) {
		double val = 0;
		for( int j = 0; j < len; j++ ) {
			val = Math.min(val, a[ai * j]); 
			c[ci+j] += val;
		}
	}

	public static void vectCumminAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		double val = 0;
		int lastIx = -1;
		for( int j = ai; j < ai+alen; j++ ) {
			//add non-existing indexes
			for( int j2=lastIx+1; j2<aix[j]; j2++ )
				c[j2] += val;
			//update value and add current index
			val = Math.min(val, a[j]);
			c[aix[j]] = val;
			lastIx = aix[j];
		}
		//add non-existing indexes
		for( int j2=lastIx+1; j2<len; j2++ )
			c[j2] += val;
	}
	
	public static double[] vectCumminWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		double val = 0;
		for( int j = 0; j < len; j++ ) {
			val = Math.min(val, a[ai+j]);
			c[j] = val;
		}
		return c;
	}

	public static double[] vectCumminWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, false);
		double val = 0;
		int lastIx = -1;
		for( int j = ai; j < ai+alen; j++ ) {
			//add non-existing indexes
			Arrays.fill(c, lastIx+1, aix[j], val);
			//update value and add current index
			val = Math.min(val, a[j]);
			c[aix[j]] = val;
			lastIx = aix[j];
		}
		//add non-existing indexes
		Arrays.fill(c, lastIx+1, len, val);
		return c;
	}
	
	//custom cummax

	public static void vectCummaxAdd(double[] a, double[] c, int ai, int ci, int len) {
		double val = 0;
		for( int j = 0; j < len; j++ ) {
			val = Math.max(val, a[ai * j]); 
			c[ci+j] += val;
		}
	}

	public static void vectCummaxAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		double val = 0;
		int lastIx = -1;
		for( int j = ai; j < ai+alen; j++ ) {
			//add non-existing indexes
			for( int j2=lastIx+1; j2<aix[j]; j2++ )
				c[j2] += val;
			//update value and add current index
			val = Math.max(val, a[j]);
			c[aix[j]] = val;
			lastIx = aix[j];
		}
		//add non-existing indexes
		for( int j2=lastIx+1; j2<len; j2++ )
			c[j2] += val;
	}
	
	public static double[] vectCummaxWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		double val = 0;
		for( int j = 0; j < len; j++ ) {
			val = Math.max(val, a[ai+j]);
			c[j] = val;
		}
		return c;
	}

	public static double[] vectCummaxWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, false);
		double val = 0;
		int lastIx = -1;
		for( int j = ai; j < ai+alen; j++ ) {
			//add non-existing indexes
			Arrays.fill(c, lastIx+1, aix[j], val);
			//update value and add current index
			val = Math.max(val, a[j]);
			c[aix[j]] = val;
			lastIx = aix[j];
		}
		//add non-existing indexes
		Arrays.fill(c, lastIx+1, len, val);
		return c;
	}
	
	//custom log
	
	public static void vectLogAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += Math.log(a[j]);
	}

	public static void vectLogAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += Math.log(a[j]);
	}
	
	public static double[] vectLogWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.log(a[ai]);
		return c;
	}

	public static double[] vectLogWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, Double.NEGATIVE_INFINITY);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.log(a[j]);
		return c;
	}
	
	//custom abs
	
	public static void vectAbsAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  Math.abs(a[j]);
	}

	public static void vectAbsAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += Math.abs(a[j]);
	}
	
	public static double[] vectAbsWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.abs(a[ai]);
		return c;
	}

	public static double[] vectAbsWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.abs(a[j]);
		return c;
	}
	
	//custom round
	
	public static void vectRoundAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  Math.round(a[j]);
	}

	public static void vectRoundAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += Math.round(a[j]);
	}
	
	public static double[] vectRoundWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.round(a[ai]);
		return c;
	}

	public static double[] vectRoundWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.round(a[j]);
		return c;
	}
	
	//custom ceil
	
	public static void vectCeilAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.ceil(a[j]);
	}

	public static void vectCeilAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.ceil(a[j]);
	}
	
	public static double[] vectCeilWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.ceil(a[ai]);
		return c;
	}

	public static double[] vectCeilWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.ceil(a[j]);
		return c;
	}
	
	//custom floor
	
	public static void vectFloorAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.floor(a[j]);
	}

	public static void vectFloorAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.floor(a[j]);
	}
	
	public static double[] vectFloorWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.floor(a[ai]);
		return c;
	}

	public static double[] vectFloorWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.floor(a[j]);
		return c;
	}
	
	//custom sin
	
	public static void vectSinAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.sin(a[j]);
	}

	public static void vectSinAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.sin(a[j]);
	}
	
	public static double[] vectSinWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.sin(a[ai]);
		return c;
	}

	public static double[] vectSinWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.sin(a[j]);
		return c;
	}
	
	//custom cos
	
	public static void vectCosAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.cos(a[j]);
	}

	public static void vectCosAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.cos(a[j]);
	}
	
	public static double[] vectCosWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.cos(a[ai]);
		return c;
	}

	public static double[] vectCosWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, 1);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.cos(a[j]);
		return c;
	}
	
	//custom tan
	
	public static void vectTanAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.tan(a[j]);
	}

	public static void vectTanAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.tan(a[j]);
	}
	
	public static double[] vectTanWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.tan(a[ai]);
		return c;
	}

	public static double[] vectTanWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.tan(a[j]);
		return c;
	}

	//custom asin
	
	public static void vectAsinAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.asin(a[j]);
	}

	public static void vectAsinAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.asin(a[j]);
	}
	
	public static double[] vectAsinWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.asin(a[ai]);
		return c;
	}

	public static double[] vectAsinWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.asin(a[j]);
		return c;
	}
	
	//custom acos
	
	public static void vectAcosAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.acos(a[j]);
	}

	public static void vectAcosAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.acos(a[j]);
	}
	
	public static double[] vectAcosWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.acos(a[ai]);
		return c;
	}

	public static double[] vectAcosWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, Math.PI/2);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.acos(a[j]);
		return c;
	}
	
	//custom atan
	
	public static void vectAtanAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  Math.atan(a[j]);
	}

	public static void vectAtanAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += Math.atan(a[j]);
	}
	
	public static double[] vectAtanWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.atan(a[ai]);
		return c;
	}

	public static double[] vectAtanWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.atan(a[j]);
		return c;
	}

	
	//custom sinh
	
	public static void vectSinhAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.sinh(a[j]);
	}

	public static void vectSinhAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.sinh(a[j]);
	}
	
	public static double[] vectSinhWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.sinh(a[ai]);
		return c;
	}

	public static double[] vectSinhWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.sinh(a[j]);
		return c;
	}
	
	//custom cosh
	
	public static void vectCoshAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.cosh(a[j]);
	}

	public static void vectCoshAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.cosh(a[j]);
	}
	
	public static double[] vectCoshWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.cosh(a[ai]);
		return c;
	}

	public static double[] vectCoshWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, 1);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.cosh(a[j]);
		return c;
	}
	
	//custom tanh
	
	public static void vectTanhAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.tanh(a[j]);
	}

	public static void vectTanhAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.tanh(a[j]);
	}
	
	public static double[] vectTanhWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.tanh(a[ai]);
		return c;
	}

	public static double[] vectTanhWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.tanh(a[j]);
		return c;
	}
	
	//custom sign
	
	public static void vectSignAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  FastMath.signum(a[j]);
	}

	public static void vectSignAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += FastMath.signum(a[j]);
	}
	
	public static double[] vectSignWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = FastMath.signum(a[ai]);
		return c;
	}

	public static double[] vectSignWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = FastMath.signum(a[j]);
		return c;
	}
	
	//custom pow2
	
	public static void vectPow2Add(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] * a[j];
	}

	public static void vectPow2Add(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += a[j] * a[j];
	}
	
	public static double[] vectPow2Write(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[ai] * a[ai];
		return c;
	}

	public static double[] vectPow2Write(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = a[j] * a[j];
		return c;
	}
	
	//custom mult2
	
	public static void vectMult2Add(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] + a[j];
	}
	
	public static void vectMult2Add(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += a[j] + a[j];
	}
	
	public static double[] vectMult2Write(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[ai] + a[ai];
		return c;
	}
	
	public static double[] vectMult2Write(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = a[j] + a[j];
		return c;
	}
	
	//custom sqrt
	
	public static void vectSqrtAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  Math.sqrt(a[j]);
	}

	public static void vectSqrtAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += Math.sqrt(a[j]);
	}
	
	public static double[] vectSqrtWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = Math.sqrt(a[ai]);
		return c;
	}

	public static double[] vectSqrtWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = Math.sqrt(a[j]);
		return c;
	}

	//custom sprop
	
	public static void vectSpropAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += a[j] * (1 - a[j]);
	}

	public static void vectSpropAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += a[j] * (1 - a[j]);
	}
	
	public static double[] vectSpropWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = a[j] * (1 - a[j]);
		return c;
	}

	public static double[] vectSpropWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = a[j] * (1 - a[j]);
		return c;
	}
	
	//custom sigmoid
	
	public static void vectSigmoidAdd(double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  1 / (1 + FastMath.exp(-a[j]));
	}

	public static void vectSigmoidAdd(double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += 1 / (1 + FastMath.exp(-a[j]));
	}
	
	public static double[] vectSigmoidWrite(double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = 1 / (1 + FastMath.exp(-a[j]));
		return c;
	}

	public static double[] vectSigmoidWrite(double[] a, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true, 0.5); //sigmoid(0) = 0.5
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = 1 / (1 + FastMath.exp(-a[j]));
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

	public static void vectEqualAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval == 0 )
			for( int j=0; j<len; j++ )
				c[j] += 1;
		else		
			for( int j = ai; j < ai+alen; j++ )
				c[ci + aix[j]] += (a[j] == bval) ? 1 : 0;
	}
	
	public static void vectEqualAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectEqualAdd(a, bval, c, aix, ai, ci, alen, len);
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

	public static double[] vectEqualWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval == 0) ? 1 : 0;
		double[] c = allocVector(len, true, init);
		if( bval != 0 )
			for( int j = ai; j < ai+alen; j++ )
				c[aix[j]] = (a[j] == bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectEqualWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectEqualWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectEqualWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j=0; j<len; j++ )
			c[j] = (b[bi+j]==0) ? 1 : 0;
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = (a[j] == b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	public static double[] vectEqualWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectEqualWrite(b, a, bix, bi, ai, blen, len);
	}
	
	//custom vector not equal
	
	public static void vectNotequalAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] != bval) ? 1 : 0;
	}
	
	public static void vectNotequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectNotequalAdd(a, bval, c, ai, ci, len);
	}

	public static void vectNotequalAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen,  int len) {
		if( bval != 0 )
			for( int j=0; j<len; j++ )
				c[j] += 1;
		double init = (bval != 0) ? 1 : 0;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += ((a[j] != bval) ? 1 : 0) - init;
	}
	
	public static void vectNotequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectNotequalAdd(a, bval, c, aix, ai, ci, alen, len);
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

	public static double[] vectNotequalWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval != 0) ? 1 : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = ((a[j] != bval) ? 1 : 0);
		return c;
	}
	
	public static double[] vectNotequalWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectNotequalWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectNotequalWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j=0; j<len; j++ )
			c[j] = (b[bi+j]!=0) ? 1 : 0;
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = (a[j] != b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	public static double[] vectNotequalWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectNotequalWrite(b, a, bix, bi, ai, blen, len);
	}
	
	//custom vector less
	
	public static void vectLessAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] < bval) ? 1 : 0;
	}
	
	public static void vectLessAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectGreaterequalAdd(a, bval, c, ai, ci, len);
	}

	public static void vectLessAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval > 0 )
			for( int j=0; j<len; j++ )
				c[j] += 1;
		double init = (bval > 0) ? 1 : 0;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += ((a[j] < bval) ? 1 : 0) - init;
	}
	
	public static void vectLessAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectGreaterequalAdd(a, bval, c, aix, ai, ci, alen, len);
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

	public static double[] vectLessWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval > 0) ? 1 : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = (a[j] < bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectLessWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectGreaterequalWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectLessWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j=0; j<len; j++ )
			c[j] = ( 0 < b[bi+j] ) ? 1 : 0; 
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = (a[j] < b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	public static double[] vectLessWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectGreaterequalWrite(b, a, bix, bi, ai, blen, len);
	}
	
	//custom vector less equal
	
	public static void vectLessequalAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] <= bval) ? 1 : 0;
	}
	
	public static void vectLessequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectGreaterAdd(a, bval, c, ai, ci, len);
	}

	public static void vectLessequalAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval >= 0 )
			for( int j=0; j<len; j++ )
				c[j] += 1;
		double init = (bval >= 0) ? 1 : 0;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += ((a[j] <= bval) ? 1 : 0) - init;
	}
	
	public static void vectLessequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectGreaterAdd(a, bval, c, aix, ai, ci, alen, len);
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

	public static double[] vectLessequalWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval >= 0) ? 1 : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = ((a[j] <= bval) ? 1 : 0);
		return c;
	}
	
	public static double[] vectLessequalWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectGreaterWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectLessequalWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j=0; j<len; j++ )
			c[j] = ( 0 <= b[bi+j] ) ? 1 : 0; 
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = (a[j] <= b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	public static double[] vectLessequalWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectGreaterWrite(b, a, bix, bi, ai, blen, len);
	}

	//custom vector greater
	
	public static void vectGreaterAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] > bval) ? 1 : 0;
	}
	
	public static void vectGreaterAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectLessequalAdd(a, bval, c, ai, ci, len);
	}

	public static void vectGreaterAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval < 0 )
			for( int j=0; j<len; j++ )
				c[j] += 1;
		double init = (bval < 0) ? 1 : 0;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += ((a[j] > bval) ? 1 : 0) - init;
	}
	
	public static void vectGreaterAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectLessequalAdd(a, bval, c, aix, ai, ci, alen, len);
	}
	
	public static double[] vectGreaterWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++)
			c[j] = (a[ai] > bval) ? 1 : 0;
		return c;
	}
	
	public static double[] vectGreaterWrite(double bval, double[] a, int ai, int len) {
		return vectLessWrite(a, bval, ai, len);
	}
	
	public static double[] vectGreaterWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++, ai++, bi++)
			c[j] = (a[ai] > b[bi]) ? 1 : 0;
		return c;
	}

	public static double[] vectGreaterWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval < 0) ? 1 : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = ((a[j] > bval) ? 1 : 0);
		return c;
	}
	
	public static double[] vectGreaterWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectLessequalWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectGreaterWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j=0; j<len; j++ )
			c[j] = ( 0 > b[bi+j] ) ? 1 : 0; 
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = (a[j] > b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	public static double[] vectGreaterWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectLessequalWrite(b, a, bix, bi, ai, blen, len);
	}
	
	//custom vector greater equal
	
	public static void vectGreaterequalAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] >= bval) ? 1 : 0;
	}
	
	public static void vectGreaterequalAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		vectLessAdd(a, bval, c, ai, ci, len);
	}

	public static void vectGreaterequalAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		if( bval <= 0 )
			for( int j=0; j<len; j++ )
				c[j] += 1;
		double init = (bval <= 0) ? 1 : 0;
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += ((a[j] >= bval) ? 1 : 0) - init;
	}
	
	public static void vectGreaterequalAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		vectLessAdd(a, bval, c, aix, ai, ci, alen, len);
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

	public static double[] vectGreaterequalWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double init = (bval < 0) ? 1 : 0;
		double[] c = allocVector(len, true, init);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = ((a[j] >= bval) ? 1 : 0) - init;
		return c;
	}
	
	public static double[] vectGreaterequalWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectLessWrite(a, bval, aix, ai, alen, len);
	}
	
	public static double[] vectGreaterequalWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, false);
		for( int j=0; j<len; j++ )
			c[j] = ( 0 >= b[bi+j] ) ? 1 : 0; 
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = (a[j] >= b[bi+aix[j]]) ? 1 : 0;
		return c;
	}
	
	public static double[] vectGreaterequalWrite(double[] a, double[] b, int ai, int[] bix, int bi, int blen, int len) {
		//invariant to the ordering of inputs
		return vectLessWrite(b, a, bix, bi, ai, blen, len);
	}

	//bitwise and
	
	//1. dense vector vs. scalar
	public static double[] vectBitwandWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++ )
			c[j] = bwAnd(a[ai+j], bval);
		return c;
	}

	//2. scalar vs. dense vector
	public static double[] vectBitwandWrite(double bval, double[] a, int ai, int len) {
		return vectBitwandWrite(a, bval, ai, len);
	}

	//3. dense vector vs. dense vector
	public static double[] vectBitwandWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c= allocVector(len, false);
		for( int j = 0; j < len; j++ )
			c[j] = bwAnd(a[ai+j], b[bi+j]);
		return c;
	}

	//4. sparse vector vs. scalar.
	public static double[] vectBitwandWrite(double[] a, double bval, int[] aix, int ai, int alen, int len) {
		double[] c = allocVector(len, true);
		int bval1 = (int)bval;
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = bwAnd(a[j], bval1);
		return c;
	}

	//5. scalar vs. sparse vector
	public static double[] vectBitwandWrite(double bval, double[] a, int[] aix, int ai, int alen, int len) {
		return vectBitwandWrite(a, bval, aix, ai, alen, len);
	}

	//6. sparse vector vs. dense vector
	public static double[] vectBitwandWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, true);
		for( int j = ai; j < ai+alen; j++ )
			c[aix[j]] = bwAnd(a[j], b[bi+aix[j]]);
		return c;
	}

	//6. sparse vector vs. dense vector
	public static double[] vectBitwandWrite(double[] a, double[] b, int ai, int[] aix, int bi, int alen, int len) {
		return vectBitwandWrite(a, b, aix, ai, bi, alen, len);
	}
	
	// bias add
	
	public static double[] vectBiasaddWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		System.arraycopy(a, ai, c, 0, len);
		LibMatrixDNN.addBias(c, b, 1, 1, b.length, len/b.length);
		return c;
	}
	
	public static double[] vectBiasaddWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, true);
		for(int k=ai; k<ai+alen; k++)
			c[aix[k]] = a[k];
		LibMatrixDNN.addBias(c, b, 1, 1, b.length, len/b.length);
		return c;
	}
	
	// bias mult
	
	public static double[] vectBiasmultWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		System.arraycopy(a, ai, c, 0, len);
		LibMatrixDNN.multBias(c, b, 1, b.length, len/b.length);
		return c;
	}
	
	public static double[] vectBiasmultWrite(double[] a, double[] b, int[] aix, int ai, int bi, int alen, int len) {
		double[] c = allocVector(len, true);
		for(int k=ai; k<ai+alen; k++)
			c[aix[k]] = a[k];
		LibMatrixDNN.multBias(c, b, 1, b.length, len/b.length);
		return c;
	}
	
	//maxpool
	
	public static double[] vectMaxpoolWrite(double[] a, int ai, int len, int rix, int C, int P, int Q, int K, int R, int S, int H, int W) {
		double[] c = allocVector(C*P*Q, true);
		LibMatrixDNNPooling.poolingDenseStride1Pad0(PoolingType.MAX,
			-Double.MAX_VALUE, 1, a, c, rix, rix+1, ai, 0, C, P, Q, R, S, H, W);
		return c;
	} 
	
	public static double[] vectMaxpoolWrite(double[] avals, int[] aix, int ai, int alen, int len, int rix, int C, int P, int Q, int K, int R, int S, int H, int W) {
		double[] a = allocVector(len, true);
		double[] c = allocVector(C*P*Q, true);
		for(int k=ai; k<ai+alen; k++)
			a[aix[k]] = avals[k];
		LibMatrixDNNPooling.poolingDenseStride1Pad0(PoolingType.MAX,
			-Double.MAX_VALUE, 1, a, c, rix, rix+1, 0, 0, C, P, Q, R, S, H, W);
		return c;
	}
	
	//avgpool

	public static double[] vectAvgpoolWrite(double[] a, int ai, int len, int rix, int C, int P, int Q, int K, int R, int S, int H, int W) {
		double[] c = allocVector(C*P*Q, true);
		LibMatrixDNNPooling.poolingDenseStride1Pad0(PoolingType.AVG,
			0, 1/(R*S), a, c, rix, rix+1, ai, 0, C, P, Q, R, S, H, W);
		return c;
	} 
	
	public static double[] vectAvgpoolWrite(double[] avals, int[] aix, int ai, int alen, int len, int rix, int C, int P, int Q, int K, int R, int S, int H, int W) {
		double[] a = allocVector(len, true);
		double[] c = allocVector(C*P*Q, true);
		for(int k=ai; k<ai+alen; k++)
			a[aix[k]] = avals[k];
		LibMatrixDNNPooling.poolingDenseStride1Pad0(PoolingType.AVG,
			0, 1/(R*S), a, c, rix, rix+1, 0, 0, C, P, Q, R, S, H, W);
		return c;
	}
	
	//im2col
	
	public static double[] vectIm2colWrite(double[] a, int ai, int len, int rix, int C, int P, int Q, int K, int R, int S, int H, int W) {
		double[] c = allocVector(C*R*S * P*Q, true);
		LibMatrixDNNIm2Col.im2colDenseStride1Pad0(a, c, ai, C, R, S, H, W, P, Q);
		return c;
	}
	
	public static double[] vectIm2colWrite(double[] avals, int[] aix, int ai, int alen, int len, int rix, int C, int P, int Q, int K, int R, int S, int H, int W) {
		double[] a = allocVector(len, true);
		double[] c = allocVector(C*R*S * P*Q, true);
		for(int k=ai; k<ai+alen; k++)
			a[aix[k]] = avals[k];
		LibMatrixDNNIm2Col.im2colDenseStride1Pad0(a, c, ai, C, R, S, H, W, P, Q);
		return c;
	}
	
	//conv2d matrix mult
	
	public static double[] vectConv2dmmWrite(double[] a, double[] b, int ai, int bi, int len, int rix, int C, int P, int Q, int K, int R, int S, int H, int W) {
		double[] c = allocVector(K*P*Q, true);
		int CRS = C*R*S, PQ = P*Q;
		LibMatrixMult.matrixMultDenseDenseMM(
			new DenseBlockFP64(new int[]{K, CRS}, a), new DenseBlockFP64(new int[]{CRS, PQ}, b),
			new DenseBlockFP64(new int[]{K, PQ}, c), PQ, CRS, 0, K, 0, PQ);
		return c;
	} 

	public static double vectVar(double[] a, int ai, int len) {
		double meanVal = Math.pow(vectMean(a, ai, len), 2);
		double[] aSqr = vectPow2Write(a, ai, len);
		return (vectSum(aSqr, 0, len)-len*meanVal)/(len-1);
	}

	public static double vectVar(double[] avals, int[] aix, int ai, int alen, int len) {
		double meanVal = Math.pow(vectMean(avals, aix, ai, alen, len), 2);
		double[] avalsSqr = vectPow2Write(avals, aix, ai, alen, len);
		return (vectSum(avalsSqr, 0, len)-len*meanVal)/(len-1);
	}

	/**
	 * Vector primitives with SparseRowVector intermediates
	 * Changes:
	 * 	- Changed method signature to avoid method duplicate conflicts
	 * 		e.g. (double[], double, int[], int, int, int) --> (int, double[], double, int[], int, int)
	 *  - Added blen for vector - vector calculations to be able to use both vectors as SparseRowVectors
	 *  - Implemented a new SparseVectorBuffer class that creates a ring buffer for SparseRowVectors in different sizes
	 */

	public static SparseRowVector vectMultWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		if( a == null ) return c;
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = a[j]*bval;
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectMultWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectMultWrite(len, a, bval, aix, ai, alen);
	}

	//version with branching
	public static SparseRowVector vectMultWriteB(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(Math.min(alen, blen));
		if( a == null || b == null ) return c;
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr] * b[bItr];
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx)
				aItr++;
			else
				bItr++;
		}
		c.setSize(index);
		return c;
	}

	//version without branching
	public static SparseRowVector vectMultWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(Math.min(alen, blen));
		int index = 0;
		int aItr = ai;
		int bItr = bi;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			indexes[index] = aIdx;
			values[index] = a[aItr] * b[bItr];
			index += aIdx == bIdx ? 1 : 0;
			aItr += aIdx <= bIdx ? 1 : 0;
			bItr += aIdx >= bIdx ? 1 : 0;
		}
		c.setSize(index);
		return c;
	}

	public static void vectWrite(double[] a, int[] aix, double[] c, int ci, int len) {
		if( a == null ) return;
		for(int j = 0; j < len; j++)
			c[ci+aix[j]] = a[j];
	}

	public static void vectWrite(double[] a, double[] c, int[] aix, int ai, int ci, int alen) {
		if( a == null ) return;
		for(int j = 0; j < ai+alen; j++)
			c[ci+aix[j]] = a[j];
	}

	public static SparseRowVector vectDivWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for( int j = 0; j < ai+alen; j++ ) {
			indexes[j] = aix[j];
			values[j] = a[j] / bval;
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectDivWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = bval / a[j];
		}
		c.setSize(alen);
		return c;
	}

	//version with branching
	public static SparseRowVector vectDivWriteB(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr] / b[bItr];
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = (a[aItr]>0) ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
				aItr++;
				index++;
			} else {
				bItr++;
			}
		}
		c.setSize(index);
		return c;
	}

	//version without branching
	public static SparseRowVector vectDivWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(Math.min(alen, blen));
		int index = 0;
		int aItr = ai;
		int bItr = bi;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			indexes[index] = aIdx;
			values[index] = a[aItr] / b[bItr];
			index += aIdx == bIdx ? 1 : 0;
			aItr += aIdx <= bIdx ? 1 : 0;
			bItr += aIdx >= bIdx ? 1 : 0;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectMinusWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen+blen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr] - b[bItr];
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr];
				aItr++;
				index++;
			} else {
				indexes[index] = bIdx;
				values[index] = -b[bItr];
				bItr++;
				index++;
			}
		}
		for (; aItr < ai+alen; aItr++) {
			indexes[index] = aix[aItr];
			values[index] = a[aItr];
			index++;
		}
		for (; bItr < bi+blen; bItr++) {
			indexes[index] = bix[bItr];
			values[index] = -b[bItr];
			index++;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectPlusWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen+blen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr] + b[bItr];
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr];
				aItr++;
				index++;
			} else {
				indexes[index] = bIdx;
				values[index] = b[bItr];
				bItr++;
				index++;
			}
		}
		for (; aItr < ai+alen; aItr++) {
			indexes[index] = aix[aItr];
			values[index] = a[aItr];
			index++;
		}
		for (; bItr < bi+blen; bItr++) {
			indexes[index] = bix[bItr];
			values[index] = b[bItr];
			index++;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectXorWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval != 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = !(a[aItr] != 0) ? 1 : 0;
					aItr++;
				} else {
					values[index] = 1;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = 1;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = (a[j] != 0) ? 1 : 0;
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectXorWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectXorWrite(len, a, bval, aix, ai, alen);
	}

	public static SparseRowVector vectXorWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen+blen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = ((a[aItr] != 0) != (b[bItr] != 0)) ? 1 : 0;
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = (a[aItr] != 0) ? 1 : 0;
				aItr++;
				index++;
			} else {
				indexes[index] = bIdx;
				values[index] = (b[bItr] != 0) ? 1 : 0;
				bItr++;
				index++;
			}
		}
		for (; aItr < ai+alen; aItr++) {
			indexes[index] = aix[aItr];
			values[index] = (a[aItr] != 0) ? 1 : 0;
			index++;
		}
		for (; bItr < bi+blen; bItr++) {
			indexes[index] = bix[bItr];
			values[index] = (b[bItr] != 0) ? 1 : 0;
			index++;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectPowWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval == 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = Math.pow(a[aItr], bval) - 1;
					aItr++;
				} else {
					values[index] = 1;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = 1;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = Math.pow(a[j], bval);
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectMinWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval < 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = Math.min(a[aItr], bval);
					aItr++;
				} else {
					values[index] = bval;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = bval;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = Math.min(a[j], bval);
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectMinWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectMinWrite(len, a, bval, aix, ai, alen);
	}

	public static SparseRowVector vectMinWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen+blen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = Math.min(a[aItr], b[bItr]);
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = Math.min(a[aItr], 0);
				aItr++;
				index++;
			} else {
				indexes[index] = bIdx;
				values[index] = Math.min(b[bItr], 0);
				bItr++;
				index++;
			}
		}
		for (; aItr < ai+alen; aItr++) {
			indexes[index] = aix[aItr];
			values[index] = Math.min(a[aItr], 0);
			index++;
		}
		for (; bItr < bi+blen; bItr++) {
			indexes[index] = bix[bItr];
			values[index] = Math.min(b[bItr], 0);
			index++;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectMaxWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval > 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = Math.max(a[aItr], bval);
					aItr++;
				} else {
					values[index] = bval;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = bval;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = Math.max(a[j], bval);
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectMaxWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectMaxWrite(len, a, bval, aix, ai, alen);
	}

	public static SparseRowVector vectMaxWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen+blen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = Math.max(a[aItr], b[bItr]);
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = Math.max(a[aItr], 0);
				aItr++;
				index++;
			} else {
				indexes[index] = bIdx;
				values[index] = Math.max(b[bItr], 0);
				bItr++;
				index++;
			}
		}
		for (; aItr < ai+alen; aItr++) {
			indexes[index] = aix[aItr];
			values[index] = Math.max(a[aItr], 0);
			index++;
		}
		for (; bItr < bi+blen; bItr++) {
			indexes[index] = bix[bItr];
			values[index] = Math.max(b[bItr], 0);
			index++;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectEqualWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval == 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = a[aItr] == bval ? 1 : 0;
					aItr++;
				} else {
					values[index] = 1;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = 1;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = a[j] == bval ? 1 : 0;
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectEqualWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectEqualWrite(len, a, bval, aix, ai, alen);
	}

	//doesn't return SparseRowVector, but still uses two sparse vectors as inputs
	public static double[] vectEqualWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		double[] c = allocVector(len, true, 1);
		int aItr = ai;
		int bItr = bi;
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if (aIdx == bIdx) {
				c[aIdx] = (a[aItr] == b[bItr]) ? 1 : 0;
				aItr++;
				bItr++;
			} else if(aIdx < bIdx) {
				c[aIdx] = a[aItr] == 0 ? 1 : 0;
				aItr++;
			} else {
				c[bIdx] = b[bItr] == 0 ? 1 : 0;
				bItr++;
			}
		}
		for (; aItr < ai+alen; aItr++) c[aix[aItr]] = 0;
		for (; bItr < bi+blen; bItr++)  c[bix[bItr]] = 0;
		return c;
	}

	public static SparseRowVector vectNotequalWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval != 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = a[aItr] != bval ? 1 : 0;
					aItr++;
				} else {
					values[index] = 1;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = 1;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = a[j] != bval ? 1 : 0;
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectNotequalWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectNotequalWrite(len, a, bval, aix, ai, alen);
	}

	public static SparseRowVector vectNotequalWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen+blen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = (a[aItr] != b[bItr]) ? 1 : 0;
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr] != 0 ? 1 : 0;
				aItr++;
				index++;
			} else {
				indexes[index] = bIdx;
				values[index] = b[bItr] != 0 ? 1 : 0;
				bItr++;
				index++;
			}
		}
		for (; aItr < ai+alen; aItr++) {
			indexes[index] = aix[aItr];
			values[index] = a[aItr] != 0 ? 1 : 0;
			index++;
		}
		for (; bItr < bi+blen; bItr++) {
			indexes[index] = bix[bItr];
			values[index] = b[bItr] != 0 ? 1 : 0;
			index++;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectLessWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval > 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = a[aItr] < bval ? 1 : 0;
					aItr++;
				} else {
					values[index] = 1;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = 1;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = a[j] < bval ? 1 : 0;
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectLessWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectGreaterequalWrite(len, a, bval, aix, ai, alen);
	}

	public static SparseRowVector vectLessWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen+blen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = (a[aItr] < b[bItr]) ? 1 : 0;
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr] < 0 ? 1 : 0;
				aItr++;
				index++;
			} else {
				indexes[index] = bIdx;
				values[index] = 0 < b[bItr] ? 1 : 0;
				bItr++;
				index++;
			}
		}
		for (; aItr < ai+alen; aItr++) {
			indexes[index] = aix[aItr];
			values[index] = a[aItr] < 0? 1 : 0;
			index++;
		}
		for (; bItr < bi+blen; bItr++) {
			indexes[index] = bix[bItr];
			values[index] = 0 < b[bItr] ? 1 : 0;
			index++;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectLessequalWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval >= 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = a[aItr] <= bval ? 1 : 0;
					aItr++;
				} else {
					values[index] = 1;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = 1;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = a[j] <= bval ? 1 : 0;
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectLessequalWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectGreaterWrite(len, a, bval, aix, ai, alen);
	}

	//doesn't return SparseRowVector, but still uses two sparse vectors as inputs
	public static double[] vectLessequalWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		double[] c = allocVector(len, true, 1);
		int aItr = ai;
		int bItr = bi;
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				c[aIdx] = (a[aItr] <= b[bItr]) ? 1 : 0;
				aItr++;
				bItr++;
			} else if(aIdx < bIdx) {
				c[aIdx] = (a[aItr] <= 0) ? 1 : 0;
				aItr++;
			} else {
				c[bIdx] = (0 <= b[bItr]) ? 1 : 0;
				bItr++;
			}
		}
		for(; aItr < ai+alen; aItr++) c[aix[aItr]] = (a[aItr] <= 0) ? 1 : 0;
		for(; bItr < bi+blen; bItr++)  c[bix[bItr]] = (0 <= b[bItr]) ? 1 : 0;
		return c;
	}

	public static SparseRowVector vectGreaterWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval < 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = a[aItr] > bval ? 1 : 0;
					aItr++;
				} else {
					values[index] = 1;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = 1;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = a[j] > bval ? 1 : 0;
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectGreaterWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectLessequalWrite(len, a, bval, aix, ai, alen);
	}

	public static SparseRowVector vectGreaterWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen+blen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = (a[aItr] > b[bItr]) ? 1 : 0;
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				indexes[index] = aIdx;
				values[index] = a[aItr] > 0 ? 1 : 0;
				aItr++;
				index++;
			} else {
				indexes[index] = bIdx;
				values[index] = 0 > b[bItr] ? 1 : 0;
				bItr++;
				index++;
			}
		}
		for (; aItr < ai+alen; aItr++) {
			indexes[index] = aix[aItr];
			values[index] = a[aItr] > 0 ? 1 : 0;
			index++;
		}
		for (; bItr < bi+blen; bItr++) {
			indexes[index] = bix[bItr];
			values[index] = 0 > b[bItr] ? 1 : 0;
			index++;
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectGreaterequalWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		if(bval <= 0) {
			SparseRowVector c = allocSparseVector(len);
			int[] indexes = c.indexes();
			double[] values = c.values();
			int index = 0;
			int aItr = 0;
			while(aItr < ai+alen && index < len) {
				indexes[index] = index;
				if(aix[aItr] == index) {
					values[index] = a[aItr] >= bval ? 1 : 0;
					aItr++;
				} else {
					values[index] = 1;
				}
				index++;
			}
			for(; index < len; index++) {
				indexes[index] = index;
				values[index] = 1;
			}
			c.setSize(len);
			return c;
		} else {
			SparseRowVector c = allocSparseVector(alen);
			int[] indexes = c.indexes();
			double[] values = c.values();
			for(int j = 0; j < ai+alen; j++) {
				indexes[j] = aix[j];
				values[j] = a[j] >= bval ? 1 : 0;
			}
			c.setSize(alen);
			return c;
		}
	}

	public static SparseRowVector vectGreaterequalWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectLessWrite(len, a, bval, aix, ai, alen);
	}

	//doesn't return SparseRowVector, but still uses two sparse vectors as inputs
	public static double[] vectGreaterequalWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		double[] c = allocVector(len, true, 1);
		int aItr = ai;
		int bItr = bi;
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				c[aIdx] = (a[aItr] >= b[bItr]) ? 1 : 0;
				aItr++;
				bItr++;
			} else if(aIdx < bIdx) {
				c[aIdx] = (a[aItr] >= 0) ? 1 : 0;
				aItr++;
			} else {
				c[bIdx] = (0 >= b[bItr]) ? 1 : 0;
				bItr++;
			}
		}
		for(; aItr < ai+alen; aItr++) c[aix[aItr]] = (a[aItr] >= 0) ? 1 : 0;
		for(; bItr < bi+blen; bItr++)  c[bix[bItr]] = (0 >= b[bItr]) ? 1 : 0;
		return c;
	}

	public static SparseRowVector vectBitwandWrite(int len, double[] a, double bval, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		int bval1 = (int) bval;
		for( int j = ai; j < ai+alen; j++ ) {
			indexes[j] = aix[j];
			values[j] = bwAnd(a[j], bval1);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectBitwandWrite(int len, double bval, double[] a, int[] aix, int ai, int alen) {
		return vectBitwandWrite(len, a, bval, aix, ai, alen);
	}

	public static SparseRowVector vectBitwandWrite(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
		SparseRowVector c = allocSparseVector(alen);
		int aItr = ai;
		int bItr = bi;
		int index = 0;
		int[] indexes = c.indexes();
		double[] values = c.values();
		while(aItr < ai+alen && bItr < bi+blen) {
			int aIdx = aix[aItr];
			int bIdx = bix[bItr];
			if(aIdx == bIdx) {
				indexes[index] = aIdx;
				values[index] = bwAnd(a[aItr], b[bItr]);
				aItr++;
				bItr++;
				index++;
			} else if(aIdx < bIdx) {
				aItr++;
			} else {
				bItr++;
			}
		}
		c.setSize(index);
		return c;
	}

	public static SparseRowVector vectSqrtWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.sqrt(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectAbsWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.abs(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectRoundWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.round(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectCeilWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.ceil(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectFloorWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.floor(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectSinWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.sin(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectTanWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.tan(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectAsinWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.asin(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectAtanWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.atan(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectSinhWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.sinh(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectTanhWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.tanh(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	public static SparseRowVector vectSignWrite(int len, double[] a, int[] aix, int ai, int alen) {
		SparseRowVector c = allocSparseVector(alen);
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = Math.signum(a[j]);
		}
		c.setSize(alen);
		return c;
	}

	//todo MatrixMult, pow2 and mult2 drafts
//	public static SparseRowVector vectMatrixMult(int len, double[] a, double[] b, int[] aix, int[] bix, int ai, int bi, int alen, int blen) {
//		//note: assumption b is already transposed for efficient dot products
//		int m2clen = b.length / len;
//		SparseRowVector c = allocSparseVector(m2clen);
//		for(int i = 0; i < m2clen; i++) {
//			c.set(bix[i], LibMatrixMult.dotProduct(a, aix, ai, alen, b, bix, bi, blen));
//		}
//		return c;
//	}
//
//	public static SparseRowVector vectPow2Write(int len, double[] a, int[] aix, int ai, int alen) {
//		SparseRowVector c = allocSparseVector(len);
//		for(int j = 0; j < ai+alen; j++)
//			c.set(aix[j], a[j] * a[j]);
//		return c;
//	}
//
//	public static SparseRowVector vectMult2Write(int len, double[] a, int[] aix, int ai, int alen) {
//		SparseRowVector c = allocSparseVector(len);
//		for(int j = 0; j < ai+alen; j++)
//			c.set(aix[j], a[j] + a[j]);
//		return c;
//	}

	//complex builtin functions that are not directly generated
	//(included here in order to reduce the number of imports)
	
	public static double intDiv(double in1, double in2) {
		return intDiv.execute(in1, in2);
	}
	
	public static double mod(double in1, double in2) {
		return mod.execute(in1, in2);
	}
	
	public static double bwAnd(double in1, double in2) {
		return bwAnd.execute(in1, in2);
	}
	
	public static boolean isFlipOuter(int len1, int len2) {
		return (len1 > 64 * len2);
	}
	
	//dynamic memory management
	
	public static void setupThreadLocalMemory(int numVectors, int len) {
		if( numVectors > 0 )
			setupThreadLocalMemory(numVectors, len, -1);
	}
	
	public static void setupThreadLocalMemory(int numVectors, int len, int len2) {
		if( numVectors > 0 )
			memPool.set(new VectorBuffer(numVectors, len, len2));
	}

	public static void setupSparseThreadLocalMemory(int numVectors, int len, int len2) {
		if( numVectors > 0 )
			sparseMemPool.set(new SparseVectorBuffer(numVectors, len, len2));
	}
	
	public static void cleanupThreadLocalMemory() {
		memPool.remove();
	}

	public static void cleanupSparseThreadLocalMemory() {
		sparseMemPool.remove();
	}
	
	public static double[] allocVector(int len, boolean reset) {
		return allocVector(len, reset, 0);
	}
	
	protected static double[] allocVector(int len, boolean reset, double resetVal) {
		VectorBuffer buff = memPool.get();
		
		//find next matching vector in ring buffer or
		//allocate new vector if required
		double[] vect = buff.next(len);
		if( vect == null )
			vect = new double[len];
		
		//reset vector if required
		if( reset )
			Arrays.fill(vect, resetVal);
		return vect;
	}

	public static SparseRowVector allocSparseVector(int len) {
		SparseVectorBuffer buff = sparseMemPool.get();

		//find next matching vector in ring buffer or
		//allocate new vector if no vector was returned
		SparseRowVector vect = buff.next(len);
		if(vect == null)
			vect = new SparseRowVector(len);
			//reset vector for normal outputs
		else if(vect.size() != 0)
			vect.reset(len, len);

		return vect;
	}
	
	/**
	 * Simple ring buffer of allocated vectors, where
	 * vectors of different sizes are interspersed.
	 */
	private static class VectorBuffer {
		private static final int MAX_SIZE = 512*1024; //4MB
		private final double[][] _data;
		private int _pos;
		private int _len1;
		private int _len2;
		
		public VectorBuffer(int num, int len1, int len2) {
			//best effort size restriction since large intermediates
			//not necessarily used (num refers to the total number)
			len1 = Math.min(len1, MAX_SIZE);
			len2 = Math.min(len2, MAX_SIZE);
			//pre-allocate ring buffer
			int lnum = (len2>0 && len1!=len2) ? 2*num : num;
			_data = new double[lnum][];
			for( int i=0; i<num; i++ ) {
				if( lnum > num ) {
					_data[2*i] = new double[len1];
					_data[2*i+1] = new double[len2];
				}
				else {
					_data[i] = new double[len1];
				}
			}
			_pos = -1;
			_len1 = len1;
			_len2 = len2;
		}
		public double[] next(int len) {
			if( _len1!=len && _len2!=len )
				return null;
			do {
				_pos = (_pos+1>=_data.length) ? 0 : _pos+1;
			} while( _data[_pos].length!=len );
			return _data[_pos];
		}
		@SuppressWarnings("unused")
		public boolean isReusable(int num, int len1, int len2) {
			int lnum = (len2>0 && len1!=len2) ? 2*num : num;
			return (_len1 == len1 && _len2 == len2
				&& _data.length == lnum);
		}
	}

	/**
	 * Simple ring buffer of allocated SparseRowVectors, where
	 * vectors of different sizes are interspersed.
	 */
	private static class SparseVectorBuffer {
		private static final int MAX_SIZE = 512*1024; //4MB
		private final SparseRowVector[] _data;
		private int _pos;
		private int _len1;
		private int _len2;

		public SparseVectorBuffer(int num, int len1, int len2) {
			//best effort size restriction since large intermediates
			//not necessarily used (num refers to the total number)
			len1 = Math.min(len1, MAX_SIZE);
			len2 = Math.min(len2, MAX_SIZE);
			//pre-allocate ring buffer
			int lnum = (len2>0 && len1!=len2) ? 2*num : num;
			_data = new SparseRowVector[lnum];
			for( int i=0; i<num; i++ ) {
				if( lnum > num ) {
					_data[2*i] = new SparseRowVector(len1);
					_data[2*i+1] = new SparseRowVector(len2);
				}
				else {
					_data[i] = new SparseRowVector(len1);
				}
			}
			_pos = -1;
			_len1 = len1;
			_len2 = len2;
		}
		public SparseRowVector next(int len) {
			if( _len1<len && _len2<len )
				return null;
			do {
				_pos = (_pos+1>=_data.length) ? 0 : _pos+1;
			} while( _data[_pos].values().length<len );
			return _data[_pos];
		}
		@SuppressWarnings("unused")
		public boolean isReusable(int num, int len1, int len2) {
			int lnum = (len2>0 && len1!=len2) ? 2*num : num;
			return (_len1 == len1 && _len2 == len2
				&& _data.length == lnum);
		}
	}
}
