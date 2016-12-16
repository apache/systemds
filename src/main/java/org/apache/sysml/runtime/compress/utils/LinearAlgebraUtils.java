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
}
