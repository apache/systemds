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

package org.apache.sysds.runtime.einsum;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.codegen.SpoofRowwise;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;

public final class EinsumSpoofRowwise extends SpoofRowwise {
    private final int _ABCount;
    private final int _BCount;
    private final int _ACount;
    private final int _ZCount;
    private final int _AZCount;
    private final int _ZSize;

    private final int _uptoBCumCount;
    private final int _uptoZCumCount;

    private final EOpNodeFuse.EinsumRewriteType _EinsumRewriteType;

    public EinsumSpoofRowwise(EOpNodeFuse.EinsumRewriteType einsumRewriteType, RowType rowType, long constDim2, boolean tb1, int reqVectMem, int abCount, int bCount, int aCount, int zCount, int azCount, int zSize) {
        super(rowType, constDim2, tb1, reqVectMem);
        _ABCount = abCount;
        _BCount = bCount;
        _uptoBCumCount = bCount+ abCount;
        _ACount = aCount;
        _ZCount = zCount;
        _uptoZCumCount = _uptoBCumCount + aCount;
        _AZCount = azCount;
        _EinsumRewriteType = einsumRewriteType;
        _ZSize = zSize;
    }
    protected void genexec(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix) {
        switch (_EinsumRewriteType) {
            case AB_BA_B_A__AB -> {
				genexec_AB(a,ai,b,scalars,c,ci,len,grix,rix);
				if (scalars.length != 0) {
					LibMatrixMult.vectMultiplyWrite(scalars[0], c,c,ci,ci, len);
				}
			}
            case AB_BA_B_A__B -> {
				genexec_B(a,ai,b,scalars,c,ci,len,grix,rix);
			}
            case AB_BA_B_A__A -> {
//				HARDCODEDgenexec_A_or_(a,ai,b,scalars,c,ci,len,grix,rix);
				genexec_A_or_(a,ai,b,scalars,c,ci,len,grix,rix);
				if (scalars.length != 0) {
					c[rix] *= scalars[0];
				}
			}
            case AB_BA_B_A__ -> {
				genexec_A_or_(a,ai,b,scalars,c,ci,len,grix,rix);
				if (scalars.length != 0) {
					c[0] *= scalars[0];
				}
			}
            case AB_BA_B_A_AZ__Z -> {
                double[] temp = {0};
                genexec_A_or_(a,ai,b,scalars,temp,0,len,grix,rix);
				if (scalars.length != 0) {
					temp[0] *= scalars[0];
				}
                LibMatrixMult.vectMultiplyAdd(temp[0], b[_uptoZCumCount].values(rix), c, _ZSize*rix,0, _ZSize);
            }
            case AB_BA_B_A_AZ__BZ -> {
                double[] temp = new double[len];
                genexec_B(a,ai,b,scalars,temp,0,len,grix,rix);
				if (scalars.length != 0) {
					LibMatrixMult.vectMultiplyWrite(scalars[0], temp,temp,0,0,len);
				}
                LibSpoofPrimitives.vectOuterMultAdd(temp, b[_uptoZCumCount].values(rix), c,0, _ZSize*rix, 0,  len,_ZSize);
            }
            case AB_BA_B_A_AZ__ZB -> {
                double[] temp = new double[len];
                genexec_B(a,ai,b,scalars,temp,0,len,grix,rix);
				if (scalars.length != 0) {
					LibMatrixMult.vectMultiplyWrite(scalars[0], temp,temp,0,0,len);
				}
                LibSpoofPrimitives.vectOuterMultAdd(b[_uptoZCumCount].values(rix),temp , c,_ZSize*rix,0, 0, _ZSize, len);
            }
            default -> throw new NotImplementedException();
        }

    }
    private void genexec_AB(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix,
		int rix) {
        int bi = 0;
        double[] TMP1 = null;
        if (_ABCount != 0){
            TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[bi++].values(rix),ai,ai,len);
            while (bi < _ABCount) {
                if(_ACount == 0 && _BCount == 0 && bi == _ABCount-1) {
                    LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), c, 0, ai, ci, len);
                }else {
                    LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
                }
            }
        }

        if(_BCount == 1) {
			if(_ACount == 1)
				if(TMP1 == null)
					vectMultiplyWrite(b[bi + 1].values(0)[rix], a, b[bi].values(0), c, ai, 0, ci, len);
				else
					vectMultiplyWrite(b[bi + 1].values(0)[rix], TMP1, b[bi].values(0), c, 0, 0, ci, len);
			else if(TMP1 == null)
				LibMatrixMult.vectMultiplyWrite(a, b[bi].values(0), c, ai, 0, ci, len);
			else
				LibMatrixMult.vectMultiplyWrite(TMP1, b[bi].values(0), c, 0, 0, ci, len);
		} else if(_ACount == 1) {
			if(TMP1 == null)
				LibMatrixMult.vectMultiplyWrite(b[bi].values(0)[rix],a,c, ai, ci, len);
			else
				LibMatrixMult.vectMultiplyWrite(b[bi].values(0)[rix],TMP1,c, 0, ci, len);
		}
    }

  private void genexec_B(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix,
	  int rix) {
    int bi = 0;
    double[] TMP1 = null;
    if(_ABCount != 0) {
      TMP1 = LibSpoofPrimitives.vectMultWrite(a, b[bi++].values(rix), ai, ai, len);
      while(bi < _ABCount) {
        if(_ACount == 0 && _BCount == 0 && bi == _ABCount - 1) {
          LibMatrixMult.vectMultiplyAdd(TMP1, b[bi++].values(rix), c, 0, ai, 0, len);
        }
        else {
          LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
        }
      }
    }

    if(_BCount == 1) {
      if(_ACount == 1)
        if(TMP1 == null)
          vectMultiplyAdd(b[bi + 1].values(0)[rix], a, b[bi].values(0), c, ai, 0, 0, len);
        else
          vectMultiplyAdd(b[bi + 1].values(0)[rix], TMP1, b[bi].values(0), c, 0, 0, 0, len);
      else if(TMP1 == null)
        LibMatrixMult.vectMultiplyAdd(a, b[bi].values(0), c, ai, 0, 0, len);
      else
        LibMatrixMult.vectMultiplyAdd(TMP1, b[bi].values(0), c, 0, 0, 0, len);
    }
    else if(_ACount == 1) {
      if(TMP1 == null)
        LibMatrixMult.vectMultiplyAdd(b[bi].values(0)[rix], a, c, ai, 0, len);
      else
        LibMatrixMult.vectMultiplyAdd(b[bi].values(0)[rix], TMP1, c, 0, 0, len);
    }
  }

    private void genexec_A_or_(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len,
		long grix, int rix) {
        int bi = 0;
        double[] TMP1 = null;
        double TMP2 = 0;
		if (_ABCount == 1 && _BCount == 0)
			TMP2 = LibSpoofPrimitives.dotProduct(TMP1,b[bi++].values(rix),0,ai,len);
		else if (_ABCount != 0) {
            TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[bi++].values(rix),ai,ai,len);
            while (bi < _ABCount) {
                if(_BCount == 0 && bi == _ABCount - 1)
                    TMP2 = LibSpoofPrimitives.dotProduct(TMP1,b[bi++].values(rix),0,ai,len);
                else
                    LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
            }
        }

        if(_BCount == 1)
			if(_ABCount != 0)
				TMP2 = LibSpoofPrimitives.dotProduct(TMP1,b[bi++].values(0),0,0,len);
			else
				TMP2 = LibSpoofPrimitives.dotProduct(a,b[bi++].values(0),ai,0,len);
		else if(_ABCount == 0)
			TMP2 = LibSpoofPrimitives.vectSum(a, ai, len);

		if(_ACount == 1)
				TMP2 *= b[bi].values(0)[rix];

        if (_EinsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__A) c[ci] = TMP2;
        else c[0] += TMP2;
    }

	private void HARDCODEDgenexec_A_or_(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci,
	  int len, long grix, int rix) {
	  double[] TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[0].values(rix),ai,ai,len);
	  double TMP2 = LibSpoofPrimitives.dotProduct(TMP1,b[1].values(0),0,ai,len);
	  TMP2 *= b[2].values(0)[rix];
	  c[rix] = TMP2;
	}

    protected void genexec(double[] avals, int[] aix, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int alen, int len, long grix, int rix) {
		throw new RuntimeException("Sparse fused einsum not implemented");
    }


	// I am not sure if it is worth copying to LibMatrixMult so for now added it here
	private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
	private static final int vLen = SPECIES.length();
	public static void vectMultiplyWrite( final double aval, double[] a, double[] b, double[] c,int ai, int bi, int ci, final int len )
	{
		final int bn = len%vLen;

		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, ai++, bi++, ci++)
			c[ ci ] = aval * b[ bi ] * a[ ai ];

		//unrolled vLen-block (for better instruction-level parallelism)
		DoubleVector avalVec = DoubleVector.broadcast(SPECIES, aval);
		for( int j = bn; j < len; j+=vLen, ai+=vLen, bi+=vLen, ci+=vLen)
		{
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai);
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi);
			avalVec.mul(bVec).mul(aVec).intoArray(c, ci);
		}
	}

	public static void vectMultiplyAdd( final double aval, double[] a, double[] b, double[] c,int ai, int bi, int ci, final int len )
	{
		final int bn = len%vLen;

		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, ai++, bi++, ci++)
			c[ ci ] += aval * b[ bi ] * a[ ai ];

		//unrolled vLen-block (for better instruction-level parallelism)
		DoubleVector avalVec = DoubleVector.broadcast(SPECIES, aval);
		for( int j = bn; j < len; j+=vLen, ai+=vLen, bi+=vLen, ci+=vLen)
		{
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai);
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi);
			DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci);
			DoubleVector tmp = aVec.mul(bVec);
			tmp.fma(avalVec, cVec).intoArray(c, ci);
		}
	}
}
