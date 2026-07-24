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
	private static final long serialVersionUID = -5957679254041639561L;
	
	private final int _ABCount;
	private final boolean _Bsupplied;
	private final int _ACount;
	private final int _AZCount;
	private final int _ZSize;
	private final int _AZStartIndex;
	private final EOpNodeFuse.EinsumRewriteType _EinsumRewriteType;

	public EinsumSpoofRowwise(EOpNodeFuse.EinsumRewriteType einsumRewriteType, RowType rowType, long constDim2,
		int abCount, boolean bSupplied, int aCount, int azCount, int zSize) {
		super(rowType, constDim2, false, 1);
		_ABCount = abCount;
		_Bsupplied = bSupplied;
		_ACount = aCount;
		_AZStartIndex = abCount + (_Bsupplied ? 1 : 0) + aCount;
		_AZCount = azCount;
		_EinsumRewriteType = einsumRewriteType;
		_ZSize = zSize;
	}

	protected void genexec(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix,
		int rix) {
		switch(_EinsumRewriteType) {
			case AB_BA_B_A__AB -> {
				genexecAB(a, ai, b, null, c, ci, len, grix, rix);
				if(scalars.length != 0) { LibMatrixMult.vectMultiplyWrite(scalars[0], c, c, ci, ci, len); }
			}
			case AB_BA_A__B -> {
				genexecB(a, ai, b, null, c, ci, len, grix, rix);
			}
			case AB_BA_B_A__A -> {
				genexecAor(a, ai, b, null, c, ci, len, grix, rix);
				if(scalars.length != 0) { c[rix] *= scalars[0]; }
			}
			case AB_BA_B_A__ -> {
				genexecAor(a, ai, b, null, c, ci, len, grix, rix);
				if(scalars.length != 0) { c[0] *= scalars[0]; }
			}
			case AB_BA_B_A_AZ__Z -> {
				double[] temp = {0};
				genexecAor(a, ai, b, null, temp, 0, len, grix, rix);
				if(scalars.length != 0) { temp[0] *= scalars[0]; }
				if(_AZCount > 1) {
					double[] temp2 = new double[_ZSize];
					int bi = _AZStartIndex;
					LibMatrixMult.vectMultiplyWrite(b[bi++].values(0), b[bi++].values(0), temp2, _ZSize * rix,
						_ZSize * rix, 0, _ZSize);
					while(bi < _AZStartIndex + _AZCount) {
						LibMatrixMult.vectMultiplyWrite(temp2, b[bi++].values(0), temp2, 0, _ZSize * rix, 0, _ZSize);
					}
					LibMatrixMult.vectMultiplyAdd(temp[0], temp2, c, 0, 0, _ZSize);
				}
				else
					LibMatrixMult.vectMultiplyAdd(temp[0], b[_AZStartIndex].values(rix), c, _ZSize * rix, 0, _ZSize);
			}
			case AB_BA_A_AZ__BZ -> {
				double[] temp = new double[len];
				genexecB(a, ai, b, null, temp, 0, len, grix, rix);
				if(scalars.length != 0) {
					LibMatrixMult.vectMultiplyWrite(scalars[0], temp, temp, 0, 0, len);
				}
				if(_AZCount > 1) {
					double[] temp2 = new double[_ZSize];
					int bi = _AZStartIndex;
					LibMatrixMult.vectMultiplyWrite(b[bi++].values(0), b[bi++].values(0), temp2, _ZSize * rix,
						_ZSize * rix, 0, _ZSize);
					while(bi < _AZStartIndex + _AZCount) {
						LibMatrixMult.vectMultiplyWrite(temp2, b[bi++].values(0), temp2, 0, _ZSize * rix, 0, _ZSize);
					}
					LibSpoofPrimitives.vectOuterMultAdd(temp, temp2, c, 0, 0, 0, len, _ZSize);
				}
				else
					LibSpoofPrimitives.vectOuterMultAdd(temp, b[_AZStartIndex].values(rix), c, 0, _ZSize * rix, 0, len, _ZSize);
			}
			case AB_BA_A_AZ__ZB -> {
				double[] temp = new double[len];
				genexecB(a, ai, b, null, temp, 0, len, grix, rix);
				if(scalars.length != 0) {
					LibMatrixMult.vectMultiplyWrite(scalars[0], temp, temp, 0, 0, len);
				}
				if(_AZCount > 1) {
					double[] temp2 = new double[_ZSize];
					int bi = _AZStartIndex;
					LibMatrixMult.vectMultiplyWrite(b[bi++].values(0), b[bi++].values(0), temp2, _ZSize * rix,
						_ZSize * rix, 0, _ZSize);
					while(bi < _AZStartIndex + _AZCount) {
						LibMatrixMult.vectMultiplyWrite(temp2, b[bi++].values(0), temp2, 0, _ZSize * rix, 0, _ZSize);
					}
					LibSpoofPrimitives.vectOuterMultAdd(temp2, temp, c, 0, 0, 0, _ZSize, len);
				}
				else
					LibSpoofPrimitives.vectOuterMultAdd(b[_AZStartIndex].values(rix), temp, c, _ZSize * rix, 0, 0, _ZSize, len);
			}
			default -> throw new NotImplementedException();
		}
	}

	private void genexecAB(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix,
		int rix) {
		int bi = 0;
		double[] TMP1 = null;
		if(_ABCount != 0) {
			if(_ABCount == 1 & _ACount == 0 && !_Bsupplied) {
				LibMatrixMult.vectMultiplyWrite(a, b[0].values(rix), c, ai, ai, ci, len);
				return;
			}
			TMP1 = LibSpoofPrimitives.vectMultWrite(a, b[bi++].values(rix), ai, ai, len);
			while(bi < _ABCount) {
				if(_ACount == 0 && !_Bsupplied && bi == _ABCount - 1) {
					LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), c, 0, ai, ci, len);
				}
				else {
					LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
				}
			}
		}

		if(_Bsupplied) {
			if(_ACount == 1)
				if(TMP1 == null)
					vectMultiplyWrite(b[bi + 1].values(0)[rix], a, b[bi].values(0), c, ai, 0, ci, len);
				else
					vectMultiplyWrite(b[bi + 1].values(0)[rix], TMP1, b[bi].values(0), c, 0, 0, ci, len);
			else if(TMP1 == null)
				LibMatrixMult.vectMultiplyWrite(a, b[bi].values(0), c, ai, 0, ci, len);
			else
				LibMatrixMult.vectMultiplyWrite(TMP1, b[bi].values(0), c, 0, 0, ci, len);
		}
		else if(_ACount == 1) {
			if(TMP1 == null)
				LibMatrixMult.vectMultiplyWrite(b[bi].values(0)[rix], a, c, ai, ci, len);
			else
				LibMatrixMult.vectMultiplyWrite(b[bi].values(0)[rix], TMP1, c, 0, ci, len);
		}
	}

	private void genexecB(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix,
		int rix) {
		int bi = 0;
		double[] TMP1 = null;
		if(_ABCount == 1 && _ACount == 0)
			LibMatrixMult.vectMultiplyAdd(a, b[bi++].values(rix), c, ai, ai, 0, len);
		else if(_ABCount != 0) {
			TMP1 = LibSpoofPrimitives.vectMultWrite(a, b[bi++].values(rix), ai, ai, len);
			while(bi < _ABCount) {
				if(_ACount == 0 && bi == _ABCount - 1) {
					LibMatrixMult.vectMultiplyAdd(TMP1, b[bi++].values(rix), c, 0, ai, 0, len);
				}
				else {
					LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
				}
			}
		}

		if(_ACount == 1) {
			if(TMP1 == null)
				LibMatrixMult.vectMultiplyAdd(b[bi].values(0)[rix], a, c, ai, 0, len);
			else
				LibMatrixMult.vectMultiplyAdd(b[bi].values(0)[rix], TMP1, c, 0, 0, len);
		}
	}

	private void genexecAor(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix) {
		int bi = 0;
		double[] TMP1 = null;
		double TMP2 = 0;
		if(_ABCount == 1 && !_Bsupplied)
			TMP2 = LibSpoofPrimitives.dotProduct(a, b[bi++].values(rix), ai, ai, len);
		else if(_ABCount != 0) {
			TMP1 = LibSpoofPrimitives.vectMultWrite(a, b[bi++].values(rix), ai, ai, len);
			while(bi < _ABCount) {
				if(!_Bsupplied && bi == _ABCount - 1)
					TMP2 = LibSpoofPrimitives.dotProduct(TMP1, b[bi++].values(rix), 0, ai, len);
				else
					LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
			}
		}

		if(_Bsupplied)
			if(_ABCount != 0) TMP2 = LibSpoofPrimitives.dotProduct(TMP1, b[bi++].values(0), 0, 0, len);
			else TMP2 = LibSpoofPrimitives.dotProduct(a, b[bi++].values(0), ai, 0, len);
		else if(_ABCount == 0) TMP2 = LibSpoofPrimitives.vectSum(a, ai, len);

		if(_ACount == 1) TMP2 *= b[bi].values(0)[rix];

		if(_EinsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__A) c[ci] = TMP2;
		else c[0] += TMP2;
	}

	protected void genexec(double[] avals, int[] aix, int ai, SideInput[] b, double[] scalars, double[] c, int ci,
		int alen, int len, long grix, int rix) {
		throw new RuntimeException("Sparse fused einsum not implemented");
	}

	// I am not sure if it is worth copying to LibMatrixMult so for now added it here
	private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
	private static final int vLen = SPECIES.length();

	public static void vectMultiplyWrite(final double aval, double[] a, double[] b, double[] c, int ai, int bi, int ci,
		final int len) {
		final int bn = len % vLen;

		//rest, not aligned to vLen-blocks
		for(int j = 0; j < bn; j++, ai++, bi++, ci++)
			c[ci] = aval * b[bi] * a[ai];

		//unrolled vLen-block (for better instruction-level parallelism)
		DoubleVector avalVec = DoubleVector.broadcast(SPECIES, aval);
		for(int j = bn; j < len; j += vLen, ai += vLen, bi += vLen, ci += vLen) {
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai);
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi);
			avalVec.mul(bVec).mul(aVec).intoArray(c, ci);
		}
	}

	public static void vectMultiplyAdd(final double aval, double[] a, double[] b, double[] c, int ai, int bi, int ci,
		final int len) {
		final int bn = len % vLen;

		//rest, not aligned to vLen-blocks
		for(int j = 0; j < bn; j++, ai++, bi++, ci++)
			c[ci] += aval * b[bi] * a[ai];

		//unrolled vLen-block (for better instruction-level parallelism)
		DoubleVector avalVec = DoubleVector.broadcast(SPECIES, aval);
		for(int j = bn; j < len; j += vLen, ai += vLen, bi += vLen, ci += vLen) {
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai);
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi);
			DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci);
			DoubleVector tmp = aVec.mul(bVec);
			tmp.fma(avalVec, cVec).intoArray(c, ci);
		}
	}
}
