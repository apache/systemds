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
package org.apache.sysml.udf.lib;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * Performs following operation:
 * # Computes the intersection ("meet") of equivalence classes for
 * # each row of A and B, excluding 0-valued cells.
 * # INPUT:
 * #   A, B = matrices whose rows contain that row's class labels;
 * #          for each i, rows A [i, ] and B [i, ] define two
 * #          equivalence relations on some of the columns, which
 * #          we want to intersect
 * #   A [i, j] == A [i, k] != 0 if and only if (j ~ k) as defined
 * #          by row A [i, ];
 * #   A [i, j] == 0 means that j is excluded by A [i, ]
 * #   B [i, j] is analogous
 * #   NOTE 1: Either nrow(A) == nrow(B), or exactly one of A or B
 * #   has one row that "applies" to each row of the other matrix.
 * #   NOTE 2: If ncol(A) != ncol(B), we pad extra 0-columns up to
 * #   max (ncol(A), ncol(B)).
 * # OUTPUT:
 * #   Both C and N have the same size as (the max of) A and B.
 * #   C = matrix whose rows contain class labels that represent
 * #       the intersection (coarsest common refinement) of the
 * #       corresponding rows of A and B.
 * #   C [i, j] == C [i, k] != 0 if and only if (j ~ k) as defined
 * #       by both A [i, ] and B [j, ]
 * #   C [i, j] == 0 if and only if A [i, j] == 0 or B [i, j] == 0
 * #       Additionally, we guarantee that non-0 labels in C [i, ]
 * #       will be integers from 1 to max (C [i, ]) without gaps.
 * #       For A and B the labels can be arbitrary.
 * #   N = matrix with class-size information for C-cells
 * #   N [i, j] = count of {C [i, k] | C [i, j] == C [i, k] != 0}
 *
 */
public class RowClassMeet extends PackageFunction {

	private static final long serialVersionUID = 1L;
	private Matrix CMat, NMat;
	private MatrixBlock A, B, C, N;
	private int nr, nc;

	@Override
	public int getNumFunctionOutputs() {
		return 2;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if(pos == 0)
			return CMat;
		else if(pos == 1)
			return NMat;
		else
			throw new RuntimeException("RowClassMeet produces only one output");
	}
	
	
	public class ClassLabels {
		public double aVal;
		public double bVal;
		public ClassLabels(double aVal, double bVal) {
			this.aVal = aVal;
			this.bVal = bVal;
		}
	}
	
	public class ClassLabelComparator implements Comparator<ClassLabels> {
		Integer tmp1, tmp2;
		@Override
		public int compare(ClassLabels o1, ClassLabels o2) {
			if(o1.aVal != o2.aVal) {
				tmp1 = (int) o1.aVal;
				tmp2 = (int) o2.aVal;
			}
			else {
				tmp1 = (int) o1.bVal;
				tmp2 = (int) o2.bVal;
			}
			return tmp1.compareTo(tmp2);
		}
	}
	
	double [] getRow(MatrixBlock B, double [] bRow, int i) {
		if(B.getNumRows() == 1) 
			i = 0;
		Arrays.fill(bRow, 0);
		if(B.isInSparseFormat()) {
			Iterator<IJV> iter = B.getSparseBlockIterator(i, i+1);
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				bRow[ijv.getJ()] = ijv.getV();
			}
		}
		else {
			double [] denseBlk = B.getDenseBlock();
			if(denseBlk != null)
				System.arraycopy(denseBlk, i*B.getNumColumns(), bRow, 0, B.getNumColumns());
		}
		return bRow;
	}
	
	@Override
	public void execute() {
		try {
			A = ((Matrix) getFunctionInput(0)).getMatrixObject().acquireRead();
			B = ((Matrix) getFunctionInput(1)).getMatrixObject().acquireRead();
			nr = Math.max(A.getNumRows(), B.getNumRows());
			nc = Math.max(A.getNumColumns(), B.getNumColumns());
			
			double [] bRow = new double[B.getNumColumns()];
			CMat = new Matrix( createOutputFilePathAndName( "TMP" ), nr, nc, ValueType.Double );
			C = new MatrixBlock(nr, nc, false);
			C.allocateDenseBlock();
			NMat = new Matrix( createOutputFilePathAndName( "TMP" ), nr, nc, ValueType.Double );
			N = new MatrixBlock(nr, nc, false);
			N.allocateDenseBlock();
			
			double [] cBlk = C.getDenseBlock();
			double [] nBlk = N.getDenseBlock();
			
			if(B.getNumRows() == 1)
				getRow(B, bRow, 0);
			
			for(int i = 0; i < A.getNumRows(); i++) {
				if(B.getNumRows() != 1)
					getRow(B, bRow, i);
				
				// Create class labels
				TreeMap<ClassLabels, ArrayList<Integer>> classLabelMapping = new TreeMap<>(new ClassLabelComparator());
				if(A.isInSparseFormat()) {
					Iterator<IJV> iter = A.getSparseBlockIterator(i, i+1);
					while(iter.hasNext()) {
						IJV ijv = iter.next();
						int j = ijv.getJ();
						double aVal = ijv.getV();
						if(aVal != 0 && bRow[j] != 0) {
							ClassLabels key = new ClassLabels(aVal, bRow[j]);
							if(!classLabelMapping.containsKey(key))
								classLabelMapping.put(key, new ArrayList<Integer>());
							classLabelMapping.get(key).add(j);
						}
					}
				}
				else {
					double [] denseBlk = A.getDenseBlock();
					if(denseBlk != null) {
						int offset = i*A.getNumColumns();
						for(int j = 0; j < A.getNumColumns(); j++) {
							double aVal = denseBlk[offset + j];
							if(aVal != 0 && bRow[j] != 0) {
								ClassLabels key = new ClassLabels(aVal, bRow[j]);
								if(!classLabelMapping.containsKey(key))
									classLabelMapping.put(key, new ArrayList<Integer>());
								classLabelMapping.get(key).add(j);
							}
						}
					}
				}
				
				
				int labelID = 1;
				for(Entry<ClassLabels, ArrayList<Integer>> entry : classLabelMapping.entrySet()) {
					double nVal = entry.getValue().size();
					for(Integer j : entry.getValue()) {
						nBlk[i*nc + j] = nVal;
						cBlk[i*nc + j] = labelID;
					}
					labelID++;
				}
			}
			
			((Matrix) getFunctionInput(0)).getMatrixObject().release();
			((Matrix) getFunctionInput(1)).getMatrixObject().release();
		} catch (CacheException e) {
			throw new RuntimeException("Error while executing RowClassMeet", e);
		} 
		
		try {
			C.recomputeNonZeros();
			C.examSparsity();
			CMat.setMatrixDoubleArray(C, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			N.recomputeNonZeros();
			N.examSparsity();
			NMat.setMatrixDoubleArray(N, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("Error while executing RowClassMeet", e);
		} catch (IOException e) {
			throw new RuntimeException("Error while executing RowClassMeet", e);
		}
	}
	
	
}
