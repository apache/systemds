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
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.codegen.SpoofOperator.SideInput;
import org.apache.sysml.runtime.compress.utils.IntArrayList;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.UtilFunctions;
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
	
	@Override
	public int getNumFunctionOutputs() {
		return 2;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		switch( pos ) {
			case 0: return CMat;
			case 1: return NMat;
			default:
				throw new RuntimeException("RowClassMeet produces only one output");
		}
	}
	
	@Override
	public void execute() {
		try 
		{
			MatrixBlock A = ((Matrix) getFunctionInput(0)).getMatrixObject().acquireRead();
			MatrixBlock B = ((Matrix) getFunctionInput(1)).getMatrixObject().acquireRead();
			int nr = Math.max(A.getNumRows(), B.getNumRows());
			int nc = Math.max(A.getNumColumns(), B.getNumColumns());
			MatrixBlock C = new MatrixBlock(nr, nc, false).allocateBlock();
			MatrixBlock N = new MatrixBlock(nr, nc, false).allocateBlock();
			double[] dC = C.getDenseBlock();
			double[] dN = N.getDenseBlock();
			//wrap both A and B into side inputs for efficient sparse access
			SideInput sB = CodegenUtils.createSideInput(B);
			boolean mv = (B.getNumRows() == 1);
			int numCols = Math.min(A.getNumColumns(), B.getNumColumns());
			
			HashMap<ClassLabel, IntArrayList> classLabelMapping = new HashMap<>();
			
			for(int i=0, ai=0; i < A.getNumRows(); i++, ai+=A.getNumColumns()) {
				classLabelMapping.clear(); sB.reset();
				if( A.isInSparseFormat() ) {
					if(A.getSparseBlock()==null || A.getSparseBlock().isEmpty(i))
						continue;
					int alen = A.getSparseBlock().size(i);
					int apos = A.getSparseBlock().pos(i);
					int[] aix = A.getSparseBlock().indexes(i);
					double[] avals = A.getSparseBlock().values(i);
					for(int k=apos; k<apos+alen; k++) {
						if( aix[k] >= numCols ) break;
						int bval = (int)sB.getValue(mv?0:i, aix[k]);
						if( bval != 0 ) {
							ClassLabel key = new ClassLabel((int)avals[k], bval);
							if(!classLabelMapping.containsKey(key))
								classLabelMapping.put(key, new IntArrayList());
							classLabelMapping.get(key).appendValue(aix[k]);
						}
					}
				}
				else {
					double [] denseBlk = A.getDenseBlock();
					if(denseBlk == null) break;
					for(int j = 0; j < numCols; j++) {
						int aVal = (int) denseBlk[ai+j];
						int bVal = (int) sB.getValue(mv?0:i, j);
						if(aVal != 0 && bVal != 0) {
							ClassLabel key = new ClassLabel(aVal, bVal);
							if(!classLabelMapping.containsKey(key))
								classLabelMapping.put(key, new IntArrayList());
							classLabelMapping.get(key).appendValue(j);
						}
					}
				}
				
				int labelID = 1;
				for(Entry<ClassLabel, IntArrayList> entry : classLabelMapping.entrySet()) {
					int nVal = entry.getValue().size();
					int[] list = entry.getValue().extractValues();
					for(int k=0, off=i*nc; k<nVal; k++) {
						dN[off+list[k]] = nVal;
						dC[off+list[k]] = labelID;
					}
					labelID++;
				}
			}
			
			((Matrix) getFunctionInput(0)).getMatrixObject().release();
			((Matrix) getFunctionInput(1)).getMatrixObject().release();
		
			//prepare outputs 
			C.recomputeNonZeros(); C.examSparsity();
			CMat = new Matrix( createOutputFilePathAndName( "TMP" ), nr, nc, ValueType.Double );
			CMat.setMatrixDoubleArray(C, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			N.recomputeNonZeros(); N.examSparsity();
			NMat = new Matrix( createOutputFilePathAndName( "TMP" ), nr, nc, ValueType.Double );
			NMat.setMatrixDoubleArray(N, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		} 
		catch (DMLRuntimeException | IOException e) {
			throw new RuntimeException("Error while executing RowClassMeet", e);
		}
	}
	
	private static class ClassLabel {
		public int aVal;
		public int bVal;
		public ClassLabel(int aVal, int bVal) {
			this.aVal = aVal;
			this.bVal = bVal;
		}
		@Override
		public int hashCode() {
			return UtilFunctions.intHashCode(aVal, bVal);
		}
		@Override
		public boolean equals(Object o) {
			if( !(o instanceof ClassLabel) )
				return false;
			ClassLabel that = (ClassLabel) o;
			return aVal == that.aVal && bVal == that.bVal;
		}
	}
}
