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

package org.apache.sysds.performance.micro;

import org.apache.sysds.runtime.data.SparseBlock.Type;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.IntegerArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;

public class FrameCompressedTransform {

	static int specCols = 5;
	static String spec = "{ids:true,dummycode:[1,2,3,4,5]}";

	public static void main(String[] args) {

		// scaleRows();
		// scaleDistinct();
		scaleCols();
	}

	public static void scaleRows() {

		System.out.println("Rows,Comp,MCSR,CSR,COO,Dense");
		for(int i = 1; i < 300; i += 1) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(i, 1000)));
		}

		for(int i = 300; i < 16000; i += 100) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(i, 1000)));
		}

		for(int i = 16000; i < 160000; i += 1000) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(i, 1000)));
		}
	}

	public static void scaleDistinct() {

		System.out.println("Distinct,Comp,MCSR,CSR,COO,Dense");
		for(int i = 1; i < 10; i += 1) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(100000, i)));
		}

		for(int i = 10; i < 100; i += 10) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(100000, i)));
		}

		for(int i = 100; i < 100000; i += 100) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(100000, i)));
		}

	}

	public static void scaleCols() {

		System.out.println("Cols,Comp,MCSR,CSR,COO,Dense");
		for(int i = 1; i < 10; i += 1) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(100000, i, 1000)));
		}

		for(int i = 10; i < 100; i += 10) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(100000, i, 1000)));
		}

		for(int i = 100; i < 1000; i += 100) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(100000, i, 1000)));
		}

		for(int i = 1000; i < 10000; i += 1000) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(100000, i, 1000)));
		}

		for(int i = 10000; i < 20000; i += 10000) {
			System.out.print(i + ",");
			System.out.println(getSize(genFrameBlock(100000, i, 1000)));
		}

	}

	private static String getSize(FrameBlock e) {
		if(specCols != e.getNumColumns())
			createSpec(e.getNumColumns());
		MultiColumnEncoder encoderCompressed = //
			EncoderFactory.createEncoder(spec, e.getColumnNames(), e.getNumColumns(), null);
		MatrixBlock outCompressed = encoderCompressed.encode(e, 16, true);

		long compSize = outCompressed.getInMemorySize();

		long denseSize = outCompressed.estimateSizeDenseInMemory();
		long csr = outCompressed.estimateSizeSparseInMemory(Type.CSR);
		long mcsr = outCompressed.estimateSizeSparseInMemory(Type.MCSR);
		long coo = outCompressed.estimateSizeSparseInMemory(Type.COO);

		// MatrixBlock uc = CompressedMatrixBlock.getUncompressed(outCompressed);
		// uc.denseToSparse(true);

		// SparseBlock sb = uc.getSparseBlock();
		// if(sb == null) {
		// System.out.println(uc);
		// System.exit(-1);
		// }

		// long csr = new MatrixBlock(uc.getNumRows(), uc.getNumColumns(), uc.getNonZeros(), new SparseBlockCSR(sb))
		// .getInMemorySize();
		// long mcsr = new MatrixBlock(uc.getNumRows(), uc.getNumColumns(), uc.getNonZeros(), new SparseBlockMCSR(sb))
		// .getInMemorySize();
		// long coo = new MatrixBlock(uc.getNumRows(), uc.getNumColumns(), uc.getNonZeros(), new SparseBlockCOO(sb))
		// .getInMemorySize();

		// long denseSize = uc.estimateSizeDenseInMemory();

		return compSize + "," + mcsr + "," + csr + "," + coo + "," + denseSize;
	}

	private static void createSpec(int nCol) {
		StringBuilder sb = new StringBuilder();
		sb.append("{ids:true,dummycode:[1");
		for(int i = 1; i < nCol; i++) {
			sb.append(",");
			sb.append(i + 1);
		}
		sb.append("]}");
		spec = sb.toString();
		specCols = nCol;
	}

	private static FrameBlock genFrameBlock(int nRow, int nDistinct) {
		IntegerArray a = new IntegerArray(new int[nRow]);
		for(int i = 0; i < nRow; i++)
			a.set(i, i % nDistinct);
		return new FrameBlock(new Array<?>[] {a, a, a, a, a});
	}

	private static FrameBlock genFrameBlock(int nRow, int cols, int nDistinct) {
		IntegerArray a = new IntegerArray(new int[nRow]);
		for(int i = 0; i < nRow; i++)
			a.set(i, i % nDistinct);
		Array<?>[] r = new Array<?>[cols];
		for(int i = 0; i < cols; i++)
			r[i] = a;

		return new FrameBlock(r);
	}
}
