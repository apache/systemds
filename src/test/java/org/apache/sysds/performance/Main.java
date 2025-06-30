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

package org.apache.sysds.performance;

import org.apache.sysds.performance.compression.IOBandwidth;
import org.apache.sysds.performance.compression.SchemaTest;
import org.apache.sysds.performance.compression.Serialize;
import org.apache.sysds.performance.compression.StreamCompress;
import org.apache.sysds.performance.compression.TransformPerf;
import org.apache.sysds.performance.frame.Transform;
import org.apache.sysds.performance.generators.ConstMatrix;
import org.apache.sysds.performance.generators.FrameFile;
import org.apache.sysds.performance.generators.FrameTransformFile;
import org.apache.sysds.performance.generators.GenMatrices;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.performance.generators.MatrixFile;
import org.apache.sysds.performance.matrix.MatrixAppend;
import org.apache.sysds.performance.matrix.MatrixBinaryCellPerf;
import org.apache.sysds.performance.matrix.MatrixMultiplicationPerf;
import org.apache.sysds.performance.matrix.MMSparsityPerformance;
import org.apache.sysds.performance.matrix.MatrixReplacePerf;
import org.apache.sysds.performance.matrix.MatrixStorage;
import org.apache.sysds.performance.matrix.ReshapePerf;
import org.apache.sysds.performance.matrix.SparseAppend;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.TestUtils;

public class Main {

	private static void exec(int prog, String[] args) throws Exception {
		switch(prog) {
			case 1:
				new StreamCompress(100, new GenMatrices(10000, 100, 32, 1.0)).run();
				break;
			case 2:
				new SchemaTest(100, new GenMatrices(10000, 1000, 32, 1.0)).run();
				break;
			case 3:
				new SchemaTest(100, new GenMatrices(1000, 1, 32, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 10, 32, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 100, 32, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 32, 1.0)).run();
				break;
			case 4:
				new SchemaTest(100, new GenMatrices(1000, 1000, 1, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 2, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 4, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 8, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 16, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 32, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 64, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 128, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 256, 1.0)).run();
				new SchemaTest(100, new GenMatrices(1000, 1000, 512, 1.0)).run();
				break;
			case 5:
				new SchemaTest(100, new ConstMatrix(1000, 100, 32, 1.0)).runCom();
				break;
			case 6:
				new SchemaTest(100, new GenMatrices(1000, 1000, 32, 0.3)).run();
				break;
			case 7:
				new SchemaTest(100, new ConstMatrix(1000, 1000, 32, 0.3)).runCom();
				break;
			case 8:
				new IOBandwidth(100, new ConstMatrix(1000, 1000, 32, 1.0)).run();
				break;
			case 9:
				run9(args);
				break;
			case 10:
				run10(args);
				break;
			case 11:
				run11(args, -1);
				break;
			case 12:
				run11(args, Integer.parseInt(args[7]));
				break;
			case 13:
				run13(args);
				break;
			case 14:
				run14(args);
				break;

			case 15:
				run15(args);
				break;
			case 16:
				run16(args);
				break;
			case 17: 
				run17(args);
				break;
			case 1000:
				run1000(args);
				break;
			case 1001:
				run1001(args);
				break;
			case 1002:
				run1002(args);
				break;
			case 1003:
				run1003(args);
				break;
			case 1004:
				run1004(args);
				break;
			case 1005:
				ReshapePerf.main(args);
				break;
			case 1006:
				MatrixBinaryCellPerf.main(args);
				break;
			case 1007:
				Transform.main(args);
				break;
			case 1008:
				MatrixAppend.main(args);
				break;
			case 1009:
				MatrixMultiplicationPerf.main(args);
				break;
			default:
				break;
		}
	}

	private static void run9(String[] args) throws Exception {
		int rows = Integer.parseInt(args[1]);
		int cols = Integer.parseInt(args[2]);
		int unique = Integer.parseInt(args[3]);
		double sparsity = Double.parseDouble(args[4]);
		int k = Integer.parseInt(args[5]);
		int n = Integer.parseInt(args[6]);
		new IOBandwidth(n, new ConstMatrix(rows, cols, unique, sparsity), k).run();
	}

	private static void run10(String[] args) throws Exception {
		int rows = Integer.parseInt(args[1]);
		int cols = Integer.parseInt(args[2]);
		int unique = Integer.parseInt(args[3]);
		double sparsity = Double.parseDouble(args[4]);
		int k = Integer.parseInt(args[5]);
		int n = Integer.parseInt(args[6]);
		new IOBandwidth(n, new ConstMatrix(rows, cols, unique, sparsity), k).runVector();
	}

	private static void run11(String[] args, int id) throws Exception {
		int rows = Integer.parseInt(args[1]);
		int cols = Integer.parseInt(args[2]);
		int unique = Integer.parseInt(args[3]);
		double sparsity = Double.parseDouble(args[4]);
		int k = Integer.parseInt(args[5]);
		int n = Integer.parseInt(args[6]);
		//args[7] is id
		Serialize s = (args.length == 9) ? //
			new Serialize(n, new ConstMatrix(rows, cols, unique, sparsity), k) : //
			new Serialize(n, new ConstMatrix(rows, cols, unique, sparsity), k, args[7], args[8]);

		if(id == -1)
			s.run();
		else
			s.run(id);
	}

	private static void run13(String[] args) throws Exception {
		int k = Integer.parseInt(args[1]);
		int n = Integer.parseInt(args[2]);
		String p = args[3];
		int id = Integer.parseInt(args[4]);
		run13A(n, MatrixFile.create(p), k, id);
	}

	private static void run14(String[] args) throws Exception {
		int k = Integer.parseInt(args[1]);
		int n = Integer.parseInt(args[2]);
		String p = args[3]; // input frame
		String s = args[4]; // spec
		int id = Integer.parseInt(args[5]);
		run13A(n, FrameTransformFile.create(p, s), k, id);
	}

	private static void run13A(int n, IGenerate<MatrixBlock> g, int k, int id) throws Exception {

		Serialize s = new Serialize(n, g, k);

		if(id == -1)
			s.run();
		else
			s.run(id);
	}

	private static void run15(String[] args) throws Exception {
		int k = Integer.parseInt(args[1]);
		int n = Integer.parseInt(args[2]);
		IGenerate<FrameBlock> g = FrameFile.create(args[3]);
		String spec = args[4];
		new TransformPerf(n, k, g, spec).run();

	}

	private static void run16(String[] args) {
		int len = Integer.parseInt(args[1]);
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(len, len, 0, 100, 0.01, len + 1));
		System.out.println(mb);
	}

	private static void run17(String[] args) throws Exception {
		int rows = Integer.parseInt(args[1]);
		int cols = Integer.parseInt(args[2]);
		double spar = Double.parseDouble(args[3]);
		int k = Integer.parseInt(args[4]);
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(rows, cols, 0, 100, spar, rows + 1));
		IGenerate<MatrixBlock> g = new ConstMatrix(mb);
		new MatrixReplacePerf(100, g, k).run();
	}

	private static void run1000(String[] args) {
		MMSparsityPerformance perf;
		if (args.length < 3) {
			perf = new MMSparsityPerformance();
		} else {
			// ... <rl> <cl> [resolution] [maxSparsity] [resolution] [warmupRuns] [repetitions]
			int rl = Integer.parseInt(args[1]);
			int cl = Integer.parseInt(args[2]);
			int resolution = 18;
			float maxSparsity = .4f;
			int warmupRuns = 30;
			int repetitions = 100;

			if (args.length > 3)
				resolution = Integer.parseInt(args[3]);
			if (args.length > 4)
				maxSparsity = Float.parseFloat(args[4]);
			if (args.length > 5)
				warmupRuns = Integer.parseInt(args[5]);
			if (args.length > 6)
				repetitions = Integer.parseInt(args[6]);

			perf = new MMSparsityPerformance(rl, cl, warmupRuns, repetitions, resolution, maxSparsity, 2f);
		}

		perf.testSparseFormat(null, null);
		perf.testSparseFormat(SparseBlock.Type.MCSR, SparseBlock.Type.MCSR);
		perf.testSparseFormat(SparseBlock.Type.CSR, SparseBlock.Type.CSR);
		perf.testSparseFormat(SparseBlock.Type.COO, SparseBlock.Type.COO);
		perf.testSparseFormat(SparseBlock.Type.DCSR, SparseBlock.Type.DCSR);
	}

	private static void run1001(String[] args) {
		// ... [rl] [cl] [repetitions] [resolution] [maxSparsity]
		MatrixStorage ms;
		int rl = 1024;
		int cl = 1024;
		int repetitions = 10;
		int resolution = 18;
		float maxSparsity = 0.4f;

		if (args.length > 1)
			rl = Integer.parseInt(args[1]);
		if (args.length > 2)
			cl = Integer.parseInt(args[2]);
		if (args.length > 3)
			repetitions = Integer.parseInt(args[3]);
		if (args.length > 4)
			resolution = Integer.parseInt(args[4]);
		if (args.length > 5)
			maxSparsity = Float.parseFloat(args[5]);

		ms = new MatrixStorage(resolution, 2f, maxSparsity);

		ms.testSparseFormat(null, rl, cl, repetitions);
		ms.testSparseFormat(SparseBlock.Type.MCSR, rl, cl, repetitions);
		ms.testSparseFormat(SparseBlock.Type.CSR, rl, cl, repetitions);
		ms.testSparseFormat(SparseBlock.Type.COO, rl, cl, repetitions);
		ms.testSparseFormat(SparseBlock.Type.DCSR, rl, cl, repetitions);
	}

	private static void run1002(String[] args) {
		// ... [sparsity] [rl] [minCl] [maxCl] [resolution] [repetitions]
		MatrixStorage ms = new MatrixStorage();
		float sparsity = 0.1f;
		int rl = 1024;
		int minCl = 50;
		int maxCl = 2048;
		int resolution = 21;
		int repetitions = 10;

		if (args.length > 1)
			sparsity = Float.parseFloat(args[1]);
		if (args.length > 2)
			rl = Integer.parseInt(args[2]);
		if (args.length > 3)
			minCl = Integer.parseInt(args[3]);
		if (args.length > 4)
			maxCl = Integer.parseInt(args[4]);
		if (args.length > 5)
			resolution = Integer.parseInt(args[5]);
		if (args.length > 6)
			repetitions = Integer.parseInt(args[6]);

		ms.testChangingDims(null, sparsity, rl, minCl, maxCl, resolution, repetitions);
		ms.testChangingDims(SparseBlock.Type.MCSR, sparsity, rl, minCl, maxCl, resolution, repetitions);
		ms.testChangingDims(SparseBlock.Type.CSR, sparsity, rl, minCl, maxCl, resolution, repetitions);
		ms.testChangingDims(SparseBlock.Type.COO, sparsity, rl, minCl, maxCl, resolution, repetitions);
		ms.testChangingDims(SparseBlock.Type.DCSR, sparsity, rl, minCl, maxCl, resolution, repetitions);
	}

	private static void run1003(String[] args) {
		// ... [sparsity] [resolution] [repetitions] [maxRowColRatio] [numMatrixEntries]
		MatrixStorage ms = new MatrixStorage();

		float sparsity = 0.1f;
		int resolution = 21;
		int repetitions = 10;
		float maxRowColRatio = 10f;
		int numEntries = 1024 * 1024;

		if (args.length > 1)
			sparsity = Float.parseFloat(args[1]);
		if (args.length > 2)
			resolution = Integer.parseInt(args[2]);
		if (args.length > 3)
			repetitions = Integer.parseInt(args[3]);
		if (args.length > 4)
			maxRowColRatio = Float.parseFloat(args[4]);
		if (args.length > 5)
			numEntries = Integer.parseInt(args[5]);

		ms.testBalancedDims(null, sparsity, numEntries, resolution, maxRowColRatio, repetitions);
		ms.testBalancedDims(SparseBlock.Type.MCSR, sparsity, numEntries, resolution, maxRowColRatio, repetitions);
		ms.testBalancedDims(SparseBlock.Type.CSR, sparsity, numEntries, resolution, maxRowColRatio, repetitions);
		ms.testBalancedDims(SparseBlock.Type.COO, sparsity, numEntries, resolution, maxRowColRatio, repetitions);
		ms.testBalancedDims(SparseBlock.Type.DCSR, sparsity, numEntries, resolution, maxRowColRatio, repetitions);
	}

	private static void run1004(String[] args){
		new SparseAppend(args);
	}

	public static void main(String[] args) {
		try {
			exec(Integer.parseInt(args[0]), args);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		finally {
			CommonThreadPool.get().shutdown();
		}
	}
}
