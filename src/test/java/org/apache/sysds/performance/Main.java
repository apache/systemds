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
import org.apache.sysds.performance.generators.ConstMatrix;
import org.apache.sysds.performance.generators.FrameFile;
import org.apache.sysds.performance.generators.GenMatrices;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.performance.generators.MatrixFile;
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

		Serialize s = new Serialize(n, new ConstMatrix(rows, cols, unique, sparsity), k);

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
		// run13A(n, FrameTransformFile.create(p, s), k, id);
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
		// new TransformPerf(n, k, g, spec).run();
	}

	private static void run16(String[] args) {
		int len = Integer.parseInt(args[1]);
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(len, len, 0, 100, 0.01, len +1));
		System.out.println(mb);
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
