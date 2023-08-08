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
import org.apache.sysds.performance.generators.GenMatrices;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class Main {

	private static void exec(int prog, String[] args) throws InterruptedException, Exception {
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
			default:
				break;
		}
	}

	private static void run9(String[] args) throws InterruptedException, Exception {
		int rows = Integer.parseInt(args[1]);
		int cols = Integer.parseInt(args[2]);
		int unique = Integer.parseInt(args[3]);
		double sparsity = Double.parseDouble(args[4]);
		int k = Integer.parseInt(args[5]);
		int n = Integer.parseInt(args[6]);
		new IOBandwidth(n, new ConstMatrix(rows, cols, unique, sparsity), k).run();
	}

	private static void run10(String[] args) throws InterruptedException, Exception {
		int rows = Integer.parseInt(args[1]);
		int cols = Integer.parseInt(args[2]);
		int unique = Integer.parseInt(args[3]);
		double sparsity = Double.parseDouble(args[4]);
		int k = Integer.parseInt(args[5]);
		int n = Integer.parseInt(args[6]);
		new IOBandwidth(n, new ConstMatrix(rows, cols, unique, sparsity), k).runVector();
	}

	private static void run11(String[] args, int id) throws InterruptedException, Exception {
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

	public static void main(String[] args) {
		try {
			exec(Integer.parseInt(args[0]), args);
		}
		catch(Exception e) {
			e.printStackTrace();
		}finally{
			CommonThreadPool.get().shutdown();
		}
	}
}
