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

package org.apache.sysds.test.component.matrix;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class TransposeInplaceTest {
	protected static final Log LOG = LogFactory.getLog(TransposeInplaceTest.class.getName());

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		
		tests.add(genIncMatrix(4, 8));
		tests.add(genIncMatrix(8, 4));
		tests.add(genIncMatrix(4, 12));
		tests.add(genIncMatrix(4, 16));
		tests.add(genIncMatrix(12, 4));
		tests.add(genIncMatrix(12, 56));
		// tests.add(genIncMatrix(3, 6));
		// tests.add(genIncMatrix(2, 5));
		// tests.add(genIncMatrix(2, 10));
		// tests.add(genIncMatrix(4, 6));
		// tests.add(genIncMatrix(13, 51));
		// tests.add(genIncMatrix(30000, 290));
		
		// tests.add(genIncMatrix(80, 40));
		// tests.add(genIncMatrix(40, 20));
		// tests.add(genIncMatrix(30000, 30));
		// tests.add(genIncMatrix(30000, 300));
		// tests.add(genIncMatrix(100000, 5000));
		// tests.add(genIncMatrix(300, 28));
		// tests.add(genIncMatrix(30, 300000));
		

		return tests;
	}

	@Parameterized.Parameter
	public MatrixBlock in;
	@Parameterized.Parameter(1)
	public int[][] org;

	private static ReorgOperator opP = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), 16);
	//private static ReorgOperator op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), 1);

	private int rep = 2;
	@Test
	public void testEstimation() {
		// for (int i = 0; i < rep; i++){

		//	 LibMatrixReorg.reorgInPlace(in, op);
		//	 Assert.assertEquals(in.quickGetValue(i%2, 1- i%2), org[1][0], 0.00001);
		// }
		// for (int i = 0; i < rep; i++){
		//	 in = LibMatrixReorg.reorg(in,new MatrixBlock(in.getNumColumns(), in.getNumRows(), false), op);
		// }
		try{
			
			Timing time = new Timing(true);
			// for (int i = 0; i < rep; i++){
			//	 in = LibMatrixReorg.reorg(in,new MatrixBlock(in.getNumColumns(), in.getNumRows(), false), opP);
			// }
	
			// System.out.println("duplicate avg: "+(time.stop() / rep)+" ms.");
			// time = new Timing(true);
			for (int i = 0; i < rep; i++){
				LibMatrixReorg.reorgInPlace(in, opP);
				// LOG.error(in);
				// LOG.error(in.slice(0, 3, 0, 3, null));
				Assert.assertEquals(org[1][0],in.quickGetValue(i%2, 1- i%2), 0.00001);
			}
			
			System.out.println("in place avg: "+(time.stop() / rep)+" ms.");
	
		} catch(Exception e){
			e.printStackTrace();
		}
		// very small test only verifying one cell...
	}

	

	private static MatrixBlock gen(int[][] v) {
		return DataConverter.convertToMatrixBlock(v);
	}

	private static Object[] genIncMatrix(int rows, int cols ){
		int[][] ret = new int[rows][cols];
		int x = 0;
		for (int i = 0; i< rows; i ++){
			for (int j = 0; j < cols; j++){
				ret[i][j] = x++;
			}
		}
		return new Object[]{gen(ret), ret};
	}
}
