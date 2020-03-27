/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.test.functions.mlcontext;

import static org.tugraz.sysds.api.mlcontext.ScriptFactory.dml;

import org.apache.spark.api.java.JavaPairRDD;
import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.mlcontext.MLResults;
import org.tugraz.sysds.api.mlcontext.Matrix;
import org.tugraz.sysds.api.mlcontext.MatrixMetadata;
import org.tugraz.sysds.api.mlcontext.Script;
import org.tugraz.sysds.api.mlcontext.MLContext.ExplainLevel;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;

public class MLContextOutputBlocksizeTest extends MLContextTestBase
{
	private final static int rows = 100;
	private final static int cols = 63;
	private final static double sparsity = 0.7;

	@Test
	public void testOutputBlocksizeTextcell() {
		runMLContextOutputBlocksizeTest("text");
	}

	@Test
	public void testOutputBlocksizeCSV() {
		runMLContextOutputBlocksizeTest("csv");
	}

	@Test
	public void testOutputBlocksizeMM() {
		runMLContextOutputBlocksizeTest("mm");
	}

	@Test
	public void testOutputBlocksizeBinary() {
		runMLContextOutputBlocksizeTest("binary");
	}


	private void runMLContextOutputBlocksizeTest(String format)
	{
		try
		{
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 76543);
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, blksz, mbA.getNonZeros());

			//create input dataset
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = SparkExecutionContext.toMatrixJavaPairRDD(sc, mbA, blksz);
			Matrix m = new Matrix(in, new MatrixMetadata(mc));

			ml.setExplain(true);
			ml.setExplainLevel(ExplainLevel.HOPS);

			//execute script
			String s ="if( sum(X) > 0 )"
					+ "   X = X/2;"
					+ "R = X;"
					+ "write(R, \"/tmp\", format=\""+format+"\");";
			Script script = dml(s).in("X", m).out("R");
			MLResults results = ml.execute(script);

			//compare output matrix characteristics
			MatrixCharacteristics mcOut = results.getMatrix("R")
				.getMatrixMetadata().asMatrixCharacteristics();
			Assert.assertEquals(blksz, mcOut.getBlocksize());
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
