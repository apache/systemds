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
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.Test;
import org.tugraz.sysds.api.mlcontext.MLResults;
import org.tugraz.sysds.api.mlcontext.MatrixFormat;
import org.tugraz.sysds.api.mlcontext.MatrixMetadata;
import org.tugraz.sysds.api.mlcontext.Script;
import org.tugraz.sysds.api.mlcontext.MLContext.ExplainLevel;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.TestUtils;


public class MLContextParforDatasetTest extends MLContextTestBase
{
	private final static int rows = 100;
	private final static int cols = 1600;
	private final static double sparsity = 0.7;

	@Test
	public void testParforDatasetVector() {
		runMLContextParforDatasetTest(true, false, false);
	}
	
	@Test
	public void testParforDatasetRow() {
		runMLContextParforDatasetTest(false, false, false);
	}
	
	@Test
	public void testParforDatasetVectorUnkownDims() {
		runMLContextParforDatasetTest(true, true, false);
	}
	
	@Test
	public void testParforDatasetRowUnknownDims() {
		runMLContextParforDatasetTest(false, true, false);
	}
	
	@Test
	public void testParforDatasetVectorMulti() {
		runMLContextParforDatasetTest(true, false, true);
	}
	
	@Test
	public void testParforDatasetRowMulti() {
		runMLContextParforDatasetTest(false, false, true);
	}
	
	@Test
	public void testParforDatasetVectorUnkownDimsMulti() {
		runMLContextParforDatasetTest(true, true, true);
	}
	
	@Test
	public void testParforDatasetRowUnknownDimsMulti() {
		runMLContextParforDatasetTest(false, true, true);
	}
	
	private void runMLContextParforDatasetTest(boolean vector, boolean unknownDims, boolean multiInputs) 
	{
		//modify memory budget to trigger fused datapartition-execute
		long oldmem = InfrastructureAnalyzer.getLocalMaxMemory();
		InfrastructureAnalyzer.setLocalMaxMemory(1*1024*1024); //1MB
		
		try
		{
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 76543); 
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A); 
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows, cols, blksz, mbA.getNonZeros());
			MatrixCharacteristics mc2 = unknownDims ? new MatrixCharacteristics() : new MatrixCharacteristics(mc1);
			
			//create input dataset
			SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = SparkExecutionContext.toMatrixJavaPairRDD(sc, mbA, blksz);
			Dataset<Row> df = RDDConverterUtils.binaryBlockToDataFrame(sparkSession, in, mc1, vector);
			MatrixMetadata mm = new MatrixMetadata(vector ? MatrixFormat.DF_VECTOR_WITH_INDEX : MatrixFormat.DF_DOUBLES_WITH_INDEX);
			mm.setMatrixCharacteristics(mc2);
			
			String s1 = "v = matrix(0, rows=nrow(X), cols=1)"
					+ "parfor(i in 1:nrow(X), log=DEBUG) {"
					+ "   v[i, ] = sum(X[i, ]);"
					+ "}"
					+ "r = sum(v);";
			String s2 = "v = matrix(0, rows=nrow(X), cols=1)"
					+"Y = X;"
					+ "parfor(i in 1:nrow(X), log=DEBUG) {"
					+ "   v[i, ] = sum(X[i, ]+Y[i, ]);"
					+ "}"
					+ "r = sum(v);";
			String s = multiInputs ? s2 : s1;
			
			ml.setExplain(true);
			ml.setExplainLevel(ExplainLevel.RUNTIME);
			ml.setStatistics(true);
			
			Script script = dml(s).in("X", df, mm).out("r");
			MLResults results = ml.execute(script);
			
			//compare aggregation results
			double sum1 = results.getDouble("r");
			double sum2 = mbA.sum() * (multiInputs ? 2 : 1);
			
			TestUtils.compareScalars(sum2, sum1, 0.000001);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);
		}
	}
}
