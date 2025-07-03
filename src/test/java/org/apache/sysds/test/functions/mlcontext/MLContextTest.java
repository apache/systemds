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

package org.apache.sysds.test.functions.mlcontext;

import static org.apache.sysds.api.mlcontext.ScriptFactory.dml;
import static org.apache.sysds.api.mlcontext.ScriptFactory.dmlFromFile;
import static org.apache.sysds.api.mlcontext.ScriptFactory.dmlFromInputStream;
import static org.apache.sysds.api.mlcontext.ScriptFactory.dmlFromLocalFile;
import static org.apache.sysds.api.mlcontext.ScriptFactory.dmlFromUrl;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysds.api.mlcontext.MLContextConversionUtil;
import org.apache.sysds.api.mlcontext.MLContextException;
import org.apache.sysds.api.mlcontext.MLContextUtil;
import org.apache.sysds.api.mlcontext.MLResults;
import org.apache.sysds.api.mlcontext.Matrix;
import org.apache.sysds.api.mlcontext.MatrixFormat;
import org.apache.sysds.api.mlcontext.MatrixMetadata;
import org.apache.sysds.api.mlcontext.Script;
import org.apache.sysds.api.mlcontext.ScriptExecutor;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import scala.Tuple1;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;
import scala.collection.Iterator;
import scala.collection.JavaConversions;
import scala.collection.Seq;

public class MLContextTest extends MLContextTestBase {

	private static final Log LOG = LogFactory.getLog(MLContextTest.class.getName());

	@Test
	public void testBuiltinConstantsTest() {
		LOG.debug("MLContextTest - basic builtin constants test");
		Script script = dmlFromFile(baseDirectory + File.separator + "builtin-constants-test.dml");
		executeAndCaptureStdOut(script);
		Assert.assertTrue(Statistics.getNoOfExecutedSPInst() == 0);
	}

	@Test
	public void testBasicExecuteEvalTest() {
		LOG.debug("MLContextTest - basic eval test");
		Script script = dmlFromFile(baseDirectory + File.separator + "eval-test.dml");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("10"));
	}

	@Test
	public void testRewriteExecuteEvalTest() {
		LOG.debug("MLContextTest - eval rewrite test");
		Script script = dmlFromFile(baseDirectory + File.separator + "eval2-test.dml");
		executeAndCaptureStdOut(script);
		Assert.assertTrue(Statistics.getNoOfExecutedSPInst() == 0);
	}

	@Test
	public void testExecuteEvalBuiltinTest() {
		runEvalTest("eval3-builtin-test.dml", "TRUE");
	}

	@Test
	public void testExecuteEvalNestedBuiltinTest() {
		runEvalTest("eval4-nested_builtin-test.dml", "TRUE");
	}

	@Test
	public void testExecuteEvalBooleanArgument_01(){
		runEvalTest("eval5-bool-not-true.dml", "FALSE");
	}

	@Test
	public void testExecuteEvalBooleanArgument_02(){
		runEvalTest("eval5-bool-not-false.dml", "TRUE");
	}

	@Test
	public void testExecuteEvalBooleanArgument_03(){
		runEvalTest("eval5-bool-allFalse-list.dml", "FALSE");
	}

	@Test
	public void testExecuteEvalBooleanArgument_04(){
		runEvalTest("eval5-bool-allFalse-list-2.dml", "TRUE");
	}

	@Test
	public void testExecuteEvalGridSearchNoDefault(){
		// grid search where all parameters are defined in parameter ranges
		runEvalTest("eval6-gridSearch-1.dml", "You Found Me! TRUE");
	}

	@Test
	public void testExecuteEvalGridSearchWithDefault(){
		// grid search where all but one boolean parameter is defined in parameter ranges
		runEvalTest("eval6-gridSearch-2.dml", "You Found Me Also! TRUE");
	}

	@Test
	public void testExecuteEvalGridSearchWithTwoDefault(){
		// grid search where two boolean parameters are not defined in parameter ranges.
		runEvalTest("eval6-gridSearch-3.dml", "Find Me! TRUE");
	}

	private void runEvalTest(String name, String outputContains){
		LOG.debug("MLContextTest - eval builtin test " + name);
		final Script script = dmlFromFile(baseDirectory + File.separator + name);
		// ml.setExplain(true);
		final String out = executeAndCaptureStdOut(script).getRight();
		// LOG.error(out);
		assertTrue(out, out.contains(outputContains));
		// ml.setExplain(false);
	}

	@Test
	public void testCreateDMLScriptBasedOnStringAndExecute() {
		LOG.debug("MLContextTest - create DML script based on string and execute");
		String testString = "Create DML script based on string and execute";
		Script script = dml("print('" + testString + "');");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains(testString));
	}

	@Test
	public void testCreateDMLScriptBasedOnFileAndExecute() {
		LOG.debug("MLContextTest - create DML script based on file and execute");
		Script script = dmlFromFile(baseDirectory + File.separator + "hello-world.dml");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("hello world"));
	}

	@Test
	public void testCreateDMLScriptBasedOnInputStreamAndExecute() throws IOException {
		LOG.debug("MLContextTest - create DML script based on InputStream and execute");
		File file = new File(baseDirectory + File.separator + "hello-world.dml");
		try(InputStream is = new FileInputStream(file)) {
			Script script = dmlFromInputStream(is);
			String out = executeAndCaptureStdOut( script).getRight();
			assertTrue(out.contains("hello world"));
		}
	}

	@Test
	public void testCreateDMLScriptBasedOnLocalFileAndExecute() {
		LOG.debug("MLContextTest - create DML script based on local file and execute");
		File file = new File(baseDirectory + File.separator + "hello-world.dml");
		Script script = dmlFromLocalFile(file);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("hello world"));
	}

	@Test
	public void testCreateDMLScriptBasedOnURL() throws MalformedURLException {
		LOG.debug("MLContextTest - create DML script based on URL");
		String urlString = "https://raw.githubusercontent.com/apache/systemml/master/src/test/scripts/applications/hits/HITS.dml";
		URL url = new URL(urlString);
		Script script = dmlFromUrl(url);
		String expectedContent = "Licensed to the Apache Software Foundation";
		String s = script.getScriptString();
		assertTrue("Script string doesn't contain expected content: " + expectedContent, s.contains(expectedContent));
	}

	@Test
	public void testCreateDMLScriptBasedOnURLString() {
		LOG.debug("MLContextTest - create DML script based on URL string");
		String urlString = "https://raw.githubusercontent.com/apache/systemml/master/src/test/scripts/applications/hits/HITS.dml";
		Script script = dmlFromUrl(urlString);
		String expectedContent = "Licensed to the Apache Software Foundation";
		String s = script.getScriptString();
		assertTrue("Script string doesn't contain expected content: " + expectedContent, s.contains(expectedContent));
	}

	@Test
	public void testExecuteDMLScript() {
		LOG.debug("MLContextTest - execute DML script");
		String testString = "hello dml world!";
		Script script = new Script("print('" + testString + "');");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains(testString));
	}

	@Test
	public void testInputParametersAddDML() {
		LOG.debug("MLContextTest - input parameters add DML");

		String s = "x = $X; y = $Y; print('x + y = ' + (x + y));";
		Script script = dml(s).in("$X", 3).in("$Y", 4);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("x + y = 7"));
	}

	@Test
	public void testJavaRDDCSVSumDML() {
		LOG.debug("MLContextTest - JavaRDD<String> CSV sum DML");

		List<String> list = new ArrayList<>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		Script script = dml("print('sum: ' + sum(M));").in("M", javaRDD);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	@Test
	public void testJavaRDDIJVSumDML() {
		LOG.debug("MLContextTest - JavaRDD<String> IJV sum DML");

		List<String> list = new ArrayList<>();
		list.add("1 1 5");
		list.add("2 2 5");
		list.add("3 3 5");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.IJV, 3, 3);

		Script script = dml("print('sum: ' + sum(M));").in("M", javaRDD, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 15.0"));
	}

	@Test
	public void testJavaRDDAndInputParameterDML() {
		LOG.debug("MLContextTest - JavaRDD<String> and input parameter DML");

		List<String> list = new ArrayList<>();
		list.add("1,2");
		list.add("3,4");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		String s = "M = M + $X; print('sum: ' + sum(M));";
		Script script = dml(s).in("M", javaRDD).in("$X", 1);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 14.0"));
	}

	@Test
	public void testInputMapDML() {
		LOG.debug("MLContextTest - input map DML");

		List<String> list = new ArrayList<>();
		list.add("10,20");
		list.add("30,40");
		final JavaRDD<String> javaRDD = sc.parallelize(list);

		Map<String, Object> inputs = new HashMap<>() {
			private static final long serialVersionUID = 1L;
			{
				put("$X", 2);
				put("M", javaRDD);
			}
		};

		String s = "M = M + $X; print('sum: ' + sum(M));";
		Script script = dml(s).in(inputs);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 108.0"));
	}

	@Test
	public void testCustomExecutionStepDML() {
		LOG.debug("MLContextTest - custom execution step DML");
		String testString = "custom execution step";
		Script script = new Script("print('" + testString + "');");

		ScriptExecutor scriptExecutor = new ScriptExecutor() {
			@Override
			protected void showExplanation() {
			}
		};
		String out = executeAndCaptureStdOut(script, scriptExecutor).getRight();
		assertTrue(out.contains(testString));
	}

	@Test
	public void testRDDSumCSVDML() {
		LOG.debug("MLContextTest - RDD<String> CSV sum DML");

		List<String> list = new ArrayList<>();
		list.add("1,1,1");
		list.add("2,2,2");
		list.add("3,3,3");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		Script script = dml("print('sum: ' + sum(M));").in("M", rdd);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 18.0"));
	}

	@Test
	public void testRDDSumIJVDML() {
		LOG.debug("MLContextTest - RDD<String> IJV sum DML");

		List<String> list = new ArrayList<>();
		list.add("1 1 1");
		list.add("2 1 2");
		list.add("1 2 3");
		list.add("3 3 4");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.IJV, 3, 3);

		Script script = dml("print('sum: ' + sum(M));").in("M", rdd, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 10.0"));
	}

	@Test
	public void testDataFrameSumDMLDoublesWithNoIDColumn() {
		LOG.debug("MLContextTest - DataFrame sum DML, doubles with no ID column");

		List<String> list = new ArrayList<>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToDoubleArrayRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.DoubleType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.DF_DOUBLES);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 450.0"));
	}

	@Test
	public void testDataFrameSumDMLDoublesWithIDColumn() {
		LOG.debug("MLContextTest - DataFrame sum DML, doubles with ID column");

		List<String> list = new ArrayList<>();
		list.add("1,1,2,3");
		list.add("2,4,5,6");
		list.add("3,7,8,9");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToDoubleArrayRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField(RDDConverterUtils.DF_ID_COLUMN, DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C1", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.DoubleType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.DF_DOUBLES_WITH_INDEX);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	@Test
	public void testDataFrameSumDMLDoublesWithIDColumnSortCheck() {
		LOG.debug("MLContextTest - DataFrame sum DML, doubles with ID column sort check");

		List<String> list = new ArrayList<>();
		list.add("3,7,8,9");
		list.add("1,1,2,3");
		list.add("2,4,5,6");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToDoubleArrayRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField(RDDConverterUtils.DF_ID_COLUMN, DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C1", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.DoubleType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.DF_DOUBLES_WITH_INDEX);

		Script script = dml("print('M[1,1]: ' + as.scalar(M[1,1]));").in("M", dataFrame, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("M[1,1]: 1.0"));
	}

	@Test
	public void testDataFrameSumDMLVectorWithIDColumn() {
		LOG.debug("MLContextTest - DataFrame sum DML, vector with ID column");

		List<Tuple2<Double, Vector>> list = new ArrayList<>();
		list.add(new Tuple2<>(1.0, Vectors.dense(1.0, 2.0, 3.0)));
		list.add(new Tuple2<>(2.0, Vectors.dense(4.0, 5.0, 6.0)));
		list.add(new Tuple2<>(3.0, Vectors.dense(7.0, 8.0, 9.0)));
		JavaRDD<Tuple2<Double, Vector>> javaRddTuple = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddTuple.map(new DoubleVectorRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField(RDDConverterUtils.DF_ID_COLUMN, DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C1", new VectorUDT(), true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.DF_VECTOR_WITH_INDEX);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	@Test
	public void testDataFrameSumDMLMllibVectorWithIDColumn() {
		LOG.debug("MLContextTest - DataFrame sum DML, mllib vector with ID column");

		List<Tuple2<Double, org.apache.spark.mllib.linalg.Vector>> list = new ArrayList<>();
		list.add(new Tuple2<>(1.0, org.apache.spark.mllib.linalg.Vectors.dense(1.0, 2.0, 3.0)));
		list.add(new Tuple2<>(2.0, org.apache.spark.mllib.linalg.Vectors.dense(4.0, 5.0, 6.0)));
		list.add(new Tuple2<>(3.0, org.apache.spark.mllib.linalg.Vectors.dense(7.0, 8.0, 9.0)));
		JavaRDD<Tuple2<Double, org.apache.spark.mllib.linalg.Vector>> javaRddTuple = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddTuple.map(new DoubleMllibVectorRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField(RDDConverterUtils.DF_ID_COLUMN, DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C1", new org.apache.spark.mllib.linalg.VectorUDT(), true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.DF_VECTOR_WITH_INDEX);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	@Test
	public void testDataFrameSumDMLVectorWithNoIDColumn() {
		LOG.debug("MLContextTest - DataFrame sum DML, vector with no ID column");

		List<Vector> list = new ArrayList<>();
		list.add(Vectors.dense(1.0, 2.0, 3.0));
		list.add(Vectors.dense(4.0, 5.0, 6.0));
		list.add(Vectors.dense(7.0, 8.0, 9.0));
		JavaRDD<Vector> javaRddVector = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddVector.map(new VectorRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", new VectorUDT(), true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.DF_VECTOR);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	@Test
	public void testDataFrameSumDMLMllibVectorWithNoIDColumn() {
		LOG.debug("MLContextTest - DataFrame sum DML, mllib vector with no ID column");

		List<org.apache.spark.mllib.linalg.Vector> list = new ArrayList<>();
		list.add(org.apache.spark.mllib.linalg.Vectors.dense(1.0, 2.0, 3.0));
		list.add(org.apache.spark.mllib.linalg.Vectors.dense(4.0, 5.0, 6.0));
		list.add(org.apache.spark.mllib.linalg.Vectors.dense(7.0, 8.0, 9.0));
		JavaRDD<org.apache.spark.mllib.linalg.Vector> javaRddVector = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddVector.map(new MllibVectorRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", new org.apache.spark.mllib.linalg.VectorUDT(), true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.DF_VECTOR);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	static class DoubleVectorRow implements Function<Tuple2<Double, Vector>, Row> {
		private static final long serialVersionUID = 3605080559931384163L;

		@Override
		public Row call(Tuple2<Double, Vector> tup) throws Exception {
			Double doub = tup._1();
			Vector vect = tup._2();
			return RowFactory.create(doub, vect);
		}
	}

	static class DoubleMllibVectorRow implements Function<Tuple2<Double, org.apache.spark.mllib.linalg.Vector>, Row> {
		private static final long serialVersionUID = -3121178154451876165L;

		@Override
		public Row call(Tuple2<Double, org.apache.spark.mllib.linalg.Vector> tup) throws Exception {
			Double doub = tup._1();
			org.apache.spark.mllib.linalg.Vector vect = tup._2();
			return RowFactory.create(doub, vect);
		}
	}

	static class VectorRow implements Function<Vector, Row> {
		private static final long serialVersionUID = 7077761802433569068L;

		@Override
		public Row call(Vector vect) throws Exception {
			return RowFactory.create(vect);
		}
	}

	static class MllibVectorRow implements Function<org.apache.spark.mllib.linalg.Vector, Row> {
		private static final long serialVersionUID = -408929813562996706L;

		@Override
		public Row call(org.apache.spark.mllib.linalg.Vector vect) throws Exception {
			return RowFactory.create(vect);
		}
	}

	static class CommaSeparatedValueStringToRow implements Function<String, Row> {
		private static final long serialVersionUID = -7871020122671747808L;

		@Override
		public Row call(String str) throws Exception {
			String[] fields = str.split(",");
			return RowFactory.create((Object[]) fields);
		}
	}

	static class CommaSeparatedValueStringToDoubleArrayRow implements Function<String, Row> {
		private static final long serialVersionUID = -8058786466523637317L;

		@Override
		public Row call(String str) throws Exception {
			String[] strings = str.split(",");
			Double[] doubles = new Double[strings.length];
			for(int i = 0; i < strings.length; i++) {
				doubles[i] = Double.parseDouble(strings[i]);
			}
			return RowFactory.create((Object[]) doubles);
		}
	}

	@Test
	public void testCSVMatrixFileInputParameterSumDML() {
		LOG.debug("MLContextTest - CSV matrix file input parameter sum DML");

		String s = "M = read($Min); print('sum: ' + sum(M));";
		String csvFile = baseDirectory + File.separator + "1234.csv";
		String out = executeAndCaptureStdOut( dml(s).in("$Min", csvFile)).getRight();
		assertTrue(out.contains("sum: 10.0"));

	}

	@Test
	public void testCSVMatrixFileInputVariableSumDML() {
		LOG.debug("MLContextTest - CSV matrix file input variable sum DML");

		String s = "M = read($Min); print('sum: ' + sum(M));";
		String csvFile = baseDirectory + File.separator + "1234.csv";
		String out = executeAndCaptureStdOut( dml(s).in("$Min", csvFile)).getRight();
		assertTrue(out.contains("sum: 10.0"));
	}

	@Test
	public void test2DDoubleSumDML() {
		LOG.debug("MLContextTest - two-dimensional double array sum DML");

		double[][] matrix = new double[][] {{10.0, 20.0}, {30.0, 40.0}};

		Script script = dml("print('sum: ' + sum(M));").in("M", matrix);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 100.0"));
	}

	@Test
	public void testAddScalarIntegerInputsDML() {
		LOG.debug("MLContextTest - add scalar integer inputs DML");
		String s = "total = in1 + in2; print('total: ' + total);";
		Script script = dml(s).in("in1", 1).in("in2", 2);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("total: 3"));
	}

	@Test
	public void testInputScalaMapDML() {
		LOG.debug("MLContextTest - input Scala map DML");

		List<String> list = new ArrayList<>();
		list.add("10,20");
		list.add("30,40");
		final JavaRDD<String> javaRDD = sc.parallelize(list);

		Map<String, Object> inputs = new HashMap<>() {
			private static final long serialVersionUID = 1L;
			{
				put("$X", 2);
				put("M", javaRDD);
			}
		};

		scala.collection.mutable.Map<String, Object> scalaMap = JavaConversions.mapAsScalaMap(inputs);

		String s = "M = M + $X; print('sum: ' + sum(M));";
		Script script = dml(s).in(scalaMap);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 108.0"));
	}

	@Test
	public void testOutputDoubleArrayMatrixDML() {
		LOG.debug("MLContextTest - output double array matrix DML");
		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		double[][] matrix = executeAndCaptureStdOut(dml(s).out("M")).getLeft().getMatrixAs2DDoubleArray("M");
		Assert.assertEquals(1.0, matrix[0][0], 0);
		Assert.assertEquals(2.0, matrix[0][1], 0);
		Assert.assertEquals(3.0, matrix[1][0], 0);
		Assert.assertEquals(4.0, matrix[1][1], 0);
	}

	@Test
	public void testOutputScalarLongDML() {
		LOG.debug("MLContextTest - output scalar long DML");
		String s = "m = 5;";
		long result = executeAndCaptureStdOut(dml(s).out("m")).getLeft().getLong("m");
		Assert.assertEquals(5, result);
	}

	@Test
	public void testOutputScalarDoubleDML() {
		LOG.debug("MLContextTest - output scalar double DML");
		String s = "m = 1.23";
		double result = executeAndCaptureStdOut(dml(s).out("m")).getLeft().getDouble("m");
		Assert.assertEquals(1.23, result, 0);
	}

	@Test
	public void testOutputScalarBooleanDML() {
		LOG.debug("MLContextTest - output scalar boolean DML");
		String s = "m = FALSE;";
		boolean result = executeAndCaptureStdOut(dml(s).out("m")).getLeft().getBoolean("m");
		Assert.assertEquals(false, result);
	}

	@Test
	public void testOutputScalarStringDML() {
		LOG.debug("MLContextTest - output scalar string DML");
		String s = "m = 'hello';";
		String result = executeAndCaptureStdOut(dml(s).out("m")).getLeft().getString("m");
		Assert.assertEquals("hello", result);
	}

	@Test
	public void testInputFrameDML() {
		LOG.debug("MLContextTest - input frame DML");

		String s = "M = read($Min, data_type='frame', format='csv'); print(toString(M));";
		String csvFile = baseDirectory + File.separator + "one-two-three-four.csv";
		Script script = dml(s).in("$Min", csvFile);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("one"));
	}

	@Test
	public void testOutputJavaRDDStringIJVDML() {
		LOG.debug("MLContextTest - output Java RDD String IJV DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		JavaRDD<String> javaRDDStringIJV = results.getJavaRDDStringIJV("M");
		List<String> lines = javaRDDStringIJV.collect();
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputJavaRDDStringCSVDenseDML() {
		LOG.debug("MLContextTest - output Java RDD String CSV Dense DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		JavaRDD<String> javaRDDStringCSV = results.getJavaRDDStringCSV("M");
		List<String> lines = javaRDDStringCSV.collect();
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	/**
	 * Reading from dense and sparse matrices is handled differently, so we have tests for both dense and sparse
	 * matrices.
	 */
	@Test
	public void testOutputJavaRDDStringCSVSparseDML() {
		LOG.debug("MLContextTest - output Java RDD String CSV Sparse DML");

		String s = "M = matrix(0, rows=10, cols=10); M[1,1]=1; M[1,2]=2; M[2,1]=3; M[2,2]=4; print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		JavaRDD<String> javaRDDStringCSV = results.getJavaRDDStringCSV("M");
		List<String> lines = javaRDDStringCSV.collect();
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputRDDStringIJVDML() {
		LOG.debug("MLContextTest - output RDD String IJV DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		RDD<String> rddStringIJV = results.getRDDStringIJV("M");
		Iterator<String> iterator = rddStringIJV.toLocalIterator();
		Assert.assertEquals("1 1 1.0", iterator.next());
		Assert.assertEquals("1 2 2.0", iterator.next());
		Assert.assertEquals("2 1 3.0", iterator.next());
		Assert.assertEquals("2 2 4.0", iterator.next());
	}

	@Test
	public void testOutputRDDStringCSVDenseDML() {
		LOG.debug("MLContextTest - output RDD String CSV Dense DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		RDD<String> rddStringCSV = results.getRDDStringCSV("M");
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testOutputRDDStringCSVSparseDML() {
		LOG.debug("MLContextTest - output RDD String CSV Sparse DML");

		String s = "M = matrix(0, rows=10, cols=10); M[1,1]=1; M[1,2]=2; M[2,1]=3; M[2,2]=4; print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		RDD<String> rddStringCSV = results.getRDDStringCSV("M");
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testOutputDataFrameDML() {
		LOG.debug("MLContextTest - output DataFrame DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		Dataset<Row> dataFrame = results.getDataFrame("M");
		List<Row> list = dataFrame.collectAsList();
		Row row1 = list.get(0);
		Assert.assertEquals(1.0, row1.getDouble(0), 0.0);
		Assert.assertEquals(1.0, row1.getDouble(1), 0.0);
		Assert.assertEquals(2.0, row1.getDouble(2), 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(2.0, row2.getDouble(0), 0.0);
		Assert.assertEquals(3.0, row2.getDouble(1), 0.0);
		Assert.assertEquals(4.0, row2.getDouble(2), 0.0);
	}

	@Test
	public void testOutputDataFrameDMLVectorWithIDColumn() {
		LOG.debug("MLContextTest - output DataFrame DML, vector with ID column");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		Dataset<Row> dataFrame = results.getDataFrameVectorWithIDColumn("M");
		List<Row> list = dataFrame.collectAsList();

		Row row1 = list.get(0);
		Assert.assertEquals(1.0, row1.getDouble(0), 0.0);
		Assert.assertArrayEquals(new double[] {1.0, 2.0}, ((Vector) row1.get(1)).toArray(), 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(2.0, row2.getDouble(0), 0.0);
		Assert.assertArrayEquals(new double[] {3.0, 4.0}, ((Vector) row2.get(1)).toArray(), 0.0);
	}

	@Test
	public void testOutputDataFrameDMLVectorNoIDColumn() {
		LOG.debug("MLContextTest - output DataFrame DML, vector no ID column");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		Dataset<Row> dataFrame = results.getDataFrameVectorNoIDColumn("M");
		List<Row> list = dataFrame.collectAsList();

		Row row1 = list.get(0);
		Assert.assertArrayEquals(new double[] {1.0, 2.0}, ((Vector) row1.get(0)).toArray(), 0.0);

		Row row2 = list.get(1);
		Assert.assertArrayEquals(new double[] {3.0, 4.0}, ((Vector) row2.get(0)).toArray(), 0.0);
	}

	@Test
	public void testOutputDataFrameDMLDoublesWithIDColumn() {
		LOG.debug("MLContextTest - output DataFrame DML, doubles with ID column");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		Dataset<Row> dataFrame = results.getDataFrameDoubleWithIDColumn("M");
		List<Row> list = dataFrame.collectAsList();

		Row row1 = list.get(0);
		Assert.assertEquals(1.0, row1.getDouble(0), 0.0);
		Assert.assertEquals(1.0, row1.getDouble(1), 0.0);
		Assert.assertEquals(2.0, row1.getDouble(2), 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(2.0, row2.getDouble(0), 0.0);
		Assert.assertEquals(3.0, row2.getDouble(1), 0.0);
		Assert.assertEquals(4.0, row2.getDouble(2), 0.0);
	}

	@Test
	public void testOutputDataFrameDMLDoublesNoIDColumn() {
		LOG.debug("MLContextTest - output DataFrame DML, doubles no ID column");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		Dataset<Row> dataFrame = results.getDataFrameDoubleNoIDColumn("M");
		List<Row> list = dataFrame.collectAsList();

		Row row1 = list.get(0);
		Assert.assertEquals(1.0, row1.getDouble(0), 0.0);
		Assert.assertEquals(2.0, row1.getDouble(1), 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(3.0, row2.getDouble(0), 0.0);
		Assert.assertEquals(4.0, row2.getDouble(1), 0.0);
	}

	@Test
	public void testTwoScriptsDML() {
		LOG.debug("MLContextTest - two scripts with inputs and outputs DML");

		double[][] m1 = new double[][] {{1.0, 2.0}, {3.0, 4.0}};
		String s1 = "sum1 = sum(m1);";
		double sum1 = executeAndCaptureStdOut(dml(s1).in("m1", m1).out("sum1")).getLeft().getDouble("sum1");
		Assert.assertEquals(10.0, sum1, 0.0);

		double[][] m2 = new double[][] {{5.0, 6.0}, {7.0, 8.0}};
		String s2 = "sum2 = sum(m2);";
		double sum2 = executeAndCaptureStdOut(dml(s2).in("m2", m2).out("sum2")).getLeft().getDouble("sum2");
		Assert.assertEquals(26.0, sum2, 0.0);
	}

	@Test
	public void testOneScriptTwoExecutionsDML() {
		LOG.debug("MLContextTest - one script with two executions DML");

		Script script = new Script();

		double[][] m1 = new double[][] {{1.0, 2.0}, {3.0, 4.0}};
		script.setScriptString("sum1 = sum(m1);").in("m1", m1).out("sum1");
		executeAndCaptureStdOut(script);
		Assert.assertEquals(10.0, script.results().getDouble("sum1"), 0.0);

		script.clearAll();

		double[][] m2 = new double[][] {{5.0, 6.0}, {7.0, 8.0}};
		script.setScriptString("sum2 = sum(m2);").in("m2", m2).out("sum2");
		executeAndCaptureStdOut(script);
		Assert.assertEquals(26.0, script.results().getDouble("sum2"), 0.0);
	}

	@Test
	public void testInputParameterBooleanDML() {
		LOG.debug("MLContextTest - input parameter boolean DML");

		String s = "x = $X; if (x == TRUE) { print('yes'); }";
		Script script = dml(s).in("$X", true);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("yes"));
	}

	@Test
	public void testMultipleOutDML() {
		LOG.debug("MLContextTest - multiple out DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); N = sum(M)";
		// alternative to .out("M").out("N")
		MLResults results = executeAndCaptureStdOut(dml(s).out("M", "N")).getLeft();
		double[][] matrix = results.getMatrixAs2DDoubleArray("M");
		double sum = results.getDouble("N");
		Assert.assertEquals(1.0, matrix[0][0], 0);
		Assert.assertEquals(2.0, matrix[0][1], 0);
		Assert.assertEquals(3.0, matrix[1][0], 0);
		Assert.assertEquals(4.0, matrix[1][1], 0);
		Assert.assertEquals(10.0, sum, 0);
	}

	@Test
	public void testOutputMatrixObjectDML() {
		LOG.debug("MLContextTest - output matrix object DML");
		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		MatrixObject mo = executeAndCaptureStdOut(dml(s).out("M")).getLeft().getMatrixObject("M");
		RDD<String> rddStringCSV = MLContextConversionUtil.matrixObjectToRDDStringCSV(mo);
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testInputMatrixBlockDML() {
		LOG.debug("MLContextTest - input MatrixBlock DML");

		List<String> list = new ArrayList<>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		Matrix m = new Matrix(dataFrame);
		MatrixBlock matrixBlock = m.toMatrixBlock();
		Script script = dml("avg = avg(M);").in("M", matrixBlock).out("avg");
		double avg = executeAndCaptureStdOut(script).getLeft().getDouble("avg");
		Assert.assertEquals(50.0, avg, 0.0);
	}

	@Test
	public void testOutputBinaryBlocksDML() {
		LOG.debug("MLContextTest - output binary blocks DML");
		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		MLResults results = executeAndCaptureStdOut(dml(s).out("M")).getLeft();
		Matrix m = results.getMatrix("M");
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks = m.toBinaryBlocks();
		MatrixMetadata mm = m.getMatrixMetadata();
		MatrixCharacteristics mc = mm.asMatrixCharacteristics();
		JavaRDD<String> javaRDDStringIJV = RDDConverterUtils.binaryBlockToTextCell(binaryBlocks, mc);

		List<String> lines = javaRDDStringIJV.collect();
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputListStringCSVDenseDML() {
		LOG.debug("MLContextTest - output List String CSV Dense DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringCSV(mo);
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputListStringCSVSparseDML() {
		LOG.debug("MLContextTest - output List String CSV Sparse DML");

		String s = "M = matrix(0, rows=10, cols=10); M[1,1]=1; M[1,2]=2; M[2,1]=3; M[2,2]=4; print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringCSV(mo);
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputListStringIJVDenseDML() {
		LOG.debug("MLContextTest - output List String IJV Dense DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringIJV(mo);
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputListStringIJVSparseDML() {
		LOG.debug("MLContextTest - output List String IJV Sparse DML");

		String s = "M = matrix(0, rows=10, cols=10); M[1,1]=1; M[1,2]=2; M[2,1]=3; M[2,2]=4; print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringIJV(mo);
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testJavaRDDGoodMetadataDML() {
		LOG.debug("MLContextTest - JavaRDD<String> good metadata DML");

		List<String> list = new ArrayList<>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = dml("print('sum: ' + sum(M));").in("M", javaRDD, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	@Test
	public void testJavaRDDBadMetadataDML() {
		LOG.debug("MLContextTest - JavaRDD<String> bad metadata DML");

		List<String> list = new ArrayList<>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(1, 1, 9);

		Script script = dml("print('sum: ' + sum(M));").in("M", javaRDD, mm);
		executeAndCaptureStdOut(script, MLContextException.class);
	}

	@Test
	public void testRDDGoodMetadataDML() {
		LOG.debug("MLContextTest - RDD<String> good metadata DML");

		List<String> list = new ArrayList<>();
		list.add("1,1,1");
		list.add("2,2,2");
		list.add("3,3,3");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = dml("print('sum: ' + sum(M));").in("M", rdd, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 18.0"));
	}

	@Test
	public void testDataFrameGoodMetadataDML() {
		LOG.debug("MLContextTest - DataFrame good metadata DML");

		List<String> list = new ArrayList<>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToDoubleArrayRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.DoubleType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 450.0"));
	}

	@SuppressWarnings({"rawtypes", "unchecked"})
	@Test
	public void testInputTupleSeqNoMetadataDML() {
		LOG.debug("MLContextTest - Tuple sequence no metadata DML");

		List<String> list1 = new ArrayList<>();
		list1.add("1,2");
		list1.add("3,4");
		JavaRDD<String> javaRDD1 = sc.parallelize(list1);
		RDD<String> rdd1 = JavaRDD.toRDD(javaRDD1);

		List<String> list2 = new ArrayList<>();
		list2.add("5,6");
		list2.add("7,8");
		JavaRDD<String> javaRDD2 = sc.parallelize(list2);
		RDD<String> rdd2 = JavaRDD.toRDD(javaRDD2);

		Tuple2 tuple1 = new Tuple2("m1", rdd1);
		Tuple2 tuple2 = new Tuple2("m2", rdd2);
		List tupleList = new ArrayList();
		tupleList.add(tuple1);
		tupleList.add(tuple2);
		Seq seq = JavaConversions.asScalaBuffer(tupleList).toSeq();

		Script script = dml("print('sums: ' + sum(m1) + ' ' + sum(m2));").in(seq);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sums: 10.0 26.0"));
		executeAndCaptureStdOut(script);
	}

	@SuppressWarnings({"rawtypes", "unchecked"})
	@Test
	public void testInputTupleSeqWithMetadataDML() {
		LOG.debug("MLContextTest - Tuple sequence with metadata DML");

		List<String> list1 = new ArrayList<>();
		list1.add("1,2");
		list1.add("3,4");
		JavaRDD<String> javaRDD1 = sc.parallelize(list1);
		RDD<String> rdd1 = JavaRDD.toRDD(javaRDD1);

		List<String> list2 = new ArrayList<>();
		list2.add("5,6");
		list2.add("7,8");
		JavaRDD<String> javaRDD2 = sc.parallelize(list2);
		RDD<String> rdd2 = JavaRDD.toRDD(javaRDD2);

		MatrixMetadata mm1 = new MatrixMetadata(2, 2);
		MatrixMetadata mm2 = new MatrixMetadata(2, 2);

		Tuple3 tuple1 = new Tuple3("m1", rdd1, mm1);
		Tuple3 tuple2 = new Tuple3("m2", rdd2, mm2);
		List tupleList = new ArrayList();
		tupleList.add(tuple1);
		tupleList.add(tuple2);
		Seq seq = JavaConversions.asScalaBuffer(tupleList).toSeq();

		Script script = dml("print('sums: ' + sum(m1) + ' ' + sum(m2));").in(seq);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sums: 10.0 26.0"));
	}

	@Test
	public void testCSVMatrixFromURLSumDML() throws MalformedURLException {
		LOG.debug("MLContextTest - CSV matrix from URL sum DML");
		String csv = "https://raw.githubusercontent.com/apache/systemml/master/src/test/scripts/functions/mlcontext/1234.csv";
		URL url = new URL(csv);
		Script script = dml("print('sum: ' + sum(M));").in("M", url);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 10.0"));
	}

	@Test
	public void testIJVMatrixFromURLSumDML() throws MalformedURLException {
		LOG.debug("MLContextTest - IJV matrix from URL sum DML");
		String ijv = "https://raw.githubusercontent.com/apache/systemml/master/src/test/scripts/functions/mlcontext/1234.ijv";
		URL url = new URL(ijv);
		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.IJV, 2, 2);
		Script script = dml("print('sum: ' + sum(M));").in("M", url, mm);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 10.0"));
	}

	@Test
	public void testDataFrameSumDMLDoublesWithNoIDColumnNoFormatSpecified() {
		LOG.debug("MLContextTest - DataFrame sum DML, doubles with no ID column, no format specified");

		List<String> list = new ArrayList<>();
		list.add("2,2,2");
		list.add("3,3,3");
		list.add("4,4,4");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToDoubleArrayRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.DoubleType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 27.0"));
	}

	@Test
	public void testDataFrameSumDMLDoublesWithIDColumnNoFormatSpecified() {
		LOG.debug("MLContextTest - DataFrame sum DML, doubles with ID column, no format specified");

		List<String> list = new ArrayList<>();
		list.add("1,2,2,2");
		list.add("2,3,3,3");
		list.add("3,4,4,4");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToDoubleArrayRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField(RDDConverterUtils.DF_ID_COLUMN, DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C1", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.DoubleType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 27.0"));
	}

	@Test
	public void testDataFrameSumDMLVectorWithIDColumnNoFormatSpecified() {
		LOG.debug("MLContextTest - DataFrame sum DML, vector with ID column, no format specified");

		List<Tuple2<Double, Vector>> list = new ArrayList<>();
		list.add(new Tuple2<>(1.0, Vectors.dense(1.0, 2.0, 3.0)));
		list.add(new Tuple2<>(2.0, Vectors.dense(4.0, 5.0, 6.0)));
		list.add(new Tuple2<>(3.0, Vectors.dense(7.0, 8.0, 9.0)));
		JavaRDD<Tuple2<Double, Vector>> javaRddTuple = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddTuple.map(new DoubleVectorRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField(RDDConverterUtils.DF_ID_COLUMN, DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C1", new VectorUDT(), true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	@Test
	public void testDataFrameSumDMLVectorWithNoIDColumnNoFormatSpecified() {
		LOG.debug("MLContextTest - DataFrame sum DML, vector with no ID column, no format specified");

		List<Vector> list = new ArrayList<>();
		list.add(Vectors.dense(1.0, 2.0, 3.0));
		list.add(Vectors.dense(4.0, 5.0, 6.0));
		list.add(Vectors.dense(7.0, 8.0, 9.0));
		JavaRDD<Vector> javaRddVector = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddVector.map(new VectorRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", new VectorUDT(), true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("sum: 45.0"));
	}

	@Test
	public void testDisplayBooleanDML() {
		LOG.debug("MLContextTest - display boolean DML");
		String s = "print(b);";
		Script script = dml(s).in("b", true);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("TRUE"));
	}

	@Test
	public void testDisplayBooleanNotDML() {
		LOG.debug("MLContextTest - display boolean 'not' DML");
		String s = "print(!b);";
		Script script = dml(s).in("b", true);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("FALSE"));
	}

	@Test
	public void testDisplayIntegerAddDML() {
		LOG.debug("MLContextTest - display integer add DML");
		String s = "print(i+j);";
		Script script = dml(s).in("i", 5).in("j", 6);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("11"));
	}

	@Test
	public void testDisplayStringConcatenationDML() {
		LOG.debug("MLContextTest - display string concatenation DML");
		String s = "print(str1+str2);";
		Script script = dml(s).in("str1", "hello").in("str2", "goodbye");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("hellogoodbye"));
	}

	@Test
	public void testDisplayDoubleAddDML() {
		LOG.debug("MLContextTest - display double add DML");
		String s = "print(i+j);";
		Script script = dml(s).in("i", 5.1).in("j", 6.2);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("11.3"));
	}

	@Test
	public void testPrintFormattingStringSubstitution() {
		LOG.debug("MLContextTest - print formatting string substitution");
		Script script = dml("print('hello %s', 'world');");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("hello world"));
	}

	@Test
	public void testPrintFormattingStringSubstitutions() {
		LOG.debug("MLContextTest - print formatting string substitutions");
		Script script = dml("print('%s %s', 'hello', 'world');");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("hello world"));
	}

	@Test
	public void testPrintFormattingStringSubstitutionAlignment() {
		LOG.debug("MLContextTest - print formatting string substitution alignment");
		Script script = dml("print(\"'%10s' '%-10s'\", \"hello\", \"world\");");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("'     hello' 'world     '"));
	}

	@Test
	public void testPrintFormattingStringSubstitutionVariables() {
		LOG.debug("MLContextTest - print formatting string substitution variables");
		Script script = dml("a='hello'; b='world'; print('%s %s', a, b);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("hello world"));
	}

	@Test
	public void testPrintFormattingIntegerSubstitution() {
		LOG.debug("MLContextTest - print formatting integer substitution");
		Script script = dml("print('int %d', 42);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("int 42"));
	}

	@Test
	public void testPrintFormattingIntegerSubstitutions() {
		LOG.debug("MLContextTest - print formatting integer substitutions");
		Script script = dml("print('%d %d', 42, 43);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("42 43"));
	}

	@Test
	public void testPrintFormattingIntegerSubstitutionAlignment() {
		LOG.debug("MLContextTest - print formatting integer substitution alignment");
		Script script = dml("print(\"'%10d' '%-10d'\", 42, 43);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("'        42' '43        '"));
	}

	@Test
	public void testPrintFormattingIntegerSubstitutionVariables() {
		LOG.debug("MLContextTest - print formatting integer substitution variables");
		Script script = dml("a=42; b=43; print('%d %d', a, b);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("42 43"));
	}

	@Test
	public void testPrintFormattingDoubleSubstitution() {
		LOG.debug("MLContextTest - print formatting double substitution");
		Script script = dml("print('double %f', 42.0);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("double 42.000000"));
	}

	@Test
	public void testPrintFormattingDoubleSubstitutions() {
		LOG.debug("MLContextTest - print formatting double substitutions");
		Script script = dml("print('%f %f', 42.42, 43.43);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("42.420000 43.430000"));
	}

	@Test
	public void testPrintFormattingDoubleSubstitutionAlignment() {
		LOG.debug("MLContextTest - print formatting double substitution alignment");
		Script script = dml("print(\"'%10.2f' '%-10.2f'\", 42.53, 43.54);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("'     42.53' '43.54     '"));
	}

	@Test
	public void testPrintFormattingDoubleSubstitutionVariables() {
		LOG.debug("MLContextTest - print formatting double substitution variables");
		Script script = dml("a=12.34; b=56.78; print('%f %f', a, b);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("12.340000 56.780000"));
	}

	@Test
	public void testPrintFormattingBooleanSubstitution() {
		LOG.debug("MLContextTest - print formatting boolean substitution");
		Script script = dml("print('boolean %b', TRUE);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("boolean true"));
	}

	@Test
	public void testPrintFormattingBooleanSubstitutions() {
		LOG.debug("MLContextTest - print formatting boolean substitutions");
		Script script = dml("print('%b %b', TRUE, FALSE);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("true false"));
	}

	@Test
	public void testPrintFormattingBooleanSubstitutionAlignment() {
		LOG.debug("MLContextTest - print formatting boolean substitution alignment");
		Script script = dml("print(\"'%10b' '%-10b'\", TRUE, FALSE);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("'      true' 'false     '"));
	}

	@Test
	public void testPrintFormattingBooleanSubstitutionVariables() {
		LOG.debug("MLContextTest - print formatting boolean substitution variables");
		Script script = dml("a=TRUE; b=FALSE; print('%b %b', a, b);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("true false"));
	}

	@Test
	public void testPrintFormattingMultipleTypes() {
		LOG.debug("MLContextTest - print formatting multiple types");
		Script script = dml("a='hello'; b=3; c=4.5; d=TRUE; print('%s %d %f %b', a, b, c, d);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("hello 3 4.500000 true"));
	}

	@Test
	public void testPrintFormattingMultipleExpressions() {
		LOG.debug("MLContextTest - print formatting multiple expressions");
		Script script = dml(
			"a='hello'; b='goodbye'; c=4; d=3; e=3.0; f=5.0; g=FALSE; print('%s %d %f %b', (a+b), (c-d), (e*f), !g);");
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("hellogoodbye 1 15.000000 true"));
	}

	@Test
	public void testPrintFormattingForLoop() {
		LOG.debug("MLContextTest - print formatting for loop");
		Script script = dml("for (i in 1:3) { print('int value %d', i); }");
		// check that one of the lines is returned
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("int value 3"));
	}

	@Test
	public void testPrintFormattingParforLoop() {
		LOG.debug("MLContextTest - print formatting parfor loop");
		Script script = dml("parfor (i in 1:3) { print('int value %d', i); }");
		// check that one of the lines is returned
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("int value 3"));
	}

	@Test
	public void testPrintFormattingForLoopMultiply() {
		LOG.debug("MLContextTest - print formatting for loop multiply");
		Script script = dml("a = 5.0; for (i in 1:3) { print('%d %f', i, a * i); }");
		// check that one of the lines is returned
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("3 15.000000"));
	}
	
	@Test
	public void testErrorHandlingTwoIdentifiers() {
		try {
			System.out.println("MLContextTest - error handling two identifiers");
			Script script = dml("foo bar");
			ml.execute(script);
		}
		catch(Exception ex) {
			Throwable t = ex;
			while( t.getCause() != null )
				t = t.getCause();
			System.out.println(t.getMessage());
			Assert.assertTrue(t.getMessage().contains("foo bar"));
			//unfortunately, the generated antlr parser creates the concatenated msg
			//we do a best effort error reporting here, by adding the offending symbol
			//Assert.assertFalse(t.getMessage().contains("foobar"));
			Assert.assertTrue(t.getMessage().contains("'bar'"));
		}
	}

	@Test
	public void testInputVariablesAddLongsDML() {
		LOG.debug("MLContextTest - input variables add longs DML");

		String s = "print('x + y = ' + (x + y));";
		Script script = dml(s).in("x", 3L).in("y", 4L);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("x + y = 7"));
	}

	@Test
	public void testInputVariablesAddFloatsDML() {
		LOG.debug("MLContextTest - input variables add floats DML");

		String s = "print('x + y = ' + (x + y));";
		Script script = dml(s).in("x", 3F).in("y", 4F);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("x + y = 7.0"));
	}

	@Test
	public void testFunctionNoReturnValueDML() {
		LOG.debug("MLContextTest - function with no return value DML");

		String s = "hello=function(){print('no return value')}\nhello();";
		Script script = dml(s);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("no return value"));
	}

	@Test
	public void testFunctionNoReturnValueForceFunctionCallDML() {
		LOG.debug("MLContextTest - function with no return value, force function call DML");

		String s = "hello=function(){\nwhile(FALSE){};\nprint('no return value, force function call');\n}\nhello();";
		Script script = dml(s);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("no return value, force function call"));
	}

	@Test
	public void testFunctionReturnValueDML() {
		LOG.debug("MLContextTest - function with return value DML");

		String s = "hello=function()return(string s){s='return value'}\na=hello();\nprint(a);";
		Script script = dml(s);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("return value"));
	}

	@Test
	public void testFunctionTwoReturnValuesDML() {
		LOG.debug("MLContextTest - function with two return values DML");

		String s = "hello=function()return(string s1,string s2){s1='return'; s2='values'}\n[a,b]=hello();\nprint(a+' '+b);";
		Script script = dml(s);
		String out = executeAndCaptureStdOut( script).getRight();
		assertTrue(out.contains("return values"));
	}

	@Test
	public void testOutputListDML() {
		LOG.debug("MLContextTest - output specified as List DML");

		List<String> outputs = Arrays.asList("x", "y");
		Script script = dml("a=1;x=a+1;y=x+1").out(outputs);
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		Assert.assertEquals(2, results.getLong("x"));
		Assert.assertEquals(3, results.getLong("y"));
	}

	@SuppressWarnings({"unchecked", "rawtypes"})
	@Test
	public void testOutputScalaSeqDML() {
		LOG.debug("MLContextTest - output specified as Scala Seq DML");

		List outputs = Arrays.asList("x", "y");
		Seq seq = JavaConversions.asScalaBuffer(outputs).toSeq();
		Script script = dml("a=1;x=a+1;y=x+1").out(seq);
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		Assert.assertEquals(2, results.getLong("x"));
		Assert.assertEquals(3, results.getLong("y"));
	}

	@Test
	public void testOutputDataFrameOfVectorsDML() {
		LOG.debug("MLContextTest - output DataFrame of vectors DML");

		String s = "m=matrix('1 2 3 4',rows=2,cols=2);";
		Script script = dml(s).out("m");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		Dataset<Row> df = results.getDataFrame("m", true);
		Dataset<Row> sortedDF = df.sort(RDDConverterUtils.DF_ID_COLUMN);

		// verify column types
		StructType schema = sortedDF.schema();
		StructField[] fields = schema.fields();
		StructField idColumn = fields[0];
		StructField vectorColumn = fields[1];
		Assert.assertTrue(idColumn.dataType() instanceof DoubleType);
		Assert.assertTrue(vectorColumn.dataType() instanceof VectorUDT);

		List<Row> list = sortedDF.collectAsList();

		Row row1 = list.get(0);
		Assert.assertEquals(1.0, row1.getDouble(0), 0.0);
		Vector v1 = (DenseVector) row1.get(1);
		double[] arr1 = v1.toArray();
		Assert.assertArrayEquals(new double[] {1.0, 2.0}, arr1, 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(2.0, row2.getDouble(0), 0.0);
		Vector v2 = (DenseVector) row2.get(1);
		double[] arr2 = v2.toArray();
		Assert.assertArrayEquals(new double[] {3.0, 4.0}, arr2, 0.0);
	}

	@Test
	public void testOutputDoubleArrayFromMatrixDML() {
		LOG.debug("MLContextTest - output double array from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		double[][] matrix = executeAndCaptureStdOut(dml(s).out("M")).getLeft().getMatrix("M").to2DDoubleArray();
		Assert.assertEquals(1.0, matrix[0][0], 0);
		Assert.assertEquals(2.0, matrix[0][1], 0);
		Assert.assertEquals(3.0, matrix[1][0], 0);
		Assert.assertEquals(4.0, matrix[1][1], 0);
	}

	@Test
	public void testOutputDataFrameFromMatrixDML() {
		LOG.debug("MLContextTest - output DataFrame from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		Dataset<Row> df = executeAndCaptureStdOut(script).getLeft().getMatrix("M").toDF();
		Dataset<Row> sortedDF = df.sort(RDDConverterUtils.DF_ID_COLUMN);
		List<Row> list = sortedDF.collectAsList();
		Row row1 = list.get(0);
		Assert.assertEquals(1.0, row1.getDouble(0), 0.0);
		Assert.assertEquals(1.0, row1.getDouble(1), 0.0);
		Assert.assertEquals(2.0, row1.getDouble(2), 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(2.0, row2.getDouble(0), 0.0);
		Assert.assertEquals(3.0, row2.getDouble(1), 0.0);
		Assert.assertEquals(4.0, row2.getDouble(2), 0.0);
	}

	@Test
	public void testOutputDataFrameDoublesNoIDColumnFromMatrixDML() {
		LOG.debug("MLContextTest - output DataFrame of doubles with no ID column from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=1, cols=4);";
		Script script = dml(s).out("M");
		Dataset<Row> df = executeAndCaptureStdOut(script).getLeft().getMatrix("M").toDFDoubleNoIDColumn();
		List<Row> list = df.collectAsList();

		Row row = list.get(0);
		Assert.assertEquals(1.0, row.getDouble(0), 0.0);
		Assert.assertEquals(2.0, row.getDouble(1), 0.0);
		Assert.assertEquals(3.0, row.getDouble(2), 0.0);
		Assert.assertEquals(4.0, row.getDouble(3), 0.0);
	}

	@Test
	public void testOutputDataFrameDoublesWithIDColumnFromMatrixDML() {
		LOG.debug("MLContextTest - output DataFrame of doubles with ID column from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		Dataset<Row> df = executeAndCaptureStdOut(script).getLeft().getMatrix("M").toDFDoubleWithIDColumn();
		Dataset<Row> sortedDF = df.sort(RDDConverterUtils.DF_ID_COLUMN);
		List<Row> list = sortedDF.collectAsList();

		Row row1 = list.get(0);
		Assert.assertEquals(1.0, row1.getDouble(0), 0.0);
		Assert.assertEquals(1.0, row1.getDouble(1), 0.0);
		Assert.assertEquals(2.0, row1.getDouble(2), 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(2.0, row2.getDouble(0), 0.0);
		Assert.assertEquals(3.0, row2.getDouble(1), 0.0);
		Assert.assertEquals(4.0, row2.getDouble(2), 0.0);
	}

	@Test
	public void testOutputDataFrameVectorsNoIDColumnFromMatrixDML() {
		LOG.debug("MLContextTest - output DataFrame of vectors with no ID column from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=1, cols=4);";
		Script script = dml(s).out("M");
		Dataset<Row> df = executeAndCaptureStdOut(script).getLeft().getMatrix("M").toDFVectorNoIDColumn();
		List<Row> list = df.collectAsList();

		Row row = list.get(0);
		Assert.assertArrayEquals(new double[] {1.0, 2.0, 3.0, 4.0}, ((Vector) row.get(0)).toArray(), 0.0);
	}

	@Test
	public void testOutputDataFrameVectorsWithIDColumnFromMatrixDML() {
		LOG.debug("MLContextTest - output DataFrame of vectors with ID column from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=1, cols=4);";
		Script script = dml(s).out("M");
		Dataset<Row> df = executeAndCaptureStdOut(script).getLeft().getMatrix("M").toDFVectorWithIDColumn();
		List<Row> list = df.collectAsList();

		Row row = list.get(0);
		Assert.assertEquals(1.0, row.getDouble(0), 0.0);
		Assert.assertArrayEquals(new double[] {1.0, 2.0, 3.0, 4.0}, ((Vector) row.get(1)).toArray(), 0.0);
	}

	@Test
	public void testOutputJavaRDDStringCSVFromMatrixDML() {
		LOG.debug("MLContextTest - output Java RDD String CSV from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=1, cols=4);";
		Script script = dml(s).out("M");
		JavaRDD<String> javaRDDStringCSV = executeAndCaptureStdOut(script).getLeft().getMatrix("M")
			.toJavaRDDStringCSV();
		List<String> lines = javaRDDStringCSV.collect();
		Assert.assertEquals("1.0,2.0,3.0,4.0", lines.get(0));
	}

	@Test
	public void testOutputJavaRDDStringIJVFromMatrixDML() {
		LOG.debug("MLContextTest - output Java RDD String IJV from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = executeAndCaptureStdOut(script).getLeft();
		JavaRDD<String> javaRDDStringIJV = results.getJavaRDDStringIJV("M");
		List<String> lines = javaRDDStringIJV.sortBy(row -> row, true, 1).collect();
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputRDDStringCSVFromMatrixDML() {
		LOG.debug("MLContextTest - output RDD String CSV from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=1, cols=4);";
		Script script = dml(s).out("M");
		RDD<String> rddStringCSV = executeAndCaptureStdOut(script).getLeft().getMatrix("M").toRDDStringCSV();
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0,3.0,4.0", iterator.next());
	}

	@Test
	public void testOutputRDDStringIJVFromMatrixDML() {
		LOG.debug("MLContextTest - output RDD String IJV from matrix DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		RDD<String> rddStringIJV = executeAndCaptureStdOut(script).getLeft().getMatrix("M").toRDDStringIJV();
		String[] rows = (String[]) rddStringIJV.collect();
		Arrays.sort(rows);
		Assert.assertEquals("1 1 1.0", rows[0]);
		Assert.assertEquals("1 2 2.0", rows[1]);
		Assert.assertEquals("2 1 3.0", rows[2]);
		Assert.assertEquals("2 2 4.0", rows[3]);
	}

	@Test
	public void testMLContextVersionMessage() {
		LOG.debug("MLContextTest - version message");

		String version = ml.version();
		// not available until jar built
		Assert.assertEquals(MLContextUtil.VERSION_NOT_AVAILABLE, version);
	}

	@Test
	public void testMLContextBuildTimeMessage() {
		LOG.debug("MLContextTest - build time message");

		String buildTime = ml.buildTime();
		// not available until jar built
		Assert.assertEquals(MLContextUtil.BUILD_TIME_NOT_AVAILABLE, buildTime);
	}

	@Test
	public void testMLContextCreateAndClose() {
		// MLContext created by the @BeforeClass method in MLContextTestBase
		// MLContext closed by the @AfterClass method in MLContextTestBase
		LOG.debug("MLContextTest - create MLContext and close (without script execution)");
	}

	@Test
	public void testDataFrameToBinaryBlocks() {
		LOG.debug("MLContextTest - DataFrame to binary blocks");

		List<String> list = new ArrayList<>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToDoubleArrayRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.DoubleType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> dataFrame = spark.createDataFrame(javaRddRow, schema);

		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks = MLContextConversionUtil
			.dataFrameToMatrixBinaryBlocks(dataFrame);
		Tuple2<MatrixIndexes, MatrixBlock> first = binaryBlocks.first();
		MatrixBlock mb = first._2();
		double[][] matrix = DataConverter.convertToDoubleMatrix(mb);
		Assert.assertArrayEquals(new double[] {1.0, 2.0, 3.0}, matrix[0], 0.0);
		Assert.assertArrayEquals(new double[] {4.0, 5.0, 6.0}, matrix[1], 0.0);
		Assert.assertArrayEquals(new double[] {7.0, 8.0, 9.0}, matrix[2], 0.0);
	}

	@Test
	public void testGetTuple1DML() {
		LOG.debug("MLContextTest - Get Tuple1<Matrix> DML");
		JavaRDD<String> javaRddString = sc
			.parallelize(Stream.of("1,2,3", "4,5,6", "7,8,9").collect(Collectors.toList()));
		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToDoubleArrayRow());
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.DoubleType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.DoubleType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> df = spark.createDataFrame(javaRddRow, schema);

		Script script = dml("N=M*2").in("M", df).out("N");
		Tuple1<Matrix> tuple = executeAndCaptureStdOut(script).getLeft().getTuple("N");
		double[][] n = tuple._1().to2DDoubleArray();
		Assert.assertEquals(2.0, n[0][0], 0);
		Assert.assertEquals(4.0, n[0][1], 0);
		Assert.assertEquals(6.0, n[0][2], 0);
		Assert.assertEquals(8.0, n[1][0], 0);
		Assert.assertEquals(10.0, n[1][1], 0);
		Assert.assertEquals(12.0, n[1][2], 0);
		Assert.assertEquals(14.0, n[2][0], 0);
		Assert.assertEquals(16.0, n[2][1], 0);
		Assert.assertEquals(18.0, n[2][2], 0);
	}

	@Test
	public void testGetTuple2DML() {
		LOG.debug("MLContextTest - Get Tuple2<Matrix,Double> DML");

		double[][] m = new double[][] {{1, 2}, {3, 4}};

		Script script = dml("N=M*2;s=sum(N)").in("M", m).out("N", "s");
		Tuple2<Matrix, Double> tuple = executeAndCaptureStdOut(script).getLeft().getTuple("N", "s");
		double[][] n = tuple._1().to2DDoubleArray();
		double s = tuple._2();
		Assert.assertArrayEquals(new double[] {2, 4}, n[0], 0.0);
		Assert.assertArrayEquals(new double[] {6, 8}, n[1], 0.0);
		Assert.assertEquals(20.0, s, 0.0);
	}

	@Test
	public void testGetTuple3DML() {
		LOG.debug("MLContextTest - Get Tuple3<Long,Double,Boolean> DML");

		Script script = dml("a=1+2;b=a+0.5;c=TRUE;").out("a", "b", "c");
		Tuple3<Long, Double, Boolean> tuple = executeAndCaptureStdOut(script).getLeft().getTuple("a", "b", "c");
		long a = tuple._1();
		double b = tuple._2();
		boolean c = tuple._3();
		Assert.assertEquals(3, a);
		Assert.assertEquals(3.5, b, 0.0);
		Assert.assertEquals(true, c);
	}

	@Test
	public void testGetTuple4DML() {
		LOG.debug("MLContextTest - Get Tuple4<Long,Double,Boolean,String> DML");

		Script script = dml("a=1+2;b=a+0.5;c=TRUE;d=\"yes it's \"+c").out("a", "b", "c", "d");
		Tuple4<Long, Double, Boolean, String> tuple = executeAndCaptureStdOut(script).getLeft()
			.getTuple("a", "b", "c", "d");
		long a = tuple._1();
		double b = tuple._2();
		boolean c = tuple._3();
		String d = tuple._4();
		Assert.assertEquals(3, a);
		Assert.assertEquals(3.5, b, 0.0);
		Assert.assertEquals(true, c);
		Assert.assertEquals("yes it's TRUE", d);
	}
	
	@Test
	public void testNNImport() {
		System.out.println("MLContextTest - NN import");
		String s =    "source(\"scripts/nn/layers/relu.dml\") as relu;\n"
					+ "X = rand(rows=100, cols=10, min=-1, max=1);\n"
					+ "R1 = relu::forward(X);\n"
					+ "R2 = max(X, 0);\n"
					+ "R = sum(R1==R2);\n";
		double ret = ml.execute(dml(s).out("R"))
			.getScalarObject("R").getDoubleValue();
		Assert.assertEquals(1000, ret, 1e-20);
	}
}
