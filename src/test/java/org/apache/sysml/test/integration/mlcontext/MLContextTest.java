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

package org.apache.sysml.test.integration.mlcontext;

import static org.apache.sysml.api.mlcontext.ScriptFactory.dml;
import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile;
import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromInputStream;
import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromLocalFile;
import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromUrl;
import static org.apache.sysml.api.mlcontext.ScriptFactory.pydml;
import static org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromFile;
import static org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromInputStream;
import static org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromLocalFile;
import static org.apache.sysml.api.mlcontext.ScriptFactory.pydmlFromUrl;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.mlcontext.BinaryBlockMatrix;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.MLContextConversionUtil;
import org.apache.sysml.api.mlcontext.MLContextException;
import org.apache.sysml.api.mlcontext.MLResults;
import org.apache.sysml.api.mlcontext.MatrixFormat;
import org.apache.sysml.api.mlcontext.MatrixMetadata;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.api.mlcontext.ScriptExecutor;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import scala.Tuple2;
import scala.Tuple3;
import scala.collection.Iterator;
import scala.collection.JavaConversions;
import scala.collection.Seq;

public class MLContextTest extends AutomatedTestBase {
	protected final static String TEST_DIR = "org/apache/sysml/api/mlcontext";
	protected final static String TEST_NAME = "MLContext";

	static SparkConf conf;
	static JavaSparkContext sc;
	MLContext ml;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);

		if (conf == null) {
			conf = new SparkConf().setAppName("MLContextTest").setMaster("local");
		}
		if (sc == null) {
			sc = new JavaSparkContext(conf);
		}
		ml = new MLContext(sc);
		// ml.setExplain(true);
	}

	@Test
	public void testCreateDMLScriptBasedOnStringAndExecute() {
		System.out.println("MLContextTest - create DML script based on string and execute");
		String testString = "Create DML script based on string and execute";
		setExpectedStdOut(testString);
		Script script = dml("print('" + testString + "');");
		ml.execute(script);
	}

	@Test
	public void testCreatePYDMLScriptBasedOnStringAndExecute() {
		System.out.println("MLContextTest - create PYDML script based on string and execute");
		String testString = "Create PYDML script based on string and execute";
		setExpectedStdOut(testString);
		Script script = pydml("print('" + testString + "')");
		ml.execute(script);
	}

	@Test
	public void testCreateDMLScriptBasedOnFileAndExecute() {
		System.out.println("MLContextTest - create DML script based on file and execute");
		setExpectedStdOut("hello world");
		Script script = dmlFromFile(baseDirectory + File.separator + "hello-world.dml");
		ml.execute(script);
	}

	@Test
	public void testCreatePYDMLScriptBasedOnFileAndExecute() {
		System.out.println("MLContextTest - create PYDML script based on file and execute");
		setExpectedStdOut("hello world");
		Script script = pydmlFromFile(baseDirectory + File.separator + "hello-world.pydml");
		ml.execute(script);
	}

	@Test
	public void testCreateDMLScriptBasedOnInputStreamAndExecute() throws FileNotFoundException {
		System.out.println("MLContextTest - create DML script based on InputStream and execute");
		setExpectedStdOut("hello world");
		File file = new File(baseDirectory + File.separator + "hello-world.dml");
		InputStream is = new FileInputStream(file);
		Script script = dmlFromInputStream(is);
		ml.execute(script);
	}

	@Test
	public void testCreatePYDMLScriptBasedOnInputStreamAndExecute() throws FileNotFoundException {
		System.out.println("MLContextTest - create PYDML script based on InputStream and execute");
		setExpectedStdOut("hello world");
		File file = new File(baseDirectory + File.separator + "hello-world.pydml");
		InputStream is = new FileInputStream(file);
		Script script = pydmlFromInputStream(is);
		ml.execute(script);
	}

	@Test
	public void testCreateDMLScriptBasedOnLocalFileAndExecute() {
		System.out.println("MLContextTest - create DML script based on local file and execute");
		setExpectedStdOut("hello world");
		File file = new File(baseDirectory + File.separator + "hello-world.dml");
		Script script = dmlFromLocalFile(file);
		ml.execute(script);
	}

	@Test
	public void testCreatePYDMLScriptBasedOnLocalFileAndExecute() {
		System.out.println("MLContextTest - create PYDML script based on local file and execute");
		setExpectedStdOut("hello world");
		File file = new File(baseDirectory + File.separator + "hello-world.pydml");
		Script script = pydmlFromLocalFile(file);
		ml.execute(script);
	}

	@Test
	public void testCreateDMLScriptBasedOnURL() throws MalformedURLException {
		System.out.println("MLContextTest - create DML script based on URL");
		String urlString = "https://raw.githubusercontent.com/apache/incubator-systemml/master/src/test/scripts/applications/hits/HITS.dml";
		URL url = new URL(urlString);
		Script script = dmlFromUrl(url);
		String expectedContent = "Licensed to the Apache Software Foundation";
		String s = script.getScriptString();
		assertTrue("Script string doesn't contain expected content: " + expectedContent, s.contains(expectedContent));
	}

	@Test
	public void testCreatePYDMLScriptBasedOnURL() throws MalformedURLException {
		System.out.println("MLContextTest - create PYDML script based on URL");
		String urlString = "https://raw.githubusercontent.com/apache/incubator-systemml/master/src/test/scripts/applications/hits/HITS.pydml";
		URL url = new URL(urlString);
		Script script = pydmlFromUrl(url);
		String expectedContent = "Licensed to the Apache Software Foundation";
		String s = script.getScriptString();
		assertTrue("Script string doesn't contain expected content: " + expectedContent, s.contains(expectedContent));
	}

	@Test
	public void testCreateDMLScriptBasedOnURLString() throws MalformedURLException {
		System.out.println("MLContextTest - create DML script based on URL string");
		String urlString = "https://raw.githubusercontent.com/apache/incubator-systemml/master/src/test/scripts/applications/hits/HITS.dml";
		Script script = dmlFromUrl(urlString);
		String expectedContent = "Licensed to the Apache Software Foundation";
		String s = script.getScriptString();
		assertTrue("Script string doesn't contain expected content: " + expectedContent, s.contains(expectedContent));
	}

	@Test
	public void testCreatePYDMLScriptBasedOnURLString() throws MalformedURLException {
		System.out.println("MLContextTest - create PYDML script based on URL string");
		String urlString = "https://raw.githubusercontent.com/apache/incubator-systemml/master/src/test/scripts/applications/hits/HITS.pydml";
		Script script = pydmlFromUrl(urlString);
		String expectedContent = "Licensed to the Apache Software Foundation";
		String s = script.getScriptString();
		assertTrue("Script string doesn't contain expected content: " + expectedContent, s.contains(expectedContent));
	}

	@Test
	public void testExecuteDMLScript() {
		System.out.println("MLContextTest - execute DML script");
		String testString = "hello dml world!";
		setExpectedStdOut(testString);
		Script script = new Script("print('" + testString + "');", org.apache.sysml.api.mlcontext.ScriptType.DML);
		ml.execute(script);
	}

	@Test
	public void testExecutePYDMLScript() {
		System.out.println("MLContextTest - execute PYDML script");
		String testString = "hello pydml world!";
		setExpectedStdOut(testString);
		Script script = new Script("print('" + testString + "')", org.apache.sysml.api.mlcontext.ScriptType.PYDML);
		ml.execute(script);
	}

	@Test
	public void testInputParametersAddDML() {
		System.out.println("MLContextTest - input parameters add DML");

		String s = "x = $X; y = $Y; print('x + y = ' + (x + y));";
		Script script = dml(s).in("$X", 3).in("$Y", 4);
		setExpectedStdOut("x + y = 7");
		ml.execute(script);
	}

	@Test
	public void testInputParametersAddPYDML() {
		System.out.println("MLContextTest - input parameters add PYDML");

		String s = "x = $X\ny = $Y\nprint('x + y = ' + (x + y))";
		Script script = pydml(s).in("$X", 3).in("$Y", 4);
		setExpectedStdOut("x + y = 7");
		ml.execute(script);
	}

	@Test
	public void testJavaRDDCSVSumDML() {
		System.out.println("MLContextTest - JavaRDD<String> CSV sum DML");

		List<String> list = new ArrayList<String>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		Script script = dml("print('sum: ' + sum(M));").in("M", javaRDD);
		setExpectedStdOut("sum: 45.0");
		ml.execute(script);
	}

	@Test
	public void testJavaRDDCSVSumPYDML() {
		System.out.println("MLContextTest - JavaRDD<String> CSV sum PYDML");

		List<String> list = new ArrayList<String>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		Script script = pydml("print('sum: ' + sum(M))").in("M", javaRDD);
		setExpectedStdOut("sum: 45.0");
		ml.execute(script);
	}

	@Test
	public void testJavaRDDIJVSumDML() {
		System.out.println("MLContextTest - JavaRDD<String> IJV sum DML");

		List<String> list = new ArrayList<String>();
		list.add("1 1 5");
		list.add("2 2 5");
		list.add("3 3 5");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.IJV, 3, 3);

		Script script = dml("print('sum: ' + sum(M));").in("M", javaRDD, mm);
		setExpectedStdOut("sum: 15.0");
		ml.execute(script);
	}

	@Test
	public void testJavaRDDIJVSumPYDML() {
		System.out.println("MLContextTest - JavaRDD<String> IJV sum PYDML");

		List<String> list = new ArrayList<String>();
		list.add("1 1 5");
		list.add("2 2 5");
		list.add("3 3 5");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.IJV, 3, 3);

		Script script = pydml("print('sum: ' + sum(M))").in("M", javaRDD, mm);
		setExpectedStdOut("sum: 15.0");
		ml.execute(script);
	}

	@Test
	public void testJavaRDDAndInputParameterDML() {
		System.out.println("MLContextTest - JavaRDD<String> and input parameter DML");

		List<String> list = new ArrayList<String>();
		list.add("1,2");
		list.add("3,4");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		String s = "M = M + $X; print('sum: ' + sum(M));";
		Script script = dml(s).in("M", javaRDD).in("$X", 1);
		setExpectedStdOut("sum: 14.0");
		ml.execute(script);
	}

	@Test
	public void testJavaRDDAndInputParameterPYDML() {
		System.out.println("MLContextTest - JavaRDD<String> and input parameter PYDML");

		List<String> list = new ArrayList<String>();
		list.add("1,2");
		list.add("3,4");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		String s = "M = M + $X\nprint('sum: ' + sum(M))";
		Script script = pydml(s).in("M", javaRDD).in("$X", 1);
		setExpectedStdOut("sum: 14.0");
		ml.execute(script);
	}

	@Test
	public void testInputMapDML() {
		System.out.println("MLContextTest - input map DML");

		List<String> list = new ArrayList<String>();
		list.add("10,20");
		list.add("30,40");
		final JavaRDD<String> javaRDD = sc.parallelize(list);

		Map<String, Object> inputs = new HashMap<String, Object>() {
			private static final long serialVersionUID = 1L;
			{
				put("$X", 2);
				put("M", javaRDD);
			}
		};

		String s = "M = M + $X; print('sum: ' + sum(M));";
		Script script = dml(s).in(inputs);
		setExpectedStdOut("sum: 108.0");
		ml.execute(script);
	}

	@Test
	public void testInputMapPYDML() {
		System.out.println("MLContextTest - input map PYDML");

		List<String> list = new ArrayList<String>();
		list.add("10,20");
		list.add("30,40");
		final JavaRDD<String> javaRDD = sc.parallelize(list);

		Map<String, Object> inputs = new HashMap<String, Object>() {
			private static final long serialVersionUID = 1L;
			{
				put("$X", 2);
				put("M", javaRDD);
			}
		};

		String s = "M = M + $X\nprint('sum: ' + sum(M))";
		Script script = pydml(s).in(inputs);
		setExpectedStdOut("sum: 108.0");
		ml.execute(script);
	}

	@Test
	public void testCustomExecutionStepDML() {
		System.out.println("MLContextTest - custom execution step DML");
		String testString = "custom execution step";
		setExpectedStdOut(testString);
		Script script = new Script("print('" + testString + "');", org.apache.sysml.api.mlcontext.ScriptType.DML);

		ScriptExecutor scriptExecutor = new ScriptExecutor() {
			// turn off global data flow optimization check
			@Override
			protected void globalDataFlowOptimization() {
				return;
			}
		};
		ml.execute(script, scriptExecutor);
	}

	@Test
	public void testCustomExecutionStepPYDML() {
		System.out.println("MLContextTest - custom execution step PYDML");
		String testString = "custom execution step";
		setExpectedStdOut(testString);
		Script script = new Script("print('" + testString + "')", org.apache.sysml.api.mlcontext.ScriptType.PYDML);

		ScriptExecutor scriptExecutor = new ScriptExecutor() {
			// turn off global data flow optimization check
			@Override
			protected void globalDataFlowOptimization() {
				return;
			}
		};
		ml.execute(script, scriptExecutor);
	}

	@Test
	public void testRDDSumCSVDML() {
		System.out.println("MLContextTest - RDD<String> CSV sum DML");

		List<String> list = new ArrayList<String>();
		list.add("1,1,1");
		list.add("2,2,2");
		list.add("3,3,3");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		Script script = dml("print('sum: ' + sum(M));").in("M", rdd);
		setExpectedStdOut("sum: 18.0");
		ml.execute(script);
	}

	@Test
	public void testRDDSumCSVPYDML() {
		System.out.println("MLContextTest - RDD<String> CSV sum PYDML");

		List<String> list = new ArrayList<String>();
		list.add("1,1,1");
		list.add("2,2,2");
		list.add("3,3,3");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		Script script = pydml("print('sum: ' + sum(M))").in("M", rdd);
		setExpectedStdOut("sum: 18.0");
		ml.execute(script);
	}

	@Test
	public void testRDDSumIJVDML() {
		System.out.println("MLContextTest - RDD<String> IJV sum DML");

		List<String> list = new ArrayList<String>();
		list.add("1 1 1");
		list.add("2 1 2");
		list.add("1 2 3");
		list.add("3 3 4");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.IJV, 3, 3);

		Script script = dml("print('sum: ' + sum(M));").in("M", rdd, mm);
		setExpectedStdOut("sum: 10.0");
		ml.execute(script);
	}

	@Test
	public void testRDDSumIJVPYDML() {
		System.out.println("MLContextTest - RDD<String> IJV sum PYDML");

		List<String> list = new ArrayList<String>();
		list.add("1 1 1");
		list.add("2 1 2");
		list.add("1 2 3");
		list.add("3 3 4");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		MatrixMetadata mm = new MatrixMetadata(MatrixFormat.IJV, 3, 3);

		Script script = pydml("print('sum: ' + sum(M))").in("M", rdd, mm);
		setExpectedStdOut("sum: 10.0");
		ml.execute(script);
	}

	@Test
	public void testDataFrameSumDML() {
		System.out.println("MLContextTest - DataFrame sum DML");

		List<String> list = new ArrayList<String>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToRow());
		SQLContext sqlContext = new SQLContext(sc);
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		DataFrame dataFrame = sqlContext.createDataFrame(javaRddRow, schema);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame);
		setExpectedStdOut("sum: 450.0");
		ml.execute(script);
	}

	@Test
	public void testDataFrameSumPYDML() {
		System.out.println("MLContextTest - DataFrame sum PYDML");

		List<String> list = new ArrayList<String>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToRow());
		SQLContext sqlContext = new SQLContext(sc);
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		DataFrame dataFrame = sqlContext.createDataFrame(javaRddRow, schema);

		Script script = pydml("print('sum: ' + sum(M))").in("M", dataFrame);
		setExpectedStdOut("sum: 450.0");
		ml.execute(script);
	}

	static class CommaSeparatedValueStringToRow implements Function<String, Row> {
		private static final long serialVersionUID = -7871020122671747808L;

		public Row call(String str) throws Exception {
			String[] fields = str.split(",");
			return RowFactory.create((Object[]) fields);
		}
	}

	@Test
	public void testCSVMatrixFileInputParameterSumDML() {
		System.out.println("MLContextTest - CSV matrix file input parameter sum DML");

		String s = "M = read($Min); print('sum: ' + sum(M));";
		String csvFile = baseDirectory + File.separator + "1234.csv";
		setExpectedStdOut("sum: 10.0");
		ml.execute(dml(s).in("$Min", csvFile));
	}

	@Test
	public void testCSVMatrixFileInputParameterSumPYDML() {
		System.out.println("MLContextTest - CSV matrix file input parameter sum PYDML");

		String s = "M = load($Min)\nprint('sum: ' + sum(M))";
		String csvFile = baseDirectory + File.separator + "1234.csv";
		setExpectedStdOut("sum: 10.0");
		ml.execute(pydml(s).in("$Min", csvFile));
	}

	@Test
	public void testCSVMatrixFileInputVariableSumDML() {
		System.out.println("MLContextTest - CSV matrix file input variable sum DML");

		String s = "M = read(Min); print('sum: ' + sum(M));";
		String csvFile = baseDirectory + File.separator + "1234.csv";
		setExpectedStdOut("sum: 10.0");
		ml.execute(dml(s).in("Min", csvFile));
	}

	@Test
	public void testCSVMatrixFileInputVariableSumPYDML() {
		System.out.println("MLContextTest - CSV matrix file input variable sum PYDML");

		String s = "M = load(Min)\nprint('sum: ' + sum(M))";
		String csvFile = baseDirectory + File.separator + "1234.csv";
		setExpectedStdOut("sum: 10.0");
		ml.execute(pydml(s).in("Min", csvFile));
	}

	@Test
	public void test2DDoubleSumDML() {
		System.out.println("MLContextTest - two-dimensional double array sum DML");

		double[][] matrix = new double[][] { { 10.0, 20.0 }, { 30.0, 40.0 } };

		Script script = dml("print('sum: ' + sum(M));").in("M", matrix);
		setExpectedStdOut("sum: 100.0");
		ml.execute(script);
	}

	@Test
	public void test2DDoubleSumPYDML() {
		System.out.println("MLContextTest - two-dimensional double array sum PYDML");

		double[][] matrix = new double[][] { { 10.0, 20.0 }, { 30.0, 40.0 } };

		Script script = pydml("print('sum: ' + sum(M))").in("M", matrix);
		setExpectedStdOut("sum: 100.0");
		ml.execute(script);
	}

	@Test
	public void testAddScalarIntegerInputsDML() {
		System.out.println("MLContextTest - add scalar integer inputs DML");
		String s = "total = in1 + in2; print('total: ' + total);";
		Script script = dml(s).in("in1", 1).in("in2", 2);
		setExpectedStdOut("total: 3");
		ml.execute(script);
	}

	@Test
	public void testAddScalarIntegerInputsPYDML() {
		System.out.println("MLContextTest - add scalar integer inputs PYDML");
		String s = "total = in1 + in2\nprint('total: ' + total)";
		Script script = pydml(s).in("in1", 1).in("in2", 2);
		setExpectedStdOut("total: 3");
		ml.execute(script);
	}

	@Test
	public void testInputScalaMapDML() {
		System.out.println("MLContextTest - input Scala map DML");

		List<String> list = new ArrayList<String>();
		list.add("10,20");
		list.add("30,40");
		final JavaRDD<String> javaRDD = sc.parallelize(list);

		Map<String, Object> inputs = new HashMap<String, Object>() {
			private static final long serialVersionUID = 1L;
			{
				put("$X", 2);
				put("M", javaRDD);
			}
		};

		scala.collection.mutable.Map<String, Object> scalaMap = JavaConversions.mapAsScalaMap(inputs);

		String s = "M = M + $X; print('sum: ' + sum(M));";
		Script script = dml(s).in(scalaMap);
		setExpectedStdOut("sum: 108.0");
		ml.execute(script);
	}

	@Test
	public void testInputScalaMapPYDML() {
		System.out.println("MLContextTest - input Scala map PYDML");

		List<String> list = new ArrayList<String>();
		list.add("10,20");
		list.add("30,40");
		final JavaRDD<String> javaRDD = sc.parallelize(list);

		Map<String, Object> inputs = new HashMap<String, Object>() {
			private static final long serialVersionUID = 1L;
			{
				put("$X", 2);
				put("M", javaRDD);
			}
		};

		scala.collection.mutable.Map<String, Object> scalaMap = JavaConversions.mapAsScalaMap(inputs);

		String s = "M = M + $X\nprint('sum: ' + sum(M))";
		Script script = pydml(s).in(scalaMap);
		setExpectedStdOut("sum: 108.0");
		ml.execute(script);
	}

	@Test
	public void testOutputDoubleArrayMatrixDML() {
		System.out.println("MLContextTest - output double array matrix DML");
		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		double[][] matrix = ml.execute(dml(s).out("M")).getDoubleMatrix("M");
		Assert.assertEquals(1.0, matrix[0][0], 0);
		Assert.assertEquals(2.0, matrix[0][1], 0);
		Assert.assertEquals(3.0, matrix[1][0], 0);
		Assert.assertEquals(4.0, matrix[1][1], 0);
	}

	@Test
	public void testOutputDoubleArrayMatrixPYDML() {
		System.out.println("MLContextTest - output double array matrix PYDML");
		String s = "M = full('1 2 3 4', rows=2, cols=2)";
		double[][] matrix = ml.execute(pydml(s).out("M")).getDoubleMatrix("M");
		Assert.assertEquals(1.0, matrix[0][0], 0);
		Assert.assertEquals(2.0, matrix[0][1], 0);
		Assert.assertEquals(3.0, matrix[1][0], 0);
		Assert.assertEquals(4.0, matrix[1][1], 0);
	}

	@Test
	public void testOutputScalarLongDML() {
		System.out.println("MLContextTest - output scalar long DML");
		String s = "m = 5;";
		long result = ml.execute(dml(s).out("m")).getLong("m");
		Assert.assertEquals(5, result);
	}

	@Test
	public void testOutputScalarLongPYDML() {
		System.out.println("MLContextTest - output scalar long PYDML");
		String s = "m = 5";
		long result = ml.execute(pydml(s).out("m")).getLong("m");
		Assert.assertEquals(5, result);
	}

	@Test
	public void testOutputScalarDoubleDML() {
		System.out.println("MLContextTest - output scalar double DML");
		String s = "m = 1.23";
		double result = ml.execute(dml(s).out("m")).getDouble("m");
		Assert.assertEquals(1.23, result, 0);
	}

	@Test
	public void testOutputScalarDoublePYDML() {
		System.out.println("MLContextTest - output scalar double PYDML");
		String s = "m = 1.23";
		double result = ml.execute(pydml(s).out("m")).getDouble("m");
		Assert.assertEquals(1.23, result, 0);
	}

	@Test
	public void testOutputScalarBooleanDML() {
		System.out.println("MLContextTest - output scalar boolean DML");
		String s = "m = FALSE;";
		boolean result = ml.execute(dml(s).out("m")).getBoolean("m");
		Assert.assertEquals(false, result);
	}

	@Test
	public void testOutputScalarBooleanPYDML() {
		System.out.println("MLContextTest - output scalar boolean PYDML");
		String s = "m = False";
		boolean result = ml.execute(pydml(s).out("m")).getBoolean("m");
		Assert.assertEquals(false, result);
	}

	@Test
	public void testOutputScalarStringDML() {
		System.out.println("MLContextTest - output scalar string DML");
		String s = "m = 'hello';";
		String result = ml.execute(dml(s).out("m")).getString("m");
		Assert.assertEquals("hello", result);
	}

	@Test
	public void testOutputScalarStringPYDML() {
		System.out.println("MLContextTest - output scalar string PYDML");
		String s = "m = 'hello'";
		String result = ml.execute(pydml(s).out("m")).getString("m");
		Assert.assertEquals("hello", result);
	}

	@Test
	public void testInputFrameDML() {
		System.out.println("MLContextTest - input frame DML");

		String s = "M = read(Min, data_type='frame', format='csv'); print(toString(M));";
		String csvFile = baseDirectory + File.separator + "one-two-three-four.csv";
		Script script = dml(s).in("Min", csvFile);
		setExpectedStdOut("one");
		ml.execute(script);
	}

	@Test
	public void testInputFramePYDML() {
		System.out.println("MLContextTest - input frame PYDML");

		String s = "M = load(Min, data_type='frame', format='csv')\nprint(toString(M))";
		String csvFile = baseDirectory + File.separator + "one-two-three-four.csv";
		Script script = pydml(s).in("Min", csvFile);
		setExpectedStdOut("one");
		ml.execute(script);
	}

	@Test
	public void testOutputFrameDML() {
		System.out.println("MLContextTest - output frame DML");

		String s = "M = read(Min, data_type='frame', format='csv');";
		String csvFile = baseDirectory + File.separator + "one-two-three-four.csv";
		Script script = dml(s).in("Min", csvFile).out("M");
		String[][] frame = ml.execute(script).getFrame("M");
		Assert.assertEquals("one", frame[0][0]);
		Assert.assertEquals("two", frame[0][1]);
		Assert.assertEquals("three", frame[1][0]);
		Assert.assertEquals("four", frame[1][1]);
	}

	@Test
	public void testOutputFramePYDML() {
		System.out.println("MLContextTest - output frame PYDML");

		String s = "M = load(Min, data_type='frame', format='csv')";
		String csvFile = baseDirectory + File.separator + "one-two-three-four.csv";
		Script script = pydml(s).in("Min", csvFile).out("M");
		String[][] frame = ml.execute(script).getFrame("M");
		Assert.assertEquals("one", frame[0][0]);
		Assert.assertEquals("two", frame[0][1]);
		Assert.assertEquals("three", frame[1][0]);
		Assert.assertEquals("four", frame[1][1]);
	}

	@Test
	public void testOutputJavaRDDStringIJVDML() {
		System.out.println("MLContextTest - output Java RDD String IJV DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		JavaRDD<String> javaRDDStringIJV = results.getJavaRDDStringIJV("M");
		List<String> lines = javaRDDStringIJV.collect();
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputJavaRDDStringIJVPYDML() {
		System.out.println("MLContextTest - output Java RDD String IJV PYDML");

		String s = "M = full('1 2 3 4', rows=2, cols=2)";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		JavaRDD<String> javaRDDStringIJV = results.getJavaRDDStringIJV("M");
		List<String> lines = javaRDDStringIJV.collect();
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputJavaRDDStringCSVDenseDML() {
		System.out.println("MLContextTest - output Java RDD String CSV Dense DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		JavaRDD<String> javaRDDStringCSV = results.getJavaRDDStringCSV("M");
		List<String> lines = javaRDDStringCSV.collect();
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputJavaRDDStringCSVDensePYDML() {
		System.out.println("MLContextTest - output Java RDD String CSV Dense PYDML");

		String s = "M = full('1 2 3 4', rows=2, cols=2)\nprint(toString(M))";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		JavaRDD<String> javaRDDStringCSV = results.getJavaRDDStringCSV("M");
		List<String> lines = javaRDDStringCSV.collect();
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	/**
	 * Reading from dense and sparse matrices is handled differently, so we have
	 * tests for both dense and sparse matrices.
	 */
	@Test
	public void testOutputJavaRDDStringCSVSparseDML() {
		System.out.println("MLContextTest - output Java RDD String CSV Sparse DML");

		String s = "M = matrix(0, rows=10, cols=10); M[1,1]=1; M[1,2]=2; M[2,1]=3; M[2,2]=4; print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		JavaRDD<String> javaRDDStringCSV = results.getJavaRDDStringCSV("M");
		List<String> lines = javaRDDStringCSV.collect();
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	/**
	 * Reading from dense and sparse matrices is handled differently, so we have
	 * tests for both dense and sparse matrices.
	 */
	@Test
	public void testOutputJavaRDDStringCSVSparsePYDML() {
		System.out.println("MLContextTest - output Java RDD String CSV Sparse PYDML");

		String s = "M = full(0, rows=10, cols=10)\nM[0,0]=1\nM[0,1]=2\nM[1,0]=3\nM[1,1]=4\nprint(toString(M))";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		JavaRDD<String> javaRDDStringCSV = results.getJavaRDDStringCSV("M");
		List<String> lines = javaRDDStringCSV.collect();
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputRDDStringIJVDML() {
		System.out.println("MLContextTest - output RDD String IJV DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		RDD<String> rddStringIJV = results.getRDDStringIJV("M");
		Iterator<String> iterator = rddStringIJV.toLocalIterator();
		Assert.assertEquals("1 1 1.0", iterator.next());
		Assert.assertEquals("1 2 2.0", iterator.next());
		Assert.assertEquals("2 1 3.0", iterator.next());
		Assert.assertEquals("2 2 4.0", iterator.next());
	}

	@Test
	public void testOutputRDDStringIJVPYDML() {
		System.out.println("MLContextTest - output RDD String IJV PYDML");

		String s = "M = full('1 2 3 4', rows=2, cols=2)";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		RDD<String> rddStringIJV = results.getRDDStringIJV("M");
		Iterator<String> iterator = rddStringIJV.toLocalIterator();
		Assert.assertEquals("1 1 1.0", iterator.next());
		Assert.assertEquals("1 2 2.0", iterator.next());
		Assert.assertEquals("2 1 3.0", iterator.next());
		Assert.assertEquals("2 2 4.0", iterator.next());
	}

	@Test
	public void testOutputRDDStringCSVDenseDML() {
		System.out.println("MLContextTest - output RDD String CSV Dense DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		RDD<String> rddStringCSV = results.getRDDStringCSV("M");
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testOutputRDDStringCSVDensePYDML() {
		System.out.println("MLContextTest - output RDD String CSV Dense PYDML");

		String s = "M = full('1 2 3 4', rows=2, cols=2)\nprint(toString(M))";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		RDD<String> rddStringCSV = results.getRDDStringCSV("M");
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testOutputRDDStringCSVSparseDML() {
		System.out.println("MLContextTest - output RDD String CSV Sparse DML");

		String s = "M = matrix(0, rows=10, cols=10); M[1,1]=1; M[1,2]=2; M[2,1]=3; M[2,2]=4; print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		RDD<String> rddStringCSV = results.getRDDStringCSV("M");
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testOutputRDDStringCSVSparsePYDML() {
		System.out.println("MLContextTest - output RDD String CSV Sparse PYDML");

		String s = "M = full(0, rows=10, cols=10)\nM[0,0]=1\nM[0,1]=2\nM[1,0]=3\nM[1,1]=4\nprint(toString(M))";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		RDD<String> rddStringCSV = results.getRDDStringCSV("M");
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testOutputDataFrameDML() {
		System.out.println("MLContextTest - output DataFrame DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		DataFrame dataFrame = results.getDataFrame("M");
		List<Row> list = dataFrame.collectAsList();
		Row row1 = list.get(0);
		Assert.assertEquals(0.0, row1.getDouble(0), 0.0);
		Assert.assertEquals(1.0, row1.getDouble(1), 0.0);
		Assert.assertEquals(2.0, row1.getDouble(2), 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(1.0, row2.getDouble(0), 0.0);
		Assert.assertEquals(3.0, row2.getDouble(1), 0.0);
		Assert.assertEquals(4.0, row2.getDouble(2), 0.0);
	}

	@Test
	public void testOutputDataFramePYDML() {
		System.out.println("MLContextTest - output DataFrame PYDML");

		String s = "M = full('1 2 3 4', rows=2, cols=2)";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		DataFrame dataFrame = results.getDataFrame("M");
		List<Row> list = dataFrame.collectAsList();
		Row row1 = list.get(0);
		Assert.assertEquals(0.0, row1.getDouble(0), 0.0);
		Assert.assertEquals(1.0, row1.getDouble(1), 0.0);
		Assert.assertEquals(2.0, row1.getDouble(2), 0.0);

		Row row2 = list.get(1);
		Assert.assertEquals(1.0, row2.getDouble(0), 0.0);
		Assert.assertEquals(3.0, row2.getDouble(1), 0.0);
		Assert.assertEquals(4.0, row2.getDouble(2), 0.0);
	}

	@Test
	public void testTwoScriptsDML() {
		System.out.println("MLContextTest - two scripts with inputs and outputs DML");

		double[][] m1 = new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } };
		String s1 = "sum1 = sum(m1);";
		double sum1 = ml.execute(dml(s1).in("m1", m1).out("sum1")).getDouble("sum1");
		Assert.assertEquals(10.0, sum1, 0.0);

		double[][] m2 = new double[][] { { 5.0, 6.0 }, { 7.0, 8.0 } };
		String s2 = "sum2 = sum(m2);";
		double sum2 = ml.execute(dml(s2).in("m2", m2).out("sum2")).getDouble("sum2");
		Assert.assertEquals(26.0, sum2, 0.0);
	}

	@Test
	public void testTwoScriptsPYDML() {
		System.out.println("MLContextTest - two scripts with inputs and outputs PYDML");

		double[][] m1 = new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } };
		String s1 = "sum1 = sum(m1)";
		double sum1 = ml.execute(pydml(s1).in("m1", m1).out("sum1")).getDouble("sum1");
		Assert.assertEquals(10.0, sum1, 0.0);

		double[][] m2 = new double[][] { { 5.0, 6.0 }, { 7.0, 8.0 } };
		String s2 = "sum2 = sum(m2)";
		double sum2 = ml.execute(pydml(s2).in("m2", m2).out("sum2")).getDouble("sum2");
		Assert.assertEquals(26.0, sum2, 0.0);
	}

	@Test
	public void testOneScriptTwoExecutionsDML() {
		System.out.println("MLContextTest - one script with two executions DML");

		Script script = new Script(org.apache.sysml.api.mlcontext.ScriptType.DML);

		double[][] m1 = new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } };
		script.setScriptString("sum1 = sum(m1);").in("m1", m1).out("sum1");
		ml.execute(script);
		Assert.assertEquals(10.0, script.results().getDouble("sum1"), 0.0);

		script.clearAll();

		double[][] m2 = new double[][] { { 5.0, 6.0 }, { 7.0, 8.0 } };
		script.setScriptString("sum2 = sum(m2);").in("m2", m2).out("sum2");
		ml.execute(script);
		Assert.assertEquals(26.0, script.results().getDouble("sum2"), 0.0);
	}

	@Test
	public void testOneScriptTwoExecutionsPYDML() {
		System.out.println("MLContextTest - one script with two executions PYDML");

		Script script = new Script(org.apache.sysml.api.mlcontext.ScriptType.PYDML);

		double[][] m1 = new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } };
		script.setScriptString("sum1 = sum(m1)").in("m1", m1).out("sum1");
		ml.execute(script);
		Assert.assertEquals(10.0, script.results().getDouble("sum1"), 0.0);

		script.clearAll();

		double[][] m2 = new double[][] { { 5.0, 6.0 }, { 7.0, 8.0 } };
		script.setScriptString("sum2 = sum(m2)").in("m2", m2).out("sum2");
		ml.execute(script);
		Assert.assertEquals(26.0, script.results().getDouble("sum2"), 0.0);
	}

	@Test
	public void testInputParameterBooleanDML() {
		System.out.println("MLContextTest - input parameter boolean DML");

		String s = "x = $X; if (x == TRUE) { print('yes'); }";
		Script script = dml(s).in("$X", true);
		setExpectedStdOut("yes");
		ml.execute(script);
	}

	@Test
	public void testInputParameterBooleanPYDML() {
		System.out.println("MLContextTest - input parameter boolean PYDML");

		String s = "x = $X\nif (x == True):\n  print('yes')";
		Script script = pydml(s).in("$X", true);
		setExpectedStdOut("yes");
		ml.execute(script);
	}

	@Test
	public void testMultipleOutDML() {
		System.out.println("MLContextTest - multiple out DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); N = sum(M)";
		// alternative to .out("M").out("N")
		MLResults results = ml.execute(dml(s).out("M", "N"));
		double[][] matrix = results.getDoubleMatrix("M");
		double sum = results.getDouble("N");
		Assert.assertEquals(1.0, matrix[0][0], 0);
		Assert.assertEquals(2.0, matrix[0][1], 0);
		Assert.assertEquals(3.0, matrix[1][0], 0);
		Assert.assertEquals(4.0, matrix[1][1], 0);
		Assert.assertEquals(10.0, sum, 0);
	}

	@Test
	public void testMultipleOutPYDML() {
		System.out.println("MLContextTest - multiple out PYDML");

		String s = "M = full('1 2 3 4', rows=2, cols=2)\nN = sum(M)";
		// alternative to .out("M").out("N")
		MLResults results = ml.execute(pydml(s).out("M", "N"));
		double[][] matrix = results.getDoubleMatrix("M");
		double sum = results.getDouble("N");
		Assert.assertEquals(1.0, matrix[0][0], 0);
		Assert.assertEquals(2.0, matrix[0][1], 0);
		Assert.assertEquals(3.0, matrix[1][0], 0);
		Assert.assertEquals(4.0, matrix[1][1], 0);
		Assert.assertEquals(10.0, sum, 0);
	}

	@Test
	public void testOutputMatrixObjectDML() {
		System.out.println("MLContextTest - output matrix object DML");
		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		MatrixObject mo = ml.execute(dml(s).out("M")).getMatrixObject("M");
		RDD<String> rddStringCSV = MLContextConversionUtil.matrixObjectToRDDStringCSV(mo);
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testOutputMatrixObjectPYDML() {
		System.out.println("MLContextTest - output matrix object PYDML");
		String s = "M = full('1 2 3 4', rows=2, cols=2);";
		MatrixObject mo = ml.execute(pydml(s).out("M")).getMatrixObject("M");
		RDD<String> rddStringCSV = MLContextConversionUtil.matrixObjectToRDDStringCSV(mo);
		Iterator<String> iterator = rddStringCSV.toLocalIterator();
		Assert.assertEquals("1.0,2.0", iterator.next());
		Assert.assertEquals("3.0,4.0", iterator.next());
	}

	@Test
	public void testInputBinaryBlockMatrixDML() {
		System.out.println("MLContextTest - input BinaryBlockMatrix DML");

		List<String> list = new ArrayList<String>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToRow());
		SQLContext sqlContext = new SQLContext(sc);
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		DataFrame dataFrame = sqlContext.createDataFrame(javaRddRow, schema);

		BinaryBlockMatrix binaryBlockMatrix = new BinaryBlockMatrix(dataFrame);
		Script script = dml("avg = avg(M);").in("M", binaryBlockMatrix).out("avg");
		double avg = ml.execute(script).getDouble("avg");
		Assert.assertEquals(50.0, avg, 0.0);
	}

	@Test
	public void testInputBinaryBlockMatrixPYDML() {
		System.out.println("MLContextTest - input BinaryBlockMatrix PYDML");

		List<String> list = new ArrayList<String>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToRow());
		SQLContext sqlContext = new SQLContext(sc);
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		DataFrame dataFrame = sqlContext.createDataFrame(javaRddRow, schema);

		BinaryBlockMatrix binaryBlockMatrix = new BinaryBlockMatrix(dataFrame);
		Script script = pydml("avg = avg(M)").in("M", binaryBlockMatrix).out("avg");
		double avg = ml.execute(script).getDouble("avg");
		Assert.assertEquals(50.0, avg, 0.0);
	}

	@Test
	public void testOutputBinaryBlockMatrixDML() {
		System.out.println("MLContextTest - output BinaryBlockMatrix DML");
		String s = "M = matrix('1 2 3 4', rows=2, cols=2);";
		BinaryBlockMatrix binaryBlockMatrix = ml.execute(dml(s).out("M")).getBinaryBlockMatrix("M");

		JavaRDD<String> javaRDDStringIJV = MLContextConversionUtil
				.binaryBlockMatrixToJavaRDDStringIJV(binaryBlockMatrix);
		List<String> lines = javaRDDStringIJV.collect();
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputBinaryBlockMatrixPYDML() {
		System.out.println("MLContextTest - output BinaryBlockMatrix PYDML");
		String s = "M = full('1 2 3 4', rows=2, cols=2);";
		BinaryBlockMatrix binaryBlockMatrix = ml.execute(pydml(s).out("M")).getBinaryBlockMatrix("M");

		JavaRDD<String> javaRDDStringIJV = MLContextConversionUtil
				.binaryBlockMatrixToJavaRDDStringIJV(binaryBlockMatrix);
		List<String> lines = javaRDDStringIJV.collect();
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputListStringCSVDenseDML() {
		System.out.println("MLContextTest - output List String CSV Dense DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringCSV(mo);
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputListStringCSVDensePYDML() {
		System.out.println("MLContextTest - output List String CSV Dense PYDML");

		String s = "M = full('1 2 3 4', rows=2, cols=2)\nprint(toString(M))";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringCSV(mo);
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputListStringCSVSparseDML() {
		System.out.println("MLContextTest - output List String CSV Sparse DML");

		String s = "M = matrix(0, rows=10, cols=10); M[1,1]=1; M[1,2]=2; M[2,1]=3; M[2,2]=4; print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringCSV(mo);
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputListStringCSVSparsePYDML() {
		System.out.println("MLContextTest - output List String CSV Sparse PYDML");

		String s = "M = full(0, rows=10, cols=10)\nM[0,0]=1\nM[0,1]=2\nM[1,0]=3\nM[1,1]=4\nprint(toString(M))";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringCSV(mo);
		Assert.assertEquals("1.0,2.0", lines.get(0));
		Assert.assertEquals("3.0,4.0", lines.get(1));
	}

	@Test
	public void testOutputListStringIJVDenseDML() {
		System.out.println("MLContextTest - output List String IJV Dense DML");

		String s = "M = matrix('1 2 3 4', rows=2, cols=2); print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringIJV(mo);
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputListStringIJVDensePYDML() {
		System.out.println("MLContextTest - output List String IJV Dense PYDML");

		String s = "M = full('1 2 3 4', rows=2, cols=2)\nprint(toString(M))";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringIJV(mo);
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputListStringIJVSparseDML() {
		System.out.println("MLContextTest - output List String IJV Sparse DML");

		String s = "M = matrix(0, rows=10, cols=10); M[1,1]=1; M[1,2]=2; M[2,1]=3; M[2,2]=4; print(toString(M));";
		Script script = dml(s).out("M");
		MLResults results = ml.execute(script);
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringIJV(mo);
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testOutputListStringIJVSparsePYDML() {
		System.out.println("MLContextTest - output List String IJV Sparse PYDML");

		String s = "M = full(0, rows=10, cols=10)\nM[0,0]=1\nM[0,1]=2\nM[1,0]=3\nM[1,1]=4\nprint(toString(M))";
		Script script = pydml(s).out("M");
		MLResults results = ml.execute(script);
		MatrixObject mo = results.getMatrixObject("M");
		List<String> lines = MLContextConversionUtil.matrixObjectToListStringIJV(mo);
		Assert.assertEquals("1 1 1.0", lines.get(0));
		Assert.assertEquals("1 2 2.0", lines.get(1));
		Assert.assertEquals("2 1 3.0", lines.get(2));
		Assert.assertEquals("2 2 4.0", lines.get(3));
	}

	@Test
	public void testJavaRDDGoodMetadataDML() {
		System.out.println("MLContextTest - JavaRDD<String> good metadata DML");

		List<String> list = new ArrayList<String>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = dml("print('sum: ' + sum(M));").in("M", javaRDD, mm);
		setExpectedStdOut("sum: 45.0");
		ml.execute(script);
	}

	@Test
	public void testJavaRDDGoodMetadataPYDML() {
		System.out.println("MLContextTest - JavaRDD<String> good metadata PYDML");

		List<String> list = new ArrayList<String>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = pydml("print('sum: ' + sum(M))").in("M", javaRDD, mm);
		setExpectedStdOut("sum: 45.0");
		ml.execute(script);
	}

	@Test(expected = MLContextException.class)
	public void testJavaRDDBadMetadataDML() {
		System.out.println("MLContextTest - JavaRDD<String> bad metadata DML");

		List<String> list = new ArrayList<String>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(1, 1, 9);

		Script script = dml("print('sum: ' + sum(M));").in("M", javaRDD, mm);
		ml.execute(script);
	}

	@Test(expected = MLContextException.class)
	public void testJavaRDDBadMetadataPYDML() {
		System.out.println("MLContextTest - JavaRDD<String> bad metadata PYML");

		List<String> list = new ArrayList<String>();
		list.add("1,2,3");
		list.add("4,5,6");
		list.add("7,8,9");
		JavaRDD<String> javaRDD = sc.parallelize(list);

		MatrixMetadata mm = new MatrixMetadata(1, 1, 9);

		Script script = dml("print('sum: ' + sum(M))").in("M", javaRDD, mm);
		ml.execute(script);
	}

	@Test
	public void testRDDGoodMetadataDML() {
		System.out.println("MLContextTest - RDD<String> good metadata DML");

		List<String> list = new ArrayList<String>();
		list.add("1,1,1");
		list.add("2,2,2");
		list.add("3,3,3");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = dml("print('sum: ' + sum(M));").in("M", rdd, mm);
		setExpectedStdOut("sum: 18.0");
		ml.execute(script);
	}

	@Test
	public void testRDDGoodMetadataPYDML() {
		System.out.println("MLContextTest - RDD<String> good metadata PYDML");

		List<String> list = new ArrayList<String>();
		list.add("1,1,1");
		list.add("2,2,2");
		list.add("3,3,3");
		JavaRDD<String> javaRDD = sc.parallelize(list);
		RDD<String> rdd = JavaRDD.toRDD(javaRDD);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = pydml("print('sum: ' + sum(M))").in("M", rdd, mm);
		setExpectedStdOut("sum: 18.0");
		ml.execute(script);
	}

	@Test
	public void testDataFrameGoodMetadataDML() {
		System.out.println("MLContextTest - DataFrame good metadata DML");

		List<String> list = new ArrayList<String>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToRow());
		SQLContext sqlContext = new SQLContext(sc);
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		DataFrame dataFrame = sqlContext.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = dml("print('sum: ' + sum(M));").in("M", dataFrame, mm);
		setExpectedStdOut("sum: 450.0");
		ml.execute(script);
	}

	@Test
	public void testDataFrameGoodMetadataPYDML() {
		System.out.println("MLContextTest - DataFrame good metadata PYDML");

		List<String> list = new ArrayList<String>();
		list.add("10,20,30");
		list.add("40,50,60");
		list.add("70,80,90");
		JavaRDD<String> javaRddString = sc.parallelize(list);

		JavaRDD<Row> javaRddRow = javaRddString.map(new CommaSeparatedValueStringToRow());
		SQLContext sqlContext = new SQLContext(sc);
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C2", DataTypes.StringType, true));
		fields.add(DataTypes.createStructField("C3", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		DataFrame dataFrame = sqlContext.createDataFrame(javaRddRow, schema);

		MatrixMetadata mm = new MatrixMetadata(3, 3, 9);

		Script script = pydml("print('sum: ' + sum(M))").in("M", dataFrame, mm);
		setExpectedStdOut("sum: 450.0");
		ml.execute(script);
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	@Test
	public void testInputTupleSeqNoMetadataDML() {
		System.out.println("MLContextTest - Tuple sequence no metadata DML");

		List<String> list1 = new ArrayList<String>();
		list1.add("1,2");
		list1.add("3,4");
		JavaRDD<String> javaRDD1 = sc.parallelize(list1);
		RDD<String> rdd1 = JavaRDD.toRDD(javaRDD1);

		List<String> list2 = new ArrayList<String>();
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
		setExpectedStdOut("sums: 10.0 26.0");
		ml.execute(script);
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	@Test
	public void testInputTupleSeqNoMetadataPYDML() {
		System.out.println("MLContextTest - Tuple sequence no metadata PYDML");

		List<String> list1 = new ArrayList<String>();
		list1.add("1,2");
		list1.add("3,4");
		JavaRDD<String> javaRDD1 = sc.parallelize(list1);
		RDD<String> rdd1 = JavaRDD.toRDD(javaRDD1);

		List<String> list2 = new ArrayList<String>();
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

		Script script = pydml("print('sums: ' + sum(m1) + ' ' + sum(m2))").in(seq);
		setExpectedStdOut("sums: 10.0 26.0");
		ml.execute(script);
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	@Test
	public void testInputTupleSeqWithMetadataDML() {
		System.out.println("MLContextTest - Tuple sequence with metadata DML");

		List<String> list1 = new ArrayList<String>();
		list1.add("1,2");
		list1.add("3,4");
		JavaRDD<String> javaRDD1 = sc.parallelize(list1);
		RDD<String> rdd1 = JavaRDD.toRDD(javaRDD1);

		List<String> list2 = new ArrayList<String>();
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
		setExpectedStdOut("sums: 10.0 26.0");
		ml.execute(script);
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	@Test
	public void testInputTupleSeqWithMetadataPYDML() {
		System.out.println("MLContextTest - Tuple sequence with metadata PYDML");

		List<String> list1 = new ArrayList<String>();
		list1.add("1,2");
		list1.add("3,4");
		JavaRDD<String> javaRDD1 = sc.parallelize(list1);
		RDD<String> rdd1 = JavaRDD.toRDD(javaRDD1);

		List<String> list2 = new ArrayList<String>();
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

		Script script = pydml("print('sums: ' + sum(m1) + ' ' + sum(m2))").in(seq);
		setExpectedStdOut("sums: 10.0 26.0");
		ml.execute(script);
	}

	// NOTE: Uncomment these tests once they work

	// @SuppressWarnings({ "rawtypes", "unchecked" })
	// @Test
	// public void testInputTupleSeqWithAndWithoutMetadataDML() {
	// System.out.println("MLContextTest - Tuple sequence with and without metadata DML");
	//
	// List<String> list1 = new ArrayList<String>();
	// list1.add("1,2");
	// list1.add("3,4");
	// JavaRDD<String> javaRDD1 = sc.parallelize(list1);
	// RDD<String> rdd1 = JavaRDD.toRDD(javaRDD1);
	//
	// List<String> list2 = new ArrayList<String>();
	// list2.add("5,6");
	// list2.add("7,8");
	// JavaRDD<String> javaRDD2 = sc.parallelize(list2);
	// RDD<String> rdd2 = JavaRDD.toRDD(javaRDD2);
	//
	// MatrixMetadata mm1 = new MatrixMetadata(2, 2);
	//
	// Tuple3 tuple1 = new Tuple3("m1", rdd1, mm1);
	// Tuple2 tuple2 = new Tuple2("m2", rdd2);
	// List tupleList = new ArrayList();
	// tupleList.add(tuple1);
	// tupleList.add(tuple2);
	// Seq seq = JavaConversions.asScalaBuffer(tupleList).toSeq();
	//
	// Script script =
	// dml("print('sums: ' + sum(m1) + ' ' + sum(m2));").in(seq);
	// setExpectedStdOut("sums: 10.0 26.0");
	// ml.execute(script);
	// }
	//
	// @SuppressWarnings({ "rawtypes", "unchecked" })
	// @Test
	// public void testInputTupleSeqWithAndWithoutMetadataPYDML() {
	// System.out.println("MLContextTest - Tuple sequence with and without metadata PYDML");
	//
	// List<String> list1 = new ArrayList<String>();
	// list1.add("1,2");
	// list1.add("3,4");
	// JavaRDD<String> javaRDD1 = sc.parallelize(list1);
	// RDD<String> rdd1 = JavaRDD.toRDD(javaRDD1);
	//
	// List<String> list2 = new ArrayList<String>();
	// list2.add("5,6");
	// list2.add("7,8");
	// JavaRDD<String> javaRDD2 = sc.parallelize(list2);
	// RDD<String> rdd2 = JavaRDD.toRDD(javaRDD2);
	//
	// MatrixMetadata mm1 = new MatrixMetadata(2, 2);
	//
	// Tuple3 tuple1 = new Tuple3("m1", rdd1, mm1);
	// Tuple2 tuple2 = new Tuple2("m2", rdd2);
	// List tupleList = new ArrayList();
	// tupleList.add(tuple1);
	// tupleList.add(tuple2);
	// Seq seq = JavaConversions.asScalaBuffer(tupleList).toSeq();
	//
	// Script script =
	// pydml("print('sums: ' + sum(m1) + ' ' + sum(m2))").in(seq);
	// setExpectedStdOut("sums: 10.0 26.0");
	// ml.execute(script);
	// }

	@After
	public void tearDown() {
		super.tearDown();
	}

}
