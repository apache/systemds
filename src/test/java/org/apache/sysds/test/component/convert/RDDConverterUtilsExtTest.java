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
 
package org.apache.sysds.test.component.convert;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkException;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtilsExt;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;


@net.jcip.annotations.NotThreadSafe
public class RDDConverterUtilsExtTest extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(RDDConverterUtilsExtTest.class.getName());

	private static SparkConf conf;
	private static JavaSparkContext sc;

	@BeforeClass
	public static void setUpClass() {
		if (conf == null)
			conf = SparkExecutionContext.createSystemDSSparkConf().setAppName("RDDConverterUtilsExtTest")
					.set("spark.port.maxRetries", "100")
					.setMaster("local")
					.set("spark.driver.bindAddress", "127.0.0.1")
					.set("SPARK_MASTER_PORT", "0")
					.set("SPARK_WORKER_PORT", "0");

		if (sc == null)
			sc = new JavaSparkContext(conf);
	}

	@Override
	public void setUp() {
		// no setup required
	}

	/**
	 * Convert a basic String to a spark.sql.Row.
	 */
	static class StringToRow implements Function<String, Row> {
		private static final long serialVersionUID = 3945939649355731805L;

		@Override
		public Row call(String str) throws Exception {
			return RowFactory.create(str);
		}
	}

	@Test
	public void testStringDataFrameToVectorDataFrame() {
		List<String> list = new ArrayList<>();
		list.add("((1.2, 4.3, 3.4))");
		list.add("(1.2, 3.4, 2.2)");
		list.add("[[1.2, 34.3, 1.2, 1.25]]");
		list.add("[1.2, 3.4]");
		JavaRDD<String> javaRddString = sc.parallelize(list);
		JavaRDD<Row> javaRddRow = javaRddString.map(new StringToRow());
		SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> inDF = sparkSession.createDataFrame(javaRddRow, schema);
		Dataset<Row> outDF = RDDConverterUtilsExt.stringDataFrameToVectorDataFrame(sparkSession, inDF);

		List<String> expectedResults = new ArrayList<>();
		expectedResults.add("[[1.2,4.3,3.4]]");
		expectedResults.add("[[1.2,3.4,2.2]]");
		expectedResults.add("[[1.2,34.3,1.2,1.25]]");
		expectedResults.add("[[1.2,3.4]]");

		List<Row> outputList = outDF.collectAsList();
		for (Row row : outputList) {
			assertTrue("Expected results don't contain: " + row, expectedResults.contains(row.toString()));
		}
	}

	@Test
	public void testStringDataFrameToVectorDataFrameNull() {
		List<String> list = new ArrayList<>();
		list.add("[1.2, 3.4]");
		list.add(null);
		JavaRDD<String> javaRddString = sc.parallelize(list);
		JavaRDD<Row> javaRddRow = javaRddString.map(new StringToRow());
		SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> inDF = sparkSession.createDataFrame(javaRddRow, schema);
		Dataset<Row> outDF = RDDConverterUtilsExt.stringDataFrameToVectorDataFrame(sparkSession, inDF);

		List<String> expectedResults = new ArrayList<>();
		expectedResults.add("[[1.2,3.4]]");
		expectedResults.add("[null]");

		List<Row> outputList = outDF.collectAsList();
		for (Row row : outputList) {
			assertTrue("Expected results don't contain: " + row, expectedResults.contains(row.toString()));
		}
	}

	@Test(expected = SparkException.class)
	public void testStringDataFrameToVectorDataFrameNonNumbers() {
		List<String> list = new ArrayList<>();
		list.add("[cheeseburger,fries]");
		JavaRDD<String> javaRddString = sc.parallelize(list);
		JavaRDD<Row> javaRddRow = javaRddString.map(new StringToRow());
		SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
		List<StructField> fields = new ArrayList<>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> inDF = sparkSession.createDataFrame(javaRddRow, schema);
		Dataset<Row> outDF = RDDConverterUtilsExt.stringDataFrameToVectorDataFrame(sparkSession, inDF);
		// trigger evaluation to throw exception
		outDF.collectAsList();
	}

	@After
	@Override
	public void tearDown() {
		super.tearDown();
	}

	@AfterClass
	public static void tearDownClass() {
		// stop spark context to allow single jvm tests (otherwise the
		// next test that tries to create a SparkContext would fail)
		try{
			sc.stop();
		}
		catch(Exception e){
			// Since it does not matter if the Spark context is closed properly if only executing component tests
			// we simply write a warning. This is because our GitHub Actions tests fail sometimes with the spark context.
			LOG.warn(e);
		}
		finally{
			sc = null;
			conf = null;
		}
	}
}
