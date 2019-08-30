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
 
package org.tugraz.sysds.test.component.convert;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

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
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDConverterUtilsExt;
import org.tugraz.sysds.test.AutomatedTestBase;

public class RDDConverterUtilsExtTest extends AutomatedTestBase {

	private static SparkConf conf;
	private static JavaSparkContext sc;

	@BeforeClass
	public static void setUpClass() {
		if (conf == null)
			conf = SparkExecutionContext.createSystemDSSparkConf().setAppName("RDDConverterUtilsExtTest")
					.setMaster("local");
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
		List<String> list = new ArrayList<String>();
		list.add("((1.2, 4.3, 3.4))");
		list.add("(1.2, 3.4, 2.2)");
		list.add("[[1.2, 34.3, 1.2, 1.25]]");
		list.add("[1.2, 3.4]");
		JavaRDD<String> javaRddString = sc.parallelize(list);
		JavaRDD<Row> javaRddRow = javaRddString.map(new StringToRow());
		SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> inDF = sparkSession.createDataFrame(javaRddRow, schema);
		Dataset<Row> outDF = RDDConverterUtilsExt.stringDataFrameToVectorDataFrame(sparkSession, inDF);

		List<String> expectedResults = new ArrayList<String>();
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
		List<String> list = new ArrayList<String>();
		list.add("[1.2, 3.4]");
		list.add(null);
		JavaRDD<String> javaRddString = sc.parallelize(list);
		JavaRDD<Row> javaRddRow = javaRddString.map(new StringToRow());
		SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> inDF = sparkSession.createDataFrame(javaRddRow, schema);
		Dataset<Row> outDF = RDDConverterUtilsExt.stringDataFrameToVectorDataFrame(sparkSession, inDF);

		List<String> expectedResults = new ArrayList<String>();
		expectedResults.add("[[1.2,3.4]]");
		expectedResults.add("[null]");

		List<Row> outputList = outDF.collectAsList();
		for (Row row : outputList) {
			assertTrue("Expected results don't contain: " + row, expectedResults.contains(row.toString()));
		}
	}

	@Test(expected = SparkException.class)
	public void testStringDataFrameToVectorDataFrameNonNumbers() {
		List<String> list = new ArrayList<String>();
		list.add("[cheeseburger,fries]");
		JavaRDD<String> javaRddString = sc.parallelize(list);
		JavaRDD<Row> javaRddRow = javaRddString.map(new StringToRow());
		SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
		List<StructField> fields = new ArrayList<StructField>();
		fields.add(DataTypes.createStructField("C1", DataTypes.StringType, true));
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> inDF = sparkSession.createDataFrame(javaRddRow, schema);
		Dataset<Row> outDF = RDDConverterUtilsExt.stringDataFrameToVectorDataFrame(sparkSession, inDF);
		// trigger evaluation to throw exception
		outDF.collectAsList();
	}

	@After
	public void tearDown() {
		super.tearDown();
	}

	@AfterClass
	public static void tearDownClass() {
		// stop spark context to allow single jvm tests (otherwise the
		// next test that tries to create a SparkContext would fail)
		sc.stop();
		sc = null;
		conf = null;
	}
}
