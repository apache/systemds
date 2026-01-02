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

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysds.api.mlcontext.FrameFormat;
import org.apache.sysds.api.mlcontext.FrameMetadata;
import org.apache.sysds.api.mlcontext.FrameSchema;
import org.apache.sysds.api.mlcontext.MLResults;
import org.apache.sysds.api.mlcontext.MatrixFormat;
import org.apache.sysds.api.mlcontext.MatrixMetadata;
import org.apache.sysds.api.mlcontext.Script;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysds.test.functions.mlcontext.MLContextTest.CommaSeparatedValueStringToDoubleArrayRow;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import scala.collection.Iterator;

public class MLContextFrameTest extends MLContextTestBase {

	protected static final Log LOG = LogFactory.getLog(MLContextFrameTest.class.getName());

	public static enum ScriptType {
		DML
	}

	public static enum IoType {
		ANY, FILE, JAVA_RDD_STR_CSV, JAVA_RDD_STR_IJV, RDD_STR_CSV, RDD_STR_IJV, DATAFRAME
	}

	private static String CSV_DELIM = ",";

	@BeforeClass
	public static void setUpClass() {
		MLContextTestBase.setUpClass();
		// ml.setExplainLevel(ExplainLevel.RECOMPILE_HOPS);
	}

	@Test
	public void testFrameJavaRDD_CSV_DML() {
		testFrame(FrameFormat.CSV, ScriptType.DML, IoType.JAVA_RDD_STR_CSV, IoType.ANY);
	}

	@Test
	public void testFrameJavaRDD_CSV_DML_OutJavaRddCSV() {
		testFrame(FrameFormat.CSV, ScriptType.DML, IoType.JAVA_RDD_STR_CSV, IoType.JAVA_RDD_STR_CSV);
	}

	@Test
	public void testFrameJavaRDD_IJV_DML() {
		testFrame(FrameFormat.IJV, ScriptType.DML, IoType.JAVA_RDD_STR_IJV, IoType.ANY);
	}

	@Test
	public void testFrameRDD_IJV_DML() {
		testFrame(FrameFormat.IJV, ScriptType.DML, IoType.RDD_STR_IJV, IoType.ANY);
	}

	@Test
	public void testFrameJavaRDD_IJV_DML_OutRddCSV() {
		testFrame(FrameFormat.IJV, ScriptType.DML, IoType.JAVA_RDD_STR_IJV, IoType.RDD_STR_CSV);
	}

	@Test
	public void testFrameFile_CSV_DML() {
		testFrame(FrameFormat.CSV, ScriptType.DML, IoType.FILE, IoType.ANY);
	}

	@Test
	public void testFrameFile_IJV_DML() {
		testFrame(FrameFormat.IJV, ScriptType.DML, IoType.FILE, IoType.ANY);
	}

	@Test
	public void testFrameDataFrame_CSV_DML() {
		testFrame(FrameFormat.CSV, ScriptType.DML, IoType.DATAFRAME, IoType.ANY);
	}

	@Test
	public void testFrameDataFrameOutDataFrame_CSV_DML() {
		testFrame(FrameFormat.CSV, ScriptType.DML, IoType.DATAFRAME, IoType.DATAFRAME);
	}

	public void testFrame(FrameFormat format, ScriptType script_type, IoType inputType, IoType outputType) {

		// System.out.println("MLContextTest - Frame JavaRDD<String> for format: " + format + " Script: " + script_type);

		List<String> listA = new ArrayList<>();
		List<String> listB = new ArrayList<>();
		FrameMetadata fmA = null, fmB = null;
		Script script = null;
		ValueType[] schemaA = { ValueType.INT64, ValueType.STRING, ValueType.FP64, ValueType.BOOLEAN };
		List<ValueType> lschemaA = Arrays.asList(schemaA);
		FrameSchema fschemaA = new FrameSchema(lschemaA);
		ValueType[] schemaB = { ValueType.STRING, ValueType.FP64, ValueType.BOOLEAN };
		List<ValueType> lschemaB = Arrays.asList(schemaB);
		FrameSchema fschemaB = new FrameSchema(lschemaB);

		if (inputType != IoType.FILE) {
			if (format == FrameFormat.CSV) {
				listA.add("1,Str2,3.0,true");
				listA.add("4,Str5,6.0,false");
				listA.add("7,Str8,9.0,true");

				listB.add("Str12,13.0,true");
				listB.add("Str25,26.0,false");

				fmA = new FrameMetadata(FrameFormat.CSV, fschemaA, 3, 4);
				fmB = new FrameMetadata(FrameFormat.CSV, fschemaB, 2, 3);
			} else if (format == FrameFormat.IJV) {
				listA.add("1 1 1");
				listA.add("1 2 Str2");
				listA.add("1 3 3.0");
				listA.add("1 4 true");
				listA.add("2 1 4");
				listA.add("2 2 Str5");
				listA.add("2 3 6.0");
				listA.add("2 4 false");
				listA.add("3 1 7");
				listA.add("3 2 Str8");
				listA.add("3 3 9.0");
				listA.add("3 4 true");

				listB.add("1 1 Str12");
				listB.add("1 2 13.0");
				listB.add("1 3 true");
				listB.add("2 1 Str25");
				listB.add("2 2 26.0");
				listB.add("2 3 false");

				fmA = new FrameMetadata(FrameFormat.IJV, fschemaA, 3, 4);
				fmB = new FrameMetadata(FrameFormat.IJV, fschemaB, 2, 3);
			}
			JavaRDD<String> javaRDDA = sc.parallelize(listA);
			JavaRDD<String> javaRDDB = sc.parallelize(listB);

			if (inputType == IoType.DATAFRAME) {
				JavaRDD<Row> javaRddRowA = FrameRDDConverterUtils.csvToRowRDD(sc, javaRDDA, CSV_DELIM, schemaA);
				JavaRDD<Row> javaRddRowB = FrameRDDConverterUtils.csvToRowRDD(sc, javaRDDB, CSV_DELIM, schemaB);

				// Create DataFrame
				StructType dfSchemaA = FrameRDDConverterUtils.convertFrameSchemaToDFSchema(schemaA, false);
				Dataset<Row> dataFrameA = spark.createDataFrame(javaRddRowA, dfSchemaA);
				StructType dfSchemaB = FrameRDDConverterUtils.convertFrameSchemaToDFSchema(schemaB, false);
				Dataset<Row> dataFrameB = spark.createDataFrame(javaRddRowB, dfSchemaB);
				if (script_type == ScriptType.DML)
					script = dml("A[2:3,2:4]=B;C=A[2:3,2:3]").in("A", dataFrameA, fmA).in("B", dataFrameB, fmB).out("A")
							.out("C");
			} else {
				if (inputType == IoType.JAVA_RDD_STR_CSV || inputType == IoType.JAVA_RDD_STR_IJV) {
					if (script_type == ScriptType.DML)
						script = dml("A[2:3,2:4]=B;C=A[2:3,2:3]").in("A", javaRDDA, fmA).in("B", javaRDDB, fmB).out("A")
								.out("C");
				} else if (inputType == IoType.RDD_STR_CSV || inputType == IoType.RDD_STR_IJV) {
					RDD<String> rddA = JavaRDD.toRDD(javaRDDA);
					RDD<String> rddB = JavaRDD.toRDD(javaRDDB);

					if (script_type == ScriptType.DML)
						script = dml("A[2:3,2:4]=B;C=A[2:3,2:3]").in("A", rddA, fmA).in("B", rddB, fmB).out("A")
								.out("C");
				}

			}

		} else { // Input type is file
			String fileA = null, fileB = null;
			if (format == FrameFormat.CSV) {
				fileA = baseDirectory + File.separator + "FrameA.csv";
				fileB = baseDirectory + File.separator + "FrameB.csv";
			} else if (format == FrameFormat.IJV) {
				fileA = baseDirectory + File.separator + "FrameA.ijv";
				fileB = baseDirectory + File.separator + "FrameB.ijv";
			}

			if (script_type == ScriptType.DML)
				script = dml("A=read($A); B=read($B);A[2:3,2:4]=B;C=A[2:3,2:3];A[1,1]=234").in("$A", fileA, fmA)
						.in("$B", fileB, fmB).out("A").out("C");
		}

		MLResults mlResults = ml.execute(script);
		
		//Validate output schema
		List<ValueType> lschemaOutA = Arrays.asList(mlResults.getFrameObject("A").getSchema());
		List<ValueType> lschemaOutC = Arrays.asList(mlResults.getFrameObject("C").getSchema());

		if(inputType != IoType.RDD_STR_IJV && inputType != IoType.JAVA_RDD_STR_IJV){

			Assert.assertEquals(ValueType.INT64, lschemaOutA.get(0));
			Assert.assertEquals(ValueType.STRING, lschemaOutA.get(1));
			Assert.assertEquals(ValueType.FP64, lschemaOutA.get(2));
			Assert.assertEquals(ValueType.BOOLEAN, lschemaOutA.get(3));
			
			Assert.assertEquals(ValueType.STRING, lschemaOutC.get(0));
			Assert.assertEquals(ValueType.FP64, lschemaOutC.get(1));
		}

		if (outputType == IoType.JAVA_RDD_STR_CSV) {

			JavaRDD<String> javaRDDStringCSVA = mlResults.getJavaRDDStringCSV("A");
			List<String> linesA = javaRDDStringCSVA.collect();
			Assert.assertEquals("1,Str2,3.0,true", linesA.get(0));
			Assert.assertEquals("4,Str12,13.0,true", linesA.get(1));
			Assert.assertEquals("7,Str25,26.0,false", linesA.get(2));

			JavaRDD<String> javaRDDStringCSVC = mlResults.getJavaRDDStringCSV("C");
			List<String> linesC = javaRDDStringCSVC.collect();
			Assert.assertEquals("Str12,13.0", linesC.get(0));
			Assert.assertEquals("Str25,26.0", linesC.get(1));
		} else if (outputType == IoType.JAVA_RDD_STR_IJV) {
			JavaRDD<String> javaRDDStringIJVA = mlResults.getJavaRDDStringIJV("A");
			List<String> linesA = javaRDDStringIJVA.collect();
			Assert.assertEquals("1 1 1", linesA.get(0));
			Assert.assertEquals("1 2 Str2", linesA.get(1));
			Assert.assertEquals("1 3 3.0", linesA.get(2));
			Assert.assertEquals("1 4 true", linesA.get(3));
			Assert.assertEquals("2 1 4", linesA.get(4));
			Assert.assertEquals("2 2 Str12", linesA.get(5));
			Assert.assertEquals("2 3 13.0", linesA.get(6));
			Assert.assertEquals("2 4 true", linesA.get(7));

			JavaRDD<String> javaRDDStringIJVC = mlResults.getJavaRDDStringIJV("C");
			List<String> linesC = javaRDDStringIJVC.collect();
			Assert.assertEquals("1 1 Str12", linesC.get(0));
			Assert.assertEquals("1 2 13.0", linesC.get(1));
			Assert.assertEquals("2 1 Str25", linesC.get(2));
			Assert.assertEquals("2 2 26.0", linesC.get(3));
		} else if (outputType == IoType.RDD_STR_CSV) {
			RDD<String> rddStringCSVA = mlResults.getRDDStringCSV("A");
			Iterator<String> iteratorA = rddStringCSVA.toLocalIterator();
			Assert.assertEquals("1,Str2,3.0,true", iteratorA.next());
			Assert.assertEquals("4,Str12,13.0,true", iteratorA.next());
			Assert.assertEquals("7,Str25,26.0,false", iteratorA.next());

			RDD<String> rddStringCSVC = mlResults.getRDDStringCSV("C");
			Iterator<String> iteratorC = rddStringCSVC.toLocalIterator();
			Assert.assertEquals("Str12,13.0", iteratorC.next());
			Assert.assertEquals("Str25,26.0", iteratorC.next());
		} else if (outputType == IoType.RDD_STR_IJV) {
			RDD<String> rddStringIJVA = mlResults.getRDDStringIJV("A");
			Iterator<String> iteratorA = rddStringIJVA.toLocalIterator();
			Assert.assertEquals("1 1 1", iteratorA.next());
			Assert.assertEquals("1 2 Str2", iteratorA.next());
			Assert.assertEquals("1 3 3.0", iteratorA.next());
			Assert.assertEquals("1 4 true", iteratorA.next());
			Assert.assertEquals("2 1 4", iteratorA.next());
			Assert.assertEquals("2 2 Str12", iteratorA.next());
			Assert.assertEquals("2 3 13.0", iteratorA.next());
			Assert.assertEquals("2 4 true", iteratorA.next());
			Assert.assertEquals("3 1 7", iteratorA.next());
			Assert.assertEquals("3 2 Str25", iteratorA.next());
			Assert.assertEquals("3 3 26.0", iteratorA.next());
			Assert.assertEquals("3 4 false", iteratorA.next());

			RDD<String> rddStringIJVC = mlResults.getRDDStringIJV("C");
			Iterator<String> iteratorC = rddStringIJVC.toLocalIterator();
			Assert.assertEquals("1 1 Str12", iteratorC.next());
			Assert.assertEquals("1 2 13.0", iteratorC.next());
			Assert.assertEquals("2 1 Str25", iteratorC.next());
			Assert.assertEquals("2 2 26.0", iteratorC.next());

		} else if (outputType == IoType.DATAFRAME) {

			Dataset<Row> dataFrameA = mlResults.getDataFrame("A").drop(RDDConverterUtils.DF_ID_COLUMN);
			StructType dfschemaA = dataFrameA.schema(); 
			StructField structTypeA = dfschemaA.apply(0);
			Assert.assertEquals(DataTypes.LongType, structTypeA.dataType());
			structTypeA = dfschemaA.apply(1);
			Assert.assertEquals(DataTypes.StringType, structTypeA.dataType());
			structTypeA = dfschemaA.apply(2);
			Assert.assertEquals(DataTypes.DoubleType, structTypeA.dataType());
			structTypeA = dfschemaA.apply(3);
			Assert.assertEquals(DataTypes.BooleanType, structTypeA.dataType());

			List<Row> listAOut = dataFrameA.collectAsList();

			Row row1 = listAOut.get(0);
			Assert.assertEquals("Mismatch with expected value", Long.valueOf(1), row1.get(0));
			Assert.assertEquals("Mismatch with expected value", "Str2", row1.get(1));
			Assert.assertEquals("Mismatch with expected value", 3.0, row1.get(2));
			Assert.assertEquals("Mismatch with expected value", true, row1.get(3));
			
			Row row2 = listAOut.get(1);
			Assert.assertEquals("Mismatch with expected value", Long.valueOf(4), row2.get(0));
			Assert.assertEquals("Mismatch with expected value", "Str12", row2.get(1));
			Assert.assertEquals("Mismatch with expected value", 13.0, row2.get(2));
			Assert.assertEquals("Mismatch with expected value", true, row2.get(3));

			Dataset<Row> dataFrameC = mlResults.getDataFrame("C").drop(RDDConverterUtils.DF_ID_COLUMN);
			StructType dfschemaC = dataFrameC.schema(); 
			StructField structTypeC = dfschemaC.apply(0);
			Assert.assertEquals(DataTypes.StringType, structTypeC.dataType());
			structTypeC = dfschemaC.apply(1);
			Assert.assertEquals(DataTypes.DoubleType, structTypeC.dataType());
			
			List<Row> listCOut = dataFrameC.collectAsList();

			Row row3 = listCOut.get(0);
			Assert.assertEquals("Mismatch with expected value", "Str12", row3.get(0));
			Assert.assertEquals("Mismatch with expected value", 13.0, row3.get(1));

			Row row4 = listCOut.get(1);
			Assert.assertEquals("Mismatch with expected value", "Str25", row4.get(0));
			Assert.assertEquals("Mismatch with expected value", 26.0, row4.get(1));
		} else {
			String[][] frameA = mlResults.getFrameAs2DStringArray("A");
			Assert.assertEquals("Str2", frameA[0][1]);
			Assert.assertEquals("3.0", frameA[0][2]);
			Assert.assertEquals("13.0", frameA[1][2]);
			Assert.assertEquals("true", frameA[1][3]);
			Assert.assertEquals("Str25", frameA[2][1]);

			String[][] frameC = mlResults.getFrameAs2DStringArray("C");
			Assert.assertEquals("Str12", frameC[0][0]);
			Assert.assertEquals("Str25", frameC[1][0]);
			Assert.assertEquals("13.0", frameC[0][1]);
			Assert.assertEquals("26.0", frameC[1][1]);
		}
	}

	@Test
	public void testOutputFrameDML() {
		// System.out.println("MLContextFrameTest - output frame DML");

		String s = "M = read($Min, data_type='frame', format='csv');";
		String csvFile = baseDirectory + File.separator + "one-two-three-four.csv";
		Script script = dml(s).in("$Min", csvFile).out("M");
		String[][] frame = ml.execute(script).getFrameAs2DStringArray("M");
		Assert.assertEquals("one", frame[0][0]);
		Assert.assertEquals("two", frame[0][1]);
		Assert.assertEquals("three", frame[1][0]);
		Assert.assertEquals("four", frame[1][1]);
	}

	@Test
	public void testInputFrameAndMatrixOutputMatrix() {
		// System.out.println("MLContextFrameTest - input frame and matrix, output matrix");

		List<String> dataA = new ArrayList<>();
		dataA.add("Test1,4.0");
		dataA.add("Test2,5.0");
		dataA.add("Test3,6.0");
		JavaRDD<String> javaRddStringA = sc.parallelize(dataA);
		ValueType[] schema = { ValueType.STRING, ValueType.FP64 };

		List<String> dataB = new ArrayList<>();
		dataB.add("1.0");
		dataB.add("2.0");
		JavaRDD<String> javaRddStringB = sc.parallelize(dataB);

		JavaRDD<Row> javaRddRowA = FrameRDDConverterUtils.csvToRowRDD(sc, javaRddStringA, CSV_DELIM, schema);
		JavaRDD<Row> javaRddRowB = javaRddStringB.map(new CommaSeparatedValueStringToDoubleArrayRow());

		List<StructField> fieldsA = new ArrayList<>();
		fieldsA.add(DataTypes.createStructField("1", DataTypes.StringType, true));
		fieldsA.add(DataTypes.createStructField("2", DataTypes.DoubleType, true));
		StructType schemaA = DataTypes.createStructType(fieldsA);
		Dataset<Row> dataFrameA = spark.createDataFrame(javaRddRowA, schemaA);

		List<StructField> fieldsB = new ArrayList<>();
		fieldsB.add(DataTypes.createStructField("1", DataTypes.DoubleType, true));
		StructType schemaB = DataTypes.createStructType(fieldsB);
		Dataset<Row> dataFrameB = spark.createDataFrame(javaRddRowB, schemaB);

		String dmlString = "[tA, tAM] = transformencode (target = A, spec = \"{ids: true ,recode: [ 1, 2 ]}\");\n"
				+ "C = tA %*% B;\n" + "M = s * C;";

		Script script = dml(dmlString)
				.in("A", dataFrameA,
						new FrameMetadata(FrameFormat.CSV, dataFrameA.count(), (long) dataFrameA.columns().length))
				.in("B", dataFrameB,
						new MatrixMetadata(MatrixFormat.CSV, dataFrameB.count(), (long) dataFrameB.columns().length))
				.in("s", 2).out("M");
		MLResults results = ml.execute(script);
		double[][] matrix = results.getMatrixAs2DDoubleArray("M");
		Assert.assertEquals(6.0, matrix[0][0], 0.0);
		Assert.assertEquals(12.0, matrix[1][0], 0.0);
		Assert.assertEquals(18.0, matrix[2][0], 0.0);
	}

	@Test
	public void testInputFrameAndMatrixOutputMatrixAndFrame() {
		// System.out.println("MLContextFrameTest - input frame and matrix, output matrix and frame");
		
		Row[] rowsA = {RowFactory.create("Doc1", "Feat1", 10), RowFactory.create("Doc1", "Feat2", 20), RowFactory.create("Doc2", "Feat1", 31)};

		JavaRDD<Row> javaRddRowA = sc. parallelize( Arrays.asList(rowsA)); 

		List<StructField> fieldsA = new ArrayList<>();
		fieldsA.add(DataTypes.createStructField("myID", DataTypes.StringType, true));
		fieldsA.add(DataTypes.createStructField("FeatureName", DataTypes.StringType, true));
		fieldsA.add(DataTypes.createStructField("FeatureValue", DataTypes.IntegerType, true));
		StructType schemaA = DataTypes.createStructType(fieldsA);
		Dataset<Row> dataFrameA = spark.createDataFrame(javaRddRowA, schemaA);

		String dmlString = "[tA, tAM] = transformencode (target = A, spec = \"{ids: false ,recode: [ myID, FeatureName ]}\");";

		Script script = dml(dmlString)
				.in("A", dataFrameA,
						new FrameMetadata(FrameFormat.CSV, dataFrameA.count(), (long) dataFrameA.columns().length))
				.out("tA").out("tAM");
		MLResults results = ml.execute(script);

		double[][] matrixtA = results.getMatrixAs2DDoubleArray("tA");
		Assert.assertEquals(10.0, matrixtA[0][2], 0.0);
		Assert.assertEquals(20.0, matrixtA[1][2], 0.0);
		Assert.assertEquals(31.0, matrixtA[2][2], 0.0);

		Dataset<Row> dataFrame_tA = results.getMatrix("tA").toDF();
		// System.out.println("Number of matrix tA rows = " + dataFrame_tA.count());
		dataFrame_tA.printSchema();
		dataFrame_tA.show();
		
		Dataset<Row> dataFrame_tAM = results.getFrame("tAM").toDF();
		// System.out.println("Number of frame tAM rows = " + dataFrame_tAM.count());
		dataFrame_tAM.printSchema();
		dataFrame_tAM.show();
	}

	@Test
	public void testTransform() {
		// System.out.println("MLContextFrameTest - transform");
		
		Row[] rowsA = {RowFactory.create("\"`@(\"(!&",2,"20news-bydate-train/comp.os.ms-windows.misc/9979"),
				RowFactory.create("\"`@(\"\"(!&\"",3,"20news-bydate-train/comp.os.ms-windows.misc/9979")};

		JavaRDD<Row> javaRddRowA = sc. parallelize( Arrays.asList(rowsA)); 

		List<StructField> fieldsA = new ArrayList<>();
		fieldsA.add(DataTypes.createStructField("featureName", DataTypes.StringType, true));
		fieldsA.add(DataTypes.createStructField("featureValue", DataTypes.IntegerType, true));
		fieldsA.add(DataTypes.createStructField("id", DataTypes.StringType, true));
		StructType schemaA = DataTypes.createStructType(fieldsA);
		Dataset<Row> dataFrameA = spark.createDataFrame(javaRddRowA, schemaA);

		String dmlString = "[tA, tAM] = transformencode (target = A, spec = \"{ids: false ,recode: [ featureName, id ]}\");";

		Script script = dml(dmlString)
				.in("A", dataFrameA,
						new FrameMetadata(FrameFormat.CSV, dataFrameA.count(), (long) dataFrameA.columns().length))
				.out("tA").out("tAM");
		// ml.setExplain(true);
		// ml.setExplainLevel(ExplainLevel.RECOMPILE_HOPS);
		MLResults results = ml.execute(script);

		double[][] matrixtA = results.getMatrixAs2DDoubleArray("tA");
		Assert.assertEquals(1.0, matrixtA[0][2], 0.0);

		Dataset<Row> dataFrame_tA = results.getMatrix("tA").toDF();
		// System.out.println("Number of matrix tA rows = " + dataFrame_tA.count());
		dataFrame_tA.printSchema();
		dataFrame_tA.show();
		
		Dataset<Row> dataFrame_tAM = results.getFrame("tAM").toDF();
		// System.out.println("Number of frame tAM rows = " + dataFrame_tAM.count());
		dataFrame_tAM.printSchema();
		dataFrame_tAM.show();
	}
}
