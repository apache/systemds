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

package org.apache.sysds.test.functions.io.parquet;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.sql.Timestamp;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.test.AutomatedTestBase;

class ParquetTestUtils {

	static class ParquetMetadataInfo {
		String[] names;
		ValueType[] schema;
		long rlen;
		long clen;
	}

	static ParquetMetadataInfo inferMetadata(String fname) throws IOException {
		Configuration conf = ConfigurationManager.getCachedJobConf();
		Path path = new Path(fname);

		ParquetMetadata metadata;
		try (ParquetFileReader r = ParquetFileReader.open(HadoopInputFile.fromPath(path, conf))) {
			metadata = r.getFooter();
		}
		MessageType parquetSchema = metadata.getFileMetaData().getSchema();

		int fieldCount = parquetSchema.getFieldCount();
		String[] names = new String[fieldCount];
		ValueType[] schema = new ValueType[fieldCount];

		for (int i = 0; i < fieldCount; i++) {
			names[i] = parquetSchema.getFieldName(i);
			PrimitiveType.PrimitiveTypeName type = parquetSchema.getType(i).asPrimitiveType().getPrimitiveTypeName();
			switch (type) {
				case INT32:    schema[i] = ValueType.INT32;    break;
				case INT64:    schema[i] = ValueType.INT64;    break;
				case FLOAT:    schema[i] = ValueType.FP32;     break;
				case DOUBLE:   schema[i] = ValueType.FP64;     break;
				case BOOLEAN:  schema[i] = ValueType.BOOLEAN;  break;
				case BINARY:   schema[i] = ValueType.STRING;   break;
				case INT96:    schema[i] = ValueType.INT64;    break;
				default:
					throw new IOException("Unsupported parquet type: " + type + " in column " + names[i]);
			}
		}

		long rlen = 0;
		for (BlockMetaData block : metadata.getBlocks())
			rlen += block.getRowCount();

		ParquetMetadataInfo info = new ParquetMetadataInfo();
		info.names = names;
		info.schema = schema;
		info.rlen = rlen;
		info.clen = fieldCount;
		return info;
	}

	/**
	 * Generates the public test files (userdata1, alltypes_plain, all) with Spark's
	 * DataFrameWriter. userdata1 and alltypes_plain each include a TimestampType column,
	 * which Spark 3.5 encodes as INT96 by default.
	 *
	 * @param outDir directory the generated files are written into
	 * @return map from file name (e.g. "userdata1") to its generated file path
	 */
	static Map<String, String> generatePublicTestFiles(File outDir) throws Exception {
		SparkSession spark = AutomatedTestBase.createSystemDSSparkSession("parquet-test-files", "local[1]");

		Map<String, String> files = new LinkedHashMap<>();
		files.put("userdata1", writeTestFile(spark, outDir, "userdata1", userdata1Rows(), userdata1Schema()));
		files.put("alltypes_plain", writeTestFile(spark, outDir, "alltypes_plain", alltypesPlainRows(), alltypesPlainSchema()));
		files.put("all", writeTestFile(spark, outDir, "all", allRows(), allSchema()));
		return files;
	}

	private static StructType userdata1Schema() {
		return new StructType(new StructField[] {
			new StructField("registration_dttm", DataTypes.TimestampType, true, Metadata.empty()),
			new StructField("id", DataTypes.IntegerType, true, Metadata.empty()),
			new StructField("first_name", DataTypes.StringType, true, Metadata.empty()),
			new StructField("salary", DataTypes.DoubleType, true, Metadata.empty()),
		});
	}

	private static List<Row> userdata1Rows() {
		return Arrays.asList(
			RowFactory.create(Timestamp.valueOf("2016-02-03 07:55:29"), 1, "Amanda", 49756.53),
			RowFactory.create(Timestamp.valueOf("2016-02-03 17:04:03"), 2, "Albert", 150280.17),
			RowFactory.create(Timestamp.valueOf("2016-02-03 01:09:31"), 3, "Evelyn", 144972.51),
			RowFactory.create((Timestamp) null, 4, "Denise", 90263.05),
			RowFactory.create(Timestamp.valueOf("2016-02-03 05:07:25"), 5, "Carlos", 75500.34)
		);
	}

	private static StructType alltypesPlainSchema() {
		return new StructType(new StructField[] {
			new StructField("id", DataTypes.IntegerType, true, Metadata.empty()),
			new StructField("bool_col", DataTypes.BooleanType, true, Metadata.empty()),
			new StructField("tinyint_col", DataTypes.IntegerType, true, Metadata.empty()),
			new StructField("smallint_col", DataTypes.IntegerType, true, Metadata.empty()),
			new StructField("bigint_col", DataTypes.LongType, true, Metadata.empty()),
			new StructField("float_col", DataTypes.FloatType, true, Metadata.empty()),
			new StructField("double_col", DataTypes.DoubleType, true, Metadata.empty()),
			new StructField("date_string_col", DataTypes.StringType, true, Metadata.empty()),
			new StructField("string_col", DataTypes.StringType, true, Metadata.empty()),
			new StructField("timestamp_col", DataTypes.TimestampType, true, Metadata.empty()),
		});
	}

	private static List<Row> alltypesPlainRows() {
		return Arrays.asList(
			RowFactory.create(1, true,  1, 10,  100L, 1.5f, 2.25,  "03/01/09", "row-1", Timestamp.valueOf("2009-03-01 00:00:00")),
			RowFactory.create(2, false, 2, 20,  200L, 2.5f, 4.5,   "03/02/09", "row-2", Timestamp.valueOf("2009-03-02 05:15:30")),
			RowFactory.create(3, true,  3, 30,  300L, 3.5f, 6.75,  "03/03/09", "row-3", Timestamp.valueOf("2009-03-03 10:30:00")),
			RowFactory.create(4, false, 4, 40,  400L, 4.5f, 9.0,   "03/04/09", "row-4", Timestamp.valueOf("2009-03-04 15:45:15")),
			RowFactory.create(5, true,  5, 50,  500L, 5.5f, 11.25, "03/05/09", "row-5", Timestamp.valueOf("2009-03-05 21:00:45")),
			RowFactory.create(6, false, 6, 60,  600L, 6.5f, 13.5,  "03/06/09", "row-6", Timestamp.valueOf("2009-03-06 02:20:10")),
			RowFactory.create(7, true,  7, 70,  700L, 7.5f, 15.75, "03/07/09", "row-7", Timestamp.valueOf("2009-03-07 08:35:50")),
			RowFactory.create(8, false, 8, 80,  800L, 8.5f, 18.0,  "03/08/09", "row-8", Timestamp.valueOf("2009-03-08 13:50:25"))
		);
	}

	private static StructType allSchema() {
		return new StructType(new StructField[] {
			new StructField("PassengerId", DataTypes.IntegerType, true, Metadata.empty()),
			new StructField("Survived", DataTypes.IntegerType, true, Metadata.empty()),
			new StructField("Pclass", DataTypes.IntegerType, true, Metadata.empty()),
			new StructField("Name", DataTypes.StringType, true, Metadata.empty()),
			new StructField("Sex", DataTypes.StringType, true, Metadata.empty()),
			new StructField("Age", DataTypes.DoubleType, true, Metadata.empty()),
			new StructField("Fare", DataTypes.DoubleType, true, Metadata.empty()),
			new StructField("Embarked", DataTypes.StringType, true, Metadata.empty()),
		});
	}

	private static List<Row> allRows() {
		return Arrays.asList(
			RowFactory.create(1, 0, 3, "Braund, Mr. Owen Harris", "male", 22.0, 7.25, "S"),
			RowFactory.create(2, 1, 1, "Cumings, Mrs. John Bradley", "female", 38.0, 71.2833, "C"),
			RowFactory.create(3, 1, 3, "Heikkinen, Miss. Laina", "female", 26.0, 7.925, "S"),
			RowFactory.create(4, 1, 1, "Futrelle, Mrs. Jacques Heath", "female", 35.0, 53.1, "S"),
			RowFactory.create(5, 0, 3, "Allen, Mr. William Henry", "male", null, 8.05, "S"),
			RowFactory.create(6, 0, 3, "Moran, Mr. James", "male", null, 8.4583, "Q"),
			RowFactory.create(7, 0, 1, "McCarthy, Mr. Timothy J", "male", 54.0, 51.8625, "S"),
			RowFactory.create(8, 0, 3, "Palsson, Master. Gosta Leonard", "male", 2.0, 21.075, "S")
		);
	}

	// Spark writes a directory of part files, so we force one partition and rename it to a single file.
	private static String writeTestFile(SparkSession spark, File outDir, String name, List<Row> rows, StructType schema) throws IOException {
		Dataset<Row> df = spark.createDataFrame(rows, schema);
		File tmpDir = new File(outDir, "_tmp_" + name);
		df.coalesce(1).write().mode(SaveMode.Overwrite).parquet(tmpDir.getPath());

		File[] parts = tmpDir.listFiles((d, n) -> n.startsWith("part-") && n.endsWith(".parquet"));
		if (parts == null || parts.length != 1)
			throw new IOException("expected exactly 1 part file in " + tmpDir);

		File dest = new File(outDir, name + ".parquet");
		Files.copy(parts[0].toPath(), dest.toPath(), StandardCopyOption.REPLACE_EXISTING);
		deleteRecursive(tmpDir);
		return dest.getPath();
	}

	private static void deleteRecursive(File f) {
		File[] children = f.listFiles();
		if (children != null)
			for (File c : children)
				deleteRecursive(c);
		f.delete();
	}
}
