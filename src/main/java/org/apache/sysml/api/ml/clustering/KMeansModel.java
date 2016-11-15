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

package org.apache.sysml.api.ml.clustering;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.functions.ConvertRowToDouble;

import scala.Tuple2;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

public class KMeansModel extends Model<KMeansModel> {

	private static final long serialVersionUID = 3975281531186536515L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();
	private String featuresCol = "";

	public KMeansModel(
			HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results,
			SparkContext sc,
			HashMap<String, String> cmdLineParams,
			String featuresCol) {
		this.results = results;
		this.sc = sc;
		this.cmdLineParams = cmdLineParams;
		this.featuresCol = featuresCol;
	}

	@Override
	public String uid() {
		return Long.toString(KMeansModel.serialVersionUID);
	}

	@Override
	public KMeansModel copy(ParamMap arg0) {
		return new KMeansModel(results, sc, cmdLineParams, getFeaturesCol());
	}

	public String getFeaturesCol() {
		return featuresCol;
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			StructType outputSchema = SchemaUtils.appendColumn(dataset.schema(),
					DataTypes.createStructField("prediction", DataTypes.DoubleType, true));

			MLContext ml = new MLContext(sc);

			MatrixCharacteristics mcXin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
			Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
					dataset,
					mcXin,
					true,
					true);

			ClassTag<String> strClassTag = ClassTag$.MODULE$.apply(String.class);
			JavaRDD<String> emptyRDD = sc.emptyRDD(strClassTag).toJavaRDD();

			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("spY", emptyRDD, "csv");
			ml.registerOutput("prY");

			cmdLineParams.put("X", "X");
			cmdLineParams.put("C", "C");
			cmdLineParams.put("prY", "prY");

			for (Map.Entry<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> entry : results
					.entrySet())
				ml.registerInput(entry.getKey(), entry.getValue()._1, entry.getValue()._2);

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "Kmeans-predict.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "Kmeans-predict.dml";
			MLOutput out = ml.execute(dmlFilePath, cmdLineParams);

			List<Row> rowList = dataset.rdd().toJavaRDD().collect();
			List<Row> resultList = new ArrayList<Row>();
			List<Double> pred = out.getStringRDD("prY", "text").map(new ConvertRowToDouble())
					.collect();

			for (long i = 0; i < pred.size(); i++) {
				int colCount = outputSchema.size() - 1;

				if (colCount == 1)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0),
							pred.get((int) i)));
				else if (colCount == 2)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0),
							rowList.get((int) i).get(1),
							pred.get((int) i)));
				else if (colCount == 3)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0),
							rowList.get((int) i).get(1),
							rowList.get((int) i).get(2),
							pred.get((int) i)));
				else if (colCount == 4)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0),
							rowList.get((int) i).get(1),
							rowList.get((int) i).get(2),
							rowList.get((int) i).get(3),
							pred.get((int) i)));
				else if (colCount == 5)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0),
							rowList.get((int) i).get(1),
							rowList.get((int) i).get(2),
							rowList.get((int) i).get(3),
							rowList.get((int) i).get(4),
							pred.get((int) i)));
			}

			@SuppressWarnings("resource")
			JavaRDD<Row> row = (new JavaSparkContext(sc)).parallelize(resultList);

			return dataset.sqlContext().createDataFrame(row, outputSchema);
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public StructType transformSchema(StructType arg0) {
		return null;
	}
}
