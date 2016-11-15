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

package org.apache.sysml.api.ml.classification;

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
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataType;
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

public class RandomForestClassificationModel extends PredictionModel<Vector, RandomForestClassificationModel> {

	private static final long serialVersionUID = 6866405690919453132L;

	private SparkContext sc = null;
	private DataFrame rDF = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();
	private String labelCol = "", rCol = "";

	public RandomForestClassificationModel(
			HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results,
			SparkContext sc, DataFrame rDF, String featuresCol, String labelCol, String rCol) {
		this.results = results;
		this.sc = sc;
		this.labelCol = labelCol;
		this.rCol = rCol;

		setFeaturesCol(featuresCol);
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public RandomForestClassificationModel copy(ParamMap paramMap) {
		return new RandomForestClassificationModel(
				results, sc, rDF, getFeaturesCol(), labelCol, rCol);
	}

	@Override
	public StructType validateAndTransformSchema(StructType arg0, boolean arg1, DataType arg2) {
		return null;
	}

	@Override
	public double predict(Vector arg0) {
		System.out.println("predict is currently not supported.");
		return 0;
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			StructType outputSchema = SchemaUtils.appendColumn(dataset.schema(), DataTypes
					.createStructField("prediction", DataTypes.DoubleType, true));
			MLContext ml = new MLContext(
					sc);

			MatrixCharacteristics mcXin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
			Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
					dataset.select(getFeaturesCol()),
					mcXin,
					false,
					true);

			boolean labelExists = false;

			for (String s : dataset.columns())
				if (s.equals(labelCol))
					labelExists = true;

			ClassTag<String> strClassTag = ClassTag$.MODULE$.apply(String.class);
			JavaRDD<String> emptyRDD = sc.emptyRDD(strClassTag).toJavaRDD();

			if (labelExists) {
				MatrixCharacteristics mcYin = new MatrixCharacteristics();
				JavaPairRDD<MatrixIndexes, MatrixBlock> Yin;

				Yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
						dataset.select(labelCol),
						mcYin,
						false,
						true);

				ml.registerInput("Y_dummy", Yin, mcYin);
				cmdLineParams.put("Y", "Y");
			} else {
				ml.registerInput("Y_dummy", emptyRDD, "csv");
				cmdLineParams.put("Y", " ");
			}

			MatrixCharacteristics mcRin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Rin = null;
			if (rDF != null) {
				try {
					Rin = RDDConverterUtils.dataFrameToBinaryBlock(
							new JavaSparkContext(sc),
							rDF.select(rCol),
							mcXin,
							false,
							true);
					ml.registerInput("R", Rin, mcRin);
					cmdLineParams.put("R", "R");
				} catch (DMLRuntimeException e1) {
					e1.printStackTrace();
					return null;
				}
			}

			for (Map.Entry<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> entry : results
					.entrySet())
				ml.registerInput(entry.getKey(), entry.getValue()._1, entry.getValue()._2);

			ml.registerInput("X", Xin, mcXin);
			ml.registerOutput("Y_predicted");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			cmdLineParams.put("X", " ");
			cmdLineParams.put("M", " ");
			cmdLineParams.put("P", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "random-forest-predict.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "random-forest-predict.dml";
			MLOutput out = ml.execute(dmlFilePath, cmdLineParams);

			List<Row> rowList = dataset.rdd().toJavaRDD().collect();
			List<Row> resultList = new ArrayList<Row>();
			List<Double> pred = out.getStringRDD("Y_predicted", "text").map(new ConvertRowToDouble())
					.collect();

			for (long i = 0; i < pred.size(); i++) {
				int colCount = outputSchema.size() - 1;

				if (colCount == 1)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0), pred.get(
							(int) i)));
				else if (colCount == 2)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0), rowList.get(
							(int) i).get(1), pred.get((int) i)));
				else if (colCount == 3)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0), rowList.get(
							(int) i).get(1), rowList.get((int) i).get(2), pred.get(
									(int) i)));
				else if (colCount == 4)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0),
							rowList.get(
									(int) i).get(1),
							rowList.get((int) i).get(2),
							rowList.get(
									(int) i).get(3),
							pred.get((int) i)));
				else if (colCount == 5)
					resultList.add(RowFactory.create(rowList.get((int) i).get(0),
							rowList.get(
									(int) i).get(1),
							rowList.get((int) i).get(2),
							rowList.get(
									(int) i).get(3),
							rowList.get((int) i).get(
									4),
							pred.get((int) i)));
			}

			@SuppressWarnings("resource")
			JavaRDD<Row> row = (new JavaSparkContext(
					sc)).parallelize(resultList);

			return dataset.sqlContext().createDataFrame(row, outputSchema);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.err.println("The trained model does not generate anything and cannot be transformed.");
		return dataset;
	}
}