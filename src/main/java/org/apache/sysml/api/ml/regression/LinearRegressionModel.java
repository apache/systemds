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

package org.apache.sysml.api.ml.regression;

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
import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.RegressionModel;
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
import org.apache.sysml.api.ml.param.LinearRegressionModelParams;

import scala.Tuple2;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

public class LinearRegressionModel extends RegressionModel<Vector, LinearRegressionModel>
		implements LinearRegressionModelParams {

	private static final long serialVersionUID = 3232209697514247658L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, String> params = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();

	private String labelCol = "";

	private IntParam dfam = new IntParam(this,
			"dfam",
			"GLM distribution family: 1 = Power, 2 = Binomial, 3 = Multinomial Logit");
	private DoubleParam vpow = new DoubleParam(this, "vpow", "Power for Variance");
	private IntParam link = new IntParam(this,
			"link",
			"Link function code: 0 = canonical (depends on distribution), 1 = Power, 2 = Logit, 3 = Probit, 4 = Cloglog, 5 = Cauchit; ignored if Multinomial");
	private DoubleParam lpow = new DoubleParam(this, "lpow", "Power for Link function");
	private DoubleParam disp = new DoubleParam(this, "disp", "Dispersion Value");

	public LinearRegressionModel(
			HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results,
			SparkContext sc,
			HashMap<String, String> params,
			String featuresCol,
			String labelCol) {
		this.results = results;
		this.sc = sc;
		this.params = params;
		this.labelCol = labelCol;

		cmdLineParams.put("dfam", params.get("dfam"));
		cmdLineParams.put("vpow", params.get("vpow"));
		cmdLineParams.put("link", params.get("link"));
		cmdLineParams.put("lpow", params.get("lpow"));
		cmdLineParams.put("disp", params.get("disp"));

		setDefault(dfam(), Integer.parseInt(params.get(dfam.name())));
		setDefault(vpow(), Double.parseDouble(params.get(vpow.name())));
		setDefault(link(), Integer.parseInt(params.get(link.name())));
		setDefault(lpow(), Double.parseDouble(params.get(lpow.name())));
		setDefault(disp(), Double.parseDouble(params.get(disp.name())));

		setFeaturesCol(featuresCol);
	}

	@Override
	public LinearRegressionModel copy(ParamMap paramMap) {
		return new LinearRegressionModel(results, sc, params, getFeaturesCol(), labelCol);
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public StructType validateAndTransformSchema(StructType arg0, boolean arg1, DataType arg2) {
		return null;
	}

	@Override
	public double predict(Vector arg0) {
		return 0.0;
	}

	@Override
	public int getDfam() {
		return Integer.parseInt(cmdLineParams.get(dfam.name()));
	}

	@Override
	public IntParam dfam() {
		return dfam;
	}

	@Override
	public double getVpow() {
		return Double.parseDouble(cmdLineParams.get(vpow.name()));
	}

	@Override
	public DoubleParam vpow() {
		return vpow;
	}

	@Override
	public int getLink() {
		return Integer.parseInt(cmdLineParams.get(link.name()));
	}

	@Override
	public IntParam link() {
		return link;
	}

	@Override
	public double getLpow() {
		return Double.parseDouble(cmdLineParams.get(vpow.name()));
	}

	@Override
	public DoubleParam lpow() {
		return lpow;
	}

	@Override
	public double getDisp() {
		return Double.parseDouble(cmdLineParams.get(disp.name()));
	}

	@Override
	public DoubleParam disp() {
		return disp;
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			StructType outputSchema = SchemaUtils.appendColumn(dataset.schema(),
					DataTypes.createStructField("prediction", DataTypes.DoubleType, true));
			MLContext ml = new MLContext(sc);

			MatrixCharacteristics mcXin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
			Xin = RDDConverterUtils.dataFrameToBinaryBlock(
					new JavaSparkContext(this.sc),
					dataset.select(getFeaturesCol()),
					mcXin,
					false,
					true);

			ClassTag<String> strClassTag = ClassTag$.MODULE$.apply(String.class);
			JavaRDD<String> emptyRDD = sc.emptyRDD(strClassTag).toJavaRDD();

			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("Y", emptyRDD, "csv");

			for (Map.Entry<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> entry : results
					.entrySet())
				ml.registerInput(entry.getKey(), entry.getValue()._1, entry.getValue()._2);

			ml.registerOutput("means");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			cmdLineParams.put("X", " ");
			cmdLineParams.put("B", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "GLM-predict.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "GLM-predict.dml";
			MLOutput out = ml.execute(dmlFilePath, cmdLineParams);

			List<Row> rowList = dataset.rdd().toJavaRDD().collect();
			List<Row> resultList = new ArrayList<Row>();
			List<Double> pred = out.getStringRDD("means", "text").map(new ConvertRowToDouble())
					.collect();
			long nRow = out.getMatrixCharacteristics("means").getRows();
			int colCount = outputSchema.size();

			for (int i = 0; i < nRow; i++) {
				Object[] rowArray = new Object[colCount];
				int j = 0;

				for (j = 0; j < colCount - 1; j++)
					rowArray[j] = rowList.get(i).get(j);

				rowArray[j++] = pred.get(i);

				resultList.add(RowFactory.create(rowArray));
			}

			@SuppressWarnings("resource")
			JavaRDD<Row> row = (new JavaSparkContext(sc)).parallelize(resultList);

			return dataset.sqlContext().createDataFrame(row, outputSchema);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return dataset;
	}
}
