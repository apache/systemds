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

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
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
import org.apache.sysml.api.ml.param.LogisticRegressionModelParams;

import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

public class LogisticRegressionModel extends PredictionModel<Vector, LogisticRegressionModel>
		implements LogisticRegressionModelParams {

	private static final long serialVersionUID = -6464693773946415027L;

	private SparkContext sc = null;
	private JavaPairRDD<MatrixIndexes, MatrixBlock> b_out;
	private MatrixCharacteristics b_outMC;
	private HashMap<String, String> params = new HashMap<String, String>();
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();

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

	public LogisticRegressionModel(JavaPairRDD<MatrixIndexes, MatrixBlock> b_out2,
			MatrixCharacteristics b_outMC, String featuresCol, String labelCol, SparkContext sc,
			HashMap<String, String> params) {
		this.b_out = b_out2;
		this.b_outMC = b_outMC;
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
	public LogisticRegressionModel copy(ParamMap paramMap) {
		return new LogisticRegressionModel(b_out, b_outMC, getFeaturesCol(), labelCol, sc, params);
	}

	@Override
	public String uid() {
		return Long.toString(LogisticRegressionModel.serialVersionUID);
	}

	@Override
	public StructType validateAndTransformSchema(StructType arg0, boolean arg1, DataType arg2) {
		return null;
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
	public double predict(Vector arg0) {
		System.out.println("Not supported yet.");
		return 0;
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			StructType outputSchema = SchemaUtils.appendColumn(dataset.schema(),
					DataTypes.createStructField("probability", new VectorUDT(), true));
			outputSchema = SchemaUtils.appendColumn(outputSchema,
					DataTypes.createStructField("prediction", DataTypes.DoubleType, true));
			MLContext ml = new MLContext(sc);

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

			MatrixCharacteristics mcYin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Yin;
			ArrayList<String> cols = new ArrayList<String>();
			cols.add(this.getLabelCol());

			ClassTag<String> strClassTag = ClassTag$.MODULE$.apply(String.class);
			JavaRDD<String> emptyRDD = sc.emptyRDD(strClassTag).toJavaRDD();

			if (labelExists) {
				Yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
						dataset.select(getLabelCol()),
						mcYin,
						false,
						false);

				ml.registerInput("Y", Yin, mcYin);
			} else
				ml.registerInput("Y", emptyRDD, "csv");

			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("B_full", b_out, b_outMC);
			ml.registerOutput("means");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in GLM-predict.dml
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
			long nCol = out.getMatrixCharacteristics("means").getCols();
			int colCount = outputSchema.size();

			for (int i = 0; i < nRow; i++) {
				Object[] rowArray = new Object[colCount];
				double[] vec = new double[(int) nCol];
				int j = 0;
				double labelVal = -Double.MAX_VALUE;
				double argmax = -Double.MAX_VALUE;

				for (j = 0; j < colCount - 2; j++)
					rowArray[j] = rowList.get(i).get(j);

				for (int k = 0; k < nCol; k++) {
					double currVal = pred.get(2 * i + k);
					vec[k] = currVal;

					if (currVal > labelVal) {
						labelVal = currVal;
						argmax = (double) (k + 1);
					}
				}

				rowArray[j++] = Vectors.dense(vec);
				rowArray[j++] = argmax;

				resultList.add(RowFactory.create(rowArray));
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
}