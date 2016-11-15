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

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Predictor;
import org.apache.spark.ml.param.DoubleParam;
//import org.apache.spark.ml.param.IntArrayParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.LongParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.param.CoxRegressionParams;

import scala.Tuple2;
import scala.collection.Seq;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

public class CoxRegression extends Predictor<Vector, CoxRegression, CoxRegressionModel> implements
		CoxRegressionParams {

	private static final long serialVersionUID = 4140518679911801960L;

	private SparkContext sc = null;
	private DataFrame rDF = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();

	private DoubleParam alpha = new DoubleParam(
			this, "alpha", "Parameter to compute a 100*(1-alpha)% "
					+ "confidence interval for the betas");
	private DoubleParam tol = new DoubleParam(
			this, "tol", "Value of tolerance (epsilon)");
	private IntParam maxOuterIter = new IntParam(
			this, "moi", "Maximum number of outer (Newton) iterations");
	private IntParam maxInnerIter = new IntParam(
			this, "mii", "Maximum number of inner (conjugate gradient) " + "iterations");
	private IntParam featureIndicesRangeStart = new IntParam(
			this, "fiRangeStart", "Starting index of the " + "feature column range");
	private IntParam featureIndicesRangeEnd = new IntParam(
			this, "fiRangeEnd", "Starting index of the feature" + " column range");
//	private IntArrayParam featureIndicesArray = new IntArrayParam(
//			this, "fiArray", "A list of feature " + "columns");
	private LongParam timestampIndex = new LongParam(
			this, "timestampIndex", "Index of the timestamp in the feature " + "vector");
	private LongParam eventIndex = new LongParam(
			this, "eventIndex", "Index of the timestamp in the feature vector");

	private Param<String> teCol = new Param<String>(
			this, "teCol", "Name of the timestamp and event column");
	private Param<String> fCol = new Param<String>(
			this, "fCol", "Name of the column that has the feature vector " + "indices");
	private Param<String> rCol = new Param<String>(
			this, "rCol", "Name of the column that has the indices of the "
					+ "categorical values in the feature vectors");

	private int start, end;
	private List<Integer> arr;

	public CoxRegression(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameter(0.05f, 0.000001f, 100, 0);
	}

	public CoxRegression(SparkContext sc, DataFrame rDF) throws DMLRuntimeException {
		this.sc = sc;
		this.rDF = rDF;
		setAllParameter(0.05f, 0.000001f, 100, 0);
	}

	public CoxRegression(SparkContext sc, double alpha, double tol, int maxOuterIter, int maxInnerIter)
			throws DMLRuntimeException {
		this.sc = sc;
		setAllParameter(alpha, tol, maxOuterIter, maxInnerIter);
	}

	public CoxRegression(SparkContext sc, DataFrame rDF, double alpha, double tol, int maxOuterIter,
			int maxInnerIter) throws DMLRuntimeException {
		this.sc = sc;
		this.rDF = rDF;
		setAllParameter(alpha, tol, maxOuterIter, maxInnerIter);
	}

	private void setAllParameter(double alpha, double tol, int maxOuterIter, int maxInnerIter) {
		setDefault(alpha(), alpha);
		cmdLineParams.put(this.alpha.name(), Double.toString(alpha));
		setDefault(tol(), tol);
		cmdLineParams.put(this.tol.name(), Double.toString(tol));
		setDefault(maxOuterIter(), maxOuterIter);
		cmdLineParams.put(this.maxOuterIter.name(), Integer.toString(maxOuterIter));
		setDefault(maxInnerIter(), maxInnerIter);
		cmdLineParams.put(this.maxInnerIter.name(), Integer.toString(maxInnerIter));
		setTECol("tsAndEventIndices");
		setFCol("featureIndices");
		setRCol("categoricalIndices");
		start = -1;
		end = -1;
		arr = new ArrayList<Integer>();
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public CoxRegression copy(ParamMap paramMap) {
		try {
			String strAlpha = paramMap.getOrElse(alpha, getAlpha()).toString();
			String strTol = paramMap.getOrElse(tol, getTol()).toString();
			String strMoi = paramMap.getOrElse(maxOuterIter, getMaxOuterIter()).toString();
			String strMii = paramMap.getOrElse(maxInnerIter, getMaxInnerIter()).toString();

			CoxRegression cr = new CoxRegression(
					sc,
					rDF,
					Double.parseDouble(strAlpha),
					Double.parseDouble(strTol),
					Integer
							.parseInt(strMoi),
					Integer.parseInt(strMii));

			cr.cmdLineParams.put(alpha.name(), strAlpha);
			cr.cmdLineParams.put(tol.name(), strTol);
			cr.cmdLineParams.put(maxOuterIter.name(), strMoi);
			cr.cmdLineParams.put(maxInnerIter.name(), strMii);
			cr.setFeaturesCol(getFeaturesCol());
			cr.setLabelCol(getLabelCol());
			cr.setTECol(getTECol());
			cr.setFCol(getFCol());
			cr.setRCol(getRCol());
			cr.setTimestampIndex(getTimestampIndex());
			cr.setEventIndex(getEventIndex());
			cr.start = start;
			cr.end = end;
			cr.arr = arr;

			return cr;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}

		return null;
	}

	@Override
	public StructType validateAndTransformSchema(StructType arg0, boolean arg1, DataType arg2) {
		return null;
	}

	@Override
	public double getAlpha() {
		return Double.parseDouble(cmdLineParams.get(alpha.name()));
	}

	@Override
	public DoubleParam alpha() {
		return alpha;
	}

	public CoxRegression setAlpha(double value) {
		cmdLineParams.put(alpha.name(), Double.toString(value));
		return (CoxRegression) setDefault(alpha, value);
	}

	@Override
	public double getTol() {
		return Double.parseDouble(cmdLineParams.get(tol.name()));
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasTol$_setter_$tol_$eq(DoubleParam arg0) {

	}

	@Override
	public DoubleParam tol() {
		return tol;
	}

	public CoxRegression setTol(double value) {
		cmdLineParams.put(tol.name(), Double.toString(value));
		return (CoxRegression) setDefault(tol, value);
	}

	@Override
	public int getMaxOuterIter() {
		return Integer.parseInt(cmdLineParams.get(maxOuterIter.name()));
	}

	@Override
	public IntParam maxOuterIter() {
		return maxOuterIter;
	}

	public CoxRegression setMaxOuterIter(int value) {
		cmdLineParams.put(maxOuterIter.name(), Integer.toString(value));
		return (CoxRegression) setDefault(maxOuterIter, value);
	}

	@Override
	public int getMaxInnerIter() {
		return Integer.parseInt(cmdLineParams.get(maxInnerIter.name()));
	}

	@Override
	public IntParam maxInnerIter() {
		return maxInnerIter;
	}

	public CoxRegression setMaxInnerIter(int value) {
		cmdLineParams.put(maxInnerIter.name(), Integer.toString(value));
		return (CoxRegression) setDefault(maxInnerIter, value);
	}

	@Override
	public String getTECol() {
		return cmdLineParams.get(teCol.name());
	}

	@Override
	public Param<String> teCol() {
		return teCol;
	}

	public CoxRegression setTECol(String value) {
		cmdLineParams.put(teCol.name(), value);
		return (CoxRegression) setDefault(teCol, value);
	}

	@Override
	public String getFCol() {
		return cmdLineParams.get(fCol.name());
	}

	@Override
	public Param<String> fCol() {
		return fCol;
	}

	public CoxRegression setFCol(String value) {
		cmdLineParams.put(fCol.name(), value);
		return (CoxRegression) setDefault(fCol, value);
	}

	@Override
	public String getRCol() {
		return cmdLineParams.get(rCol.name());
	}

	@Override
	public Param<String> rCol() {
		return rCol;
	}

	public CoxRegression setRCol(String value) {
		cmdLineParams.put(rCol.name(), value);
		return (CoxRegression) setDefault(rCol, value);
	}

	public CoxRegression setFeatureIndicesRangeStart(int value) {
		start = value;
		return (CoxRegression) setDefault(featureIndicesRangeStart, value);
	}

	@Override
	public IntParam featureIndicesRangeStart() {
		return featureIndicesRangeStart;
	}

	@Override
	public int getFeatureIndicesRangeStart() {
		return start;
	}

	public CoxRegression setFeatureIndicesRangeEnd(int value) {
		end = value;
		return (CoxRegression) setDefault(featureIndicesRangeEnd, value);
	}

	@Override
	public IntParam featureIndicesRangeEnd() {
		return featureIndicesRangeEnd;
	}

	@Override
	public int getFeatureIndicesRangeEnd() {
		return end;
	}

//	@Override
//	public IntArrayParam featureIndices() {
//		return featureIndicesArray;
//	}

	@Override
	public List<Integer> getFeatureIndices() {
		return arr;
	}

	public CoxRegression setFeatureIndices(Seq<Integer> value) {
		arr = scala.collection.JavaConversions.asJavaList(value);
		return (CoxRegression) setDefault(featureIndicesRangeEnd, value);
	}

	public CoxRegression setFeatureIndices(List<Integer> value) {
		arr = value;
		return (CoxRegression) setDefault(featureIndicesRangeEnd, value);
	}

	public CoxRegression setTimestampIndex(long value) {
		cmdLineParams.put(timestampIndex.name(), Long.toString(value));
		return (CoxRegression) setDefault(timestampIndex, value);
	}

	@Override
	public LongParam timestampIndex() {
		return timestampIndex;
	}

	@Override
	public long getTimestampIndex() {
		return Long.parseLong(cmdLineParams.get(timestampIndex.name()));
	}

	public CoxRegression setEventIndex(long value) {
		cmdLineParams.put(eventIndex.name(), Long.toString(value));
		return (CoxRegression) setDefault(eventIndex, value);
	}

	@Override
	public LongParam eventIndex() {
		return eventIndex;
	}

	@Override
	public long getEventIndex() {
		return Long.parseLong(cmdLineParams.get(eventIndex.name()));
	}

	@Override
	public CoxRegressionModel train(DataFrame df) {
		MLContext ml = null;
		MLOutput out = null;
		JavaSparkContext jsc = new JavaSparkContext(
				sc);

		String featuresColName = getFeaturesCol();
		String teColName = getTECol();
		String fColName = getFCol();
		String rColName = getRCol();

		try {
			ml = new MLContext(
					sc);
		} catch (DMLRuntimeException e1) {
			e1.printStackTrace();
		}

		// Convert input data to format that SystemML accepts
		MatrixCharacteristics mcXin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;

		Xin = RDDConverterUtils.dataFrameToBinaryBlock(jsc, df.filter(featuresColName
				+ " is not null").select(featuresColName), mcXin, false, true);

		List<Row> teRowList = new ArrayList<Row>();
		long tsValue = getTimestampIndex();
		long eventValue = getEventIndex();

		if (tsValue < 0 || eventValue < 0)
			System.err.println(
					"The indices of the Timestamp column and the Event column have to be positive "
							+ "values");
		else {
			teRowList.add(RowFactory.create((double) tsValue));
			teRowList.add(RowFactory.create((double) eventValue));
		}

		JavaRDD<Row> teRow = jsc.parallelize(teRowList);
		List<StructField> teFields = new ArrayList<StructField>();
		teFields.add(DataTypes.createStructField(getTECol(), DataTypes.DoubleType, true));
		StructType teSchema = DataTypes.createStructType(teFields);
		DataFrame teDF = df.sqlContext().createDataFrame(teRow, teSchema);

		// Convert input data to format that SystemML accepts
		MatrixCharacteristics mcTEin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> TEin;
		TEin = RDDConverterUtils.dataFrameToBinaryBlock(jsc,
				teDF.select(teColName),
				mcTEin,
				false,
				false);

		List<Row> fiRowList = new ArrayList<Row>();

		if (start != -1 && end != -1)
			for (int i = start; i <= end; i++)
				fiRowList.add(RowFactory.create((double) i));
		else if (!arr.isEmpty())
			for (Integer i : arr)
				fiRowList.add(RowFactory.create((double) i));
		else
			System.err.println("Please provide range of integers or an array(list) of integers");

		JavaRDD<Row> fiRow = jsc.parallelize(fiRowList);
		List<StructField> fiFields = new ArrayList<StructField>();
		fiFields.add(DataTypes.createStructField(getFCol(), DataTypes.DoubleType, true));
		StructType fiSchema = DataTypes.createStructType(fiFields);
		DataFrame fiDF = df.sqlContext().createDataFrame(fiRow, fiSchema);

		// Convert input data to format that SystemML accepts
		MatrixCharacteristics mcFin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Fin;
		Fin = RDDConverterUtils.dataFrameToBinaryBlock(jsc,
				fiDF.select(fColName),
				mcFin,
				false,
				false);

		ClassTag<String> strClassTag = ClassTag$.MODULE$.apply(String.class);
		JavaRDD<String> emptyRDD = sc.emptyRDD(strClassTag).toJavaRDD();

		if (rDF != null && rDF.count() > 0) {
			// Convert input data to format that SystemML accepts
			MatrixCharacteristics mcRin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Rin;
			
			try {
				Rin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(
						sc),
						rDF.filter(rColName + " is not null").select(rColName),
						mcRin,
						false,
						true);
				ml.registerInput("R", Rin, mcRin);
				cmdLineParams.put("R", "R");
			} catch (DMLRuntimeException e) {
				e.printStackTrace();
			}
		} else {
			try {
				ml.registerInput("R", emptyRDD, "csv");
				cmdLineParams.put("R", " ");
			} catch (DMLRuntimeException e1) {
				e1.printStackTrace();
			}
		}

		try {
			// Register the input/output variables of script
			// 'Cox.dml'
			ml.registerInput("X_orig", Xin, mcXin);
			ml.registerInput("TE", TEin, mcTEin);
			ml.registerInput("F", Fin, mcFin);
			ml.registerOutput("TE_F");
			ml.registerOutput("M");
			ml.registerOutput("RT");
			ml.registerOutput("H_inv");
			ml.registerOutput("X_orig");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in Cox.dml
			cmdLineParams.put("X", " ");
			cmdLineParams.put("F", "F");
			cmdLineParams.put("TE", " ");
			cmdLineParams.put("RT", " ");
			cmdLineParams.put("MF", " ");
			cmdLineParams.put("M", " ");
			cmdLineParams.put("XO", " ");
			cmdLineParams.put("COV", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "Cox.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "Cox.dml";

			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, cmdLineParams);
			}

			System.out.println("Training has completed. Next will be modeling......");
			
			results.put("X_orig",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							Xin, mcXin));
			results.put("RT_X",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("RT"), out
									.getMatrixCharacteristics("RT")));
			results.put("M",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("M"),
							out.getMatrixCharacteristics("M")));
			results.put("COV",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("H_inv"), out
									.getMatrixCharacteristics("H_inv")));
			results.put("col_ind",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("TE_F"), out
									.getMatrixCharacteristics("TE_F")));

			return new CoxRegressionModel(results, sc, cmdLineParams, featuresColName, getLabelCol())
					.setParent(this);
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		}
	}
}
