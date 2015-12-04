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

package org.apache.sysml.api.ml;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegressionParams;
import org.apache.spark.ml.classification.ProbabilisticClassifier;
import org.apache.spark.ml.param.BooleanParam;
import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.StringArrayParam;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.api.ml.LogisticRegressionModel;
import org.apache.sysml.api.ml.functions.ConvertSingleColumnToString;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

/**
 * 
 * This class shows how SystemML can be integrated into MLPipeline. Note, it has not been optimized for performance and 
 * is implemented as a proof of concept. An optimized pipeline can be constructed by usage of DML's 'parfor' construct.
 * 
 * TODO: 
 * - Please note that this class expects 1-based labels. To run below example,
 * please set environment variable 'SYSTEMML_HOME' and create folder 'algorithms' 
 * and place atleast two scripts in that folder 'MultiLogReg.dml' and 'GLM-predict.dml'
 * - It is not yet optimized for performance. 
 * - Also, it needs to be extended to surface all the parameters of MultiLogReg.dml
 * 
 * Example usage:
 * <pre><code>
 * // Code to demonstrate usage of pipeline
 * import org.apache.spark.ml.Pipeline
 * import org.apache.sysml.api.ml.LogisticRegression
 * import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
 * import org.apache.spark.mllib.linalg.Vector
 * case class LabeledDocument(id: Long, text: String, label: Double)
 * case class Document(id: Long, text: String)
 * val training = sc.parallelize(Seq(
 *      LabeledDocument(0L, "a b c d e spark", 1.0),
 *      LabeledDocument(1L, "b d", 2.0),
 *      LabeledDocument(2L, "spark f g h", 1.0),
 *      LabeledDocument(3L, "hadoop mapreduce", 2.0),
 *      LabeledDocument(4L, "b spark who", 1.0),
 *      LabeledDocument(5L, "g d a y", 2.0),
 *      LabeledDocument(6L, "spark fly", 1.0),
 *      LabeledDocument(7L, "was mapreduce", 2.0),
 *      LabeledDocument(8L, "e spark program", 1.0),
 *      LabeledDocument(9L, "a e c l", 2.0),
 *      LabeledDocument(10L, "spark compile", 1.0),
 *      LabeledDocument(11L, "hadoop software", 2.0)))
 * val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
 * val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
 * val lr = new LogisticRegression(sc, sqlContext)
 * val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
 * val model = pipeline.fit(training.toDF)
 * val test = sc.parallelize(Seq(
 *       Document(12L, "spark i j k"),
 *       Document(13L, "l m n"),
 *       Document(14L, "mapreduce spark"),
 *       Document(15L, "apache hadoop")))
 * model.transform(test.toDF).show
 * 
 * // Code to demonstrate usage of cross-validation
 * import org.apache.spark.ml.Pipeline
 * import org.apache.sysml.api.ml.LogisticRegression
 * import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
 * import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
 * import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
 * import org.apache.spark.mllib.linalg.Vector
 * case class LabeledDocument(id: Long, text: String, label: Double)
 * case class Document(id: Long, text: String)
 * val training = sc.parallelize(Seq(
 *      LabeledDocument(0L, "a b c d e spark", 1.0),
 *      LabeledDocument(1L, "b d", 2.0),
 *      LabeledDocument(2L, "spark f g h", 1.0),
 *      LabeledDocument(3L, "hadoop mapreduce", 2.0),
 *      LabeledDocument(4L, "b spark who", 1.0),
 *      LabeledDocument(5L, "g d a y", 2.0),
 *      LabeledDocument(6L, "spark fly", 1.0),
 *      LabeledDocument(7L, "was mapreduce", 2.0),
 *      LabeledDocument(8L, "e spark program", 1.0),
 *      LabeledDocument(9L, "a e c l", 2.0),
 *      LabeledDocument(10L, "spark compile", 1.0),
 *      LabeledDocument(11L, "hadoop software", 2.0)))
 * val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
 * val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
 * val lr = new LogisticRegression(sc, sqlContext)
 * val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
 * val crossval = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator)
 * val paramGrid = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid(lr.regParam, Array(0.1, 0.01)).build()
 * crossval.setEstimatorParamMaps(paramGrid)
 * crossval.setNumFolds(2)
 * val cvModel = crossval.fit(training.toDF)
 * val test = sc.parallelize(Seq(
 *       Document(12L, "spark i j k"),
 *       Document(13L, "l m n"),
 *       Document(14L, "mapreduce spark"),
 *       Document(15L, "apache hadoop")))
 * cvModel.transform(test.toDF).show
 * </code></pre>
 * 
 */
public class LogisticRegression extends ProbabilisticClassifier<Vector, LogisticRegression, LogisticRegressionModel>
	implements LogisticRegressionParams {

	private static final long serialVersionUID = 7763813395635870734L;
	
	private SparkContext sc = null;
	private SQLContext sqlContext = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();

	private IntParam icpt = new IntParam(this, "icpt", "Value of intercept");
	private DoubleParam reg = new DoubleParam(this, "reg", "Value of regularization parameter");
	private DoubleParam tol = new DoubleParam(this, "tol", "Value of tolerance");
	private IntParam moi = new IntParam(this, "moi", "Max outer iterations");
	private IntParam mii = new IntParam(this, "mii", "Max inner iterations");
	private IntParam labelIndex = new IntParam(this, "li", "Index of the label column");
	private StringArrayParam inputCol = new StringArrayParam(this, "icname", "Feature column name");
	private StringArrayParam outputCol = new StringArrayParam(this, "ocname", "Label column name");
	private int intMin = Integer.MIN_VALUE;
	@SuppressWarnings("unused")
	private int li = 0;
	private String[] icname = new String[1];
	private String[] ocname = new String[1];
	
	public LogisticRegression()  {
	}
	
	public LogisticRegression(String uid)  {
	}
	
	@Override
	public LogisticRegression copy(org.apache.spark.ml.param.ParamMap paramMap) {
		try {
			// Copy deals with command-line parameter of script MultiLogReg.dml
			LogisticRegression lr = new LogisticRegression(sc, sqlContext);
			lr.cmdLineParams.put(icpt.name(), paramMap.getOrElse(icpt, 0).toString());
			lr.cmdLineParams.put(reg.name(), paramMap.getOrElse(reg, 0.0f).toString());
			lr.cmdLineParams.put(tol.name(), paramMap.getOrElse(tol, 0.000001f).toString());
			lr.cmdLineParams.put(moi.name(), paramMap.getOrElse(moi, 100).toString());
			lr.cmdLineParams.put(mii.name(), paramMap.getOrElse(mii, 0).toString());
			
			return lr;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}
		return null;
		
	}
	
	public LogisticRegression(SparkContext sc, SQLContext sqlContext) throws DMLRuntimeException {
		this.sc = sc;
		this.sqlContext = sqlContext;
		
		setDefault(intercept(), 0);
		cmdLineParams.put(icpt.name(), "0");
		setDefault(regParam(), 0.0f);
		cmdLineParams.put(reg.name(), "0.0f");
		setDefault(tol(), 0.000001f);
		cmdLineParams.put(tol.name(), "0.000001f");
		setDefault(maxOuterIter(), 100);
		cmdLineParams.put(moi.name(), "100");
		setDefault(maxInnerIter(), 0);
		cmdLineParams.put(mii.name(), "0");
		setDefault(labelIdx(), intMin);
		li = intMin;
		setDefault(inputCol(), icname);
		icname[0] = "";
		setDefault(outputCol(), ocname);
		ocname[0] = "";
	}
	
	public LogisticRegression(SparkContext sc, SQLContext sqlContext, int icpt, double reg, double tol, int moi, int mii) throws DMLRuntimeException {
		this.sc = sc;
		this.sqlContext = sqlContext;

		setDefault(intercept(), icpt);
		cmdLineParams.put(this.icpt.name(), Integer.toString(icpt));
		setDefault(regParam(), reg);
		cmdLineParams.put(this.reg.name(), Double.toString(reg));
		setDefault(tol(), tol);
		cmdLineParams.put(this.tol.name(), Double.toString(tol));
		setDefault(maxOuterIter(), moi);
		cmdLineParams.put(this.moi.name(), Integer.toString(moi));
		setDefault(maxInnerIter(), mii);
		cmdLineParams.put(this.mii.name(), Integer.toString(mii));
		setDefault(labelIdx(), intMin);
		li = intMin;
		setDefault(inputCol(), icname);
		icname[0] = "";
		setDefault(outputCol(), ocname);
		ocname[0] = "";
	}

	@Override
	public String uid() {
		return Long.toString(LogisticRegression.serialVersionUID);
	}

	public LogisticRegression setRegParam(double value) {
		cmdLineParams.put(reg.name(), Double.toString(value));
		return (LogisticRegression) setDefault(reg, value);
	}
	
	@Override
	public org.apache.spark.sql.types.StructType validateAndTransformSchema(org.apache.spark.sql.types.StructType arg0, boolean arg1, org.apache.spark.sql.types.DataType arg2) {
		return null;
	}
	
	@Override
	public double getRegParam() {
		return Double.parseDouble(cmdLineParams.get(reg.name()));
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasRegParam$_setter_$regParam_$eq(DoubleParam arg0) {
		
	}

	@Override
	public DoubleParam regParam() {
		return reg;
	}

	@Override
	public DoubleParam elasticNetParam() {
		return null;
	}

	@Override
	public double getElasticNetParam() {
		return 0.0f;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasElasticNetParam$_setter_$elasticNetParam_$eq(DoubleParam arg0) {
		
	}

	@Override
	public int getMaxIter() {
		return 0;
	}

	@Override
	public IntParam maxIter() {
		return null;
	}
	
	public LogisticRegression setMaxOuterIter(int value) {
		cmdLineParams.put(moi.name(), Integer.toString(value));
		return (LogisticRegression) setDefault(moi, value);
	}
	
	public int getMaxOuterIter() {
		return Integer.parseInt(cmdLineParams.get(moi.name()));
	}

	public IntParam maxOuterIter() {
		return this.moi;
	}

	public LogisticRegression setMaxInnerIter(int value) {
		cmdLineParams.put(mii.name(), Integer.toString(value));
		return (LogisticRegression) setDefault(mii, value);
	}
	
	public int getMaxInnerIter() {
		return Integer.parseInt(cmdLineParams.get(mii.name()));
	}

	public IntParam maxInnerIter() {
		return mii;
	}
	
	@Override
	public void org$apache$spark$ml$param$shared$HasMaxIter$_setter_$maxIter_$eq(IntParam arg0) {
		
	}
	
	public LogisticRegression setIntercept(int value) {
		cmdLineParams.put(icpt.name(), Integer.toString(value));
		return (LogisticRegression) setDefault(icpt, value);
	}
	
	public int getIntercept() {
		return Integer.parseInt(cmdLineParams.get(icpt.name()));
	}

	public IntParam intercept() {
		return icpt;
	}
	
	@Override
	public BooleanParam fitIntercept() {
		return null;
	}

	@Override
	public boolean getFitIntercept() {
		return false;
	}
	
	@Override
	public void org$apache$spark$ml$param$shared$HasFitIntercept$_setter_$fitIntercept_$eq(BooleanParam arg0) {
		
	}

	public LogisticRegression setTol(double value) {
		cmdLineParams.put(tol.name(), Double.toString(value));
		return (LogisticRegression) setDefault(tol, value);
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

	@Override
	public double getThreshold() {
		return 0;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasThreshold$_setter_$threshold_$eq(DoubleParam arg0) {
		
	}

	@Override
	public DoubleParam threshold() {
		return null;
	}
	
	public LogisticRegression setLabelIndex(int value) {
		li = value;
		return (LogisticRegression) setDefault(labelIndex, value);
	}
	
	public int getLabelIndex() {
		return Integer.parseInt(cmdLineParams.get(labelIndex.name()));
	}

	public IntParam labelIdx() {
		return labelIndex;
	}
	
	public LogisticRegression setInputCol(String[] value) {
		icname[0] = value[0];
		return (LogisticRegression) setDefault(inputCol, value);
	}
	
	public String getInputCol() {
		return icname[0];
	}

	public StringArrayParam inputCol() {
		return inputCol;
	}
	
	public LogisticRegression setOutputCol(String[] value) {
		ocname[0] = value[0];
		return (LogisticRegression) setDefault(outputCol, value);
	}
	
	public String getOutputCol() {
		return ocname[0];
	}

	public StringArrayParam outputCol() {
		return outputCol;
	}
	
	@Override
	public LogisticRegressionModel train(DataFrame df) {
		MLContext ml = null;
		MLOutput out = null;
		
		try {
			ml = new MLContext(this.sc);
		} catch (DMLRuntimeException e1) {
			e1.printStackTrace();
			return null;
		}
		
		// Convert input data to format that SystemML accepts 
		MatrixCharacteristics mcXin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
		try {
			Xin = RDDConverterUtilsExt.vectorDataFrameToBinaryBlock(new JavaSparkContext(this.sc), df, mcXin, false, "features");
		} catch (DMLRuntimeException e1) {
			e1.printStackTrace();
			return null;
		}
		
		JavaRDD<String> yin = df.select("label").rdd().toJavaRDD().map(new ConvertSingleColumnToString());
		
		try {
			// Register the input/output variables of script 'MultiLogReg.dml'
			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("Y_vec", yin, "csv");
			ml.registerOutput("B_out");
			
			// Or add ifdef in MultiLogReg.dml
			cmdLineParams.put("X", " ");
			cmdLineParams.put("Y", " ");
			cmdLineParams.put("B", " ");
			
			
			// ------------------------------------------------------------------------------------
			// Please note that this logic is subject to change and is put as a placeholder
			String systemmlHome = System.getenv("SYSTEMML_HOME");
			if(systemmlHome == null) {
				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
				return null;
			}
			
			String dmlFilePath = systemmlHome + File.separator + "algorithms" + File.separator + "MultiLogReg.dml";
			// ------------------------------------------------------------------------------------
			
			synchronized(MLContext.class) { 
				// static synchronization is necessary before execute call
			    out = ml.execute(dmlFilePath, cmdLineParams);
			}
			
			JavaPairRDD<MatrixIndexes, MatrixBlock> b_out = out.getBinaryBlockedRDD("B_out");
			MatrixCharacteristics b_outMC = out.getMatrixCharacteristics("B_out");
			return new LogisticRegressionModel(b_out, b_outMC, sc).setParent(this);
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		} catch (ParseException e) {
			throw new RuntimeException(e);
		}
	}
}