package org.apache.sysml.api.ml.classification;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Predictor;
import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.param.SVMParams;

import scala.Tuple2;

public class MultiClassSVMClassifier
		extends Predictor<Vector, MultiClassSVMClassifier, MultiClassSVMClassificationModel>
		implements SVMParams {

	private static final long serialVersionUID = 8341575574238779488L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();

	private IntParam intercept = new IntParam(this, "icpt", "Value of intercept");
	private DoubleParam regParam = new DoubleParam(this, "reg", "Value of regularization parameter");
	private DoubleParam tol = new DoubleParam(this, "tol", "Value of tolerance");
	private IntParam maxIter =
			new IntParam(this, "maxiter", "Maximum number of conjugate gradient iterations");

	public MultiClassSVMClassifier(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(0, 1.0f, 0.001f, 100);
	}

	public MultiClassSVMClassifier(SparkContext sc, int intercept, double regParam, double tol, int maxIter)
			throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(intercept, regParam, tol, maxIter);
	}

	private void setAllParameters(int intercept, double regParam, double tol, int maxIter) {
		setDefault(intercept(), intercept);
		cmdLineParams.put(this.intercept.name(), Integer.toString(intercept));
		setDefault(regParam(), regParam);
		cmdLineParams.put(this.regParam.name(), Double.toString(regParam));
		setDefault(tol(), tol);
		cmdLineParams.put(this.tol.name(), Double.toString(tol));
		setDefault(maxIter(), maxIter);
		cmdLineParams.put(this.maxIter.name(), Integer.toString(maxIter));
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public MultiClassSVMClassifier copy(ParamMap paramMap) {
		try {
			String strIntercept = paramMap.getOrElse(intercept, getIntercept()).toString();
			String strRegParam = paramMap.getOrElse(regParam, getRegParam()).toString();
			String strTol = paramMap.getOrElse(tol, getTol()).toString();
			String strMaxIter = paramMap.getOrElse(maxIter, getMaxIter()).toString();

			// Copy deals with command-line parameter of script
			// m-svm.dml
			MultiClassSVMClassifier svm = new MultiClassSVMClassifier(sc,
					Integer.parseInt(strIntercept),
					Double.parseDouble(strRegParam),
					Double.parseDouble(strTol),
					Integer.parseInt(strMaxIter));

			svm.cmdLineParams.put(intercept.name(), strIntercept);
			svm.cmdLineParams.put(regParam.name(), strRegParam);
			svm.cmdLineParams.put(tol.name(), strTol);
			svm.cmdLineParams.put(maxIter.name(), strMaxIter);
			svm.setFeaturesCol(getFeaturesCol());
			svm.setLabelCol(getLabelCol());

			return svm;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}
		return null;
	}

	public MultiClassSVMClassifier setRegParam(double value) {
		cmdLineParams.put(regParam.name(), Double.toString(value));
		return (MultiClassSVMClassifier) setDefault(regParam, value);
	}

	@Override
	public double getRegParam() {
		return Double.parseDouble(cmdLineParams.get(regParam.name()));
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasRegParam$_setter_$regParam_$eq(DoubleParam arg0) {

	}

	@Override
	public DoubleParam regParam() {
		return regParam;
	}

	public MultiClassSVMClassifier setTol(double value) {
		cmdLineParams.put(tol.name(), Double.toString(value));
		return (MultiClassSVMClassifier) setDefault(tol, value);
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

	public MultiClassSVMClassifier setMaxIter(int value) {
		cmdLineParams.put(maxIter.name(), Integer.toString(value));
		return (MultiClassSVMClassifier) setDefault(maxIter, value);
	}

	@Override
	public int getMaxIter() {
		return Integer.parseInt(cmdLineParams.get(maxIter.name()));
	}

	@Override
	public IntParam maxIter() {
		return maxIter;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasMaxIter$_setter_$maxIter_$eq(IntParam arg0) {

	}

	public MultiClassSVMClassifier setIntercept(int value) {
		cmdLineParams.put(intercept.name(), Integer.toString(value));
		return (MultiClassSVMClassifier) setDefault(intercept, value);
	}

	@Override
	public int getIntercept() {
		return Integer.parseInt(cmdLineParams.get(intercept.name()));
	}

	@Override
	public IntParam intercept() {
		return intercept;
	}

	@Override
	public MultiClassSVMClassificationModel train(DataFrame df) {
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
		Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(this.sc),
				df.select(getFeaturesCol()),
				mcXin,
				false,
				true);

		// Convert input data to format that SystemML accepts
		MatrixCharacteristics mcYin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Yin;
		Yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(this.sc),
				df.select(getLabelCol()),
				mcYin,
				false,
				false);

		try {
			// Register the input/output variables of script
			// 'm-svm.dml'
			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("Y", Yin, mcYin);
			ml.registerOutput("w");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in m-svm.dml
			cmdLineParams.put("X", " ");
			cmdLineParams.put("Y", " ");
			cmdLineParams.put("model", " ");
			cmdLineParams.put("Log", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "m-svm.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "m-svm.dml";
			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, cmdLineParams);
			}

			results.put("W",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("w"),
							out.getMatrixCharacteristics("w")));

			return new MultiClassSVMClassificationModel(results, sc, getFeaturesCol(), getLabelCol())
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
