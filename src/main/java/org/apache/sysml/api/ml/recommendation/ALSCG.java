package org.apache.sysml.api.ml.recommendation;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.param.BooleanParam;
import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.param.ALSParams;

import scala.Tuple2;

public class ALSCG extends Estimator<ALSModel> implements ALSParams {

	private static final long serialVersionUID = -3907594318853617552L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();

	private IntParam rank = new IntParam(this, "rank", "Rank of the factorization");
	private Param<String> reg = new Param<String>(this, "reg", "Regularization: (L2 = L2 regularization," +
			" wL2 = weighted L2 regularization)");
	private DoubleParam lambda =
			new DoubleParam(this, "lambda", "Regularization parameter, no regularization if 0.0");
	private IntParam maxIter = new IntParam(this, "maxi", "Maximum number of iterations");
	private BooleanParam check = new BooleanParam(this, "check", "Check for convergence after every " +
			"iteration, i.e., updating U and V once");
	private DoubleParam threshold =
			new DoubleParam(this, "thr", "Assuming check is set to TRUE, the algorithm " +
					"stops and convergence is declared if the decrease in loss " +
					"in any two consecutive iterations falls below this threshold " +
					"if check is FALSE thr is ignored");
	private Param<String> featuresCol = new Param<String>(this, "featuresCol", "Name of the features column");

	public ALSCG(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(10, "L2", 0.000001f, 50, false, 0.0001f);
	}

	public ALSCG(SparkContext sc, int rank, String reg, double lambda, int maxIter, boolean check,
			double threshold) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(rank, reg, lambda, maxIter, check, threshold);
	}

	private void setAllParameters(
			int rank,
			String reg,
			double lambda,
			int maxIter,
			boolean check,
			double threshold) {
		setDefault(rank(), rank);
		cmdLineParams.put(this.rank.name(), Integer.toString(rank));
		setDefault(reg(), reg);
		cmdLineParams.put(this.reg.name(), reg);
		setDefault(lambda(), lambda);
		cmdLineParams.put(this.lambda.name(), Double.toString(lambda));
		setDefault(maxIter(), maxIter);
		cmdLineParams.put(this.maxIter.name(), Integer.toString(maxIter));
		setDefault(check(), check);
		cmdLineParams.put(this.check.name(), Boolean.toString(check).toUpperCase());
		setDefault(threshold(), threshold);
		cmdLineParams.put(this.threshold.name(), Double.toString(threshold));
		setFeaturesCol("features");
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public ALSCG copy(ParamMap paramMap) {
		try {
			String strRank = paramMap.getOrElse(rank, getRank()).toString();
			String strReg = paramMap.getOrElse(reg, getReg()).toString();
			String strLambda = paramMap.getOrElse(lambda, getLambda()).toString();
			String strMaxIter = paramMap.getOrElse(maxIter, getMaxIter()).toString();
			String strCheck = paramMap.getOrElse(check, getCheck()).toString();
			String strThreshold = paramMap.getOrElse(threshold, getThreshold()).toString();

			ALSCG als = new ALSCG(sc,
					Integer.parseInt(strRank),
					strReg,
					Double.parseDouble(strLambda),
					Integer.parseInt(strMaxIter),
					Boolean.parseBoolean(strCheck),
					Double.parseDouble(strThreshold));

			als.cmdLineParams.put(rank.name(), strRank);
			als.cmdLineParams.put(reg.name(), strReg);
			als.cmdLineParams.put(lambda.name(), strLambda);
			als.cmdLineParams.put(maxIter.name(), strMaxIter);
			als.cmdLineParams.put(check.name(), strCheck.toUpperCase());
			als.cmdLineParams.put(threshold.name(), strThreshold);
			als.setFeaturesCol(getFeaturesCol());

			return als;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}
		return null;
	}

	public ALSCG setRank(int value) {
		cmdLineParams.put(rank.name(), Integer.toString(value));
		return (ALSCG) setDefault(rank, value);
	}

	@Override
	public IntParam rank() {
		return rank;
	}

	@Override
	public int getRank() {
		return Integer.parseInt(cmdLineParams.get(rank.name()));
	}

	public ALSCG setReg(String value) {
		cmdLineParams.put(reg.name(), value);
		return (ALSCG) setDefault(reg, value);
	}

	@Override
	public Param<String> reg() {
		return reg;
	}

	@Override
	public String getReg() {
		return cmdLineParams.get(reg.name());
	}

	public ALSCG setLambda(double value) {
		cmdLineParams.put(lambda.name(), Double.toString(value));
		return (ALSCG) setDefault(lambda, value);
	}

	@Override
	public DoubleParam lambda() {
		return lambda;
	}

	@Override
	public double getLambda() {
		return Double.parseDouble(cmdLineParams.get(lambda.name()));
	}

	public ALSCG setMaxIter(int value) {
		cmdLineParams.put(maxIter.name(), Integer.toString(value));
		return (ALSCG) setDefault(maxIter, value);
	}

	@Override
	public IntParam maxIter() {
		return maxIter;
	}

	@Override
	public int getMaxIter() {
		return Integer.parseInt(cmdLineParams.get(maxIter.name()));
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasMaxIter$_setter_$maxIter_$eq(IntParam arg0) {

	}

	public ALSCG setCheck(boolean value) {
		cmdLineParams.put(check.name(), Boolean.toString(value).toUpperCase());
		return (ALSCG) setDefault(check, value);
	}

	@Override
	public BooleanParam check() {
		return check;
	}

	@Override
	public boolean getCheck() {
		return Boolean.parseBoolean(cmdLineParams.get(check.name()).toLowerCase());
	}

	public ALSCG setThreshold(double value) {
		cmdLineParams.put(threshold.name(), Double.toString(value));
		return (ALSCG) setDefault(threshold, value);
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasThreshold$_setter_$threshold_$eq(DoubleParam arg0) {

	}

	@Override
	public DoubleParam threshold() {
		return threshold;
	}

	@Override
	public double getThreshold() {
		return Double.parseDouble(cmdLineParams.get(threshold.name()));
	}

	public ALSCG setFeaturesCol(String value) {
		cmdLineParams.put(featuresCol.name(), value);
		return (ALSCG) setDefault(featuresCol, value);
	}

	@Override
	public Param<String> featuresCol() {
		return featuresCol;
	}

	@Override
	public String getFeaturesCol() {
		return cmdLineParams.get(featuresCol.name());
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasFeaturesCol$_setter_$featuresCol_$eq(Param arg0) {

	}

	@Override
	public ALSModel fit(DataFrame df) {
		MLContext ml = null;
		MLOutput out = null;

		try {
			ml = new MLContext(sc);
		} catch (DMLRuntimeException e1) {
			e1.printStackTrace();
			return null;
		}

		// Convert input data to format that SystemML accepts
		MatrixCharacteristics mcXin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
		Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
				df,
				mcXin,
				false,
				true);

		try {
			// Register the input/output variables of script
			// 'ALS-CG.dml'
			ml.registerInput("X", Xin, mcXin);
			ml.registerOutput("U");
			ml.registerOutput("V");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			DenseVector vector;
			String vec = df.select(getFeaturesCol()).first().get(0).toString();

			if (vec.startsWith("("))
				vector = ((SparseVector) df.select(getFeaturesCol()).first().get(0)).toDense();
			else
				vector = (DenseVector) df.select(getFeaturesCol()).first().get(0);

			long Vcols = (long) vector.size();
			long Vrows = df.count();

			// Or add ifdef in ALS-CG.dml
			cmdLineParams.put("X", " ");
			cmdLineParams.put("U", " ");
			cmdLineParams.put("V", " ");
			cmdLineParams.put("Vrows", Long.toString(Vrows));
			cmdLineParams.put("Vcols", Long.toString(Vcols));

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "ALS-CG.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "ALS-CG.dml";

			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, cmdLineParams);
			}

			results.put("L",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("U"),
							out.getMatrixCharacteristics("U")));
			results.put("R",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("V"),
							out.getMatrixCharacteristics("V")));

			return new ALSModel(results, sc, cmdLineParams, getFeaturesCol()).setParent(this);
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
