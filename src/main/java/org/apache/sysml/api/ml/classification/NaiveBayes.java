package org.apache.sysml.api.ml.classification;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Predictor;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.functions.ConvertSingleColumnToString;
import org.apache.sysml.api.ml.param.NaiveBayesParams;

import scala.Tuple2;

public class NaiveBayes extends Predictor<Vector, NaiveBayes, NaiveBayesModel> implements NaiveBayesParams {

	private static final long serialVersionUID = -8443724084168253344L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();

	private IntParam laplaceCorr = new IntParam(this, "laplace", "Value of laplace correction");

	public NaiveBayes(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;

		setDefault(laplaceCorrection(), 1); 
		cmdLineParams.put("laplace", "1");
	}

	public NaiveBayes(SparkContext sc, int laplace) throws DMLRuntimeException {
		this.sc = sc;

		setDefault(laplaceCorrection(), laplace);
		cmdLineParams.put("laplace", Integer.toString(laplace));
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public NaiveBayes copy(ParamMap paramMap) {
		try {
			String strLaplace = paramMap.getOrElse(laplaceCorr, getLaplaceCorrection()).toString();
			NaiveBayes nb = new NaiveBayes(sc, Integer.parseInt(strLaplace));

			nb.cmdLineParams.put(laplaceCorr.name(), strLaplace);
			nb.setFeaturesCol(getFeaturesCol());
			nb.setLabelCol(getLabelCol());

			return nb;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}
		return null;
	}

	public NaiveBayesParams setLaplaceCorrection(int value) {
		cmdLineParams.put("laplace", Integer.toString(value));
		return (NaiveBayesParams) setDefault(laplaceCorr, value);
	}

	@Override
	public IntParam laplaceCorrection() {
		return laplaceCorr;
	}

	@Override
	public int getLaplaceCorrection() {
		return Integer.parseInt(cmdLineParams.get(laplaceCorr.name()));
	}

	@Override
	public NaiveBayesModel train(DataFrame df) {
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
				df.select(getFeaturesCol()),
				mcXin,
				false,
				true);
		
		MatrixCharacteristics mcYin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> yin;
		yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(this.sc),
				df.select(getLabelCol()),
				mcYin,
				false,
				false);

		try {
			// Register the input/output variables of script
			// "naive-bayes.dml"
			ml.registerInput("D", Xin, mcXin);
			ml.registerInput("C", yin, mcYin);
			ml.registerOutput("classPrior");
			ml.registerOutput("classConditionals");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			cmdLineParams.put("X", " ");
			cmdLineParams.put("Y", " ");
			cmdLineParams.put("prior", " ");
			cmdLineParams.put("conditionals", " ");
			cmdLineParams.put("accuracy", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "naive-bayes.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "naive-bayes.dml";

			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, cmdLineParams);
			}
			System.out.println("Execution of script completed.");
			results.put("prior",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("classPrior"),
							out.getMatrixCharacteristics("classPrior")));
			results.put("conditionals",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("classConditionals"),
							out.getMatrixCharacteristics("classConditionals")));

			return new NaiveBayesModel(results, getFeaturesCol(), getLabelCol(), sc, cmdLineParams)
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
