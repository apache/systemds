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
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.param.RandomForestClassifierParams;

import scala.Tuple2;

public class RandomForestClassifier extends
		Predictor<Vector, RandomForestClassifier, RandomForestClassificationModel> implements
		RandomForestClassifierParams {

	private static final long serialVersionUID = 4944943651757609396L;

	private SparkContext sc = null;
	private DataFrame rDF = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();

	private IntParam bins = new IntParam(
			this, "bins", "Number of equiheight bins per scale feature to choose thresholds");
	private IntParam depth = new IntParam(
			this, "depth", "Maximum depth of the learned tree");
	private IntParam numLeaf = new IntParam(
			this, "num_leaf", "Number of samples when splitting stops and leaf node is added");
	private IntParam numSamples = new IntParam(
			this, "num_samples", "Number of samples at which point we switch "
					+ "to in-memory subtree building");
	private IntParam numTrees = new IntParam(
			this, "num_trees", "Number of trees to be learned in the random forest model");
	private DoubleParam subSampleRate = new DoubleParam(
			this,
			"subsamp_rate",
			"Parameter controlling the size of each tree in the forest; samples are selected from a Poisson distribution with parameter subsamp_rate (the default value is 1.0)");
	private DoubleParam featureSubset = new DoubleParam(
			this,
			"feature_subset",
			"Parameter that controls the number of feature used as candidates for splitting at each tree node as a power of number of features in the dataset; by default square root of features (i.e., feature_subset = 0.5) are used at each tree node ");
	private Param<String> impurity = new Param<String>(
			this, "impurity", "Impurity measure: entropy or Gini (the default)");
	private Param<String> rCol = new Param<String>(
			this, "rCol", "Name of the column that has categorical data indices in features vector");

	public RandomForestClassifier(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(20, 25, 10, 3000, 10, 1.0f, 0.5f, "Gini");
	}

	public RandomForestClassifier(SparkContext sc, DataFrame rDF) throws DMLRuntimeException {
		this.sc = sc;
		this.rDF = rDF;
		setAllParameters(20, 25, 10, 3000, 10, 1.0f, 0.5f, "Gini");
	}

	public RandomForestClassifier(SparkContext sc, int maxBins, int maxDepth, int numOfLeaf, int numOfSamples,
			int numOfTrees, double subSamplingRate, double featureSub, String imp)
			throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(maxBins,
				maxDepth,
				numOfLeaf,
				numOfSamples,
				numOfTrees,
				subSamplingRate,
				featureSub,
				imp);
	}

	public RandomForestClassifier(SparkContext sc, DataFrame rDF, int maxBins, int maxDepth, int numOfLeaf,
			int numOfSamples, int numOfTrees, double subSamplingRate, double featureSub, String imp)
			throws DMLRuntimeException {
		this.sc = sc;
		this.rDF = rDF;
		setAllParameters(maxBins,
				maxDepth,
				numOfLeaf,
				numOfSamples,
				numOfTrees,
				subSamplingRate,
				featureSub,
				imp);
	}

	private void setAllParameters(
			int maxBins,
			int maxDepth,
			int numOfLeaf,
			int numOfSamples,
			int numOfTrees,
			double subSamplingRate,
			double featureSub,
			String imp) {
		setDefault(maxBins(), maxBins);
		cmdLineParams.put(bins.name(), Integer.toString(maxBins));
		setDefault(maxDepth(), maxDepth);
		cmdLineParams.put(depth.name(), Integer.toString(maxDepth));
		setDefault(numLeaf(), numOfLeaf);
		cmdLineParams.put(numLeaf.name(), Integer.toString(numOfLeaf));
		setDefault(numSamples(), numOfSamples);
		cmdLineParams.put(numSamples.name(), Integer.toString(numOfSamples));
		setDefault(numOfTrees(), numOfTrees);
		cmdLineParams.put(numTrees.name(), Integer.toString(numOfTrees));
		setDefault(subSampleRate(), subSamplingRate);
		cmdLineParams.put(subSampleRate.name(), Double.toString(subSamplingRate));
		setDefault(featureSubset(), featureSub);
		cmdLineParams.put(featureSubset.name(), Double.toString(featureSub));
		setDefault(impurity(), imp);
		cmdLineParams.put(impurity.name(), imp);
		setRCol("catIndices");
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public RandomForestClassifier copy(ParamMap paramMap) {
		try {
			String strBins = paramMap.getOrElse(bins, getMaxBins()).toString();
			String strDepth = paramMap.getOrElse(depth, getMaxDepth()).toString();
			String strNumLeaf = paramMap.getOrElse(numLeaf, getNumLeaf()).toString();
			String strNumSamples = paramMap.getOrElse(numSamples, getNumSamples()).toString();
			String strNumTrees = paramMap.getOrElse(numTrees, getNumOfTrees()).toString();
			String strSubSampleRate = paramMap.getOrElse(subSampleRate, getSubSampleRate()).toString();
			String strFeatureSubset = paramMap.getOrElse(featureSubset, getFeatureSubset()).toString();
			String strImpurity = paramMap.getOrElse(impurity, getImpurity());

			RandomForestClassifier rf = new RandomForestClassifier(
					sc,
					rDF,
					Integer.parseInt(strBins),
					Integer.parseInt(strDepth),
					Integer
							.parseInt(strNumLeaf),
					Integer.parseInt(strNumSamples),
					Integer.parseInt(strNumTrees),
					Double.parseDouble(strSubSampleRate),
					Double
							.parseDouble(strFeatureSubset),
					strImpurity);

			rf.cmdLineParams.put(bins.name(), strBins);
			rf.cmdLineParams.put(depth.name(), strDepth);
			rf.cmdLineParams.put(numLeaf.name(), strNumLeaf);
			rf.cmdLineParams.put(numSamples.name(), strNumSamples);
			rf.cmdLineParams.put(numTrees.name(), strNumLeaf);
			rf.cmdLineParams.put(subSampleRate.name(), strSubSampleRate);
			rf.cmdLineParams.put(featureSubset.name(), strFeatureSubset);
			rf.cmdLineParams.put(impurity.name(), strImpurity);
			rf.setFeaturesCol(getFeaturesCol());
			rf.setLabelCol(getLabelCol());
			rf.setRCol(getRCol());

			return rf;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}

		return null;
	}

	@Override
	public StructType validateAndTransformSchema(StructType arg0, boolean arg1, DataType arg2) {
		return null;
	}

	public RandomForestClassifier setMaxBins(int value) {
		cmdLineParams.put(bins.name(), Integer.toString(value));
		return (RandomForestClassifier) setDefault(bins, value);
	}

	@Override
	public IntParam maxBins() {
		return bins;
	}

	@Override
	public int getMaxBins() {
		return Integer.parseInt(cmdLineParams.get(bins.name()));
	}

	public RandomForestClassifier setMaxDepth(int value) {
		cmdLineParams.put(depth.name(), Integer.toString(value));
		return (RandomForestClassifier) setDefault(depth, value);
	}

	@Override
	public IntParam maxDepth() {
		return depth;
	}

	@Override
	public int getMaxDepth() {
		return Integer.parseInt(cmdLineParams.get(depth.name()));
	}

	public RandomForestClassifier setNumLeaf(int value) {
		cmdLineParams.put(numLeaf.name(), Integer.toString(value));
		return (RandomForestClassifier) setDefault(numLeaf, value);
	}

	@Override
	public IntParam numLeaf() {
		return numLeaf;
	}

	@Override
	public int getNumLeaf() {
		return Integer.parseInt(cmdLineParams.get(numLeaf.name()));
	}

	public RandomForestClassifier setNumSamples(int value) {
		cmdLineParams.put(numSamples.name(), Integer.toString(value));
		return (RandomForestClassifier) setDefault(numSamples, value);
	}

	@Override
	public IntParam numSamples() {
		return numSamples;
	}

	@Override
	public int getNumSamples() {
		return Integer.parseInt(cmdLineParams.get(numSamples.name()));
	}

	public RandomForestClassifier setNumOfTrees(int value) {
		cmdLineParams.put(numTrees.name(), Integer.toString(value));
		return (RandomForestClassifier) setDefault(numTrees, value);
	}

	@Override
	public IntParam numOfTrees() {
		return numTrees;
	}

	@Override
	public int getNumOfTrees() {
		return Integer.parseInt(cmdLineParams.get(numTrees.name()));
	}

	public RandomForestClassifier setSubSampleRate(double value) {
		cmdLineParams.put(subSampleRate.name(), Double.toString(value));
		return (RandomForestClassifier) setDefault(subSampleRate, value);
	}

	@Override
	public DoubleParam subSampleRate() {
		return subSampleRate;
	}

	@Override
	public double getSubSampleRate() {
		return Double.parseDouble(cmdLineParams.get(subSampleRate.name()));
	}

	public RandomForestClassifier setFeatureSubset(double value) {
		cmdLineParams.put(featureSubset.name(), Double.toString(value));
		return (RandomForestClassifier) setDefault(featureSubset, value);
	}

	@Override
	public DoubleParam featureSubset() {
		return featureSubset;
	}

	@Override
	public double getFeatureSubset() {
		return Double.parseDouble(cmdLineParams.get(featureSubset.name()));
	}

	public RandomForestClassifier setImpurity(String value) {
		cmdLineParams.put(impurity.name(), value);
		return (RandomForestClassifier) setDefault(impurity, value);
	}

	@Override
	public Param<String> impurity() {
		return impurity;
	}

	@Override
	public String getImpurity() {
		return cmdLineParams.get(impurity.name());
	}

	public RandomForestClassifier setRCol(String value) {
		cmdLineParams.put(rCol.name(), value);
		return (RandomForestClassifier) setDefault(rCol, value);
	}

	@Override
	public Param<String> rCol() {
		return rCol;
	}

	@Override
	public String getRCol() {
		return cmdLineParams.get(rCol.name());
	}

	@Override
	public RandomForestClassificationModel train(DataFrame df) {
		MLContext ml = null;
		MLOutput out = null;

		try {
			ml = new MLContext(
					sc);
		} catch (DMLRuntimeException e1) {
			e1.printStackTrace();
			return null;
		}

		// Convert input data to format that SystemML accepts
		MatrixCharacteristics mcXin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Xin = null;
		Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(
		sc), df.select(getFeaturesCol()), mcXin, false, true);

		MatrixCharacteristics mcYin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> yin = null;
		yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(
				sc), df.select(getLabelCol()), mcYin, false, true);

		MatrixCharacteristics mcRin = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> Rin = null;
		if (rDF != null) {
			try {
				Rin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(
						sc), rDF.select(getRCol()), mcXin, false, true);
				ml.registerInput("R", Rin, mcRin);
				cmdLineParams.put("R", "R");
			} catch (DMLRuntimeException e1) {
				e1.printStackTrace();
				return null;
			}
		}

		try {
			// Register the input/output variables of script
			// 'decision-tree.dml'
			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("Y_bin", yin, mcYin);
			ml.registerOutput("M");
			ml.registerOutput("C");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in decision-tree.dml
			cmdLineParams.put("C", "C");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "random-forest.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "random-forest.dml";

			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, cmdLineParams);
			}

			results.put("M",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("M"),
							out.getMatrixCharacteristics("M")));
			results.put("C",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("C"),
							out.getMatrixCharacteristics("C")));

			return new RandomForestClassificationModel(
					results, sc, rDF, getFeaturesCol(), getLabelCol(), getRCol()).setParent(
							this);
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		}
	}
}
