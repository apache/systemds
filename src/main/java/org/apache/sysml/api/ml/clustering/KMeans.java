package org.apache.sysml.api.ml.clustering;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
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

import scala.Tuple2;

public class KMeans extends Estimator<KMeansModel> {

	private static final long serialVersionUID = 49295392421141796L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();

	private IntParam k = new IntParam(this, "k", "Number of centroids");
	private IntParam runs = new IntParam(this, "runs", "Number of runs (with different initial centroids)");
	private IntParam maxi = new IntParam(this, "maxi", "Maximum number of iterations per run");
	private DoubleParam tol = new DoubleParam(this, "tol", "Tolerance (epsilon) for WCSS change ratio");
	private IntParam samp =
			new IntParam(this, "samp", "Average number of records per centroid in data samples");
	private IntParam isY = new IntParam(this, "isY", "0 = do not write Y,  1 = write Y");
	private IntParam verb = new IntParam(this, "verb", "0 = do not print per-iteration stats, 1 = print them");
	private Param<String> featuresCol = new Param<String>(this, "featuresCol", "Name of feature column");

	public KMeans(SparkContext sc) {
		this.sc = sc;
		setAllParameters(0, 10, 1000, 0.000001f, 50, 0, 0);
	}

	public KMeans(SparkContext sc, int k, int runs, int maxi, double tol, int samp, int isY, int verb) {
		this.sc = sc;
		setAllParameters(k, runs, maxi, tol, samp, isY, verb);
	}

	private void setAllParameters(int k, int runs, int maxi, double tol, int samp, int isY, int verb) {
		setDefault(k(), k);
		cmdLineParams.put(this.k.name(), Integer.toString(k));
		setDefault(runs(), runs);
		cmdLineParams.put(this.runs.name(), Integer.toString(runs));
		setDefault(maxIter(), maxi);
		cmdLineParams.put(this.maxi.name(), Integer.toString(maxi));
		setDefault(tol(), tol);
		cmdLineParams.put(this.tol.name(), Double.toString(tol));
		setDefault(samp(), samp);
		cmdLineParams.put(this.samp.name(), Integer.toString(samp));
		setDefault(isY(), isY);
		cmdLineParams.put(this.isY.name(), Integer.toString(isY));
		setDefault(verb(), verb);
		cmdLineParams.put(this.verb.name(), Integer.toString(verb));
		setFeaturesCol("features");
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public KMeans copy(ParamMap paramMap) {
		try {
			KMeans km = new KMeans(sc);

			km.cmdLineParams.put(k.name(), paramMap.getOrElse(k, 0).toString());
			km.cmdLineParams.put(tol.name(), paramMap.getOrElse(tol, 0.000001f).toString());
			km.cmdLineParams.put(maxi.name(), paramMap.getOrElse(maxi, 1000).toString());
			km.cmdLineParams.put(runs.name(), paramMap.getOrElse(runs, 10).toString());
			km.cmdLineParams.put(samp.name(), paramMap.getOrElse(samp, 50).toString());
			km.cmdLineParams.put(isY.name(), paramMap.getOrElse(isY, 0).toString());
			km.cmdLineParams.put(verb.name(), paramMap.getOrElse(verb, 0).toString());
			km.setFeaturesCol(getFeaturesCol());

			return km;
		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}

	// support parameter
	public KMeans setRuns(int value) {
		cmdLineParams.put(runs.name(), Integer.toString(value));
		return (KMeans) setDefault(runs, value);
	}

	public double getRuns() {
		return Integer.parseInt(cmdLineParams.get(runs.name()));
	}

	public IntParam runs() {
		return runs;
	}

	public KMeans setSamp(int value) {
		cmdLineParams.put(samp.name(), Integer.toString(value));
		return (KMeans) setDefault(samp, value);
	}

	public double getSamp() {
		return Integer.parseInt(cmdLineParams.get(samp.name()));
	}

	public IntParam samp() {
		return samp;
	}

	// Do not need maxi here . support maxIter
	public KMeans setMaxIter(int value) {
		cmdLineParams.put(maxi.name(), Integer.toString(value));
		return (KMeans) setDefault(maxi, value);
	}

	public int getMaxIter() {
		return Integer.parseInt(cmdLineParams.get(maxi.name()));
	}

	public IntParam maxIter() {
		return maxi;
	}

	// K
	public KMeans setK(int value) {
		cmdLineParams.put(k.name(), Integer.toString(value));
		return (KMeans) setDefault(k, value);
	}

	public int getK() {
		return Integer.parseInt(cmdLineParams.get(k.name()));
	}

	public IntParam k() {
		return k;
	}

	// tol
	public KMeans setTol(int value) {
		cmdLineParams.put(tol.name(), Double.toString(value));
		return (KMeans) setDefault(tol, value);
	}

	public double getTol() {
		return Double.parseDouble(cmdLineParams.get(tol.name()));
	}

	public DoubleParam tol() {
		return tol;
	}

	public KMeans setIsY(int value) {
		cmdLineParams.put(isY.name(), Integer.toString(value));
		return (KMeans) setDefault(isY, value);
	}

	public int getIsY() {
		return Integer.parseInt(cmdLineParams.get(isY.name()));
	}

	public IntParam isY() {
		return isY;
	}

	public KMeans setVerb(int value) {
		cmdLineParams.put(verb.name(), Integer.toString(value));
		return (KMeans) setDefault(verb, value);
	}

	public int getVerb() {
		return Integer.parseInt(cmdLineParams.get(verb.name()));
	}

	public IntParam verb() {
		return verb;
	}

	public KMeans setFeaturesCol(String value) {
		cmdLineParams.put(featuresCol.name(), value);
		return (KMeans) setDefault(featuresCol, value);
	}

	public String getFeaturesCol() {
		return cmdLineParams.get(featuresCol.name());
	}

	public Param<String> featuresCol() {
		return featuresCol;
	}

	@Override
	public KMeansModel fit(DataFrame df) {
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
				true,
				true);

		int yVal = getIsY();

		try {
			// Register the input/output variables of script
			// 'Kmeans.dml'
			ml.registerInput("X", Xin, mcXin);
			ml.registerOutput("C");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in Kmeans.dml
			cmdLineParams.put("X", " ");

			if (yVal == 1) {
				cmdLineParams.put("isY", "1");
				ml.registerOutput("Y");
			}

			if (cmdLineParams.get(verb.name()) == "1")
				cmdLineParams.put("verb", "1");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "Kmeans.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "Kmeans.dml";

			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, cmdLineParams);
			}

			results.put("C",
					new Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>(
							out.getBinaryBlockedRDD("C"),
							out.getMatrixCharacteristics("C")));

			return new KMeansModel(results, sc, cmdLineParams, getFeaturesCol()).setParent(this);
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