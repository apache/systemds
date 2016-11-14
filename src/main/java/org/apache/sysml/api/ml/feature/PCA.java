package org.apache.sysml.api.ml.feature;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Transformer;
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
import org.apache.sysml.api.ml.param.PCAParams;

public class PCA extends Transformer implements PCAParams {

	private static final long serialVersionUID = 8173247745642668218L;

	private SparkContext sc = null;

	private HashMap<String, String> params = new HashMap<String, String>();
	private Param<String> inputCol = new Param<String>(this, "inputCol", "Input column name");
	private Param<String> outputCol = new Param<String>(this, "outputCol", "Output column name");
	private IntParam k = new IntParam(this, "K", "Number of principal components");
	private IntParam center = new IntParam(this, "CENTER", "Indicates whether or not to center data");
	private IntParam scale = new IntParam(this, "SCALE", "Indicates whether or not to scale data");
	private IntParam projected =
			new IntParam(this, "PROJDATA", "Indicates if the data should be projected or not");

	public PCA(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(1, 0, 0, 0);
	}

	public PCA(SparkContext sc, int k, int center, int scale, int projected) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(k, center, scale, projected);
	}

	private void setAllParameters(int k, int center, int scale, int projected) {
		params.put(this.k.name(), Integer.toString(k));
		params.put(this.center.name(), Integer.toString(center));
		params.put(this.scale.name(), Integer.toString(scale));
		params.put(this.projected.name(), Integer.toString(projected));
		params.put(this.inputCol.name(), " ");
		params.put(this.outputCol.name(), " ");
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public Transformer copy(ParamMap paramMap) {
		try {
			String strK = paramMap.getOrElse(k, getK()).toString();
			String strCenter = paramMap.getOrElse(center, isCenter()).toString();
			String strScale = paramMap.getOrElse(scale, isScale()).toString();
			String strProjected = paramMap.getOrElse(projected, isProjectedData()).toString();
			String strInputCol = paramMap.getOrElse(inputCol, getInputCol()).toString();
			String strOutputCol = paramMap.getOrElse(outputCol, getOutputCol()).toString();

			PCA pca = new PCA(sc,
					Integer.parseInt(strK),
					Integer.parseInt(strCenter),
					Integer.parseInt(strScale),
					Integer.parseInt(strProjected));

			pca.params.put(k.name(), strK);
			pca.params.put(center.name(), strCenter);
			pca.params.put(scale.name(), strScale);
			pca.params.put(projected.name(), strProjected);
			//pca.params.put(inputCol.name(), paramMap.g)
			pca.setInputCol(strInputCol);
			pca.setOutputCol(strOutputCol);

			return pca;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}

		return null;
	}

	public PCA setInputCol(String value) {
		params.put(inputCol.name(), value);
		return (PCA) setDefault(inputCol, value);
	}

	@Override
	public Param<String> inputCol() {
		return inputCol;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasInputCol$_setter_$inputCol_$eq(Param arg0) {

	}

	@Override
	public String getInputCol() {
		return params.get(inputCol.name());
	}

	public PCA setOutputCol(String value) {
		params.put(outputCol.name(), value);
		return (PCA) setDefault(outputCol, value);
	}

	@Override
	public Param<String> outputCol() {
		return outputCol;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasOutputCol$_setter_$outputCol_$eq(Param arg0) {

	}

	@Override
	public String getOutputCol() {
		return params.get(outputCol.name());
	}

	public PCA setK(int value) {
		params.put(k.name(), Integer.toString(value));
		return (PCA) setDefault(k, value);
	}

	@Override
	public IntParam k() {
		return k;
	}

	@Override
	public int getK() {
		return Integer.parseInt(params.get(k.name()));
	}

	public PCA setCenter(int value) {
		params.put(center.name(), Integer.toString(value));
		return (PCA) setDefault(center, value);
	}

	@Override
	public IntParam center() {
		return center;
	}

	@Override
	public int isCenter() {
		return Integer.parseInt(params.get(center.name()));
	}

	public PCA setProjectedData(int value) {
		params.put(projected.name(), Integer.toString(value));
		return (PCA) setDefault(projected, value);
	}

	@Override
	public IntParam projectedData() {
		return projected;
	}

	@Override
	public int isProjectedData() {
		return Integer.parseInt(params.get(projected.name()));
	}

	public PCA setScale(int value) {
		params.put(scale.name(), Integer.toString(value));
		return (PCA) setDefault(scale, value);
	}

	@Override
	public IntParam scale() {
		return scale;
	}

	@Override
	public int isScale() {
		return Integer.parseInt(params.get(scale.name()));
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			for(String paramkey : params.keySet()){
				params.put(paramkey, getOrDefault(getParam(paramkey)).toString());
			}
			MLContext ml = new MLContext(sc);
			MLOutput out = null;
			int projVal = isProjectedData();

			MatrixCharacteristics mcXin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
			Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
					dataset,
					mcXin,
					false,
					true);

			ml.registerInput("A", Xin, mcXin);
			if (projVal == 0) {
				ml.registerOutput("eval_stdev_dominant");
				ml.registerOutput("eval_dominant");
				ml.registerOutput("evec_dominant");
			} else if (projVal == 1)
				ml.registerOutput("newA");
			else
				System.err.println("The value of projectedData has to be either 0 or 1.");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in PCA.dml
			params.put("INPUT", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "PCA.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "PCA.dml";

			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, params);
			}
			return out.getDF(dataset.sqlContext(), "eval_stdev_dominant");
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public StructType transformSchema(StructType schema) {
		return null;
	}
}
