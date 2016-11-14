package org.apache.sysml.api.ml.recommendation;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Model;
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

public class ALSTopKModel extends Model<ALSTopKModel> {

	private static final long serialVersionUID = -6400141718671329172L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();
	private String featuresCol = "";
	private int k = 0;

	public ALSTopKModel(
			HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results,
			SparkContext sc,
			String featuresCol,
			int k) {
		this.results = results;
		this.sc = sc;
		this.featuresCol = featuresCol;
		this.k = k;
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public ALSTopKModel copy(ParamMap arg0) {
		return new ALSTopKModel(results, sc, featuresCol, k);
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			MLContext ml = new MLContext(sc);

			MatrixCharacteristics mcXin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
			Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
					dataset,
					mcXin,
					false,
					true);

			ml.registerInput("X", Xin, mcXin);
			ml.registerOutput("V_prime");

			cmdLineParams.put("X", " ");
			cmdLineParams.put("Y", " ");
			cmdLineParams.put("L", " ");
			cmdLineParams.put("R", " ");
			cmdLineParams.put("V", " ");
			cmdLineParams.put("K", Integer.toString(k));

			for (Map.Entry<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> entry : results
					.entrySet())
				ml.registerInput(entry.getKey(), entry.getValue()._1, entry.getValue()._2);

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "ALS_predict.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "ALS_predict.dml";
			MLOutput out = ml.execute(dmlFilePath, cmdLineParams);

			DataFrame resultDF = out.getDF(dataset.sqlContext(), "V_prime");
			return resultDF;
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
