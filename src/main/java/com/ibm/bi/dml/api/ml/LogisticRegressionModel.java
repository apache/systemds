package com.ibm.bi.dml.api.ml;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.api.MLContext;
import com.ibm.bi.dml.api.MLOutput;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDConverterUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class LogisticRegressionModel extends ProbabilisticClassificationModel<Vector, LogisticRegressionModel> {

	private static final long serialVersionUID = -6464693773946415027L;
	private JavaPairRDD<MatrixIndexes, MatrixBlock> b_out;
	private SparkContext sc;
	private MatrixCharacteristics b_outMC;
	@Override
	public LogisticRegressionModel copy(ParamMap paramMap) {
		return this;
	}
	
	public LogisticRegressionModel(JavaPairRDD<MatrixIndexes, MatrixBlock> b_out2, MatrixCharacteristics b_outMC, SparkContext sc) {
		this.b_out = b_out2;
		this.b_outMC = b_outMC;
		this.sc = sc;
		//this.cmdLineParams = cmdLineParams;
	}
	
	public LogisticRegressionModel() {
	}
	
	public LogisticRegressionModel(String uid) {
	}

	@Override
	public String uid() {
		return Long.toString(LogisticRegressionModel.serialVersionUID);
	}

	@Override
	public Vector raw2probabilityInPlace(Vector arg0) {
		return arg0;
	}

	@Override
	public int numClasses() {
		return 2;
	}

	@Override
	public Vector predictRaw(Vector arg0) {
		return arg0;
	}
	
	
	@Override
	public double predict(Vector features) {
		return super.predict(features);
	}
	
	@Override
	public double raw2prediction(Vector rawPrediction) {
		return super.raw2prediction(rawPrediction);
	}
	
	@Override
	public double probability2prediction(Vector probability) {
		return super.probability2prediction(probability);
	}
	
	public static class ConvertIntToRow implements Function<Integer, Row> {

		private static final long serialVersionUID = -3480953015655773622L;

		@Override
		public Row call(Integer arg0) throws Exception {
			Object[] row_fields = new Object[1];
			row_fields[0] = new Double(arg0);
			return RowFactory.create(row_fields);
		}
		
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			MatrixCharacteristics mcXin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
			try {
				Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(new JavaSparkContext(this.sc), dataset, mcXin, false, "features");
			} catch (DMLRuntimeException e1) {
				e1.printStackTrace();
				return null;
			}
			MLContext ml = new MLContext(sc);
			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("B_full", b_out, b_outMC); // Changed MLContext for this method
			ml.registerOutput("means");
			HashMap<String, String> param = new HashMap<String, String>();
			param.put("dfam", "3");
			
			// ------------------------------------------------------------------------------------
			// Please note that this logic is subject to change and is put as a placeholder
			String systemmlHome = System.getenv("SYSTEMML_HOME");
			if(systemmlHome == null) {
				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
				return null;
			}
			// Or add ifdef in GLM-predict.dml
			param.put("X", " ");
			param.put("B", " ");
						
			String dmlFilePath = systemmlHome + File.separator + "algorithms" + File.separator + "GLM-predict.dml";
			// ------------------------------------------------------------------------------------
			MLOutput out = ml.execute(dmlFilePath, param);
			
			SQLContext sqlContext = new SQLContext(sc);
			DataFrame prob = out.getDF(sqlContext, "means", true).withColumnRenamed("C1", "probability");
			
			MLContext mlNew = new MLContext(sc);
			mlNew.registerInput("X", Xin, mcXin);
			mlNew.registerInput("B_full", b_out, b_outMC);
			mlNew.registerInput("Prob", out.getBinaryBlockedRDD("means"), out.getMatrixCharacteristics("means"));
			mlNew.registerOutput("Prediction");
			mlNew.registerOutput("rawPred");
			MLOutput outNew = mlNew.executeScript("Prob = read(\"temp1\"); "
					+ "Prediction = rowIndexMax(Prob); "
					+ "write(Prediction, \"tempOut\", \"csv\")"
					+ "X = read(\"temp2\");"
					+ "B_full = read(\"temp3\");"
					+ "rawPred = 1 / (1 + exp(- X * t(B_full)) );" // Raw prediction logic: 
					+ "write(rawPred, \"tempOut1\", \"csv\")");
			
			// TODO: Perform joins in the DML
			DataFrame pred = outNew.getDF(sqlContext, "Prediction").withColumnRenamed("C1", "prediction").withColumnRenamed("ID", "ID1");
			DataFrame rawPred = outNew.getDF(sqlContext, "rawPred", true).withColumnRenamed("C1", "rawPrediction").withColumnRenamed("ID", "ID2");
			DataFrame predictionsNProb = prob.join(pred, prob.col("ID").equalTo(pred.col("ID1"))).select("ID", "probability", "prediction");
			predictionsNProb = predictionsNProb.join(rawPred, predictionsNProb.col("ID").equalTo(rawPred.col("ID2"))).select("ID", "probability", "prediction", "rawPrediction");
			DataFrame dataset1 = RDDConverterUtils.addIDToDataFrame(dataset, sqlContext, "ID");			
			return dataset1.join(predictionsNProb, dataset1.col("ID").equalTo(predictionsNProb.col("ID"))).orderBy("id");
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
