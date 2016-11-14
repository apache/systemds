package org.apache.sysml.api.ml.regression;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PredictionModel;
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
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.functions.ConvertRowToDouble;

import scala.Tuple2;

public class CoxRegressionModel extends PredictionModel<Vector, CoxRegressionModel> {

	private static final long serialVersionUID = 2074066962664167846L;

	private SparkContext sc = null;
	private HashMap<String, String> params;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();
	private String labelCol = "";

	public CoxRegressionModel(
			HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results,
			SparkContext sc, HashMap<String, String> params, String featuresCol, String labelCol) {
		this.results = results;
		this.sc = sc;
		this.params = params;
		this.labelCol = labelCol;

		setFeaturesCol(featuresCol);
	}

	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public CoxRegressionModel copy(ParamMap paramMap) {
		return new CoxRegressionModel(
				results, sc, params, getFeaturesCol(), labelCol);
	}

	@Override
	public StructType validateAndTransformSchema(StructType arg0, boolean arg1, DataType arg2) {
		return null;
	}

	@Override
	public double predict(Vector arg0) {
		System.out.println("predict is currently not supported");
		return 0;
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			scala.collection.immutable.List<StructField> schema = dataset.schema().toList();
			List<StructField> fields = new ArrayList<StructField>();
			int requiredCol = 0;
			boolean labelExists = false;

			for (int i = 0; i < schema.size(); i++) {
				StructField element = schema.apply(i);

				if (element.name() == "id") {
					fields.add(element);
					requiredCol++;
				} else if (element.name() == labelCol) {
					fields.add(element);
					requiredCol++;
					labelExists = true;
				}
			}

			fields.add(DataTypes.createStructField("Linear Predictors(LP)",
					DataTypes.DoubleType,
					true));
			fields.add(DataTypes.createStructField("Standard Error of LP",
					DataTypes.DoubleType,
					true));
			fields.add(DataTypes.createStructField("Risk", DataTypes.DoubleType, true));
			fields.add(DataTypes.createStructField("Standard Error of Risk",
					DataTypes.DoubleType,
					true));
			fields.add(DataTypes.createStructField("Cumulative Hazard(CH)",
					DataTypes.DoubleType,
					true));
			fields.add(DataTypes.createStructField("Standard Error of CH",
					DataTypes.DoubleType,
					true));

			StructType outputSchema = DataTypes.createStructType(fields);

			MLContext ml = new MLContext(sc);

			if (!labelExists)
				System.err.println("The dataset doesn't have the label column.");

			MatrixCharacteristics mcYin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Yin;

			Yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
					dataset.filter(labelCol + " is not null")
							.select(labelCol),
					mcYin,
					false,
					true);

			ml.registerInput("Y_orig", Yin, mcYin);

			for (Map.Entry<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> entry : results
					.entrySet())
				ml.registerInput(entry.getKey(), entry.getValue()._1, entry.getValue()._2);

			ml.registerOutput("P");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			cmdLineParams.put("X", " ");
			cmdLineParams.put("RT", " ");
			cmdLineParams.put("M", " ");
			cmdLineParams.put("Y", " ");
			cmdLineParams.put("COV", " ");
			cmdLineParams.put("MF", " ");
			cmdLineParams.put("P", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "Cox-predict.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "Cox-predict.dml";
			MLOutput out = ml.execute(dmlFilePath, cmdLineParams);

			List<Row> rowList = dataset.rdd().toJavaRDD().collect();
			List<Row> resultList = new ArrayList<Row>();
			List<Double> pred = out.getStringRDD("P", "text").map(new ConvertRowToDouble()).collect();
			long nRow = out.getMatrixCharacteristics("P").getRows();
			long nCol = out.getMatrixCharacteristics("P").getCols();
			int colCount = outputSchema.size();

			for (int i = 0; i < nRow; i++) {
				Object[] rowArray = new Object[colCount];

				for (int j = 0; j < requiredCol; j++)
					rowArray[j] = rowList.get(i).get(j);

				for (int k = requiredCol; k < colCount; k++)
					rowArray[k] = pred.get((i * (int) nCol) + (k - requiredCol));

				resultList.add(RowFactory.create(rowArray));
			}

			@SuppressWarnings("resource")
			JavaRDD<Row> row = (new JavaSparkContext(sc)).parallelize(resultList);

			return dataset.sqlContext().createDataFrame(row, outputSchema);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}