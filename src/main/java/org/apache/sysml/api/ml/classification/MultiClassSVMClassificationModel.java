package org.apache.sysml.api.ml.classification;

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
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
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
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

public class MultiClassSVMClassificationModel extends PredictionModel<Vector, MultiClassSVMClassificationModel> {

	private static final long serialVersionUID = -3961142158204470999L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();
	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();
	// private JavaPairRDD<MatrixIndexes, MatrixBlock> model;
	// private MatrixCharacteristics modelMC;
	private String labelCol = "";

	// public MultiClassSVMClassificationModel(JavaPairRDD<MatrixIndexes,
	// MatrixBlock> model, MatrixCharacteristics modelMC, String
	// featuresCol, String labelCol, SparkContext sc) {
	public MultiClassSVMClassificationModel(
			HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results,
			SparkContext sc,
			String featuresCol,
			String labelCol) {
		// this.model = model;
		// this.modelMC = modelMC;
		this.results = results;
		this.labelCol = labelCol;
		this.sc = sc;

		setFeaturesCol(featuresCol);
	}

	@Override
	public StructType validateAndTransformSchema(StructType arg0, boolean arg1, DataType arg2) {
		return null;
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public MultiClassSVMClassificationModel copy(ParamMap arg0) {
		return new MultiClassSVMClassificationModel(results, sc, getFeaturesCol(), getLabelCol());
	}

	@Override
	public double predict(Vector arg0) {
		return 0.0;
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			MLContext ml = new MLContext(sc);

			MatrixCharacteristics mcXin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
			Xin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
					dataset.select(getFeaturesCol()),
					mcXin,
					false,
					true);

			boolean labelExists = false;

			for (String s : dataset.columns())
				if (s.equals(labelCol))
					labelExists = true;

			MatrixCharacteristics mcYin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Yin;
			ArrayList<String> cols = new ArrayList<String>();
			cols.add(labelCol);

			ClassTag<String> strClassTag = ClassTag$.MODULE$.apply(String.class);
			JavaRDD<String> emptyRDD = sc.emptyRDD(strClassTag).toJavaRDD();

			StructType outputSchema = SchemaUtils.appendColumn(dataset.schema(),
					DataTypes.createStructField("prediction", DataTypes.DoubleType, true));

			if (labelExists) {
				Yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
						dataset.select(getLabelCol()),
						mcYin,
						false,
						false);

				ml.registerInput("y", Yin, mcYin);
				ml.registerOutput("confusion_mat");
				cmdLineParams.put("Y", "Y");
				cmdLineParams.put("confusion", "confusion");
				outputSchema = SchemaUtils.appendColumn(outputSchema,
						DataTypes.createStructField("confusion_matrix",
								new VectorUDT(),
								true));
			} else
				ml.registerInput("y", emptyRDD, "csv");

			ml.registerInput("X", Xin, mcXin);
			ml.registerOutput("scores");

			for (Map.Entry<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> entry : results
					.entrySet())
				ml.registerInput(entry.getKey(), entry.getValue()._1, entry.getValue()._2);

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			cmdLineParams.put("X", " ");
			cmdLineParams.put("model", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "m-svm-predict.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "m-svm-predict.dml";
			MLOutput out = ml.execute(dmlFilePath, cmdLineParams);
			SQLContext sqlContext = dataset.sqlContext();

			List<Row> rowList = dataset.rdd().toJavaRDD().collect();
			List<Row> resultList = new ArrayList<Row>();
			List<Double> pred = out.getStringRDD("scores", "text").map(new ConvertRowToDouble())
					.collect();
			int colCount = outputSchema.size();

			long nRow = out.getMatrixCharacteristics("scores").getRows();
			long confRow = 0;
			List<Row> conf = null;

			if (labelExists) {
				confRow = out.getMatrixCharacteristics("confusion_mat").getRows();
				conf = out.getDF(sqlContext, "confusion_mat").rdd().toJavaRDD()
						.collect();
			}

			for (int i = 0; i < nRow; i++) {
				Object[] rowArray = new Object[colCount];
				int j = 0;
				int copyColCount = (labelExists) ? colCount - 2 : colCount - 1;

				for (j = 0; j < copyColCount; j++)
					rowArray[j] = rowList.get(i).get(j);

				rowArray[j++] = pred.get(i);

				if (labelExists) {
					if (i < confRow)
						rowArray[j++] = Vectors.parse(conf.get(i).toString());
					else
						rowArray[j++] = null;
				}

				resultList.add(RowFactory.create(rowArray));
			}

			@SuppressWarnings("resource")
			JavaRDD<Row> row = (new JavaSparkContext(sc)).parallelize(resultList);

			return sqlContext.createDataFrame(row, outputSchema);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}
