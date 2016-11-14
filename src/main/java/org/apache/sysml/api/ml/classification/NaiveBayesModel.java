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

public class NaiveBayesModel extends PredictionModel<Vector, NaiveBayesModel> {

	private static final long serialVersionUID = -3602071591280338973L;
	private SparkContext sc = null;

	private HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results =
			new HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>>();
	private HashMap<String, String> params = new HashMap<String, String>();
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();

	private String labelCol = "";

	public NaiveBayesModel(
			HashMap<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> results,
			String featuresCol, String labelCol, SparkContext sc, HashMap<String, String> params) {
		this.sc = sc;
		this.results = results;
		this.labelCol = labelCol;
		this.params = params;

		setFeaturesCol(featuresCol);
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public NaiveBayesModel copy(ParamMap paramMap) {
		return new NaiveBayesModel(results, getFeaturesCol(), labelCol, sc, params);
	}

	@Override
	public double predict(Vector arg0) {
		System.out.println("predict is not supported yet.");
		return 0;
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			StructType outputSchema = SchemaUtils.appendColumn(dataset.schema(),
					DataTypes.createStructField("probability", new VectorUDT(), true));
			outputSchema = SchemaUtils.appendColumn(outputSchema,
					DataTypes.createStructField("prediction", DataTypes.DoubleType, true));
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
			cols.add(this.getLabelCol());

			ClassTag<String> strClassTag = ClassTag$.MODULE$.apply(String.class);
			JavaRDD<String> emptyRDD = sc.emptyRDD(strClassTag).toJavaRDD();

			if (labelExists) {
				try {
					Yin = RDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(sc),
							dataset.select(getLabelCol()),
							mcYin,
							false,
							false);
					cmdLineParams.put("Y", "Y");
					cmdLineParams.put("accuracy", " ");
					cmdLineParams.put("confusion", " ");
					ml.registerInput("C", Yin, mcYin);
				} catch (DMLRuntimeException e1) {
					e1.printStackTrace();
					return null;
				}
			} else
				ml.registerInput("C", emptyRDD, "csv");

			for (Map.Entry<String, Tuple2<JavaPairRDD<MatrixIndexes, MatrixBlock>, MatrixCharacteristics>> entry : results
					.entrySet())
				ml.registerInput(entry.getKey(), entry.getValue()._1, entry.getValue()._2);

			ml.registerInput("D", Xin, mcXin);
			ml.registerOutput("probs");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in naive-bayes-predict.dml
			cmdLineParams.put("X", " ");
			cmdLineParams.put("prior", " ");
			cmdLineParams.put("conditionals", " ");
			cmdLineParams.put("probabilities", "probabilities");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "naive-bayes-predict.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "naive-bayes-predict.dml";
			MLOutput out = ml.execute(dmlFilePath, cmdLineParams);

			List<Row> rowList = dataset.rdd().toJavaRDD().collect();
			List<Row> resultList = new ArrayList<Row>();
			List<Double> pred = out.getStringRDD("probs", "text").map(new ConvertRowToDouble())
					.collect();
			long nRow = out.getMatrixCharacteristics("probs").getRows();
			long nCol = out.getMatrixCharacteristics("probs").getCols();
			int colCount = outputSchema.size();

			for (int i = 0; i < nRow; i++) {
				Object[] rowArray = new Object[colCount];
				double[] vec = new double[(int) nCol];
				int j = 0;
				double labelVal = -Double.MAX_VALUE;
				double argmax = -Double.MAX_VALUE;

				for (j = 0; j < colCount - 2; j++)
					rowArray[j] = rowList.get(i).get(j);

				for (int k = 0; k < nCol; k++) {
					double currVal = pred.get(2 * i + k);
					vec[k] = currVal;

					if (currVal > labelVal) {
						labelVal = currVal;
						argmax = (double) (k + 1);
					}
				}

				rowArray[j++] = Vectors.dense(vec);
				rowArray[j++] = argmax;

				resultList.add(RowFactory.create(rowArray));
			}

			@SuppressWarnings("resource")
			JavaRDD<Row> row = (new JavaSparkContext(sc)).parallelize(resultList);

			return dataset.sqlContext().createDataFrame(row, outputSchema);
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		}
	}
}
