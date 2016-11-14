package org.apache.sysml.api.ml.feature;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Transformer;
//import org.apache.spark.ml.param.IntArrayParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.runtime.DMLRuntimeException;

import org.apache.sysml.api.ml.param.FeatureIndexerParams;

import scala.collection.Seq;

public class FeatureIndexer extends Transformer implements FeatureIndexerParams {

	private static final long serialVersionUID = 5251196526063795635L;

	private SparkContext sc = null;
	private HashMap<String, String> params = new HashMap<String, String>();
	private Param<String> inputCol = new Param<String>(this, "inputCol", "Input column name");
	private Param<String> outputCol = new Param<String>(this, "outputCol", "Output column name");
	private IntParam featureIndicesRangeStart =
			new IntParam(this, "fiRangeStart", "Starting index of the feature column range");
	private IntParam featureIndicesRangeEnd =
			new IntParam(this, "fiRangeEnd", "Starting index of the feature column range");
//	private IntArrayParam featureIndicesArray =
//			new IntArrayParam(this, "fiArray", "A list of feature columns");

	private int start, end;
	private List<Integer> arr;

	public FeatureIndexer(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;
		setInputCol("features");
		setOutputCol("featureIndices");
		start = -1;
		end = -1;
		arr = new ArrayList<Integer>();
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public Transformer copy(ParamMap paramMap) {
		try {
			FeatureIndexer fi = new FeatureIndexer(sc);

			fi.start = start;
			fi.end = end;
			fi.arr = arr;
			fi.setInputCol(getInputCol());
			fi.setOutputCol(getOutputCol());

			return fi;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}

		return null;
		// return defaultCopy(paramMap);
	}

	public FeatureIndexer setInputCol(String value) {
		params.put(inputCol.name(), value);
		return (FeatureIndexer) setDefault(inputCol, value);
	}

	@Override
	public Param<String> inputCol() {
		return inputCol;
	}

	@Override
	public String getInputCol() {
		return params.get(inputCol.name());
	}

	public FeatureIndexer setOutputCol(String value) {
		params.put(outputCol.name(), value);
		return (FeatureIndexer) setDefault(outputCol, value);
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

	@Override
	public void org$apache$spark$ml$param$shared$HasInputCol$_setter_$inputCol_$eq(Param arg0) {

	}

	public FeatureIndexer setFeatureIndicesRangeStart(int value) {
		start = value;
		return (FeatureIndexer) setDefault(featureIndicesRangeStart, value);
	}

	@Override
	public IntParam featureIndicesRangeStart() {
		return featureIndicesRangeStart;
	}

	@Override
	public int getFeatureIndicesRangeStart() {
		return start;
	}

	public FeatureIndexer setFeatureIndicesRangeEnd(int value) {
		end = value;
		return (FeatureIndexer) setDefault(featureIndicesRangeEnd, value);
	}

	@Override
	public IntParam featureIndicesRangeEnd() {
		return featureIndicesRangeEnd;
	}

	@Override
	public int getFeatureIndicesRangeEnd() {
		return end;
	}

	public FeatureIndexer setFeatureIndices(Seq<Integer> value) {
		arr = scala.collection.JavaConversions.asJavaList(value);
		return (FeatureIndexer) setDefault(featureIndicesRangeEnd, value);
	}

	public FeatureIndexer setFeatureIndices(List<Integer> value) {
		arr = value;
		return (FeatureIndexer) setDefault(featureIndicesRangeEnd, value);
	}

//	@Override
//	public IntArrayParam featureIndices() {
//		return featureIndicesArray;
//	}

//	@Override
//	public List<Integer> getFeatureIndices() {
//		return arr;
//	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		StructType outputSchema = transformSchema(dataset.schema());
		List<Double> indicesList = new ArrayList<Double>();
		List<Row> featuresList = dataset.rdd().toJavaRDD().collect();
		List<Row> resultList = new ArrayList<Row>();

		long rowCount = dataset.count();
		long rangeCount = 0, countDiff = 0;

		if (start != -1 && end != -1) {
			rangeCount = (long) end - (long) start + 1;

			for (int i = start; i <= end; i++)
				indicesList.add((double) i);
		} else if (!arr.isEmpty()) {
			rangeCount = arr.size();

			for (Integer a : arr)
				indicesList.add((double) a);
		}

		long lastIndex = featuresList.get((int) rowCount - 1).getLong(0);

		countDiff = Math.abs(rangeCount - rowCount);

		if (rowCount > rangeCount)
			for (long i = 0; i < countDiff; i++)
				indicesList.add(Double.NaN);
		else
			for (long i = 0; i < countDiff; i++)
				featuresList.add(RowFactory.create(lastIndex + i + 1, null));

		if (rangeCount == 0)
			System.err.println("Please provide range of integers or an array(list) of doubles");

		long featuresListLength = featuresList.size();

		if (indicesList.size() != featuresListLength)
			System.err.println(
					"Lengths of encoded input column and encoded feature indices column do not match");

		for (int i = 0; i < (int) featuresListLength; i++) {
			long idx = featuresList.get(i).getLong(0);
			if (idx > lastIndex)
				resultList.add(RowFactory.create(featuresList.get(i).get(0),
						null,
						indicesList.get(i)));
			else
				resultList.add(RowFactory.create(featuresList.get(i).get(0),
						featuresList.get(i).get(1),
						indicesList.get(i)));
		}

		@SuppressWarnings("resource")
		JavaSparkContext jsc = new JavaSparkContext(sc);
		JavaRDD<Row> row = jsc.parallelize(resultList);

		return dataset.sqlContext().createDataFrame(row, outputSchema);
	}

	@Override
	public StructType transformSchema(StructType schema) {
		return SchemaUtils.appendColumn(schema,
				DataTypes.createStructField(getOutputCol(), DataTypes.DoubleType, true));
	}
}
