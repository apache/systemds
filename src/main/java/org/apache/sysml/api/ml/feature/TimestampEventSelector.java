package org.apache.sysml.api.ml.feature;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.LongParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import org.apache.sysml.api.ml.param.TimestampEventSelectorParams;

public class TimestampEventSelector extends Transformer implements TimestampEventSelectorParams {

	private static final long serialVersionUID = -2112395361172795955L;

	private SparkContext sc = null;
	private HashMap<String, String> params = new HashMap<String, String>();
	private Param<String> inputCol = new Param<String>(this, "inputCol", "Input column name");
	private Param<String> outputCol = new Param<String>(this, "outputCol", "Output column name");
	private LongParam timestampIndex =
			new LongParam(this, "timestampIndex", "Index of the timestamp in the feature vector");
	private LongParam eventIndex =
			new LongParam(this, "eventIndex", "Index of the timestamp in the feature vector");

	public TimestampEventSelector(SparkContext sc) {
		this.sc = sc;
		setInputCol("featureIndices");
		setOutputCol("tsAndEventIndices");
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public Transformer copy(ParamMap paramMap) {
		return defaultCopy(paramMap);
	}

	public TimestampEventSelector setInputCol(String value) {
		params.put(inputCol.name(), value);
		return (TimestampEventSelector) setDefault(inputCol, value);
	}

	@Override
	public Param<String> inputCol() {
		return inputCol;
	}

	@Override
	public String getInputCol() {
		return params.get(inputCol.name());
	}

	public TimestampEventSelector setOutputCol(String value) {
		params.put(outputCol.name(), value);
		return (TimestampEventSelector) setDefault(outputCol, value);
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

	public TimestampEventSelector setTimestampIndex(long value) {
		params.put(timestampIndex.name(), Long.toString(value));
		return (TimestampEventSelector) setDefault(timestampIndex, value);
	}

	@Override
	public LongParam timestampIndex() {
		return timestampIndex;
	}

	@Override
	public long getTimestampIndex() {
		return Long.parseLong(params.get(timestampIndex.name()));
	}

	public TimestampEventSelector setEventIndex(long value) {
		params.put(eventIndex.name(), Long.toString(value));
		return (TimestampEventSelector) setDefault(eventIndex, value);
	}

	@Override
	public LongParam eventIndex() {
		return eventIndex;
	}

	@Override
	public long getEventIndex() {
		return Long.parseLong(params.get(eventIndex.name()));
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		StructType outputSchema = transformSchema(dataset.schema());
		List<Double> timestampAndEventList = new ArrayList<Double>();
		List<Row> featuresList = dataset.rdd().toJavaRDD().collect();
		List<Row> resultList = new ArrayList<Row>();

		long rowCount = dataset.count();
		timestampAndEventList.add((double) getTimestampIndex());
		timestampAndEventList.add((double) getEventIndex());
		long tsAndEventCount = timestampAndEventList.size();

		long countDiff = Math.max(rowCount, tsAndEventCount) - Math.min(rowCount, tsAndEventCount);
		long lastIndex = featuresList.get((int) rowCount - 1).getLong(0);

		if (rowCount > tsAndEventCount)
			for (long i = 0; i < countDiff; i++)
				timestampAndEventList.add(Double.NaN);
		else
			for (long i = 0; i < countDiff; i++)
				featuresList.add(RowFactory.create(lastIndex + i + 1, null));

		long featuresListLength = featuresList.size();

		if (timestampAndEventList.size() != featuresListLength)
			System.err.println(
					"Lengths of encoded input column and encoded feature indices column do not match");

		for (int i = 0; i < (int) featuresListLength; i++) {
			long idx = featuresList.get(i).getLong(0);
			if (idx > lastIndex)
				resultList.add(RowFactory.create(featuresList.get(i).get(0),
						null,
						Double.NaN,
						timestampAndEventList.get(i)));
			else
				resultList.add(RowFactory.create(featuresList.get(i).get(0),
						featuresList.get(i).get(1),
						featuresList.get(i).get(2),
						timestampAndEventList.get(i)));
		}

		@SuppressWarnings("resource")
		JavaRDD<Row> row = (new JavaSparkContext(sc)).parallelize(resultList);
		DataFrame result = dataset.sqlContext().createDataFrame(row, outputSchema);

		return result;
	}

	@Override
	public StructType transformSchema(StructType schema) {
		return SchemaUtils.appendColumn(schema,
				DataTypes.createStructField(getOutputCol(), DataTypes.DoubleType, true));
	}
}
