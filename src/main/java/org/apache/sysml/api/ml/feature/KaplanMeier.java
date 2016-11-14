package org.apache.sysml.api.ml.feature;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.DoubleParam;
//import org.apache.spark.ml.param.IntArrayParam;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.LongParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.api.ml.param.KaplanMeierParams;

import scala.collection.Seq;

public class KaplanMeier extends Transformer implements KaplanMeierParams {

	private static final long serialVersionUID = 6216389064589166876L;

	private SparkContext sc = null;
	private HashMap<String, String> cmdLineParams = new HashMap<String, String>();

	private Param<String> inputCol = new Param<String>(
			this, "inputCol", "Input column name");
	private Param<String> outputCol = new Param<String>(
			this, "outputCol", "Output column name");
	private DoubleParam alpha = new DoubleParam(
			this, "alpha", "Parameter to compute a 100*(1-alpha)% confidence interval for the betas");
	private Param<String> errorType = new Param<String>(
			this, "etype", "The error type, either \"greenwood\" or \"peto\"");
	private Param<String> ciType = new Param<String>(
			this, "ctype", "Parameter to modify the confidence interval; \"plain\" keeps the lower "
					+ "and upper bound of the confidence interval unmodified, \"log\" (the "
					+ "default) corresponds to logistic transformation and \"log-log\" "
					+ "corresponds to the complementary log-log transformation");
	private Param<String> testType = new Param<String>(
			this, "ttype", "If survival data for multiple groups is available specifies which test "
					+ "to perform for comparing survival data across multiple groups: \"none\""
					+ " (the default) \"log-rank\" or \"wilcoxon\" test");
	private Param<String> giCol = new Param<String>(
			this, "giCol", "The name of the column that has indices of the feature vector "
					+ "corresponding to the factors to be used for grouping");
	private Param<String> siCol = new Param<String>(
			this, "siCol", "The name of the column that has indices of the feature vector "
					+ "corresponding to the factors to be used for stratifying");
	private Param<String> teCol = new Param<String>(
			this, "teCol", "The name of the column that has indices of the timestamps (first entry) "
					+ "and event information (second entry) in the feature vector");
	private IntParam groupIndicesRangeStart = new IntParam(
			this, "giRangeStart", "Starting index of the " + "group indices range");
	private IntParam groupIndicesRangeEnd = new IntParam(
			this, "giRangeEnd", "Starting index of the feature" + " indices range");
//	private IntArrayParam groupIndicesArray = new IntArrayParam(
//			this, "giArray", "A list of group " + "indices");
	private IntParam stratifyIndicesRangeStart = new IntParam(
			this, "siRangeStart", "Starting index of the " + "stratify indices range");
	private IntParam stratifyIndicesRangeEnd = new IntParam(
			this, "siRangeEnd", "Starting index of the stratify" + " indices range");
//	private IntArrayParam stratifyIndicesArray = new IntArrayParam(
//			this, "siArray", "A list of stratify indices");
	private LongParam timestampIndex = new LongParam(
			this, "timestampIndex", "Index of the timestamp in the feature " + "vector");
	private LongParam eventIndex = new LongParam(
			this, "eventIndex", "Index of the timestamp in the feature vector");

	private int giStart, giEnd, siStart, siEnd;
	private List<Integer> giList, siList;

	public KaplanMeier(SparkContext sc) throws DMLRuntimeException {
		this.sc = sc;
		setAllParameters(0.05f, "greenwood", "log", "none");
	}

	public KaplanMeier(SparkContext sc, double alpha, String errorType, String ciType, String testType)
			throws DMLRuntimeException {
		this.sc = sc;

		setAllParameters(alpha, errorType, ciType, testType);
	}

	private void setAllParameters(double alpha, String errorType, String ciType, String testType) {
		setDefault(alpha(), alpha);
		cmdLineParams.put(this.alpha.name(), Double.toString(alpha));
		setDefault(errorType(), errorType);
		cmdLineParams.put(this.errorType.name(), errorType);
		setDefault(CIType(), ciType);
		cmdLineParams.put(this.ciType.name(), ciType);
		setDefault(testType(), testType);
		cmdLineParams.put(this.testType.name(), testType);
		setGICol("groupIndices");
		setSICol("stratifyIndices");
		setTECol("tsAndEventIndices");
		giStart = -1;
		giEnd = -1;
		siStart = -1;
		siEnd = -1;
		giList = new ArrayList<Integer>();
		siList = new ArrayList<Integer>();
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public KaplanMeier copy(ParamMap paramMap) {
		try {
			String strAlpha = paramMap.getOrElse(alpha, getAlpha()).toString();
			String strErrorType = paramMap.getOrElse(errorType, getErrorType()).toString();
			String strCIType = paramMap.getOrElse(ciType, getCIType()).toString();
			String strTestType = paramMap.getOrElse(testType, getTestType()).toString();

			KaplanMeier km = new KaplanMeier(
					sc, Double.parseDouble(strAlpha), strErrorType, strCIType, strTestType);

			km.cmdLineParams.put(alpha.name(), strAlpha);
			km.cmdLineParams.put(errorType.name(), strErrorType);
			km.cmdLineParams.put(ciType.name(), strCIType);
			km.cmdLineParams.put(testType.name(), strTestType);
			km.setInputCol(getInputCol());
			km.setOutputCol(getOutputCol());
			km.setTECol(getTECol());
			km.setGICol(getGICol());
			km.setSICol(getSICol());
			km.giStart = giStart;
			km.siStart = siStart;
			km.giEnd = giEnd;
			km.siEnd = siEnd;
			km.giList = giList;
			km.siList = siList;

			return km;
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
		}

		return null;
	}

	public KaplanMeier setInputCol(String value) {
		cmdLineParams.put(inputCol.name(), value);
		return (KaplanMeier) setDefault(inputCol, value);
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
		return cmdLineParams.get(inputCol.name());
	}

	public KaplanMeier setOutputCol(String value) {
		cmdLineParams.put(outputCol.name(), value);
		return (KaplanMeier) setDefault(outputCol, value);
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
		return cmdLineParams.get(outputCol.name());
	}

	public KaplanMeier setAlpha(double value) {
		cmdLineParams.put(alpha.name(), Double.toString(value));
		return (KaplanMeier) setDefault(alpha, value);
	}

	@Override
	public DoubleParam alpha() {
		return alpha;
	}

	@Override
	public double getAlpha() {
		return Double.parseDouble(cmdLineParams.get(alpha.name()));
	}

	public KaplanMeier setErrorType(String value) {
		cmdLineParams.put(errorType.name(), value);
		return (KaplanMeier) setDefault(errorType, value);
	}

	@Override
	public Param<String> errorType() {
		return errorType;
	}

	@Override
	public String getErrorType() {
		return cmdLineParams.get(errorType.name());
	}

	public KaplanMeier setCIType(String value) {
		cmdLineParams.put(ciType.name(), value);
		return (KaplanMeier) setDefault(ciType, value);
	}

	@Override
	public Param<String> CIType() {
		return ciType;
	}

	@Override
	public String getCIType() {
		return cmdLineParams.get(ciType.name());
	}

	public KaplanMeier setTestType(String value) {
		cmdLineParams.put(testType.name(), value);
		return (KaplanMeier) setDefault(testType, value);
	}

	@Override
	public Param<String> testType() {
		return testType;
	}

	@Override
	public String getTestType() {
		return cmdLineParams.get(testType.name());
	}

	public KaplanMeier setTECol(String value) {
		cmdLineParams.put(teCol.name(), value);
		return (KaplanMeier) setDefault(teCol, value);
	}

	@Override
	public Param<String> teCol() {
		return teCol;
	}

	@Override
	public String getTECol() {
		return cmdLineParams.get(teCol.name());
	}

	public KaplanMeier setGICol(String value) {
		cmdLineParams.put(giCol.name(), value);
		return (KaplanMeier) setDefault(giCol, value);
	}

	@Override
	public Param<String> giCol() {
		return giCol;
	}

	@Override
	public String getGICol() {
		return cmdLineParams.get(giCol.name());
	}

	public KaplanMeier setSICol(String value) {
		cmdLineParams.put(siCol.name(), value);
		return (KaplanMeier) setDefault(siCol, value);
	}

	@Override
	public Param<String> siCol() {
		return siCol;
	}

	@Override
	public String getSICol() {
		return cmdLineParams.get(siCol.name());
	}

	public KaplanMeier setGroupIndicesRangeStart(int value) {
		giStart = value;
		return (KaplanMeier) setDefault(groupIndicesRangeStart, value);
	}

	@Override
	public IntParam groupIndicesRangeStart() {
		return groupIndicesRangeStart;
	}

	@Override
	public int getGroupIndicesRangeStart() {
		return giStart;
	}

	public KaplanMeier setGroupIndicesRangeEnd(int value) {
		giEnd = value;
		return (KaplanMeier) setDefault(groupIndicesRangeEnd, value);
	}

	@Override
	public IntParam groupIndicesRangeEnd() {
		return groupIndicesRangeEnd;
	}

	@Override
	public int getGroupIndicesRangeEnd() {
		return giEnd;
	}

	@Override
	public List<Integer> getGroupIndices() {
		return giList;
	}

	public KaplanMeier setGroupIndices(Seq<Integer> value) {
		giList = scala.collection.JavaConversions.asJavaList(value);
		return (KaplanMeier) setDefault(groupIndicesRangeEnd, value);
	}

	public KaplanMeier setGroupIndices(List<Integer> value) {
		giList = value;
		return (KaplanMeier) setDefault(groupIndicesRangeEnd, value);
	}

	public KaplanMeier setStratifyIndicesRangeStart(int value) {
		siStart = value;
		return (KaplanMeier) setDefault(stratifyIndicesRangeStart, value);
	}

	@Override
	public IntParam stratifyIndicesRangeStart() {
		return stratifyIndicesRangeStart;
	}

	@Override
	public int getStratifyIndicesRangeStart() {
		return siStart;
	}

	public KaplanMeier setStratifyIndicesRangeEnd(int value) {
		siEnd = value;
		return (KaplanMeier) setDefault(stratifyIndicesRangeEnd, value);
	}

	@Override
	public IntParam stratifyIndicesRangeEnd() {
		return stratifyIndicesRangeEnd;
	}

	@Override
	public int getStratifyIndicesRangeEnd() {
		return siEnd;
	}

	@Override
	public List<Integer> getStratifyIndices() {
		return siList;
	}

	public KaplanMeier setStratifyIndices(Seq<Integer> value) {
		siList = scala.collection.JavaConversions.asJavaList(value);
		return (KaplanMeier) setDefault(stratifyIndicesRangeEnd, value);
	}

	public KaplanMeier setStratifyIndices(List<Integer> value) {
		siList = value;
		return (KaplanMeier) setDefault(stratifyIndicesRangeEnd, value);
	}

	public KaplanMeier setTimestampIndex(long value) {
		cmdLineParams.put(timestampIndex.name(), Long.toString(value));
		return (KaplanMeier) setDefault(timestampIndex, value);
	}

	@Override
	public LongParam timestampIndex() {
		return timestampIndex;
	}

	@Override
	public long getTimestampIndex() {
		return Long.parseLong(cmdLineParams.get(timestampIndex.name()));
	}

	public KaplanMeier setEventIndex(long value) {
		cmdLineParams.put(eventIndex.name(), Long.toString(value));
		return (KaplanMeier) setDefault(eventIndex, value);
	}

	@Override
	public LongParam eventIndex() {
		return eventIndex;
	}

	@Override
	public long getEventIndex() {
		return Long.parseLong(cmdLineParams.get(eventIndex.name()));
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		try {
			MLContext ml = null;
			MLOutput out = null;
			JavaSparkContext jsc = new JavaSparkContext(
					sc);

			String inputColName = getInputCol();
			String giColName = getGICol();
			String siColName = getSICol();
			String teColName = getTECol();

			try {
				ml = new MLContext(
						sc);
			} catch (DMLRuntimeException e1) {
				e1.printStackTrace();
			}

			// Convert input data to format that SystemML accepts
			MatrixCharacteristics mcXin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> Xin;
			Xin = RDDConverterUtils.dataFrameToBinaryBlock(jsc,
					dataset.filter(
							inputColName + " is not null")
							.select(inputColName),
					mcXin,
					false,
					true);

			List<Row> teRowList = new ArrayList<Row>();
			long tsValue = getTimestampIndex();
			long eventValue = getEventIndex();

			if (tsValue < 0 || eventValue < 0)
				System.err.println(
						"The indices of the Timestamp column and the Event column have to be positive "
								+ "values");
			else {
				teRowList.add(RowFactory.create((double) tsValue));
				teRowList.add(RowFactory.create((double) eventValue));
			}

			JavaRDD<Row> teRow = jsc.parallelize(teRowList);
			List<StructField> teFields = new ArrayList<StructField>();
			teFields.add(DataTypes.createStructField(getTECol(), DataTypes.DoubleType, true));
			StructType teSchema = DataTypes.createStructType(teFields);
			DataFrame teDF = dataset.sqlContext().createDataFrame(teRow, teSchema);

			// Convert input data to format that SystemML accepts
			MatrixCharacteristics mcTEin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> TEin;
			TEin = RDDConverterUtils.dataFrameToBinaryBlock(jsc,
					teDF.select(teColName),
					mcTEin,
					false,
					true);

			List<Row> giRowList = new ArrayList<Row>();

			if (giStart != -1 && giEnd != -1)
				for (int i = giStart; i <= giEnd; i++)
					giRowList.add(RowFactory.create((double) i));
			else if (!giList.isEmpty())
				for (Integer i : giList)
					giRowList.add(RowFactory.create((double) i));
			else
				System.err.println(
						"Please provide range of integers or an array(list) of integers");

			JavaRDD<Row> giRow = jsc.parallelize(giRowList);
			List<StructField> giFields = new ArrayList<StructField>();
			giFields.add(DataTypes.createStructField(getGICol(), DataTypes.DoubleType, true));
			StructType giSchema = DataTypes.createStructType(giFields);
			DataFrame giDF = dataset.sqlContext().createDataFrame(giRow, giSchema);

			// Convert input data to format that SystemML accepts
			MatrixCharacteristics mcGIin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> GIin;
			try {
				GIin = RDDConverterUtils.dataFrameToBinaryBlock(jsc,
						giDF.select(giColName),
						mcGIin,
						false,
						true);
				cmdLineParams.put("GI", "GI");
				ml.registerInput("GI", GIin, mcGIin);
			} catch (DMLRuntimeException e1) {
				e1.printStackTrace();
				return null;
			}

			List<Row> siRowList = new ArrayList<Row>();

			if (siStart != -1 && siEnd != -1)
				for (int i = siStart; i <= siEnd; i++)
					siRowList.add(RowFactory.create((double) i));
			else if (!siList.isEmpty())
				for (Integer i : siList)
					siRowList.add(RowFactory.create((double) i));
			else
				System.err.println(
						"Please provide range of integers or an array(list) of integers");

			JavaRDD<Row> siRow = jsc.parallelize(siRowList);
			List<StructField> siFields = new ArrayList<StructField>();
			siFields.add(DataTypes.createStructField(getSICol(), DataTypes.DoubleType, true));
			StructType siSchema = DataTypes.createStructType(siFields);
			DataFrame siDF = dataset.sqlContext().createDataFrame(siRow, siSchema);

			// Convert input data to format that SystemML accepts
			MatrixCharacteristics mcSIin = new MatrixCharacteristics();
			JavaPairRDD<MatrixIndexes, MatrixBlock> SIin;
			try {
				SIin = RDDConverterUtils.dataFrameToBinaryBlock(jsc,
						siDF.select(siColName),
						mcSIin,
						false,
						true);
				cmdLineParams.put("SI", "SI");
				ml.registerInput("SI", SIin, mcSIin);
			} catch (DMLRuntimeException e1) {
				e1.printStackTrace();
				return null;
			}

			// Register the input/output variables of script
			// 'Cox.dml'
			ml.registerInput("X", Xin, mcXin);
			ml.registerInput("TE", TEin, mcTEin);
			ml.registerOutput("M");
			ml.registerOutput("KM");

//			String systemmlHome = System.getenv("SYSTEMML_HOME");
//			if (systemmlHome == null) {
//				System.err.println("ERROR: The environment variable SYSTEMML_HOME is not set.");
//				return null;
//			}

			// Or add ifdef in Cox.dml
			cmdLineParams.put("X", " ");
			cmdLineParams.put("TE", " ");
			cmdLineParams.put("M", " ");
			cmdLineParams.put("O", " ");

//			String dmlFilePath = systemmlHome + File.separator + "scripts" + File.separator + "algorithms" + File.separator + "KM.dml";
			String dmlFilePath = "scripts" + File.separator + "algorithms" + File.separator + "KM.dml";

			synchronized (MLContext.class) {
				// static synchronization is necessary before
				// execute call
				out = ml.execute(dmlFilePath, cmdLineParams);
			}

			DataFrame results = out.getDF(dataset.sqlContext(), "M");
			results = results.withColumnRenamed("C1", "timestamp");
			results = results.withColumnRenamed("C2", "no. at risk");
			results = results.withColumnRenamed("C3", "no. of events");
			results = results.withColumnRenamed("C4", "surv estimation(surv)");
			results = results.withColumnRenamed("C5", "SE of surv");
			results = results.withColumnRenamed("C6", "lower CI or surv");
			results = results.withColumnRenamed("C7", "upper CI or surv");

			return results;
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		} catch (DMLException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public StructType transformSchema(StructType schema) {
		return null;
	}
}
