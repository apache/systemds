/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Scanner;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.api.jmlc.JMLCUtils;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.rewrite.ProgramRewriter;
import com.ibm.bi.dml.hops.rewrite.RewriteRemovePersistentReadWrite;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.antlr4.DMLParserWrapper;
import com.ibm.bi.dml.parser.python.PyDMLParserWrapper;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContextFactory;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.spark.data.RDDObject;
import com.ibm.bi.dml.runtime.instructions.spark.data.RDDProperties;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertRowToCSVString;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertStringToLongTextPair;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CopyBlockFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CopyTextInputFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.SparkListener;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.utils.Explain;

import org.apache.spark.sql.DataFrame;
import scala.collection.Seq;
import scala.xml.Node;

/**
 * Typical usage for MLContext is as follows:
 * scala> import com.ibm.bi.dml.api.MLContext
 * 
 * Create input DataFrame from CSV file and potentially perform some feature transformation
 * scala> val W = sqlContext.load("com.databricks.spark.csv", Map("path" -> "W.csv", "header" -> "false"))
 * scala> val H = sqlContext.load("com.databricks.spark.csv", Map("path" -> "H.csv", "header" -> "false"))
 * scala> val V = sqlContext.load("com.databricks.spark.csv", Map("path" -> "V.csv", "header" -> "false"))
 * 
 * Create MLContext
 * scala> val ml = new MLContext(sc)
 * To monitor performance (only supported for Spark 1.4.0 or higher),
 * scala> val ml = new MLContext(sc, true)
 * To run SystemML using execution types different than Spark (for example: hybrid, hadoop or single node mode),
 * scala> val ml = new MLContext("hybrid") 
 * scala> val ml = new MLContext("hadoop")
 * scala> val ml = new MLContext("singlenode")
 * 
 * Register input and output DataFrame/RDD 
 * Supported format: 
 * 1. DataFrame
 * 2. CSV/Text (as JavaRDD<String> or JavaPairRDD<LongWritable, Text>)
 * 3. Binary blocked RDD (JavaPairRDD<MatrixIndexes,MatrixBlock>))
 * Also overloaded to support metadata information such as format, rlen, clen, ...
 * Please note the variable names given below in quotes correspond to the variables in DML script.
 * These variables need to have corresponding read/write associated in DML script.
 * Currently, only matrix variables are supported through registerInput/registerOutput interface.
 * To pass scalar variables, use named/positional arguments (described later) or wrap them into matrix variable.
 * scala> ml.registerInput("V", V)
 * scala> ml.registerInput("W", W)
 * scala> ml.registerInput("H", H)
 * scala> ml.registerOutput("H")
 * scala> ml.registerOutput("W")
 * 
 * Call script with default arguments:
 * scala> val outputs = ml.execute("GNMF.dml")
 * 
 * Also supported: calling script with positional arguments (args) and named arguments (nargs): 
 * scala> val args = Array("V.mtx", "W.mtx",  "H.mtx",  "2000", "1500",  "50",  "1",  "WOut.mtx",  "HOut.mtx")
 * scala> val nargs = Map("maxIter"->"1") 
 * scala> val outputs = ml.execute("GNMF.dml", args) # or ml.execute("GNMF.dml", nargs)  
 *  
 * If monitoring performance is enabled,
 * scala> print(ml.getExplainOutput())
 * scala> val stageId = 6                      # Assuming one of the stage id displayed above is '6'  
 * scala> print(ml.getStageDAGs(stageId))
 * scala> print(ml.getStageTimeLine(stageId))
 * 
 * To run the script again using different (or even same arguments), but using same registered input/outputs:
 * scala> val new_outputs = ml.execute("GNMF.dml", new_args)
 * 
 * However, to register new input/outputs, you need to first reset MLContext
 * scala> ml.reset()
 * scala> ml.registerInput("V", newV)
 * 
 */
public class MLContext {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	// ----------------------------------------------------
	public static SparkContext _sc = null; // Read while creating SystemML's spark context
	// ---------------------------------------------------- 
	
	private ArrayList<String> _inVarnames = null;
	private ArrayList<String> _outVarnames = null;
	private LocalVariableMap _variables = null; // temporary symbol table
	private boolean _parsePyDML = false;
	
	private SparkListener _sparkListener = null;
	private boolean _monitorPerformance = false;
	private String _explainOutput = "";
	
	public MLContext(SparkContext sc) throws DMLRuntimeException {
		initializeSpark(sc, false, false);
	}
	
	/**
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc
	 * @param monitorPerformance
	 * @throws DMLRuntimeException 
	 */
	public MLContext(SparkContext sc, boolean monitorPerformance) throws DMLRuntimeException {
		initializeSpark(sc, monitorPerformance, false);
	}
	
	public MLContext(JavaSparkContext sc) throws DMLRuntimeException {
		initializeSpark(sc.sc(), false, false);
	}
	
	public MLContext(JavaSparkContext sc, boolean monitorPerformance) throws DMLRuntimeException {
		initializeSpark(sc.sc(), monitorPerformance, false);
	}
	
	public MLContext(SparkContext sc, boolean monitorPerformance, boolean setForcedSparkExecType) throws DMLRuntimeException {
		initializeSpark(sc, monitorPerformance, setForcedSparkExecType);
	}
	
	public MLContext(JavaSparkContext sc, boolean monitorPerformance, boolean setForcedSparkExecType) throws DMLRuntimeException {
		initializeSpark(sc.sc(), monitorPerformance, setForcedSparkExecType);
	}
	
	
	private int compareVersion(String versionStr1, String versionStr2) {
		Scanner s1 = null;
		Scanner s2 = null;
		try {
			s1 = new Scanner(versionStr1); s1.useDelimiter("\\.");
			s2 = new Scanner(versionStr2); s2.useDelimiter("\\.");
			while(s1.hasNextInt() && s2.hasNextInt()) {
			    int version1 = s1.nextInt();
			    int version2 = s2.nextInt();
			    if(version1 < version2) {
			        return -1;
			    } else if(version1 > version2) {
			        return 1;
			    }
			}
	
			if(s1.hasNextInt()) return 1;
		}
		finally {
			if(s1 != null) s1.close();
			if(s2 != null) s2.close();
		}
		
		return 0;
	}
	
	private void initializeSpark(SparkContext sc, boolean monitorPerformance, boolean setForcedSparkExecType) throws DMLRuntimeException {
		if(MLContext._sc != null) {
			throw new DMLRuntimeException("Creating multiple MLContexts is not allowed in single process.");
		}
		MLContext._sc = sc;
		
		if(compareVersion(sc.version(), "1.3.0")  < 0 ) {
			throw new DMLRuntimeException("Expected spark version >= 1.3.0 for running SystemML");
		}
		
		if(setForcedSparkExecType)
			DMLScript.rtplatform = RUNTIME_PLATFORM.SPARK;
		else
			DMLScript.rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
		
		this._monitorPerformance = monitorPerformance;
		if(_monitorPerformance) {
			initializeSparkListener(sc);
		}
	}
	
	private void initializeSparkListener(SparkContext sc) throws DMLRuntimeException {
		if(compareVersion(sc.version(), "1.4.0")  < 0 ) {
			throw new DMLRuntimeException("Expected spark version >= 1.4.0 for monitoring MLContext performance");
		}
		_sparkListener = new SparkListener(sc);
		sc.addSparkListener(_sparkListener);
	}
	
	public MLContext(String execType) throws DMLRuntimeException {
		if(execType.compareTo("hybrid") == 0) {
			DMLScript.rtplatform = RUNTIME_PLATFORM.HYBRID;
		}
		else if(execType.compareTo("hadoop") == 0) {
			DMLScript.rtplatform = RUNTIME_PLATFORM.HADOOP;
		}
		else if(execType.compareTo("singlenode") == 0) {
			DMLScript.rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
		}
		else if(execType.compareTo("spark") == 0 || execType.compareTo("hybrid_spark") == 0) {
			throw new DMLRuntimeException("Error: The constructor MLContext(String) is not applicable for spark. Please use MLContext(sc) instead.");
		}
		else {
			throw new DMLRuntimeException("Unsupported execution type:" + execType + ". Valid options are: hybrid, hadoop, singlenode.");
		}
	}
	

	
	/**
	 * Call this method if you want to clear any RDDs set via registerInput, registerOutput.
	 * @throws DMLRuntimeException 
	 */
	public void reset() throws DMLRuntimeException {
		_inVarnames = null;
		_outVarnames = null;
		_variables = null;
	}
	
	private void resetMonitoringData() {
		if(_sparkListener != null && _sparkListener.stageDAGs != null)
			_sparkListener.stageDAGs.clear();
		if(_sparkListener != null && _sparkListener.stageTimeline != null)
			_sparkListener.stageTimeline.clear();
	}
	

	public Seq<Node> getStageDAGs(int stageIDs) {
		if(_sparkListener == null || _sparkListener.stageDAGs == null)
			return null;
		else
			return _sparkListener.stageDAGs.get(stageIDs);
	}
	
	public Seq<Node> getStageTimeLine(int stageIDs) {
		if(_sparkListener == null || _sparkListener.stageTimeline == null)
			return null;
		else
			return _sparkListener.stageTimeline.get(stageIDs);
	}
	
	// Note: Converting Java string to Python string might be expensive if this String is long 
	public String getExplainOutput() {
		return _explainOutput;
		// return explainOutput.replaceAll(Explain.lineSeparator, "\n").replaceAll(Explain.entrySeparator, "\n");
	}
		
	// ====================================================================================
	
	
	// 1. DataFrame
	public void registerInput(String varName, DataFrame df) throws DMLRuntimeException {
		JavaRDD<String> rdd = df.javaRDD().map(new ConvertRowToCSVString());
		registerInput(varName, rdd, "csv", -1, -1);
	}
	
	public void registerInput(String varName, MLMatrix df) throws DMLRuntimeException {
		registerInput(varName, MLMatrix.getRDDLazily(df), df.rlen, df.clen, df.brlen, df.bclen);
	}
	
	// ------------------------------------------------------------------------------------
	// 2. CSV/Text: Usually JavaRDD<String>, but also supports JavaPairRDD<LongWritable, Text>
	public void registerInput(String varName, JavaRDD<String> rdd, String format, boolean hasHeader, String delim, boolean fill, double missingValue) throws DMLRuntimeException {
		RDDProperties properties = new RDDProperties();
		properties.setHasHeader(hasHeader);
		properties.setFill(fill);
		properties.setDelim(delim);
		properties.setMissingValue(missingValue);
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, -1, -1, properties);
	}
	public void registerInput(String varName, RDD<String> rdd, String format) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD().mapToPair(new ConvertStringToLongTextPair()), format, -1, -1, null);
	}
	public void registerInput(String varName, JavaRDD<String> rdd, String format) throws DMLRuntimeException {
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, -1, -1, null);
	}
	public void registerInput(String varName, JavaRDD<String> rdd, String format, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, null);
	}
	public void registerInput(String varName, RDD<String> rdd, String format, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD().mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, null);
	}
	// Register input for csv/text format
	public void registerInput(String varName, JavaPairRDD<LongWritable, Text> textOrCsv_rdd, String format, long rlen, long clen, RDDProperties properties) throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("The registerInput functionality only supported for spark runtime. Please use MLContext(sc) instead of default constructor.");
		}
		
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		
		MatrixObject mo = null;
		if(format.compareTo("csv") == 0) {
			MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			mo = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(mc, OutputInfo.CSVOutputInfo, InputInfo.CSVInputInfo));
		}
		else if(format.compareTo("text") == 0) {
			MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			mo = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(mc, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo));
		}
		else {
			throw new DMLRuntimeException("Incorrect format in registerInput: " + format);
		}
		
		JavaPairRDD<LongWritable, Text> rdd = textOrCsv_rdd.mapToPair(new CopyTextInputFunction());
		if(properties != null) {
			mo.setRddProperties(properties);
		}
		mo.setRDDHandle(new RDDObject(rdd, varName));
		_variables.put(varName, mo);
		_inVarnames.add(varName);
	}
	// ------------------------------------------------------------------------------------
	
	// 3. Binary blocked RDD: Support JavaPairRDD<MatrixIndexes,MatrixBlock> and also RDD<Tuple2<MatrixIndexes,MatrixBlock>>
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd) throws DMLRuntimeException {
		registerInput(varName, rdd, -1, -1);
	}
	public void registerInput(String varName, RDD<Tuple2<MatrixIndexes,MatrixBlock>> rdd) throws DMLRuntimeException {
		registerInput(varName, org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(rdd.toJavaRDD()), -1, -1);
	}
	public void registerInput(String varName, RDD<Tuple2<MatrixIndexes,MatrixBlock>> rdd, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(rdd.toJavaRDD()), rlen, clen);
	}
	// Register input for binary format
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd1, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, rdd1, rlen, clen, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
	}
	
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd1, long rlen, long clen, int brlen, int bclen) throws DMLRuntimeException {
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		// Bug in Spark is messing up blocks and indexes due to too eager reuse of data structures
		JavaPairRDD<MatrixIndexes, MatrixBlock> rdd = rdd1.mapToPair( new CopyBlockFunction() );
		
		MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, brlen, bclen);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, "temp", new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		mo.setRDDHandle(new RDDObject(rdd, varName));
		_variables.put(varName, mo);
		_inVarnames.add(varName);
	}
	
	// =============================================================================================
	
	public void registerOutput(String varName) throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("The registerOutput functionality only supported for spark runtime. Please use MLContext(sc) instead of default constructor.");
		}
		if(_outVarnames == null)
			_outVarnames = new ArrayList<String>();
		_outVarnames.add(varName);
	}
	
	/**
	 * Execute DML script by passing named arguments
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param namedArgs
	 * @throws IOException
	 * @throws DMLException
	 * @throws ParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, HashMap<String, String> namedArgs) throws IOException, DMLException, ParseException {
		String [] args = new String[namedArgs.size()];
		int i = 0;
		for(Entry<String, String> entry : namedArgs.entrySet()) {
			args[i] = entry.getKey() + "=" + entry.getValue();
			i++;
		}
		return compileAndExecuteScript(dmlScriptFilePath, args, true);
	}
	
	public MLOutput execute(String dmlScriptFilePath, scala.collection.immutable.Map<String, String> namedArgs) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)));
	}

	public MLOutput execute(String dmlScriptFilePath, HashMap<String, String> namedArgs, boolean parsePyDML) throws IOException, DMLException, ParseException {
		this._parsePyDML = parsePyDML;
		return execute(dmlScriptFilePath, namedArgs);
	}
	
	public MLOutput execute(String dmlScriptFilePath, scala.collection.immutable.Map<String, String> namedArgs, boolean parsePyDML) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)), parsePyDML);
	}
	
	/**
	 * Execute DML script by passing positional arguments
	 * @param dmlScriptFilePath
	 * @param args
	 * @throws IOException
	 * @throws DMLException
	 * @throws ParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args) throws IOException, DMLException, ParseException {
		return compileAndExecuteScript(dmlScriptFilePath, args, false);
	}
	
	public MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML) throws IOException, DMLException, ParseException {
		this._parsePyDML = parsePyDML;
		return compileAndExecuteScript(dmlScriptFilePath, args, false);
	}
	
	/**
	 * Execute DML script without any arguments
	 * @param dmlScriptFilePath
	 * @throws IOException
	 * @throws DMLException
	 * @throws ParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath) throws IOException, DMLException, ParseException {
		return compileAndExecuteScript(dmlScriptFilePath, null, false);
	}
	
	public MLOutput execute(String dmlScriptFilePath, boolean parsePyDML) throws IOException, DMLException, ParseException {
		this._parsePyDML = parsePyDML;
		return compileAndExecuteScript(dmlScriptFilePath, null, false);
	}
	
	public MLOutput executeScript(String dmlScript) throws IOException, DMLException, ParseException {
		return compileAndExecuteScript(dmlScript, null, false, false);
	}
	
	// Only file based input
	private synchronized MLOutput compileAndExecuteScript(String dmlScriptFilePath, String [] args, boolean isNamedArgument) throws IOException, DMLException, ParseException {
		return compileAndExecuteScript(dmlScriptFilePath, args, isNamedArgument, true);
	}
	
	/**
	 * All the execute() methods call this, which  after setting appropriate input/output variables
	 * calls _compileAndExecuteScript
	 * We have explicitly synchronized this function because MLContext/SystemML does not yet support multi-threading.
	 * @param dmlScriptFilePath
	 * @param args
	 * @param isNamedArgument
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws ParseException
	 */
	private synchronized MLOutput compileAndExecuteScript(String dmlScriptFilePath, String [] args, boolean isNamedArgument, boolean isFile) throws IOException, DMLException, ParseException {
		resetMonitoringData();
		
		if(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK) {
			
			HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>> retVal = null;
			
			// Depending on whether registerInput/registerOutput was called initialize the variables 
			String[] inputs = null; String[] outputs = null;
			if(_inVarnames != null) {
				inputs = _inVarnames.toArray(new String[0]);
			}
			else {
				inputs = new String[0];
			}
			if(_outVarnames != null) {
				outputs = _outVarnames.toArray(new String[0]);
			}
			else {
				outputs = new String[0];
			}
			HashMap<String, MatrixCharacteristics> outMetadata = new HashMap<String, MatrixCharacteristics>();
			
			HashMap<String, String> argVals = DMLScript.createArgumentsMap(isNamedArgument, args);
			
			// Run the DML script
			ExecutionContext ec = _compileAndExecuteScript(dmlScriptFilePath, isFile, argVals, _parsePyDML, inputs, outputs, _variables);
			
			// Now collect the output
			if(_outVarnames != null) {
				for( String ovar : _outVarnames ) {
					if( _variables.keySet().contains(ovar) ) {
						if(retVal == null) {
							retVal = new HashMap<String, JavaPairRDD<MatrixIndexes,MatrixBlock>>();
						}
						retVal.put(ovar, ((SparkExecutionContext) ec).getBinaryBlockRDDHandleForVariable(ovar));
						outMetadata.put(ovar, ((SparkExecutionContext) ec).getMatrixCharacteristics(ovar)); // For converting output to dataframe
					}
					else {
						throw new DMLException("Error: The variable " + ovar + " is not available as output after the execution of the DMLScript.");
					}
				}
			}
			
			return new MLOutput(retVal, outMetadata);
		}
		else if(DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID ||
				DMLScript.rtplatform == RUNTIME_PLATFORM.HADOOP ||
				DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE) {
			// Instead of calling DMLScript directly, create a new process.
			// This ensures that environment variables, security permission are preserved as well as
			// memory budgets are not messed up.
			List<String> commandLineArgs = new ArrayList<String>();
			commandLineArgs.add("hadoop"); commandLineArgs.add("jar");
			commandLineArgs.add("SystemML.jar");
			commandLineArgs.add("-f"); commandLineArgs.add(dmlScriptFilePath);
			commandLineArgs.add("-exec"); 
			if(DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID) {
				commandLineArgs.add("hybrid");
			}
			else if(DMLScript.rtplatform == RUNTIME_PLATFORM.HADOOP) {
				commandLineArgs.add("hadoop");
			}
			else if(DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE) {
				commandLineArgs.add("singlenode");
			}
			else {
				throw new DMLRuntimeException("Unsupported execution type");
			}
			commandLineArgs.add("-explain");
			if(args != null && args.length > 0) {
				if(isNamedArgument) {
					commandLineArgs.add("-nvargs");
				}
				else {
					commandLineArgs.add("-args");
				}
				for(String arg : args) {
					commandLineArgs.add(arg);
				}
			}
			ProcessBuilder builder = new ProcessBuilder(commandLineArgs);
			final Process process = builder.start();
			InputStream is = process.getInputStream();
		    InputStreamReader isr = new InputStreamReader(is);
		    BufferedReader br = new BufferedReader(isr);
		    String line;
		    while ((line = br.readLine()) != null) {
		      System.out.println(line);
		    }
		    
			return null;
		}
		else {
			throw new DMLRuntimeException("Unsupported runtime.");
		}
	}
	
	
	
	
	/**
	 * This runs the DML script and returns the ExecutionContext for the caller to extract the output variables.
	 * The caller (which is compileAndExecuteScript) is expected to set inputSymbolTable with appropriate matrix representation (RDD, MatrixObject).
	 * 
	 * @param dmlScriptFilePath
	 * @param args
	 * @param isNamedArgument
	 * @param parsePyDML
	 * @param inputs
	 * @param outputs
	 * @param inputSymbolTable
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws ParseException
	 */
	private ExecutionContext _compileAndExecuteScript(String dmlScriptFilePath, boolean isFile, HashMap<String, String> argVals, boolean parsePyDML, 
			String[] inputs, String[] outputs, LocalVariableMap inputSymbolTable) throws IOException, DMLException, ParseException {
		DMLConfig config = new DMLConfig();
		ConfigurationManager.setConfig(config);
		
		String dmlScriptStr = null;
		if(isFile)
			dmlScriptStr = DMLScript.readDMLScript("-f", dmlScriptFilePath);
		else 
			dmlScriptStr = DMLScript.readDMLScript("-s", dmlScriptFilePath);
			
		DataExpression.REJECT_READ_UNKNOWN_SIZE = false;
		
		//simplified compilation chain
		Program rtprog = null;
		
		//parsing
		DMLProgram prog = null;
		if(parsePyDML) {
			PyDMLParserWrapper parser = new PyDMLParserWrapper();
			prog = parser.parse(dmlScriptFilePath, dmlScriptStr, argVals);
		}
		else {
			DMLParserWrapper parser = new DMLParserWrapper();
			prog = parser.parse(dmlScriptFilePath, dmlScriptStr, argVals);
		}
		
		if(prog == null) {
			throw new ParseException("Couldnot parse the file:" + dmlScriptFilePath);
		}
		
		//language validate
		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.liveVariableAnalysis(prog);			
		dmlt.validateParseTree(prog);
		
		//hop construct/rewrite
		dmlt.constructHops(prog);
		dmlt.rewriteHopsDAG(prog);
		
		//rewrite persistent reads/writes
		if(inputSymbolTable != null) {
			RewriteRemovePersistentReadWrite rewrite = new RewriteRemovePersistentReadWrite(inputs, outputs);
			ProgramRewriter rewriter2 = new ProgramRewriter(rewrite);
			rewriter2.rewriteProgramHopDAGs(prog);
		}
		
		//lop construct and runtime prog generation
		dmlt.constructLops(prog);
		rtprog = prog.getRuntimeProgram(config);
		
		//final cleanup runtime prog
		JMLCUtils.cleanupRuntimeProgram(rtprog, outputs);
		
		// TODO: As first step, we will always print the plan, will revisit this later
		if(this._monitorPerformance)
			Explain.PRINT_EXPLAIN_WITH_LINEAGE = true;
				
		//create and populate execution context
		ExecutionContext ec = ExecutionContextFactory.createContext(rtprog);
		if(inputSymbolTable != null) {
			ec.setVariables(inputSymbolTable);
		}
		
		System.out.println(Explain.explain(rtprog));
		
		//core execute runtime program	
		rtprog.execute( ec );
		
		_explainOutput = Explain.explain(rtprog);
		
		if(this._monitorPerformance)
			Explain.PRINT_EXPLAIN_WITH_LINEAGE = false;
		
		return ec;
	}
}
