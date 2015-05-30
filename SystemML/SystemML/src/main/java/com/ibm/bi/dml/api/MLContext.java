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
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.utils.Explain;

import org.apache.spark.sql.DataFrame;


/**
 * This is initial mockup API for Spark integration. Typical usage is as follows:
 * scala> import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes
 * scala> import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock
 * scala> import com.ibm.bi.dml.api.MLContext
 * scala> val args = Array("V.mtx", "W.mtx",  "H.mtx",  "2000", "1500",  "50",  "1",  "WOut.mtx",  "HOut.mtx")
 * scala> val V = sc.sequenceFile[MatrixIndexes, MatrixBlock]("hdfs://curly.almaden.ibm.com:9000/user/biadmin/V.mtx")
 * scala> val W = sc.sequenceFile[MatrixIndexes, MatrixBlock]("hdfs://curly.almaden.ibm.com:9000/user/biadmin/W.mtx")
 * scala> val H = sc.sequenceFile[MatrixIndexes, MatrixBlock]("hdfs://curly.almaden.ibm.com:9000/user/biadmin/H.mtx")
 * scala> val ml = new MLContext(sc)
 * scala> val V1 = org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(V.toJavaRDD())
 * scala> val W1 = org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(W.toJavaRDD())
 * scala> val H1 = org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(H.toJavaRDD())
 * scala> ml.registerInput("V", V1)
 * scala> ml.registerInput("W", W1)
 * scala> ml.registerInput("H", H1)
 * scala> ml.registerOutput("H")
 * scala> ml.registerOutput("W")
 * scala> val outputs = ml.execute("GNMF.dml", args) 
 */
public class MLContext {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	// ----------------------------------------------------
	// TODO: Remove static by passing these variables through DMLScript
	public static SparkContext _sc = null; // Read while creating SystemML's spark context
	// ----------------------------------------------------
	
	private ArrayList<String> _inVarnames = null;
	private ArrayList<String> _outVarnames = null;
	private LocalVariableMap _variables = null; // temporary symbol table
	boolean parsePyDML = false;
	
	// To check if Py4J is working:
	// >>> str1 = sc._jvm.java.lang.Class.forName("java.lang.String")
	// >>> str1.getName()
	
	public MLContext(SparkContext sc) {
		MLContext._sc = sc;
		DMLScript.rtplatform = RUNTIME_PLATFORM.SPARK;
	}
	
	public MLContext(JavaSparkContext sc) {
		MLContext._sc = sc.sc();
		DMLScript.rtplatform = RUNTIME_PLATFORM.SPARK;
	}
	
	// This code
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
	
	public void clear() {
		_inVarnames = null;
		_outVarnames = null;
		_variables = null;
		parsePyDML = false;
	}
	
	
	// ====================================================================================
	
	// 1. DataFrame
	public void registerInput(String varName, DataFrame df) throws DMLRuntimeException {
		JavaRDD<String> rdd = df.javaRDD().map(new ConvertRowToCSVString());
		registerInput(varName, rdd, "csv", -1, -1);
	}
	
	// ------------------------------------------------------------------------------------
	// 2. CSV/Text: Usually JavaRDD<String>, but also supports JavaPairRDD<LongWritable, Text>
	public void registerInput(String varName, JavaRDD<String> rdd, String format, boolean hasHeader, String delim, boolean fill, double missingValue) throws DMLRuntimeException {
		RDDProperties properties = new RDDProperties();
		properties.setHasHeader(hasHeader);
		properties.setDelim(delim);
		properties.setDelim(delim);
		properties.setMissingValue(missingValue);
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, -1, -1, properties);
	}
	public void registerInput(String varName, JavaRDD<String> rdd, String format) throws DMLRuntimeException {
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, -1, -1, null);
	}
	public void registerInput(String varName, JavaRDD<String> rdd, String format, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, null);
	}
	// Register input for csv/text format
	public void registerInput(String varName, JavaPairRDD<LongWritable, Text> textOrCsv_rdd, String format, long rlen, long clen, RDDProperties properties) throws DMLRuntimeException {
		if(DMLScript.rtplatform != RUNTIME_PLATFORM.SPARK) {
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
		mo.setRDDHandle(new RDDObject(rdd));
		_variables.put(varName, mo);
		_inVarnames.add(varName);
	}
	// ------------------------------------------------------------------------------------
	
	// 3. Binary blocked RDD: Support JavaPairRDD<MatrixIndexes,MatrixBlock> and also RDD<Tuple2<MatrixIndexes,MatrixBlock>>
	public void registerInput(String varName, RDD<Tuple2<MatrixIndexes,MatrixBlock>> rdd) throws DMLRuntimeException {
		registerInput(varName, org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(rdd.toJavaRDD()), -1, -1);
	}
	public void registerInput(String varName, RDD<Tuple2<MatrixIndexes,MatrixBlock>> rdd, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, org.apache.spark.api.java.JavaPairRDD.fromJavaRDD(rdd.toJavaRDD()), rlen, clen);
	}
	// Register input for binary format
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd1, long rlen, long clen) throws DMLRuntimeException {
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		// Bug in Spark is messing up blocks and indexes due to too eager reuse of data structures
		JavaPairRDD<MatrixIndexes, MatrixBlock> rdd = rdd1.mapToPair( new CopyBlockFunction() );
		
		MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		mo.setRDDHandle(new RDDObject(rdd));
		_variables.put(varName, mo);
		_inVarnames.add(varName);
	}
	
	// =============================================================================================
	
	public void registerOutput(String varName) throws DMLRuntimeException {
		if(DMLScript.rtplatform != RUNTIME_PLATFORM.SPARK) {
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
		this.parsePyDML = parsePyDML;
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
		this.parsePyDML = parsePyDML;
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
		this.parsePyDML = parsePyDML;
		return compileAndExecuteScript(dmlScriptFilePath, null, false);
	}
	
	private MLOutput compileAndExecuteScript(String dmlScriptFilePath, String [] args, boolean isNamedArgument) throws IOException, DMLException, ParseException {
		if(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK) {
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
			ExecutionContext ec = compileAndExecuteScript(dmlScriptFilePath, argVals, parsePyDML, inputs, outputs, _variables);
			
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
			
			MLOutput out = new MLOutput(retVal, outMetadata);
			out.setExplainOutput(explainOutput);
			return out;
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
	
	private String explainOutput = "";
	
	
	/**
	 * This runs the DML script and returns the ExecutionContext for the caller to extract the output variables.
	 * The caller is expected to set inputSymbolTable with appropriate matrix representation (RDD, MatrixObject).
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
	public ExecutionContext compileAndExecuteScript(String dmlScriptFilePath, HashMap<String, String> argVals, boolean parsePyDML, 
			String[] inputs, String[] outputs, LocalVariableMap inputSymbolTable) throws IOException, DMLException, ParseException {
		DMLConfig config = new DMLConfig();
		ConfigurationManager.setConfig(config);
		
		String dmlScriptStr = DMLScript.readDMLScript("-f", dmlScriptFilePath);
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
		
		//create and populate execution context
		ExecutionContext ec = ExecutionContextFactory.createContext(rtprog);
		if(inputSymbolTable != null) {
			ec.setVariables(inputSymbolTable);
		}
		
		// TODO: As first step, we will always print the plan, will revisit this later
		Explain.PRINT_EXPLAIN_WITH_LINEAGE = true;
		
		//core execute runtime program	
		rtprog.execute( ec );
		
		explainOutput = Explain.explain(rtprog);
		Explain.PRINT_EXPLAIN_WITH_LINEAGE = false;
		
		return ec;
	}
}
