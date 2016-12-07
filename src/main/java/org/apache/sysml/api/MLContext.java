/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.api;


import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.api.jmlc.JMLCUtils;
import org.apache.sysml.api.mlcontext.ScriptType;
import org.apache.sysml.api.monitoring.SparkMonitoringUtil;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.OptimizerUtils.OptimizationLevel;
import org.apache.sysml.hops.globalopt.GlobalOptimizerWrapper;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteRemovePersistentReadWrite;
import org.apache.sysml.parser.AParserWrapper;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.IntIdentifier;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.StringIdentifier;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.instructions.spark.functions.ConvertStringToLongTextPair;
import org.apache.sysml.runtime.instructions.spark.functions.CopyTextInputFunction;
import org.apache.sysml.runtime.instructions.spark.functions.SparkListener;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.Explain.ExplainCounts;
import org.apache.sysml.utils.Statistics;

/**
 * The MLContext API has been redesigned and this API will be deprecated.
 * Please migrate to {@link org.apache.sysml.api.mlcontext.MLContext}.
 * <p>
 * 
 * MLContext is useful for passing RDDs as input/output to SystemML. This API avoids the need to read/write
 * from HDFS (which is another way to pass inputs to SystemML).
 * <p>
 * Typical usage for MLContext is as follows:
 * <pre><code>
 * scala&gt; import org.apache.sysml.api.MLContext
 * </code></pre>
 * <p>
 * Create input DataFrame from CSV file and potentially perform some feature transformation
 * <pre><code>
 * scala&gt; val W = sqlContext.load("com.databricks.spark.csv", Map("path" -&gt; "W.csv", "header" -&gt; "false"))
 * scala&gt; val H = sqlContext.load("com.databricks.spark.csv", Map("path" -&gt; "H.csv", "header" -&gt; "false"))
 * scala&gt; val V = sqlContext.load("com.databricks.spark.csv", Map("path" -&gt; "V.csv", "header" -&gt; "false"))
 * </code></pre>
 * <p>
 * Create MLContext
 * <pre><code>
 * scala&gt; val ml = new MLContext(sc)
 * </code></pre>
 * <p>
 * Register input and output DataFrame/RDD 
 * Supported format: 
 * <ol>
 * <li> DataFrame
 * <li> CSV/Text (as JavaRDD&lt;String&gt; or JavaPairRDD&lt;LongWritable, Text&gt;)
 * <li> Binary blocked RDD (JavaPairRDD&lt;MatrixIndexes,MatrixBlock&gt;))
 * </ol>
 * Also overloaded to support metadata information such as format, rlen, clen, ...
 * Please note the variable names given below in quotes correspond to the variables in DML script.
 * These variables need to have corresponding read/write associated in DML script.
 * Currently, only matrix variables are supported through registerInput/registerOutput interface.
 * To pass scalar variables, use named/positional arguments (described later) or wrap them into matrix variable.
 * <pre><code>
 * scala&gt; ml.registerInput("V", V)
 * scala&gt; ml.registerInput("W", W)
 * scala&gt; ml.registerInput("H", H)
 * scala&gt; ml.registerOutput("H")
 * scala&gt; ml.registerOutput("W")
 * </code></pre>
 * <p>
 * Call script with default arguments:
 * <pre><code>
 * scala&gt; val outputs = ml.execute("GNMF.dml")
 * </code></pre>
 * <p>
 * Also supported: calling script with positional arguments (args) and named arguments (nargs):
 * <pre><code> 
 * scala&gt; val args = Array("V.mtx", "W.mtx",  "H.mtx",  "2000", "1500",  "50",  "1",  "WOut.mtx",  "HOut.mtx")
 * scala&gt; val nargs = Map("maxIter"-&gt;"1", "V" -&gt; "")
 * scala&gt; val outputs = ml.execute("GNMF.dml", args) # or ml.execute("GNMF_namedArgs.dml", nargs)
 * </code></pre>  
 * <p>
 * To run the script again using different (or even same arguments), but using same registered input/outputs:
 * <pre><code> 
 * scala&gt; val new_outputs = ml.execute("GNMF.dml", new_args)
 * </code></pre>
 * <p>
 * However, to register new input/outputs, you need to first reset MLContext
 * <pre><code> 
 * scala&gt; ml.reset()
 * scala&gt; ml.registerInput("V", newV)
 * </code></pre>
 * <p>
 * Experimental API:
 * To monitor performance (only supported for Spark 1.4.0 or higher),
 * <pre><code>
 * scala&gt; val ml = new MLContext(sc, true)
 * </code></pre>
 * <p>
 * If monitoring performance is enabled,
 * <pre><code> 
 * scala&gt; print(ml.getMonitoringUtil().getExplainOutput())
 * scala&gt; ml.getMonitoringUtil().getRuntimeInfoInHTML("runtime.html")
 * </code></pre>
 * <p>
 * Note: The execute(...) methods does not support parallel calls from same or different MLContext.
 * This is because current SystemML engine does not allow multiple invocation in same JVM.
 * So, if you plan to create a system which potentially creates multiple MLContext, 
 * it is recommended to guard the execute(...) call using
 * <pre><code>  
 * synchronized(MLContext.class) { ml.execute(...); }
 * </code></pre>
 */
public class MLContext {
	
	// ----------------------------------------------------
	// TODO: To make MLContext multi-threaded, track getCurrentMLContext and also all singletons and
	// static variables in SystemML codebase.
	private static MLContext _activeMLContext = null;
	
	// Package protected so as to maintain a clean public API for MLContext.
	// Use MLContextProxy.getActiveMLContext() if necessary
	static MLContext getActiveMLContext() {
		return _activeMLContext;
	}
	// ----------------------------------------------------
	
	private SparkContext _sc = null; // Read while creating SystemML's spark context
	public SparkContext getSparkContext() {
		if(_sc == null) {
			throw new RuntimeException("No spark context set in MLContext");
		}
		return _sc;
	}
	private ArrayList<String> _inVarnames = null;
	private ArrayList<String> _outVarnames = null;
	private LocalVariableMap _variables = null; // temporary symbol table
	private Program _rtprog = null;
	
	private Map<String, String> _additionalConfigs = new HashMap<String, String>();
	
	// --------------------------------------------------
	// _monitorUtils is set only when MLContext(sc, true)
	private SparkMonitoringUtil _monitorUtils = null;
	
	/**
	 * Experimental API. Not supported in Python MLContext API.
	 * @return SparkMonitoringUtil the Spark monitoring util
	 */
	public SparkMonitoringUtil getMonitoringUtil() {
		return _monitorUtils;
	}
	// --------------------------------------------------
	
	/**
	 * Create an associated MLContext for given spark session.
	 * @param sc SparkContext
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public MLContext(SparkContext sc) throws DMLRuntimeException {
		initializeSpark(sc, false, false);
	}
	
	/**
	 * Create an associated MLContext for given spark session.
	 * @param sc JavaSparkContext
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public MLContext(JavaSparkContext sc) throws DMLRuntimeException {
		initializeSpark(sc.sc(), false, false);
	}
	
	/**
	 * Allow users to provide custom named-value configuration.
	 * @param paramName parameter name
	 * @param paramVal parameter value
	 */
	public void setConfig(String paramName, String paramVal) {
		_additionalConfigs.put(paramName, paramVal);
	}
	
	// ====================================================================================
	// Register input APIs
	// 1. DataFrame
	
	/**
	 * Register DataFrame as input. DataFrame is assumed to be in row format and each cell can be converted into double 
	 * through  Double.parseDouble(cell.toString()). This is suitable for passing dense matrices. For sparse matrices,
	 * consider passing through text format (using JavaRDD&lt;String&gt;, format="text")
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param df the DataFrame
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, DataFrame df) throws DMLRuntimeException {
		registerInput(varName, df, false);
	}
	
	/**
	 * Register DataFrame as input. DataFrame is assumed to be in row format and each cell can be converted into 
	 * SystemML frame row. Each column could be of type, Double, Float, Long, Integer, String or Boolean.  
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param df the DataFrame
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerFrameInput(String varName, DataFrame df) throws DMLRuntimeException {
		registerFrameInput(varName, df, false);
	}
	
	/**
	 * Register DataFrame as input. 
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.  
	 * @param varName variable name
	 * @param df the DataFrame
	 * @param containsID false if the DataFrame has an column ID which denotes the row ID.
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, DataFrame df, boolean containsID) throws DMLRuntimeException {
		int blksz = ConfigurationManager.getBlocksize();
		MatrixCharacteristics mcOut = new MatrixCharacteristics(-1, -1, blksz, blksz);
		JavaPairRDD<MatrixIndexes, MatrixBlock> rdd = RDDConverterUtils
				.dataFrameToBinaryBlock(new JavaSparkContext(_sc), df, mcOut, containsID, false);
		registerInput(varName, rdd, mcOut);
	}
	
	/**
	 * Register DataFrame as input. DataFrame is assumed to be in row format and each cell can be converted into 
	 * SystemML frame row. Each column could be of type, Double, Float, Long, Integer, String or Boolean.  
	 * <p>
	 * @param varName variable name
	 * @param df the DataFrame
	 * @param containsID false if the DataFrame has an column ID which denotes the row ID.
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerFrameInput(String varName, DataFrame df, boolean containsID) throws DMLRuntimeException {
		int blksz = ConfigurationManager.getBlocksize();
		MatrixCharacteristics mcOut = new MatrixCharacteristics(-1, -1, blksz, blksz);
		JavaPairRDD<Long, FrameBlock> rdd = FrameRDDConverterUtils.dataFrameToBinaryBlock(new JavaSparkContext(_sc), df, mcOut, containsID);
		registerInput(varName, rdd, mcOut.getRows(), mcOut.getCols(), null);
	}
	
	/**
	 * Experimental API. Not supported in Python MLContext API.
	 * @param varName variable name
	 * @param df the DataFrame
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, MLMatrix df) throws DMLRuntimeException {
		registerInput(varName, MLMatrix.getRDDLazily(df), df.mc);
	}
	
	// ------------------------------------------------------------------------------------
	// 2. CSV/Text: Usually JavaRDD<String>, but also supports JavaPairRDD<LongWritable, Text>
	/**
	 * Register CSV/Text as inputs: Method for supplying csv file format properties, but without dimensions or nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the RDD
	 * @param format the format
	 * @param hasHeader is there a header
	 * @param delim the delimiter
	 * @param fill if true, fill, otherwise don't fill
	 * @param fillValue the fill value
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaRDD<String> rdd, String format, boolean hasHeader, 
			String delim, boolean fill, double fillValue) throws DMLRuntimeException {
		registerInput(varName, rdd, format, hasHeader, delim, fill, fillValue, -1, -1, -1);
	}
	
	/**
	 * Register CSV/Text as inputs: Method for supplying csv file format properties, but without dimensions or nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the RDD
	 * @param format the format
	 * @param hasHeader is there a header
	 * @param delim the delimiter
	 * @param fill if true, fill, otherwise don't fill
	 * @param fillValue the fill value
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, RDD<String> rdd, String format, boolean hasHeader, 
			String delim, boolean fill, double fillValue) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD(), format, hasHeader, delim, fill, fillValue, -1, -1, -1);
	}
	
	/**
	 * Register CSV/Text as inputs: Method for supplying csv file format properties along with dimensions or nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the RDD
	 * @param format the format
	 * @param hasHeader is there a header
	 * @param delim the delimiter
	 * @param fill if true, fill, otherwise don't fill
	 * @param fillValue the fill value
	 * @param rlen rows
	 * @param clen columns
	 * @param nnz non-zeros
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, RDD<String> rdd, String format, boolean hasHeader, 
			String delim, boolean fill, double fillValue, long rlen, long clen, long nnz) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD(), format, hasHeader, delim, fill, fillValue, -1, -1, -1);
	}
	
	/**
	 * Register CSV/Text as inputs: Method for supplying csv file format properties along with dimensions or nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the JavaRDD
	 * @param format the format
	 * @param hasHeader is there a header
	 * @param delim the delimiter
	 * @param fill if true, fill, otherwise don't fill
	 * @param fillValue the fill value
	 * @param rlen rows
	 * @param clen columns
	 * @param nnz non-zeros
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaRDD<String> rdd, String format, boolean hasHeader, 
			String delim, boolean fill, double fillValue, long rlen, long clen, long nnz) throws DMLRuntimeException {
		CSVFileFormatProperties props = new CSVFileFormatProperties(hasHeader, delim, fill, fillValue, "");
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, nnz, props);
	} 
	
	/**
	 * Register CSV/Text as inputs: Convenience method without dimensions and nnz. It uses default file properties (example: delim, fill, ..)
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the RDD
	 * @param format the format
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, RDD<String> rdd, String format) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD().mapToPair(new ConvertStringToLongTextPair()), format, -1, -1, -1, null);
	}
	
	/**
	 * Register CSV/Text as inputs: Convenience method without dimensions and nnz. It uses default file properties (example: delim, fill, ..)
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the JavaRDD
	 * @param format the format
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaRDD<String> rdd, String format) throws DMLRuntimeException {
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, -1, -1, -1, null);
	}
	
	/**
	 * Register CSV/Text as inputs: Convenience method with dimensions and but no nnz. It uses default file properties (example: delim, fill, ..)
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file. 
	 * @param varName variable name
	 * @param rdd the JavaRDD
	 * @param format the format
	 * @param rlen rows
	 * @param clen columns
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaRDD<String> rdd, String format, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, -1, null);
	}
	
	/**
	 * Register CSV/Text as inputs: Convenience method with dimensions and but no nnz. It uses default file properties (example: delim, fill, ..)
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the RDD
	 * @param format the format
	 * @param rlen rows
	 * @param clen columns
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, RDD<String> rdd, String format, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD().mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, -1, null);
	}
	
	/**
	 * Register CSV/Text as inputs: with dimensions and nnz. It uses default file properties (example: delim, fill, ..)
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the JavaRDD
	 * @param format the format
	 * @param rlen rows
	 * @param clen columns
	 * @param nnz non-zeros
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaRDD<String> rdd, String format, long rlen, long clen, long nnz) throws DMLRuntimeException {
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, nnz, null);
	}
	
	/**
	 * Register CSV/Text as inputs: with dimensions and nnz. It uses default file properties (example: delim, fill, ..)
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the JavaRDD
	 * @param format the format
	 * @param rlen rows
	 * @param clen columns
	 * @param nnz non-zeros
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, RDD<String> rdd, String format, long rlen, long clen, long nnz) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD().mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, nnz, null);
	}
	
	// All CSV related methods call this ... It provides access to dimensions, nnz, file properties.
	private void registerInput(String varName, JavaPairRDD<LongWritable, Text> textOrCsv_rdd, String format, long rlen, long clen, long nnz, FileFormatProperties props) throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("The registerInput functionality only supported for spark runtime. Please use MLContext(sc) instead of default constructor.");
		}
		
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		
		MatrixObject mo;
		if( format.equals("csv") ) {
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, blksz, blksz, nnz);
			mo = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(mc, OutputInfo.CSVOutputInfo, InputInfo.CSVInputInfo));
		}
		else if( format.equals("text") ) {
			if(rlen == -1 || clen == -1) {
				throw new DMLRuntimeException("The metadata is required in registerInput for format:" + format);
			}
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, blksz, blksz, nnz);
			mo = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(mc, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo));
		}
		else if( format.equals("mm") ) {
			// TODO: Handle matrix market
			throw new DMLRuntimeException("Matrixmarket format is not yet implemented in registerInput: " + format);
		}
		else {
			
			throw new DMLRuntimeException("Incorrect format in registerInput: " + format);
		}
		
		JavaPairRDD<LongWritable, Text> rdd = textOrCsv_rdd.mapToPair(new CopyTextInputFunction());
		if(props != null)
			mo.setFileFormatProperties(props);
		mo.setRDDHandle(new RDDObject(rdd, varName));
		_variables.put(varName, mo);
		_inVarnames.add(varName);
		checkIfRegisteringInputAllowed();
	}
	
	/**
	 * Register Frame with CSV/Text as inputs: with dimensions. 
	 * File properties (example: delim, fill, ..) can be specified through props else defaults will be used.
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rddIn the JavaPairRDD
	 * @param format the format
	 * @param rlen rows
	 * @param clen columns
	 * @param props properties
	 * @param schema List of column types
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaRDD<String> rddIn, String format, long rlen, long clen, FileFormatProperties props, 
			List<ValueType> schema) throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("The registerInput functionality only supported for spark runtime. Please use MLContext(sc) instead of default constructor.");
		}
		
		long nnz = -1;
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		
		JavaPairRDD<LongWritable, Text> rddText = rddIn.mapToPair(new ConvertStringToLongTextPair());
		
		int blksz = ConfigurationManager.getBlocksize();
		MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, blksz, blksz, nnz);
		FrameObject fo = null;
		if( format.equals("csv") ) {
			CSVFileFormatProperties csvprops = (props!=null) ? (CSVFileFormatProperties)props: new CSVFileFormatProperties();
			fo = new FrameObject(OptimizerUtils.getUniqueTempFileName(), new MatrixFormatMetaData(mc, OutputInfo.CSVOutputInfo, InputInfo.CSVInputInfo));
			fo.setFileFormatProperties(csvprops);
		}
		else if( format.equals("text") ) {
			if(rlen == -1 || clen == -1) {
				throw new DMLRuntimeException("The metadata is required in registerInput for format:" + format);
			}
			fo = new FrameObject(null, new MatrixFormatMetaData(mc, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo));
		}
		else {
			
			throw new DMLRuntimeException("Incorrect format in registerInput: " + format);
		}
		if(props != null)
			fo.setFileFormatProperties(props);
		
		fo.setRDDHandle(new RDDObject(rddText, varName));
		fo.setSchema("String");		//TODO fix schema 
		_variables.put(varName, fo);
		_inVarnames.add(varName);
		checkIfRegisteringInputAllowed();
	}
	
	private void registerInput(String varName, JavaPairRDD<Long, FrameBlock> rdd, long rlen, long clen, FileFormatProperties props) throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("The registerInput functionality only supported for spark runtime. Please use MLContext(sc) instead of default constructor.");
		}
		
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		
		int blksz = ConfigurationManager.getBlocksize();
		MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, blksz, blksz, -1);
		FrameObject fo = new FrameObject(OptimizerUtils.getUniqueTempFileName(), new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		
		if(props != null)
			fo.setFileFormatProperties(props);
		
		fo.setRDDHandle(new RDDObject(rdd, varName));
		_variables.put(varName, fo);
		_inVarnames.add(varName);
		checkIfRegisteringInputAllowed();
	}
	
	// ------------------------------------------------------------------------------------
	
	// 3. Binary blocked RDD: Support JavaPairRDD<MatrixIndexes,MatrixBlock> 
	
	/**
	 * Register binary blocked RDD with given dimensions, default block sizes and no nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file. 
	 * @param varName variable name
	 * @param rdd the JavaPairRDD
	 * @param rlen rows
	 * @param clen columns
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, long rlen, long clen) throws DMLRuntimeException {
		//TODO replace default blocksize
		registerInput(varName, rdd, rlen, clen, OptimizerUtils.DEFAULT_BLOCKSIZE, OptimizerUtils.DEFAULT_BLOCKSIZE);
	}
	
	/**
	 * Register binary blocked RDD with given dimensions, given block sizes and no nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the JavaPairRDD
	 * @param rlen rows
	 * @param clen columns
	 * @param brlen block rows
	 * @param bclen block columns
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, long rlen, long clen, int brlen, int bclen) throws DMLRuntimeException {
		registerInput(varName, rdd, rlen, clen, brlen, bclen, -1);
	}
	
	
	/**
	 * Register binary blocked RDD with given dimensions, given block sizes and given nnz (preferred).
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName variable name
	 * @param rdd the JavaPairRDD
	 * @param rlen rows
	 * @param clen columns
	 * @param brlen block rows
	 * @param bclen block columns
	 * @param nnz non-zeros
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, long rlen, long clen, int brlen, int bclen, long nnz) throws DMLRuntimeException {
		if(rlen == -1 || clen == -1) {
			throw new DMLRuntimeException("The metadata is required in registerInput for binary format");
		}
		
		MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, brlen, bclen, nnz);
		registerInput(varName, rdd, mc);
	}
	
	// All binary blocked method call this.
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, MatrixCharacteristics mc) throws DMLRuntimeException {
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		// Bug in Spark is messing up blocks and indexes due to too eager reuse of data structures
		JavaPairRDD<MatrixIndexes, MatrixBlock> copyRDD = SparkUtils.copyBinaryBlockMatrix(rdd);
		
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), 
				new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		mo.setRDDHandle(new RDDObject(copyRDD, varName));
		_variables.put(varName, mo);
		_inVarnames.add(varName);
		checkIfRegisteringInputAllowed();
	}
	
	public void registerInput(String varName, MatrixBlock mb) throws DMLRuntimeException {
		int blksz = ConfigurationManager.getBlocksize();
		MatrixCharacteristics mc = new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(), blksz, blksz, mb.getNonZeros());
		registerInput(varName, mb, mc);
	}
	
	public void registerInput(String varName, MatrixBlock mb, MatrixCharacteristics mc) throws DMLRuntimeException {
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), 
				new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		mo.acquireModify(mb); 
		mo.release();
		_variables.put(varName, mo);
		_inVarnames.add(varName);
		checkIfRegisteringInputAllowed();
	}
	
	// =============================================================================================
	
	/**
	 * Marks the variable in the DML script as output variable.
	 * Note that this expects a "write(varName, ...)" statement in the DML script which through non-MLContext invocation
	 * would have written the matrix to HDFS.
	 * @param varName variable name
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void registerOutput(String varName) throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("The registerOutput functionality only supported for spark runtime. Please use MLContext(sc) instead of default constructor.");
		}
		if(_outVarnames == null)
			_outVarnames = new ArrayList<String>();
		_outVarnames.add(varName);
		if(_variables == null)
			_variables = new LocalVariableMap();
	}
	
	// =============================================================================================
	
	/**
	 * Execute DML script by passing named arguments using specified config file.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param namedArgs named arguments
	 * @param parsePyDML true if pydml, false otherwise
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, Map<String, String> namedArgs, boolean parsePyDML, String configFilePath) throws IOException, DMLException, ParseException {
		String [] args = new String[namedArgs.size()];
		int i = 0;
		for(Entry<String, String> entry : namedArgs.entrySet()) {
			if(entry.getValue().trim().isEmpty())
				args[i] = entry.getKey() + "=\"" + entry.getValue() + "\"";
			else
				args[i] = entry.getKey() + "=" + entry.getValue();
			i++;
		}
		return compileAndExecuteScript(dmlScriptFilePath, args, true, parsePyDML, configFilePath);
	}
	
	/**
	 * Execute DML script by passing named arguments using specified config file.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param namedArgs named arguments
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, Map<String, String> namedArgs, String configFilePath) throws IOException, DMLException, ParseException {
		String [] args = new String[namedArgs.size()];
		int i = 0;
		for(Entry<String, String> entry : namedArgs.entrySet()) {
			if(entry.getValue().trim().isEmpty())
				args[i] = entry.getKey() + "=\"" + entry.getValue() + "\"";
			else
				args[i] = entry.getKey() + "=" + entry.getValue();
			i++;
		}
		
		return compileAndExecuteScript(dmlScriptFilePath, args, true, false, configFilePath);
	}
	
	/**
	 * Execute DML script by passing named arguments with default configuration.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param namedArgs named arguments
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, Map<String, String> namedArgs) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, namedArgs, false, null);
	}
	
	/**
	 * Execute DML script by passing named arguments.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param namedArgs named arguments
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, scala.collection.immutable.Map<String, String> namedArgs) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)));
	}

	/**
	 * Experimental: Execute PyDML script by passing named arguments if parsePyDML=true.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param namedArgs named arguments
	 * @param parsePyDML true if pydml, false otherwise
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, Map<String, String> namedArgs, boolean parsePyDML) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, namedArgs, parsePyDML, null);
	}
	
	/**
	 * Experimental: Execute PyDML script by passing named arguments if parsePyDML=true.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param namedArgs named arguments
	 * @param parsePyDML true if pydml, false otherwise
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, scala.collection.immutable.Map<String, String> namedArgs, boolean parsePyDML) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)), parsePyDML);
	}
	
	/**
	 * Execute DML script by passing positional arguments using specified config file
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param args arguments
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args, String configFilePath) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, args, false, configFilePath);
	}
	
	/**
	 * Execute DML script by passing positional arguments using specified config file
	 * This method is implemented for compatibility with Python MLContext.
	 * Java/Scala users should use 'MLOutput execute(String dmlScriptFilePath, String [] args, String configFilePath)' instead as
	 * equivalent scala collections (Seq/ArrayBuffer) is not implemented.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param args arguments
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> args, String configFilePath) throws IOException, DMLException, ParseException {
		String [] argsArr = new String[args.size()];
		argsArr = args.toArray(argsArr);
		return execute(dmlScriptFilePath, argsArr, false, configFilePath);
	}
	
	/**
	 * Execute DML script by passing positional arguments using default configuration
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param args arguments
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, args, false, null);
	}
	
	/**
	 * Execute DML script by passing positional arguments using default configuration.
	 * This method is implemented for compatibility with Python MLContext.
	 * Java/Scala users should use 'MLOutput execute(String dmlScriptFilePath, String [] args)' instead as
	 * equivalent scala collections (Seq/ArrayBuffer) is not implemented.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param args arguments
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> args) throws IOException, DMLException, ParseException {
		String [] argsArr = new String[args.size()];
		argsArr = args.toArray(argsArr);
		return execute(dmlScriptFilePath, argsArr, false, null);
	}
	
	/**
	 * Experimental: Execute DML script by passing positional arguments if parsePyDML=true, using default configuration.
	 * This method is implemented for compatibility with Python MLContext.
	 * Java/Scala users should use 'MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML)' instead as
	 * equivalent scala collections (Seq/ArrayBuffer) is not implemented.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param args arguments
	 * @param parsePyDML true if pydml, false otherwise
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> args, boolean parsePyDML) throws IOException, DMLException, ParseException {
		String [] argsArr = new String[args.size()];
		argsArr = args.toArray(argsArr);
		return execute(dmlScriptFilePath, argsArr, parsePyDML, null);
	}
	
	/**
	 * Experimental: Execute DML script by passing positional arguments if parsePyDML=true, using specified config file.
	 * This method is implemented for compatibility with Python MLContext.
	 * Java/Scala users should use 'MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML, String configFilePath)' instead as
	 * equivalent scala collections (Seq/ArrayBuffer) is not implemented.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param args arguments
	 * @param parsePyDML true if pydml, false otherwise
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> args, boolean parsePyDML, String configFilePath) throws IOException, DMLException, ParseException {
		String [] argsArr = new String[args.size()];
		argsArr = args.toArray(argsArr);
		return execute(dmlScriptFilePath, argsArr, parsePyDML, configFilePath);
	}

	/*
	  @NOTE: from calling with the SparkR , somehow Map passing from R to java
	   is not working and hence we pass in two  arrays each representing keys
	   and values
	 */
	/**
	 * Execute DML script by passing positional arguments using specified config file
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param argsName argument names
	 * @param argsValues argument values
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> argsName,
							ArrayList<String> argsValues, String configFilePath)
			throws IOException, DMLException, ParseException  {
		HashMap<String, String> newNamedArgs = new HashMap<String, String>();
		if (argsName.size() != argsValues.size()) {
			throw new DMLException("size of argsName " + argsName.size() +
					" is diff than " + " size of argsValues");
		}
		for (int i = 0; i < argsName.size(); i++) {
			String k = argsName.get(i);
			String v = argsValues.get(i);
			newNamedArgs.put(k, v);
		}
		return execute(dmlScriptFilePath, newNamedArgs, configFilePath);
	}
	/**
	 * Execute DML script by passing positional arguments using specified config file
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param argsName argument names
	 * @param argsValues argument values
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> argsName,
							ArrayList<String> argsValues)
			throws IOException, DMLException, ParseException  {
		return execute(dmlScriptFilePath, argsName, argsValues, null);
	}

	/**
	 * Experimental: Execute DML script by passing positional arguments if parsePyDML=true, using specified config file.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param args arguments
	 * @param parsePyDML true if pydml, false otherwise
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML, String configFilePath) throws IOException, DMLException, ParseException {
		return compileAndExecuteScript(dmlScriptFilePath, args, false, parsePyDML, configFilePath);
	}
	
	/**
	 * Experimental: Execute DML script by passing positional arguments if parsePyDML=true, using default configuration.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param args arguments
	 * @param parsePyDML true if pydml, false otherwise
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, args, parsePyDML, null);
	}
	
	/**
	 * Execute DML script without any arguments using specified config path
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, String configFilePath) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, false, configFilePath);
	}
	
	/**
	 * Execute DML script without any arguments using default configuration.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, false, null);
	}
	
	/**
	 * Experimental: Execute DML script without any arguments if parsePyDML=true, using specified config path.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param parsePyDML true if pydml, false otherwise
	 * @param configFilePath path to config file
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, boolean parsePyDML, String configFilePath) throws IOException, DMLException, ParseException {
		return compileAndExecuteScript(dmlScriptFilePath, null, false, parsePyDML, configFilePath);
	}
	
	/**
	 * Experimental: Execute DML script without any arguments if parsePyDML=true, using default configuration.
	 * @param dmlScriptFilePath the dml script can be in local filesystem or in HDFS
	 * @param parsePyDML true if pydml, false otherwise
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput execute(String dmlScriptFilePath, boolean parsePyDML) throws IOException, DMLException, ParseException {
		return execute(dmlScriptFilePath, parsePyDML, null);
	}
	
	// -------------------------------- Utility methods begins ----------------------------------------------------------
	
	
	/**
	 * Call this method if you want to clear any RDDs set via registerInput, registerOutput.
	 * This is required if ml.execute(..) has been called earlier and you want to call a new DML script. 
	 * Note: By default this doesnot clean up configuration set using setConfig method. 
	 * To clean the configuration as along with registered input/outputs, please use reset(true);
	 * @throws DMLRuntimeException if DMLException occurs
	 */
	public void reset() 
			throws DMLRuntimeException 
	{
		reset(false);
	}
	
	public void reset(boolean cleanupConfig) 
			throws DMLRuntimeException 
	{
		//cleanup variables from bufferpool, incl evicted files 
		//(otherwise memory leak because bufferpool holds references)
		CacheableData.cleanupCacheDir();

		//clear mlcontext state
		_inVarnames = null;
		_outVarnames = null;
		_variables = null;
		if(cleanupConfig)
			_additionalConfigs.clear();
	}
	
	/**
	 * Used internally
	 * @param source the expression
	 * @param target the target
	 * @throws LanguageException if LanguageException occurs
	 */
	void setAppropriateVarsForRead(Expression source, String target) 
		throws LanguageException 
	{
		boolean isTargetRegistered = isRegisteredAsInput(target);
		boolean isReadExpression = (source instanceof DataExpression && ((DataExpression) source).isRead());
		if(isTargetRegistered && isReadExpression) {
			// Do not check metadata file for registered reads 
			((DataExpression) source).setCheckMetadata(false);
			
		 	if (((DataExpression)source).getDataType() == Expression.DataType.MATRIX) {

				MatrixObject mo = null;
				
				try {
					mo = getMatrixObject(target);
					int blp = source.getBeginLine(); int bcp = source.getBeginColumn();
					int elp = source.getEndLine(); int ecp = source.getEndColumn();
					((DataExpression) source).addVarParam(DataExpression.READROWPARAM, new IntIdentifier(mo.getNumRows(), source.getFilename(), blp, bcp, elp, ecp));
					((DataExpression) source).addVarParam(DataExpression.READCOLPARAM, new IntIdentifier(mo.getNumColumns(), source.getFilename(), blp, bcp, elp, ecp));
					((DataExpression) source).addVarParam(DataExpression.READNUMNONZEROPARAM, new IntIdentifier(mo.getNnz(), source.getFilename(), blp, bcp, elp, ecp));
					((DataExpression) source).addVarParam(DataExpression.DATATYPEPARAM, new StringIdentifier("matrix", source.getFilename(), blp, bcp, elp, ecp));
					((DataExpression) source).addVarParam(DataExpression.VALUETYPEPARAM, new StringIdentifier("double", source.getFilename(), blp, bcp, elp, ecp));
					
					if(mo.getMetaData() instanceof MatrixFormatMetaData) {
						MatrixFormatMetaData metaData = (MatrixFormatMetaData) mo.getMetaData();
						if(metaData.getOutputInfo() == OutputInfo.CSVOutputInfo) {
							((DataExpression) source).addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_CSV, source.getFilename(), blp, bcp, elp, ecp));
						}
						else if(metaData.getOutputInfo() == OutputInfo.TextCellOutputInfo) {
							((DataExpression) source).addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_TEXT, source.getFilename(), blp, bcp, elp, ecp));
						}
						else if(metaData.getOutputInfo() == OutputInfo.BinaryBlockOutputInfo) {
							((DataExpression) source).addVarParam(DataExpression.ROWBLOCKCOUNTPARAM, new IntIdentifier(mo.getNumRowsPerBlock(), source.getFilename(), blp, bcp, elp, ecp));
							((DataExpression) source).addVarParam(DataExpression.COLUMNBLOCKCOUNTPARAM, new IntIdentifier(mo.getNumColumnsPerBlock(), source.getFilename(), blp, bcp, elp, ecp));
							((DataExpression) source).addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_BINARY, source.getFilename(), blp, bcp, elp, ecp));
						}
						else {
							throw new LanguageException("Unsupported format through MLContext");
						}
					}
				} catch (DMLRuntimeException e) {
					throw new LanguageException(e);
				}
		 	} else if (((DataExpression)source).getDataType() == Expression.DataType.FRAME) {
				FrameObject mo = null;
				try {
					mo = getFrameObject(target);
					int blp = source.getBeginLine(); int bcp = source.getBeginColumn();
					int elp = source.getEndLine(); int ecp = source.getEndColumn();
					((DataExpression) source).addVarParam(DataExpression.READROWPARAM, new IntIdentifier(mo.getNumRows(), source.getFilename(), blp, bcp, elp, ecp));
					((DataExpression) source).addVarParam(DataExpression.READCOLPARAM, new IntIdentifier(mo.getNumColumns(), source.getFilename(), blp, bcp, elp, ecp));
					((DataExpression) source).addVarParam(DataExpression.DATATYPEPARAM, new StringIdentifier("frame", source.getFilename(), blp, bcp, elp, ecp));
					((DataExpression) source).addVarParam(DataExpression.VALUETYPEPARAM, new StringIdentifier("double", source.getFilename(), blp, bcp, elp, ecp));	//TODO change to schema
					
					if(mo.getMetaData() instanceof MatrixFormatMetaData) {
						MatrixFormatMetaData metaData = (MatrixFormatMetaData) mo.getMetaData();
						if(metaData.getOutputInfo() == OutputInfo.CSVOutputInfo) {
							((DataExpression) source).addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_CSV, source.getFilename(), blp, bcp, elp, ecp));
						}
						else if(metaData.getOutputInfo() == OutputInfo.TextCellOutputInfo) {
							((DataExpression) source).addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_TEXT, source.getFilename(), blp, bcp, elp, ecp));
						}
						else if(metaData.getOutputInfo() == OutputInfo.BinaryBlockOutputInfo) {
							((DataExpression) source).addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_BINARY, source.getFilename(), blp, bcp, elp, ecp));
						}
						else {
							throw new LanguageException("Unsupported format through MLContext");
						}
					}
				} catch (DMLRuntimeException e) {
					throw new LanguageException(e);
				}
		 	}
		}
	}
	
	/**
	 * Used internally
	 * @param tmp list of instructions
	 * @return list of instructions
	 */
	ArrayList<Instruction> performCleanupAfterRecompilation(ArrayList<Instruction> tmp) {
		String [] outputs = (_outVarnames != null) ? _outVarnames.toArray(new String[0]) : new String[0];
		return JMLCUtils.cleanupRuntimeInstructions(tmp, outputs);
	}
	
	// -------------------------------- Utility methods ends ----------------------------------------------------------
		
	
	// -------------------------------- Experimental API begins ----------------------------------------------------------
	/**
	 * Experimental api:
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc SparkContext
	 * @param monitorPerformance if true, monitor performance, otherwise false
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public MLContext(SparkContext sc, boolean monitorPerformance) throws DMLRuntimeException {
		initializeSpark(sc, monitorPerformance, false);
	}
	
	/**
	 * Experimental api:
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc JavaSparkContext
	 * @param monitorPerformance if true, monitor performance, otherwise false
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public MLContext(JavaSparkContext sc, boolean monitorPerformance) throws DMLRuntimeException {
		initializeSpark(sc.sc(), monitorPerformance, false);
	}
	
	/**
	 * Experimental api:
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc SparkContext
	 * @param monitorPerformance if true, monitor performance, otherwise false
	 * @param setForcedSparkExecType set forced spark exec type
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public MLContext(SparkContext sc, boolean monitorPerformance, boolean setForcedSparkExecType) throws DMLRuntimeException {
		initializeSpark(sc, monitorPerformance, setForcedSparkExecType);
	}
	
	/**
	 * Experimental api:
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc JavaSparkContext
	 * @param monitorPerformance if true, monitor performance, otherwise false
	 * @param setForcedSparkExecType set forced spark exec type
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public MLContext(JavaSparkContext sc, boolean monitorPerformance, boolean setForcedSparkExecType) throws DMLRuntimeException {
		initializeSpark(sc.sc(), monitorPerformance, setForcedSparkExecType);
	}
	
	// -------------------------------- Experimental API ends ----------------------------------------------------------
	
	// -------------------------------- Private methods begins ----------------------------------------------------------
	private boolean isRegisteredAsInput(String varName) {
		if(_inVarnames != null) {
			for(String v : _inVarnames) {
				if(v.equals(varName)) {
					return true;
				}
			}
		}
		return false;
	}
	
	private MatrixObject getMatrixObject(String varName) throws DMLRuntimeException {
		if(_variables != null) {
			Data mo = _variables.get(varName);
			if(mo instanceof MatrixObject) {
				return (MatrixObject) mo;
			}
			else {
				throw new DMLRuntimeException("ERROR: Incorrect type");
			}
		}
		throw new DMLRuntimeException("ERROR: getMatrixObject not set for variable:" + varName);
	}
	
	private FrameObject getFrameObject(String varName) throws DMLRuntimeException {
		if(_variables != null) {
			Data mo = _variables.get(varName);
			if(mo instanceof FrameObject) {
				return (FrameObject) mo;
			}
			else {
				throw new DMLRuntimeException("ERROR: Incorrect type");
			}
		}
		throw new DMLRuntimeException("ERROR: getMatrixObject not set for variable:" + varName);
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
		MLContextProxy.setActive(true);
		
		this._sc = sc;
		
		if(compareVersion(sc.version(), "1.3.0")  < 0 ) {
			throw new DMLRuntimeException("Expected spark version >= 1.3.0 for running SystemML");
		}
		
		if(setForcedSparkExecType)
			DMLScript.rtplatform = RUNTIME_PLATFORM.SPARK;
		else
			DMLScript.rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
		
		if(monitorPerformance) {
			initializeSparkListener(sc);
		}
	}
	
	private void initializeSparkListener(SparkContext sc) throws DMLRuntimeException {
		if(compareVersion(sc.version(), "1.4.0")  < 0 ) {
			throw new DMLRuntimeException("Expected spark version >= 1.4.0 for monitoring MLContext performance");
		}
		SparkListener sparkListener = new SparkListener(sc);
		_monitorUtils = new SparkMonitoringUtil(sparkListener);
		sc.addSparkListener(sparkListener);
	}
	
	/**
	 * Execute a script stored in a string.
	 *
	 * @param dmlScript the script
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLOutput executeScript(String dmlScript)
			throws IOException, DMLException {
		return executeScript(dmlScript, false);
	}

	public MLOutput executeScript(String dmlScript, boolean isPyDML)
			throws IOException, DMLException {
		return executeScript(dmlScript, isPyDML, null);
	}

	public MLOutput executeScript(String dmlScript, String configFilePath)
			throws IOException, DMLException {
		return executeScript(dmlScript, false, configFilePath);
	}

	public MLOutput executeScript(String dmlScript, boolean isPyDML, String configFilePath)
			throws IOException, DMLException {
		return compileAndExecuteScript(dmlScript, null, false, false, isPyDML, configFilePath);
	}

	/*
	  @NOTE: from calling with the SparkR , somehow HashMap passing from R to java
	   is not working and hence we pass in two  arrays each representing keys
	   and values
	 */
	public MLOutput executeScript(String dmlScript, ArrayList<String> argsName,
								  ArrayList<String> argsValues, String configFilePath)
			throws IOException, DMLException, ParseException  {
		HashMap<String, String> newNamedArgs = new HashMap<String, String>();
		if (argsName.size() != argsValues.size()) {
			throw new DMLException("size of argsName " + argsName.size() +
					" is diff than " + " size of argsValues");
		}
		for (int i = 0; i < argsName.size(); i++) {
			String k = argsName.get(i);
			String v = argsValues.get(i);
			newNamedArgs.put(k, v);
		}
		return executeScript(dmlScript, newNamedArgs, configFilePath);
	}

	public MLOutput executeScript(String dmlScript, ArrayList<String> argsName,
								  ArrayList<String> argsValues)
			throws IOException, DMLException, ParseException  {
		return executeScript(dmlScript, argsName, argsValues, null);
	}


	public MLOutput executeScript(String dmlScript, scala.collection.immutable.Map<String, String> namedArgs)
			throws IOException, DMLException {
		return executeScript(dmlScript, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)), null);
	}

	public MLOutput executeScript(String dmlScript, scala.collection.immutable.Map<String, String> namedArgs, boolean isPyDML)
			throws IOException, DMLException {
		return executeScript(dmlScript, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)), isPyDML, null);
	}

	public MLOutput executeScript(String dmlScript, scala.collection.immutable.Map<String, String> namedArgs, String configFilePath)
			throws IOException, DMLException {
		return executeScript(dmlScript, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)), configFilePath);
	}

	public MLOutput executeScript(String dmlScript, scala.collection.immutable.Map<String, String> namedArgs, boolean isPyDML, String configFilePath)
			throws IOException, DMLException {
		return executeScript(dmlScript, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)), isPyDML, configFilePath);
	}

	public MLOutput executeScript(String dmlScript, Map<String, String> namedArgs)
			throws IOException, DMLException {
		return executeScript(dmlScript, namedArgs, null);
	}

	public MLOutput executeScript(String dmlScript, Map<String, String> namedArgs, boolean isPyDML)
			throws IOException, DMLException {
		return executeScript(dmlScript, namedArgs, isPyDML, null);
	}

	public MLOutput executeScript(String dmlScript, Map<String, String> namedArgs, String configFilePath)
			throws IOException, DMLException {
		return executeScript(dmlScript, namedArgs, false, configFilePath);
	}

	public MLOutput executeScript(String dmlScript, Map<String, String> namedArgs, boolean isPyDML, String configFilePath)
			throws IOException, DMLException {
		String [] args = new String[namedArgs.size()];
		int i = 0;
		for(Entry<String, String> entry : namedArgs.entrySet()) {
			if(entry.getValue().trim().isEmpty())
				args[i] = entry.getKey() + "=\"" + entry.getValue() + "\"";
			else
				args[i] = entry.getKey() + "=" + entry.getValue();
			i++;
		}
		return compileAndExecuteScript(dmlScript, args, false, true, isPyDML, configFilePath);
	}

	private void checkIfRegisteringInputAllowed() throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("ERROR: registerInput is only allowed for spark execution mode");
		}
	}
	
	private MLOutput compileAndExecuteScript(String dmlScriptFilePath, String [] args, boolean isNamedArgument, boolean isPyDML, String configFilePath) throws IOException, DMLException {
		return compileAndExecuteScript(dmlScriptFilePath, args, true, isNamedArgument, isPyDML, configFilePath);
	}
	
	/**
	 * All the execute() methods call this, which  after setting appropriate input/output variables
	 * calls _compileAndExecuteScript
	 * We have explicitly synchronized this function because MLContext/SystemML does not yet support multi-threading.
	 * @param dmlScriptFilePath script file path
	 * @param args arguments
	 * @param isNamedArgument is named argument
	 * @return output as MLOutput
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	private synchronized MLOutput compileAndExecuteScript(String dmlScriptFilePath, String [] args,  boolean isFile, boolean isNamedArgument, boolean isPyDML, String configFilePath) throws IOException, DMLException {
		try {

			DMLScript.SCRIPT_TYPE = isPyDML ? ScriptType.PYDML : ScriptType.DML;

			if(getActiveMLContext() != null) {
				throw new DMLRuntimeException("SystemML (and hence by definition MLContext) doesnot support parallel execute() calls from same or different MLContexts. "
						+ "As a temporary fix, please do explicit synchronization, i.e. synchronized(MLContext.class) { ml.execute(...) } ");
			}
			
			// Set active MLContext.
			_activeMLContext = this;
			if(_monitorUtils != null) {
				_monitorUtils.resetMonitoringData();
			}
			
			if( OptimizerUtils.isSparkExecutionMode() ) {
				// Depending on whether registerInput/registerOutput was called initialize the variables 
				String[] inputs = (_inVarnames != null) ? _inVarnames.toArray(new String[0]) : new String[0];
				String[] outputs = (_outVarnames != null) ? _outVarnames.toArray(new String[0]) : new String[0];
				Map<String, JavaPairRDD<?,?>> retVal = (_outVarnames!=null && !_outVarnames.isEmpty()) ? 
						retVal = new HashMap<String, JavaPairRDD<?,?>>() : null;
				Map<String, MatrixCharacteristics> outMetadata = new HashMap<String, MatrixCharacteristics>();
				Map<String, String> argVals = DMLScript.createArgumentsMap(isNamedArgument, args);
				
				// Run the DML script
				ExecutionContext ec = executeUsingSimplifiedCompilationChain(dmlScriptFilePath, isFile, argVals, isPyDML, inputs, outputs, _variables, configFilePath);
				SparkExecutionContext sec = (SparkExecutionContext) ec;
				
				// Now collect the output
				if(_outVarnames != null) {
					if(_variables == null)
						throw new DMLRuntimeException("The symbol table returned after executing the script is empty");			
					
					for( String ovar : _outVarnames ) {
						if( !_variables.keySet().contains(ovar) )
							throw new DMLException("Error: The variable " + ovar + " is not available as output after the execution of the DMLScript.");
						
						retVal.put(ovar, sec.getRDDHandleForVariable(ovar, InputInfo.BinaryBlockInputInfo));
						outMetadata.put(ovar, ec.getMatrixCharacteristics(ovar)); // For converting output to dataframe
					}
				}
				
				return new MLOutput(retVal, outMetadata);
			}
			else {
				throw new DMLRuntimeException("Unsupported runtime:" + DMLScript.rtplatform.name());
			}
		}
		finally {
			// Remove global dml config and all thread-local configs
			// TODO enable cleanup whenever invalid GNMF MLcontext is fixed 
			// (the test is invalid because it assumes that status of previous execute is kept)
			//ConfigurationManager.setGlobalConfig(new DMLConfig());
			//ConfigurationManager.clearLocalConfigs();
			
			// Reset active MLContext.
			_activeMLContext = null;	
		}
	}
	
	
	/**
	 * This runs the DML script and returns the ExecutionContext for the caller to extract the output variables.
	 * The caller (which is compileAndExecuteScript) is expected to set inputSymbolTable with appropriate matrix representation (RDD, MatrixObject).
	 * 
	 * @param dmlScriptFilePath script file path
	 * @param isFile true if file, false otherwise
	 * @param argVals map of args
	 * @param parsePyDML  true if pydml, false otherwise
	 * @param inputs the inputs
	 * @param outputs the outputs
	 * @param inputSymbolTable the input symbol table
	 * @param configFilePath path to config file
	 * @return the execution context
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	private ExecutionContext executeUsingSimplifiedCompilationChain(String dmlScriptFilePath, boolean isFile, Map<String, String> argVals, boolean parsePyDML, 
			String[] inputs, String[] outputs, LocalVariableMap inputSymbolTable, String configFilePath) 
		throws IOException, DMLException
	{
		//construct dml configuration
		DMLConfig config = (configFilePath == null) ? new DMLConfig() : new DMLConfig(configFilePath);
		for(Entry<String, String> param : _additionalConfigs.entrySet()) {
			config.setTextValue(param.getKey(), param.getValue());
		}
		
		//set global dml and specialized compiler configurations
		ConfigurationManager.setGlobalConfig(config);
		CompilerConfig cconf = new CompilerConfig();
		cconf.set(ConfigType.IGNORE_UNSPECIFIED_ARGS, true);
		cconf.set(ConfigType.REJECT_READ_WRITE_UNKNOWNS, false);
		cconf.set(ConfigType.ALLOW_CSE_PERSISTENT_READS, false);
		ConfigurationManager.setGlobalConfig(cconf);
		
		//read dml script string
		String dmlScriptStr = DMLScript.readDMLScript( isFile?"-f":"-s", dmlScriptFilePath);
		if(_monitorUtils != null) {
			_monitorUtils.setDMLString(dmlScriptStr);
		}
		
		//simplified compilation chain
		_rtprog = null;
		
		//parsing
		AParserWrapper parser = AParserWrapper.createParser(parsePyDML);
		DMLProgram prog;
		if (isFile) {
			prog = parser.parse(dmlScriptFilePath, null, argVals);
		} else {
			prog = parser.parse(null, dmlScriptStr, argVals);
		}
		
		//language validate
		DMLTranslator dmlt = new DMLTranslator(prog);
		dmlt.liveVariableAnalysis(prog);			
		dmlt.validateParseTree(prog);
		
		//hop construct/rewrite
		dmlt.constructHops(prog);
		dmlt.rewriteHopsDAG(prog);
		
		Explain.explain(prog);
		
		//rewrite persistent reads/writes
		if(inputSymbolTable != null) {
			RewriteRemovePersistentReadWrite rewrite = new RewriteRemovePersistentReadWrite(inputs, outputs, inputSymbolTable);
			ProgramRewriter rewriter2 = new ProgramRewriter(rewrite);
			rewriter2.rewriteProgramHopDAGs(prog);
		}
		
		//lop construct and runtime prog generation
		dmlt.constructLops(prog);
		_rtprog = prog.getRuntimeProgram(config);
		
		//optional global data flow optimization
		if(OptimizerUtils.isOptLevel(OptimizationLevel.O4_GLOBAL_TIME_MEMORY) ) {
			_rtprog = GlobalOptimizerWrapper.optimizeProgram(prog, _rtprog);
		}
		
		// launch SystemML appmaster not required as it is already launched
		
		//count number compiled MR jobs / SP instructions	
		ExplainCounts counts = Explain.countDistributedOperations(_rtprog);
		Statistics.resetNoOfCompiledJobs( counts.numJobs );
		
		// Initialize caching and scratch space
		DMLScript.initHadoopExecution(config);
		
		//final cleanup runtime prog
		JMLCUtils.cleanupRuntimeProgram(_rtprog, outputs);
				
		//create and populate execution context
		ExecutionContext ec = ExecutionContextFactory.createContext(_rtprog);
		if(inputSymbolTable != null) {
			ec.setVariables(inputSymbolTable);
		}
		
		//core execute runtime program	
		_rtprog.execute( ec );
		
		if(_monitorUtils != null)
			_monitorUtils.setExplainOutput(Explain.explain(_rtprog));
		
		return ec;
	}
	
	// -------------------------------- Private methods ends ----------------------------------------------------------
	
	// TODO: Add additional create to provide sep, missing values, etc. for CSV
	/**
	 * Experimental API: Might be discontinued in future release
	 * @param sqlContext the SQLContext
	 * @param filePath the file path
	 * @param format the format
	 * @return the MLMatrix
	 * @throws IOException if IOException occurs
	 * @throws DMLException if DMLException occurs
	 * @throws ParseException if ParseException occurs
	 */
	public MLMatrix read(SQLContext sqlContext, String filePath, String format) throws IOException, DMLException, ParseException {
		this.reset();
		this.registerOutput("output");
		MLOutput out = this.executeScript("output = read(\"" + filePath + "\", format=\"" + format + "\"); " + MLMatrix.writeStmt);
		JavaPairRDD<MatrixIndexes, MatrixBlock> blocks = out.getBinaryBlockedRDD("output");
		MatrixCharacteristics mcOut = out.getMatrixCharacteristics("output");
		return MLMatrix.createMLMatrix(this, sqlContext, blocks, mcOut);
	}	
}