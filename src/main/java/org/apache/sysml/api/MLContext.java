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
import org.apache.sysml.api.monitoring.SparkMonitoringUtil;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.OptimizerUtils.OptimizationLevel;
import org.apache.sysml.hops.globalopt.GlobalOptimizerWrapper;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteRemovePersistentReadWrite;
import org.apache.sysml.parser.AParserWrapper;
import org.apache.sysml.parser.DMLParseException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.IntIdentifier;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.StringIdentifier;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.instructions.spark.data.RDDProperties;
import org.apache.sysml.runtime.instructions.spark.functions.ConvertStringToLongTextPair;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBlockPairFunction;
import org.apache.sysml.runtime.instructions.spark.functions.CopyTextInputFunction;
import org.apache.sysml.runtime.instructions.spark.functions.SparkListener;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.Explain.ExplainCounts;
import org.apache.sysml.utils.Statistics;

/**
 * MLContext is useful for passing RDDs as input/output to SystemML. This API avoids the need to read/write
 * from HDFS (which is another way to pass inputs to SystemML).
 * <p>
 * Typical usage for MLContext is as follows:
 * <pre><code>
 * scala> import org.apache.sysml.api.MLContext
 * </code></pre>
 * <p>
 * Create input DataFrame from CSV file and potentially perform some feature transformation
 * <pre><code>
 * scala> val W = sqlContext.load("com.databricks.spark.csv", Map("path" -> "W.csv", "header" -> "false"))
 * scala> val H = sqlContext.load("com.databricks.spark.csv", Map("path" -> "H.csv", "header" -> "false"))
 * scala> val V = sqlContext.load("com.databricks.spark.csv", Map("path" -> "V.csv", "header" -> "false"))
 * </code></pre>
 * <p>
 * Create MLContext
 * <pre><code>
 * scala> val ml = new MLContext(sc)
 * </code></pre>
 * <p>
 * Register input and output DataFrame/RDD 
 * Supported format: 
 * <ol>
 * <li> DataFrame
 * <li> CSV/Text (as JavaRDD<String> or JavaPairRDD<LongWritable, Text>)
 * <li> Binary blocked RDD (JavaPairRDD<MatrixIndexes,MatrixBlock>))
 * </ol>
 * Also overloaded to support metadata information such as format, rlen, clen, ...
 * Please note the variable names given below in quotes correspond to the variables in DML script.
 * These variables need to have corresponding read/write associated in DML script.
 * Currently, only matrix variables are supported through registerInput/registerOutput interface.
 * To pass scalar variables, use named/positional arguments (described later) or wrap them into matrix variable.
 * <pre><code>
 * scala> ml.registerInput("V", V)
 * scala> ml.registerInput("W", W)
 * scala> ml.registerInput("H", H)
 * scala> ml.registerOutput("H")
 * scala> ml.registerOutput("W")
 * </code></pre>
 * <p>
 * Call script with default arguments:
 * <pre><code>
 * scala> val outputs = ml.execute("GNMF.dml")
 * </code></pre>
 * <p>
 * Also supported: calling script with positional arguments (args) and named arguments (nargs):
 * <pre><code> 
 * scala> val args = Array("V.mtx", "W.mtx",  "H.mtx",  "2000", "1500",  "50",  "1",  "WOut.mtx",  "HOut.mtx")
 * scala> val nargs = Map("maxIter"->"1", "V" -> "") 
 * scala> val outputs = ml.execute("GNMF.dml", args) # or ml.execute("GNMF_namedArgs.dml", nargs)
 * </code></pre>  
 * <p>
 * To run the script again using different (or even same arguments), but using same registered input/outputs:
 * <pre><code> 
 * scala> val new_outputs = ml.execute("GNMF.dml", new_args)
 * </code></pre>
 * <p>
 * However, to register new input/outputs, you need to first reset MLContext
 * <pre><code> 
 * scala> ml.reset()
 * scala> ml.registerInput("V", newV)
 * </code></pre>
 * <p>
 * Experimental API:
 * To monitor performance (only supported for Spark 1.4.0 or higher),
 * <pre><code>
 * scala> val ml = new MLContext(sc, true)
 * </code></pre>
 * <p>
 * If monitoring performance is enabled,
 * <pre><code> 
 * scala> print(ml.getMonitoringUtil().getExplainOutput())
 * scala> ml.getMonitoringUtil().getRuntimeInfoInHTML("runtime.html")
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
	
	// --------------------------------------------------
	// _monitorUtils is set only when MLContext(sc, true)
	private SparkMonitoringUtil _monitorUtils = null;
	
	/**
	 * Experimental API. Not supported in Python MLContext API.
	 * @return
	 */
	public SparkMonitoringUtil getMonitoringUtil() {
		return _monitorUtils;
	}
	// --------------------------------------------------
	
	/**
	 * Create an associated MLContext for given spark session.
	 * @param sc
	 * @throws DMLRuntimeException
	 */
	public MLContext(SparkContext sc) throws DMLRuntimeException {
		initializeSpark(sc, false, false);
	}
	
	/**
	 * Create an associated MLContext for given spark session.
	 * @param sc
	 * @throws DMLRuntimeException
	 */
	public MLContext(JavaSparkContext sc) throws DMLRuntimeException {
		initializeSpark(sc.sc(), false, false);
	}
	
	// ====================================================================================
	// Register input APIs
	// 1. DataFrame
	
	/**
	 * Register DataFrame as input. DataFrame is assumed to be in row format and each cell can be converted into double 
	 * through  Double.parseDouble(cell.toString()). This is suitable for passing dense matrices. For sparse matrices,
	 * consider passing through text format (using JavaRDD<String>, format="text")
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName
	 * @param df
	 * @throws DMLRuntimeException
	 */
	public void registerInput(String varName, DataFrame df) throws DMLRuntimeException {
		registerInput(varName, df, false);
	}
	
	/**
	 * Register DataFrame as input. 
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.  
	 * @param varName
	 * @param df
	 * @param containsID false if the DataFrame has an column ID which denotes the row ID.
	 * @throws DMLRuntimeException
	 */
	public void registerInput(String varName, DataFrame df, boolean containsID) throws DMLRuntimeException {
		MatrixCharacteristics mcOut = new MatrixCharacteristics();
		JavaPairRDD<MatrixIndexes, MatrixBlock> rdd = RDDConverterUtilsExt.dataFrameToBinaryBlock(new JavaSparkContext(_sc), df, mcOut, containsID);
		registerInput(varName, rdd, mcOut);
	}
	
	/**
	 * Experimental API. Not supported in Python MLContext API.
	 * @param varName
	 * @param df
	 * @throws DMLRuntimeException
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
	 * @param varName
	 * @param rdd
	 * @param format
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param missingValue
	 * @throws DMLRuntimeException
	 */
	public void registerInput(String varName, JavaRDD<String> rdd, String format, boolean hasHeader, 
			String delim, boolean fill, double missingValue) throws DMLRuntimeException {
		registerInput(varName, rdd, format, hasHeader, delim, fill, missingValue, -1, -1, -1);
	}
	
	/**
	 * Register CSV/Text as inputs: Method for supplying csv file format properties, but without dimensions or nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName
	 * @param rdd
	 * @param format
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param missingValue
	 * @throws DMLRuntimeException
	 */
	public void registerInput(String varName, RDD<String> rdd, String format, boolean hasHeader, 
			String delim, boolean fill, double missingValue) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD(), format, hasHeader, delim, fill, missingValue, -1, -1, -1);
	}
	
	/**
	 * Register CSV/Text as inputs: Method for supplying csv file format properties along with dimensions or nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName
	 * @param rdd
	 * @param format
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param missingValue
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws DMLRuntimeException
	 */
	public void registerInput(String varName, RDD<String> rdd, String format, boolean hasHeader, 
			String delim, boolean fill, double missingValue, long rlen, long clen, long nnz) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD(), format, hasHeader, delim, fill, missingValue, -1, -1, -1);
	}
	
	/**
	 * Register CSV/Text as inputs: Method for supplying csv file format properties along with dimensions or nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName
	 * @param rdd
	 * @param format
	 * @param hasHeader
	 * @param delim
	 * @param fill
	 * @param missingValue
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws DMLRuntimeException
	 */
	public void registerInput(String varName, JavaRDD<String> rdd, String format, boolean hasHeader, 
			String delim, boolean fill, double missingValue, long rlen, long clen, long nnz) throws DMLRuntimeException {
		RDDProperties properties = new RDDProperties();
		properties.setHasHeader(hasHeader);
		properties.setFill(fill);
		properties.setDelim(delim);
		properties.setMissingValue(missingValue);
		registerInput(varName, rdd.mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, nnz, properties);
	} 
	
	/**
	 * Register CSV/Text as inputs: Convenience method without dimensions and nnz. It uses default file properties (example: delim, fill, ..)
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName
	 * @param rdd
	 * @param format
	 * @throws DMLRuntimeException
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
	 * @param varName
	 * @param rdd
	 * @param format
	 * @throws DMLRuntimeException
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
	 * @param varName
	 * @param rdd
	 * @param format
	 * @param rlen
	 * @param clen
	 * @throws DMLRuntimeException
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
	 * @param varName
	 * @param rdd
	 * @param format
	 * @param rlen
	 * @param clen
	 * @throws DMLRuntimeException
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
	 * @param varName
	 * @param rdd
	 * @param format
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws DMLRuntimeException
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
	 * @param varName
	 * @param rdd
	 * @param format
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws DMLRuntimeException
	 */
	public void registerInput(String varName, RDD<String> rdd, String format, long rlen, long clen, long nnz) throws DMLRuntimeException {
		registerInput(varName, rdd.toJavaRDD().mapToPair(new ConvertStringToLongTextPair()), format, rlen, clen, nnz, null);
	}
	
	// All CSV related methods call this ... It provides access to dimensions, nnz, file properties.
	private void registerInput(String varName, JavaPairRDD<LongWritable, Text> textOrCsv_rdd, String format, long rlen, long clen, long nnz, RDDProperties properties) throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("The registerInput functionality only supported for spark runtime. Please use MLContext(sc) instead of default constructor.");
		}
		
		if(_variables == null)
			_variables = new LocalVariableMap();
		if(_inVarnames == null)
			_inVarnames = new ArrayList<String>();
		
		MatrixObject mo = null;
		if(format.compareTo("csv") == 0) {
			MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, nnz);
			mo = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(mc, OutputInfo.CSVOutputInfo, InputInfo.CSVInputInfo));
		}
		else if(format.compareTo("text") == 0) {
			if(rlen == -1 || clen == -1) {
				throw new DMLRuntimeException("The metadata is required in registerInput for format:" + format);
			}
			MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, nnz);
			mo = new MatrixObject(ValueType.DOUBLE, null, new MatrixFormatMetaData(mc, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo));
		}
		else if(format.compareTo("mm") == 0) {
			// TODO: Handle matrix market
			throw new DMLRuntimeException("Matrixmarket format is not yet implemented in registerInput: " + format);
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
	 * @param varName
	 * @param rdd
	 * @param rlen
	 * @param clen
	 * @throws DMLRuntimeException
	 */
	public void registerInput(String varName, JavaPairRDD<MatrixIndexes,MatrixBlock> rdd, long rlen, long clen) throws DMLRuntimeException {
		registerInput(varName, rdd, rlen, clen, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
	}
	
	/**
	 * Register binary blocked RDD with given dimensions, given block sizes and no nnz
	 * <p>
	 * Marks the variable in the DML script as input variable.
	 * Note that this expects a "varName = read(...)" statement in the DML script which through non-MLContext invocation
	 * would have been created by reading a HDFS file.
	 * @param varName
	 * @param rdd
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws DMLRuntimeException
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
	 * @param varName
	 * @param rdd
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param nnz
	 * @throws DMLRuntimeException
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
		JavaPairRDD<MatrixIndexes, MatrixBlock> copyRDD = rdd.mapToPair( new CopyBlockPairFunction() );
		
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, "temp", new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		mo.setRDDHandle(new RDDObject(copyRDD, varName));
		_variables.put(varName, mo);
		_inVarnames.add(varName);
		checkIfRegisteringInputAllowed();
	}
	
	// =============================================================================================
	
	/**
	 * Marks the variable in the DML script as output variable.
	 * Note that this expects a "write(varName, ...)" statement in the DML script which through non-MLContext invocation
	 * would have written the matrix to HDFS.
	 * @param varName
	 * @throws DMLRuntimeException
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
	 * @param namedArgs
	 * @param parsePyDML
	 * @param configFilePath
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, HashMap<String, String> namedArgs, boolean parsePyDML, String configFilePath) throws IOException, DMLException, DMLParseException {
		String [] args = new String[namedArgs.size()];
		int i = 0;
		for(Entry<String, String> entry : namedArgs.entrySet()) {
			if(entry.getValue().trim().compareTo("") == 0)
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
	 * @param namedArgs
	 * @param configFilePath
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, HashMap<String, String> namedArgs, String configFilePath) throws IOException, DMLException, DMLParseException {
		String [] args = new String[namedArgs.size()];
		int i = 0;
		for(Entry<String, String> entry : namedArgs.entrySet()) {
			if(entry.getValue().trim().compareTo("") == 0)
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
	 * @param namedArgs
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, HashMap<String, String> namedArgs) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, namedArgs, false, null);
	}
	
	/**
	 * Execute DML script by passing named arguments.
	 * @param dmlScriptFilePath
	 * @param namedArgs
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, scala.collection.immutable.Map<String, String> namedArgs) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)));
	}

	/**
	 * Experimental: Execute PyDML script by passing named arguments if parsePyDML=true.
	 * @param dmlScriptFilePath
	 * @param namedArgs
	 * @param parsePyDML
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, HashMap<String, String> namedArgs, boolean parsePyDML) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, namedArgs, parsePyDML, null);
	}
	
	/**
	 * Experimental: Execute PyDML script by passing named arguments if parsePyDML=true.
	 * @param dmlScriptFilePath
	 * @param namedArgs
	 * @param parsePyDML
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, scala.collection.immutable.Map<String, String> namedArgs, boolean parsePyDML) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, new HashMap<String, String>(scala.collection.JavaConversions.mapAsJavaMap(namedArgs)), parsePyDML);
	}
	
	/**
	 * Execute DML script by passing positional arguments using specified config file
	 * @param dmlScriptFilePath
	 * @param args
	 * @param configFilePath
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args, String configFilePath) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, args, false, configFilePath);
	}
	
	/**
	 * Execute DML script by passing positional arguments using specified config file
	 * This method is implemented for compatibility with Python MLContext.
	 * Java/Scala users should use 'MLOutput execute(String dmlScriptFilePath, String [] args, String configFilePath)' instead as
	 * equivalent scala collections (Seq/ArrayBuffer) is not implemented.
	 * @param dmlScriptFilePath
	 * @param args
	 * @param configFilePath
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> args, String configFilePath) throws IOException, DMLException, DMLParseException {
		String [] argsArr = new String[args.size()];
		argsArr = args.toArray(argsArr);
		return execute(dmlScriptFilePath, argsArr, false, configFilePath);
	}
	
	/**
	 * Execute DML script by passing positional arguments using default configuration
	 * @param dmlScriptFilePath
	 * @param args
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, args, false, null);
	}
	
	/**
	 * Execute DML script by passing positional arguments using default configuration.
	 * This method is implemented for compatibility with Python MLContext.
	 * Java/Scala users should use 'MLOutput execute(String dmlScriptFilePath, String [] args)' instead as
	 * equivalent scala collections (Seq/ArrayBuffer) is not implemented.
	 * @param dmlScriptFilePath
	 * @param args
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> args) throws IOException, DMLException, DMLParseException {
		String [] argsArr = new String[args.size()];
		argsArr = args.toArray(argsArr);
		return execute(dmlScriptFilePath, argsArr, false, null);
	}
	
	/**
	 * Experimental: Execute DML script by passing positional arguments if parsePyDML=true, using default configuration.
	 * This method is implemented for compatibility with Python MLContext.
	 * Java/Scala users should use 'MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML)' instead as
	 * equivalent scala collections (Seq/ArrayBuffer) is not implemented.
	 * @param dmlScriptFilePath
	 * @param args
	 * @param parsePyDML
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> args, boolean parsePyDML) throws IOException, DMLException, DMLParseException {
		String [] argsArr = new String[args.size()];
		argsArr = args.toArray(argsArr);
		return execute(dmlScriptFilePath, argsArr, parsePyDML, null);
	}
	
	/**
	 * Experimental: Execute DML script by passing positional arguments if parsePyDML=true, using specified config file.
	 * This method is implemented for compatibility with Python MLContext.
	 * Java/Scala users should use 'MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML, String configFilePath)' instead as
	 * equivalent scala collections (Seq/ArrayBuffer) is not implemented.
	 * @param dmlScriptFilePath
	 * @param args
	 * @param parsePyDML
	 * @param configFilePath
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, ArrayList<String> args, boolean parsePyDML, String configFilePath) throws IOException, DMLException, DMLParseException {
		String [] argsArr = new String[args.size()];
		argsArr = args.toArray(argsArr);
		return execute(dmlScriptFilePath, argsArr, parsePyDML, configFilePath);
	}
	
	/**
	 * Experimental: Execute DML script by passing positional arguments if parsePyDML=true, using specified config file.
	 * @param dmlScriptFilePath
	 * @param args
	 * @param parsePyDML
	 * @param configFilePath
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML, String configFilePath) throws IOException, DMLException, DMLParseException {
		return compileAndExecuteScript(dmlScriptFilePath, args, false, parsePyDML, configFilePath);
	}
	
	/**
	 * Experimental: Execute DML script by passing positional arguments if parsePyDML=true, using default configuration.
	 * @param dmlScriptFilePath
	 * @param args
	 * @param parsePyDML
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, String [] args, boolean parsePyDML) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, args, parsePyDML, null);
	}
	
	/**
	 * Execute DML script without any arguments using specified config path
	 * @param dmlScriptFilePath
	 * @param configFilePath
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath, String configFilePath) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, false, configFilePath);
	}
	
	/**
	 * Execute DML script without any arguments using default configuration.
	 * @param dmlScriptFilePath
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException 
	 */
	public MLOutput execute(String dmlScriptFilePath) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, false, null);
	}
	
	/**
	 * Experimental: Execute DML script without any arguments if parsePyDML=true, using specified config path.
	 * @param dmlScriptFilePath
	 * @param parsePyDML
	 * @param configFilePath
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, boolean parsePyDML, String configFilePath) throws IOException, DMLException, DMLParseException {
		return compileAndExecuteScript(dmlScriptFilePath, null, false, parsePyDML, configFilePath);
	}
	
	/**
	 * Experimental: Execute DML script without any arguments if parsePyDML=true, using default configuration.
	 * @param dmlScriptFilePath
	 * @param parsePyDML
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput execute(String dmlScriptFilePath, boolean parsePyDML) throws IOException, DMLException, DMLParseException {
		return execute(dmlScriptFilePath, parsePyDML, null);
	}
	
	// -------------------------------- Utility methods begins ----------------------------------------------------------
	
	
	/**
	 * Call this method if you want to clear any RDDs set via registerInput, registerOutput.
	 * This is required if ml.execute(..) has been called earlier and you want to call a new DML script. 
	 * @throws DMLRuntimeException 
	 */
	public void reset() 
			throws DMLRuntimeException 
	{
		//cleanup variables from bufferpool, incl evicted files 
		//(otherwise memory leak because bufferpool holds references)
		CacheableData.cleanupCacheDir();

		//clear mlcontext state
		_inVarnames = null;
		_outVarnames = null;
		_variables = null;
	}
	
	/**
	 * Used internally
	 * @param source
	 * @param target
	 * @throws LanguageException
	 */
	void setAppropriateVarsForRead(Expression source, String target) 
		throws LanguageException 
	{
		boolean isTargetRegistered = isRegisteredAsInput(target);
		boolean isReadExpression = (source instanceof DataExpression && ((DataExpression) source).isRead());
		if(isTargetRegistered && isReadExpression) {
			// Do not check metadata file for registered reads 
			((DataExpression) source).setCheckMetadata(false);
			
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
		}
	}
	
	/**
	 * Used internally
	 * @param tmp
	 * @return
	 */
	ArrayList<Instruction> performCleanupAfterRecompilation(ArrayList<Instruction> tmp) {
		String [] outputs = null;
		if(_outVarnames != null) {
			outputs = _outVarnames.toArray(new String[0]);
		}
		else {
			outputs = new String[0];
		}
		
		// No need to clean up entire program as this method is only called for last level program block
//		JMLCUtils.cleanupRuntimeProgram(_rtprog, outputs);
		
		for( int i=0; i<tmp.size(); i++ )
		{
			Instruction linst = tmp.get(i);
			if( linst instanceof VariableCPInstruction && ((VariableCPInstruction)linst).isRemoveVariable() )
			{
				VariableCPInstruction varinst = (VariableCPInstruction) linst;
				for( String var : outputs )
					if( varinst.isRemoveVariable(var) )
					{
						tmp.remove(i);
						i--;
						break;
					}
			}
		}
		
		return tmp;
	}
	
	// -------------------------------- Utility methods ends ----------------------------------------------------------
		
	
	// -------------------------------- Experimental API begins ----------------------------------------------------------
	/**
	 * Experimental api:
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc
	 * @param monitorPerformance
	 * @throws DMLRuntimeException 
	 */
	public MLContext(SparkContext sc, boolean monitorPerformance) throws DMLRuntimeException {
		initializeSpark(sc, monitorPerformance, false);
	}
	
	/**
	 * Experimental api:
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc
	 * @param monitorPerformance
	 * @throws DMLRuntimeException
	 */
	public MLContext(JavaSparkContext sc, boolean monitorPerformance) throws DMLRuntimeException {
		initializeSpark(sc.sc(), monitorPerformance, false);
	}
	
	/**
	 * Experimental api:
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc
	 * @param monitorPerformance
	 * @param setForcedSparkExecType
	 * @throws DMLRuntimeException
	 */
	public MLContext(SparkContext sc, boolean monitorPerformance, boolean setForcedSparkExecType) throws DMLRuntimeException {
		initializeSpark(sc, monitorPerformance, setForcedSparkExecType);
	}
	
	/**
	 * Experimental api:
	 * Setting monitorPerformance to true adds additional overhead of storing state. So, use it only if necessary.
	 * @param sc
	 * @param monitorPerformance
	 * @param setForcedSparkExecType
	 * @throws DMLRuntimeException
	 */
	public MLContext(JavaSparkContext sc, boolean monitorPerformance, boolean setForcedSparkExecType) throws DMLRuntimeException {
		initializeSpark(sc.sc(), monitorPerformance, setForcedSparkExecType);
	}
	
	// -------------------------------- Experimental API ends ----------------------------------------------------------
	
	// -------------------------------- Private methods begins ----------------------------------------------------------
	private boolean isRegisteredAsInput(String varName) {
		if(_inVarnames != null) {
			for(String v : _inVarnames) {
				if(v.compareTo(varName) == 0) {
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
	 * Experimental API. Not supported in Python MLContext API.
	 * @param dmlScript
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLOutput executeScript(String dmlScript) throws IOException, DMLException, DMLParseException {
		return compileAndExecuteScript(dmlScript, null, false, false, false, null);
	}
	
	public MLOutput executeScript(String dmlScript, String configFilePath) throws IOException, DMLException, DMLParseException {
		return compileAndExecuteScript(dmlScript, null, false, false, false, configFilePath);
	}
	
	private void checkIfRegisteringInputAllowed() throws DMLRuntimeException {
		if(!(DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)) {
			throw new DMLRuntimeException("ERROR: registerInput is only allowed for spark execution mode");
		}
	}
	
	private MLOutput compileAndExecuteScript(String dmlScriptFilePath, String [] args, boolean isNamedArgument, boolean isPyDML, String configFilePath) throws IOException, DMLException, DMLParseException {
		return compileAndExecuteScript(dmlScriptFilePath, args, true, isNamedArgument, isPyDML, configFilePath);
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
	 * @throws DMLParseException
	 */
	private synchronized MLOutput compileAndExecuteScript(String dmlScriptFilePath, String [] args,  boolean isFile, boolean isNamedArgument, boolean isPyDML, String configFilePath) throws IOException, DMLException, DMLParseException {
		try {
			if(getActiveMLContext() != null) {
				throw new DMLRuntimeException("SystemML (and hence by definition MLContext) doesnot support parallel execute() calls from same or different MLContexts. "
						+ "As a temporary fix, please do explicit synchronization, i.e. synchronized(MLContext.class) { ml.execute(...) } ");
			}
			else {
				// Set active MLContext.
				_activeMLContext = this;
			}
			
			
			if(_monitorUtils != null) {
				_monitorUtils.resetMonitoringData();
			}
			
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
				ExecutionContext ec = executeUsingSimplifiedCompilationChain(dmlScriptFilePath, isFile, argVals, isPyDML, inputs, outputs, _variables, configFilePath);
				
				// Now collect the output
				if(_outVarnames != null) {
					if(_variables == null) {
						throw new DMLRuntimeException("The symbol table returned after executing the script is empty");
					}
					
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
			else {
				throw new DMLRuntimeException("Unsupported runtime:" + DMLScript.rtplatform.name());
			}
		
		}
		finally {
			// Reset active MLContext.
			_activeMLContext = null;
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
	 * @throws DMLParseException
	 */
	private ExecutionContext executeUsingSimplifiedCompilationChain(String dmlScriptFilePath, boolean isFile, HashMap<String, String> argVals, boolean parsePyDML, 
			String[] inputs, String[] outputs, LocalVariableMap inputSymbolTable, String configFilePath) throws IOException, DMLException, DMLParseException {
		DMLConfig config = null;
		if(configFilePath == null) {
			config = new DMLConfig();
		}
		else {
			config = new DMLConfig(configFilePath);
		}
		
		ConfigurationManager.setConfig(config);
		
		String dmlScriptStr = null;
		if(isFile)
			dmlScriptStr = DMLScript.readDMLScript("-f", dmlScriptFilePath);
		else 
			dmlScriptStr = DMLScript.readDMLScript("-s", dmlScriptFilePath);
			
		if(_monitorUtils != null) {
			_monitorUtils.setDMLString(dmlScriptStr);
		}
		
		DataExpression.REJECT_READ_UNKNOWN_SIZE = false;
		
		//simplified compilation chain
		_rtprog = null;
		
		//parsing
		AParserWrapper parser = AParserWrapper.createParser(parsePyDML);
		DMLProgram prog = parser.parse(dmlScriptFilePath, dmlScriptStr, argVals);
		if(prog == null) {
			throw new DMLParseException("Couldnot parse the file:" + dmlScriptFilePath);
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
			RewriteRemovePersistentReadWrite rewrite = new RewriteRemovePersistentReadWrite(inputs, outputs);
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
		
		// System.out.println(Explain.explain(_rtprog));
		
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
	 * @param sqlContext
	 * @param filePath
	 * @param format
	 * @return
	 * @throws IOException
	 * @throws DMLException
	 * @throws DMLParseException
	 */
	public MLMatrix read(SQLContext sqlContext, String filePath, String format) throws IOException, DMLException, DMLParseException {
		this.reset();
		this.registerOutput("output");
		MLOutput out = this.executeScript("output = read(\"" + filePath + "\", format=\"" + format + "\"); " + MLMatrix.writeStmt);
		JavaPairRDD<MatrixIndexes, MatrixBlock> blocks = out.getBinaryBlockedRDD("output");
		MatrixCharacteristics mcOut = out.getMatrixCharacteristics("output");
		return MLMatrix.createMLMatrix(this, sqlContext, blocks, mcOut);
	}
	
//	// TODO: Test this in different scenarios: sparse/dense/mixed
//	/**
//	 * Experimental unstable API: Might be discontinued in future release
//	 * @param ml
//	 * @param sqlContext
//	 * @param mllibMatrix
//	 * @return
//	 * @throws DMLRuntimeException
//	 */
//	public MLMatrix read(SQLContext sqlContext, BlockMatrix mllibMatrix) throws DMLRuntimeException {
//		long nnz = -1; // TODO: Find number of non-zeros from mllibMatrix ... This is important !!
//		
//		JavaPairRDD<Tuple2<Object, Object>, Matrix> mllibBlocks = JavaPairRDD.fromJavaRDD(mllibMatrix.blocks().toJavaRDD());
//		long rlen = mllibMatrix.numRows(); long clen = mllibMatrix.numCols();
//		int brlen = mllibMatrix.numRowBlocks();
//		int bclen = mllibMatrix.numColBlocks();
//		if(mllibMatrix.numRowBlocks() != DMLTranslator.DMLBlockSize && mllibMatrix.numColBlocks() != DMLTranslator.DMLBlockSize) {
//			System.err.println("WARNING: Since the block size of mllib matrix is not " + DMLTranslator.DMLBlockSize + ", it may cause "
//					+ "reblocks");
//		}
//		
//		JavaPairRDD<MatrixIndexes, MatrixBlock> blocks = mllibBlocks
//				.mapToPair(new ConvertMLLibBlocksToBinaryBlocks(rlen, clen, brlen, bclen));
//		
//		MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, brlen, bclen, nnz);
//		return MLMatrix.createMLMatrix(this, sqlContext, blocks, mc);
//	}
	
}
