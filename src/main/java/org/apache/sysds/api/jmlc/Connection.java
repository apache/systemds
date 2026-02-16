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

package org.apache.sysds.api.jmlc;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.meta.MetaDataAll;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.hops.rewrite.RewriteRemovePersistentReadWrite;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.CollectionUtils;
import org.apache.sysds.runtime.util.DataConverter;

/**
 * Interaction with SystemDS using the JMLC (Java Machine Learning Connector) API is initiated with
 * a {@link Connection} object. The JMLC API is designed after JDBC. A DML script is precompiled by calling
 * the {@link #prepareScript(String, String[], String[])}
 * method or the {@link #prepareScript(String, Map, String[], String[])}
 * method on the {@link Connection} object, which returns a
 * {@link PreparedScript} object. Note that this is similar to calling
 * a {@code prepareStatement} method on a JDBC {@code Connection} object.
 * 
 * <p>
 * Following this, input variable data is passed to the script by calling the
 * {@code setFrame}, {@code setMatrix}, and {@code setScalar} methods of the {@link PreparedScript}
 * object. The script is executed via {@link PreparedScript}'s
 * {@link PreparedScript#executeScript() executeScript} method,
 * which returns a {@link ResultVariables} object, which is similar to a JDBC
 * {@code ResultSet}. Data can be read from a {@link ResultVariables} object by calling
 * its {@link ResultVariables#getFrame(String) getFrame} and
 * {@link ResultVariables#getMatrix(String) getMatrix} methods.
 */
public class Connection implements Closeable
{
	private final DMLConfig _dmlconf;
	private final CompilerConfig _cconf;
	private static FileSystem fs = null;
	
	/**
	 * Connection constructor, the starting point for any other JMLC API calls.
	 * 
	 */
	public Connection() {
		 //with default dml configuration
		this(new DMLConfig());
	}
	
	/**
	 * Connection constructor, the starting point for any other JMLC API calls.
	 * This variant allows to enable a set of boolean compiler configurations.
	 * 
	 * @param cconfigs one or many boolean compiler configurations to enable.
	 */
	public Connection(CompilerConfig.ConfigType... cconfigs) {
		//basic constructor, which also constructs the compiler config
		this(new DMLConfig()); //with default dml configuration
		
		//set optional compiler configurations in current config
		setConfigTypes(true, cconfigs);
		setLocalConfigs();
	}
	
	/**
	 * Connection constructor, the starting point for any other JMLC API calls.
	 * This variant allows to pass a global dml configuration and enable a set
	 * of boolean compiler configurations.
	 * 
	 * @param dmlconfig a dml configuration.
	 * @param cconfigs one or many boolean compiler configurations to enable.
	 */
	public Connection(DMLConfig dmlconfig, CompilerConfig.ConfigType... cconfigs) {
		//basic constructor, which also constructs the compiler config
		this(dmlconfig); 
		
		//set optional compiler configurations in current config
		setConfigTypes(true, cconfigs);
		setLocalConfigs();
	}
	
	/**
	 * Connection constructor, the starting point for any other JMLC API calls.
	 * This variant allows to pass a global dml configuration.
	 * 
	 * @param dmlconfig a dml configuration.
	 */
	public Connection(DMLConfig dmlconfig) {
		DMLScript.setGlobalExecMode(ExecMode.SINGLE_NODE);
		
		//setup basic parameters for embedded execution
		//(parser, compiler, and runtime parameters)
		_cconf = OptimizerUtils.constructCompilerConfig(dmlconfig);
		_cconf.set(ConfigType.IGNORE_UNSPECIFIED_ARGS, true);
		_cconf.set(ConfigType.IGNORE_READ_WRITE_METADATA, true);
		_cconf.set(ConfigType.IGNORE_TEMPORARY_FILENAMES, true);
		_cconf.set(ConfigType.REJECT_READ_WRITE_UNKNOWNS, false);
		_cconf.set(ConfigType.ALLOW_CSE_PERSISTENT_READS, false);
		_cconf.set(ConfigType.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS, false);

		//disable caching globally 
		CacheableData.disableCaching();
		
		//assign the given configuration
		_dmlconf = dmlconfig;
		
		setLocalConfigs();
	}

	/**
	 * Sets compiler configs.
	 * @param activate activate or disable
	 * @param cconfigs the configs to set
	 */
	public void setConfigTypes(boolean activate, CompilerConfig.ConfigType... cconfigs) {
		for( ConfigType configType : cconfigs )
			_cconf.set(configType, activate);
	}

	/**
	 * Sets a boolean flag indicating if runtime statistics should be gathered
	 * Same behavior as in "MLContext.setStatistics()"
	 *
	 * @param stats boolean value with true indicating statistics should be gathered
	 */
	public void setStatistics(boolean stats) { DMLScript.STATISTICS = stats; }

	/**
	 * Sets a boolean flag indicating if lineage trace should be captured 
	 * @param lt boolean value with true indicating lineage should be captured 
	 */
	public void setLineage(boolean lt) { 
		DMLScript.LINEAGE = lt; 
		Lineage.resetInternalState();
	}

	/**
	 * Sets a boolean flag indicating if memory profiling statistics should be
	 * gathered. The option is false by default.
	 * @param stats boolean value with true indicating memory statistics should be gathered
	 */
	public void gatherMemStats(boolean stats) {
		DMLScript.STATISTICS = stats || DMLScript.STATISTICS;
		DMLScript.JMLC_MEM_STATISTICS = stats;
	}
	
	/**
	 * Prepares (precompiles) a script and registers input and output variables.
	 * 
	 * @param script string representing the DML or PyDML script
	 * @param inputs string array of input variables to register
	 * @param outputs string array of output variables to register
	 * @return PreparedScript object representing the precompiled script
	 */
	public PreparedScript prepareScript( String script, String[] inputs, String[] outputs) {
		return prepareScript(script, Collections.emptyMap(), inputs, outputs);
	}
	
	/**
	 * Prepares (precompiles) a script, sets input parameter values, and registers input and output variables.
	 * 
	 * @param script string representing the DML or PyDML script
	 * @param args map of input parameters ($) and their values
	 * @param inputs string array of input variables to register
	 * @param outputs string array of output variables to register
	 * @return PreparedScript object representing the precompiled script
	 */
	public PreparedScript prepareScript( String script, Map<String, String> args, String[] inputs, String[] outputs) {
		return prepareScript(script, Collections.emptyMap(), args, inputs, outputs);
	}
	
	/**
	 * Prepares (precompiles) a script, sets input parameter values, and registers input and output variables.
	 * 
	 * @param script string representing of the DML or PyDML script
	 * @param nsscripts map (name, script) of the DML or PyDML namespace scripts
	 * @param args map of input parameters ($) and their values
	 * @param inputs string array of input variables to register
	 * @param outputs string array of output variables to register
	 * @return PreparedScript object representing the precompiled script
	 */
	public PreparedScript prepareScript(String script, Map<String,String> nsscripts, Map<String, String> args, String[] inputs, String[] outputs) {
		//check for valid names of passed arguments
		String[] invalidArgs = args.keySet().stream()
			.filter(k -> k==null || !k.startsWith("$")).toArray(String[]::new);
		if( invalidArgs.length > 0 )
			throw new LanguageException("Invalid argument names: "+Arrays.toString(invalidArgs));
		
		//check for valid names of input and output variables
		String[] invalidVars = CollectionUtils.asSet(inputs, outputs).stream()
			.filter(k -> k==null || k.startsWith("$")).toArray(String[]::new);
		if( invalidVars.length > 0 )
			throw new LanguageException("Invalid variable names: "+Arrays.toString(invalidVars));
		
		setLocalConfigs();
		
		//simplified compilation chain
		Program rtprog = null;
		try {
			//parsing
			ParserWrapper parser = ParserFactory.createParser(nsscripts);
			DMLProgram prog = parser.parse(null, script, args);
			
			//language validate
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			
			//hop construct/rewrite
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);
			
			//rewrite persistent reads/writes
			RewriteRemovePersistentReadWrite rewrite = new RewriteRemovePersistentReadWrite(inputs, outputs);
			ProgramRewriter rewriter2 = new ProgramRewriter(rewrite);
			rewriter2.rewriteProgramHopDAGs(prog);
			
			//lop construct and runtime prog generation
			dmlt.constructLops(prog);
			rtprog = dmlt.getRuntimeProgram(prog, _dmlconf);
			
			//final cleanup runtime prog
			JMLCUtils.cleanupRuntimeProgram(rtprog, outputs);
		}
		catch(ParseException pe) {
			// don't chain ParseException (for cleaner error output)
			throw pe;
		}
		catch(Exception ex) {
			throw new DMLException(ex);
		}
		
		//return newly create precompiled script 
		return new PreparedScript(rtprog, inputs, outputs, _dmlconf, _cconf);
	}
	
	/**
	 * Close connection to SystemDS, which clears the
	 * thread-local DML and compiler configurations.
	 */
	@Override
	public void close() {
		//clear thread-local configurations
		ConfigurationManager.clearLocalConfigs();
		if( ConfigurationManager.isCodegenEnabled() )
			SpoofCompiler.cleanupCodeGenerator();
	}
	
	/**
	 * Read a DML or PyDML file as a string.
	 * 
	 * @param fname the filename of the script
	 * @return string content of the script file
	 * @throws IOException if IOException occurs
	 */
	public String readScript(String fname) 
		throws IOException
	{
		StringBuilder sb = new StringBuilder();
		BufferedReader in = null;
		try 
		{
			//read from hdfs or gpfs file system
			if(    fname.startsWith("hdfs:") || fname.startsWith("gpfs:")
				|| IOUtilFunctions.isObjectStoreFileScheme(new Path(fname)) ) 
			{ 
				Path scriptPath = new Path(fname);
				fs = IOUtilFunctions.getFileSystem(scriptPath);
				in = new BufferedReader(new InputStreamReader(fs.open(scriptPath)));

			}
			// from local file system
			else { 
				in = new BufferedReader(new FileReader(fname));
			}
			
			//core script reading
			String tmp = null;
			while ((tmp = in.readLine()) != null) {
				sb.append( tmp );
				sb.append( "\n" );
			}
		}
		finally {
			if(fs != null)
				fs.close();
			IOUtilFunctions.closeSilently(in);
		}
		
		return sb.toString();
	}
	
	////////////////////////////////////////////
	// Read matrices
	////////////////////////////////////////////
	
	/**
	 * Reads an input matrix in arbitrary format from HDFS into a dense double array.
	 * NOTE: this call currently only supports default configurations for CSV.
	 * 
	 * @param fname the filename of the input matrix
	 * @return matrix as a two-dimensional double array
	 * @throws IOException if IOException occurs
	 */
	public double[][] readDoubleMatrix(String fname) 
		throws IOException
	{
		try {
			//read json meta data 
			String fnamemtd = DataExpression.getMTDFileName(fname);
			MetaDataAll metaObj = new MetaDataAll(fnamemtd, false, true);
			
			//parse meta data
			long rows = metaObj.getDim1();
			long cols = metaObj.getDim2();
			int blen = metaObj.getBlocksize();
			long nnz = metaObj.getNnz();
			FileFormat fmt = metaObj.getFileFormat();
		
			//read matrix file
			return readDoubleMatrix(fname, fmt, rows, cols, blen, nnz);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * Reads an input matrix in arbitrary format from HDFS into a dense double array.
	 * NOTE: this call currently only supports default configurations for CSV.
	 * 
	 * @param fname the filename of the input matrix
	 * @param fmt file format
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @param blen block length
	 * @param nnz number of non-zero values, -1 indicates unknown
	 * @return matrix as a two-dimensional double array
	 * @throws IOException if IOException occurs
	 */
	public double[][] readDoubleMatrix(String fname, FileFormat fmt, long rows, long cols, int blen, long nnz) 
		throws IOException
	{
		setLocalConfigs();
		
		try {
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(fmt);
			MatrixBlock mb = reader.readMatrixFromHDFS(fname, rows, cols, blen, nnz);
			return DataConverter.convertToDoubleMatrix(mb);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * Converts an input string representation of a matrix in csv or textcell format
	 * into a dense double array. The meta data string is the SystemDS generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input string matrix in csv or textcell format
	 * @param meta string representing SystemDS matrix metadata in JSON format
	 * @return matrix as a two-dimensional double array
	 * @throws IOException if IOException occurs
	 */
	public double[][] convertToDoubleMatrix(String input, String meta) 
		throws IOException
	{
		MatrixBlock mb = convertToMatrix(input, meta);
		return DataConverter.convertToDoubleMatrix(mb);
	}
	
	/**
	 * Converts an input string representation of a matrix in textcell format
	 * into a dense double array. 
	 * 
	 * @param input string matrix in textcell format
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @return matrix as a two-dimensional double array
	 * @throws IOException if IOException occurs
	 */
	public double[][] convertToDoubleMatrix(String input, int rows, int cols) throws IOException {
		try( InputStream is = IOUtilFunctions.toInputStream(input) ) {
			return convertToDoubleMatrix(is, rows, cols);
		}
	}
	
	/**
	 * Converts an input stream of a string matrix in textcell format
	 * into a dense double array. 
	 * 
	 * @param input InputStream to a string matrix in textcell format
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @return matrix as a two-dimensional double array
	 * @throws IOException if IOException occurs
	 */
	public double[][] convertToDoubleMatrix(InputStream input, int rows, int cols) throws IOException {
		return convertToDoubleMatrix(input, rows, cols, FileFormat.defaultFormatString());
	}
	
	/**
	 * Converts an input stream of a string matrix in csv or textcell format
	 * into a dense double array. 
	 * 
	 * @param input InputStream to a string matrix in csv or textcell format
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @param format input format of the given stream
	 * @return matrix as a two-dimensional double array
	 * @throws IOException if IOException occurs
	 */
	public double[][] convertToDoubleMatrix(InputStream input, int rows, int cols, String format) 
		throws IOException
	{
		MatrixBlock mb = convertToMatrix(input, rows, cols, format);
		return DataConverter.convertToDoubleMatrix(mb);
	}
	
	/**
	 * Converts an input string representation of a matrix in csv or textcell format
	 * into a matrix block. The meta data string is the SystemDS generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input string matrix in csv or textcell format
	 * @param meta string representing SystemDS matrix metadata in JSON format
	 * @return matrix as a matrix block
	 * @throws IOException if IOException occurs
	 */
	public MatrixBlock convertToMatrix(String input, String meta) throws IOException {
		try( InputStream is = IOUtilFunctions.toInputStream(input) ) {
			return convertToMatrix(is, meta);
		}
	}
	
	/**
	 * Converts an input stream of a string matrix in csv or textcell format
	 * into a matrix block. The meta data string is the SystemDS generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input InputStream to a string matrix in csv or textcell format
	 * @param meta string representing SystemDS matrix metadata in JSON format
	 * @return matrix as a matrix block
	 * @throws IOException if IOException occurs
	 */
	public MatrixBlock convertToMatrix(InputStream input, String meta) 
		throws IOException
	{
		try {
			//parse meta data
			MetaDataAll mtd = new MetaDataAll(meta);
			int rows = (int) mtd.getDim1();
			int cols = (int) mtd.getDim2();
			String format = mtd.getFormatTypeString();
			
			//parse the input matrix
			return convertToMatrix(input, rows, cols, format);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * Converts an input string representation of a matrix in textcell format
	 * into a matrix block. 
	 * 
	 * @param input string matrix in textcell format
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @return matrix as a matrix block
	 * @throws IOException if IOException occurs
	 */
	public MatrixBlock convertToMatrix(String input, int rows, int cols) throws IOException {
		try( InputStream is = IOUtilFunctions.toInputStream(input) ) {
			return convertToMatrix(is, rows, cols);
		}
	}
	
	/**
	 * Converts an input stream of a string matrix in text format
	 * into a matrix block. 
	 * 
	 * @param input InputStream to a string matrix in text format
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @return matrix as a matrix block
	 * @throws IOException if IOException occurs
	 */
	public MatrixBlock convertToMatrix(InputStream input, int rows, int cols) throws IOException {
		return convertToMatrix(input, rows, cols, FileFormat.defaultFormatString());
	}
	
	/**
	 * Converts an input stream of a string matrix in csv or text format
	 * into a matrix block. 
	 * 
	 * @param input InputStream to a string matrix in csv or text format
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @param format input format of the given stream
	 * @return matrix as a matrix block
	 * @throws IOException if IOException occurs
	 */
	public MatrixBlock convertToMatrix(InputStream input, int rows, int cols, String format) 
		throws IOException
	{
		MatrixBlock ret = null;

		//sanity check input format
		if(!(FileFormat.TEXT.toString().equals(format)
			||FileFormat.MM.toString().equals(format)
			||FileFormat.CSV.toString().equals(format)) ) {
			throw new IOException("Invalid input format (expected: csv, text or mm): "+format);
		}
		
		setLocalConfigs();
		
		try {
			//read input matrix
			FileFormat fmt = FileFormat.safeValueOf(format);
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(fmt);
			int blksz = ConfigurationManager.getBlocksize();
			ret = reader.readMatrixFromInputStream(
				input, rows, cols, blksz, (long)rows*cols);
		}
		catch(DMLRuntimeException rex) {
			throw new IOException(rex);
		}
		
		return ret;
	}
	
	////////////////////////////////////////////
	// Read frames
	////////////////////////////////////////////

	/**
	 * Reads an input frame in arbitrary format from HDFS into a dense string array.
	 * NOTE: this call currently only supports default configurations for CSV.
	 * 
	 * @param fname the filename of the input frame
	 * @return frame as a two-dimensional string array
	 * @throws IOException if IOException occurs
	 */
	public String[][] readStringFrame(String fname) 
		throws IOException
	{
		try {
			//read json meta data 
			String fnamemtd = DataExpression.getMTDFileName(fname);

			MetaDataAll metaObj = new MetaDataAll(fnamemtd, false, true);

			//parse meta data
			long rows = metaObj.getDim1();
			long cols = metaObj.getDim2();
			FileFormat fmt = metaObj.getFileFormat();
		
			//read frame file
			return readStringFrame(fname, fmt, rows, cols);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * Reads an input frame in arbitrary format from HDFS into a dense string array.
	 * NOTE: this call currently only supports default configurations for CSV.
	 * 
	 * @param fname the filename of the input frame
	 * @param fmt file format type
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @return frame as a two-dimensional string array
	 * @throws IOException if IOException occurs
	 */
	public String[][] readStringFrame(String fname, FileFormat fmt, long rows, long cols) 
		throws IOException
	{
		setLocalConfigs();
		
		try {
			FrameReader reader = FrameReaderFactory.createFrameReader(fmt);
			FrameBlock mb = reader.readFrameFromHDFS(fname, rows, cols);
			return DataConverter.convertToStringFrame(mb);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * Converts an input string representation of a frame in csv or textcell format
	 * into a dense string array. The meta data string is the SystemDS generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input string frame in csv or textcell format
	 * @param meta string representing SystemDS frame metadata in JSON format
	 * @return frame as a two-dimensional string array
	 * @throws IOException if IOException occurs
	 */
	public String[][] convertToStringFrame(String input, String meta) 
		throws IOException
	{
		FrameBlock fb = convertToFrame(input, meta);
		return DataConverter.convertToStringFrame(fb);
	}
	
	/**
	 * Converts an input stream of a string frame in textcell format
	 * into a dense string array. 
	 * 
	 * @param input string frame in textcell format
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @return frame as a two-dimensional string array
	 * @throws IOException if IOException occurs
	 */
	public String[][] convertToStringFrame(String input, int rows, int cols) throws IOException {
		try( InputStream is = IOUtilFunctions.toInputStream(input) ) {
			return convertToStringFrame(is, rows, cols);
		}
	}
	
	/**
	 * Converts an input stream of a string frame in textcell format
	 * into a dense string array. 
	 * 
	 * @param input InputStream to a string frame in text format
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @return frame as a two-dimensional string array
	 * @throws IOException if IOException occurs
	 */
	public String[][] convertToStringFrame(InputStream input, int rows, int cols) throws IOException {
		return convertToStringFrame(input, rows, cols, FileFormat.defaultFormatString());
	}
	
	/**
	 * Converts an input stream of a string frame in csv or text format
	 * into a dense string array. 
	 * 
	 * @param input InputStream to a string frame in csv or text format
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @param format input format of the given stream
	 * @return frame as a two-dimensional string array
	 * @throws IOException if IOException occurs
	 */
	public String[][] convertToStringFrame(InputStream input, int rows, int cols, String format) 
		throws IOException
	{
		FrameBlock fb = convertToFrame(input, rows, cols, format);
		return DataConverter.convertToStringFrame(fb);
	}
	
	/**
	 * Converts an input string representation of a frame in csv or textcell format
	 * into a frame block. The meta data string is the SystemDS generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input string frame in csv or textcell format
	 * @param meta string representing SystemDS frame metadata in JSON format
	 * @return frame as a frame block
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock convertToFrame(String input, String meta) throws IOException {
		try( InputStream is = IOUtilFunctions.toInputStream(input) ) {
			return convertToFrame(is, meta);
		}
	}
	
	/**
	 * Converts an input stream of a string frame in csv or textcell format
	 * into a frame block. The meta data string is the SystemDS generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input InputStream to a string frame in csv or textcell format
	 * @param meta string representing SystemDS frame metadata in JSON format
	 * @return frame as a frame block
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock convertToFrame(InputStream input, String meta) 
		throws IOException
	{
		try {
			//parse meta data
			MetaDataAll mtd = new MetaDataAll(meta);
			int rows = (int) mtd.getDim1();
			int cols = (int) mtd.getDim2();
			String format = mtd.getFormatTypeString();
			
			//parse the input frame
			return convertToFrame(input, rows, cols, format);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * Converts an input string representation of a frame in textcell format
	 * into a frame block. 
	 * 
	 * @param input string frame in textcell format
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @return frame as a frame block
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock convertToFrame(String input, int rows, int cols) throws IOException {
		try( InputStream is = IOUtilFunctions.toInputStream(input) ) {
			return convertToFrame(is, rows, cols);
		}
	}
	
	/**
	 * Converts an input stream of a string frame in textcell format
	 * into a frame block. 
	 * 
	 * @param input InputStream to a string frame in textcell format
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @return frame as a frame block
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock convertToFrame(InputStream input, int rows, int cols) throws IOException {
		return convertToFrame(input, rows, cols, FileFormat.TEXT.toString());
	}
	
	/**
	 * Converts an input stream of a frame in csv or textcell format
	 * into a frame block. 
	 * 
	 * @param input InputStream to a string frame in csv or textcell format
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @param format input format of the given stream
	 * @return frame as a frame block
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock convertToFrame(InputStream input, int rows, int cols, String format) 
		throws IOException
	{
		FrameBlock ret = null;
	
		//sanity check input format
		if(!(FileFormat.TEXT.toString().equals(format)
			||FileFormat.MM.toString().equals(format)
			||FileFormat.CSV.toString().equals(format))) {
			throw new IOException("Invalid input format (expected: csv, text or mm): "+format);
		}
		
		setLocalConfigs();
		
		try {
			//read input frame
			FileFormat fmt = FileFormat.safeValueOf(format);
			FrameReader reader = FrameReaderFactory.createFrameReader(fmt);
			ret = reader.readFrameFromInputStream(input, rows, cols);
		}
		catch(DMLRuntimeException rex) {
			throw new IOException(rex);
		}
		
		return ret;
	}
	
	////////////////////////////////////////////
	// Read transform meta data
	////////////////////////////////////////////
	
	/**
	 * Reads transform meta data from an HDFS file path and converts it into an in-memory
	 * FrameBlock object. The column names in the meta data file 'column.names' is processed
	 * with default separator ','.
	 * 
	 * @param spec      transform specification as json string
	 * @param metapath  hdfs file path to meta data directory
	 * @return FrameBlock object representing transform metadata
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock readTransformMetaDataFromFile(String spec, String metapath) throws IOException {
		return readTransformMetaDataFromFile(spec, metapath, TfUtils.TXMTD_SEP);
	}
	
	/**
	 * Reads transform meta data from an HDFS file path and converts it into an in-memory
	 * FrameBlock object.
	 * 
	 * @param spec      transform specification as json string
	 * @param metapath  hdfs file path to meta data directory
	 * @param colDelim  separator for processing column names in the meta data file 'column.names'
	 * @return FrameBlock object representing transform metadata
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock readTransformMetaDataFromFile(String spec, String metapath, String colDelim) throws IOException {
		return TfMetaUtils.readTransformMetaDataFromFile(spec, metapath, colDelim);
	}
	
	/**
	 * Reads transform meta data from the class path and converts it into an in-memory
	 * FrameBlock object. The column names in the meta data file 'column.names' is processed
	 * with default separator ','.
	 * 
	 * @param metapath  resource path to meta data directory
	 * @return FrameBlock object representing transform metadata
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock readTransformMetaDataFromPath(String metapath) throws IOException {
		return readTransformMetaDataFromPath(null, metapath, TfUtils.TXMTD_SEP);
	}
	
	/**
	 * Reads transform meta data from the class path and converts it into an in-memory
	 * FrameBlock object. The column names in the meta data file 'column.names' is processed
	 * with default separator ','.
	 * 
	 * @param spec      transform specification as json string
	 * @param metapath  resource path to meta data directory
	 * @return FrameBlock object representing transform metadata
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock readTransformMetaDataFromPath(String spec, String metapath) throws IOException {
		return readTransformMetaDataFromPath(spec, metapath, TfUtils.TXMTD_SEP);
	}
	
	/**
	 * Reads transform meta data from the class path and converts it into an in-memory
	 * FrameBlock object.
	 * 
	 * @param spec      transform specification as json string
	 * @param metapath  resource path to meta data directory
	 * @param colDelim  separator for processing column names in the meta data file 'column.names'
	 * @return FrameBlock object representing transform metadata
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock readTransformMetaDataFromPath(String spec, String metapath, String colDelim) throws IOException {
		return TfMetaUtils.readTransformMetaDataFromPath(spec, metapath, colDelim);
	}
	
	private void setLocalConfigs() {
		//set thread-local configurations for compilation and read
		ConfigurationManager.setLocalConfig(_dmlconf);
		ConfigurationManager.setLocalConfig(_cconf);
	}
}
