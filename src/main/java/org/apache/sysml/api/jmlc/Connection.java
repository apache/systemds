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

package org.apache.sysml.api.jmlc;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.api.mlcontext.ScriptType;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteRemovePersistentReadWrite;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.ParserFactory;
import org.apache.sysml.parser.ParserWrapper;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.io.MatrixReader;
import org.apache.sysml.runtime.io.MatrixReaderFactory;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.wink.json4j.JSONObject;

/**
 * Interaction with SystemML using the JMLC (Java Machine Learning Connector) API is initiated with
 * a {@link Connection} object. The JMLC API is patterned
 * after JDBC. A DML script is precompiled by calling
 * the {@link #prepareScript(String, String[], String[], boolean)}
 * method or the {@link #prepareScript(String, Map, String[], String[], boolean)}
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
 * 
 * <p>
 * For examples, please see the following:
 * <ul>
 *   <li>JMLC JUnit test cases (org.apache.sysml.test.integration.functions.jmlc)</li>
 *   <li><a target="_blank" href="http://apache.github.io/systemml/jmlc.html">JMLC section
 *   of SystemML online documentation</a></li>
 * </ul>
 */
public class Connection implements Closeable
{		
	private DMLConfig _dmlconf = null;

	/**
	 * Connection constructor, the starting point for any other JMLC API calls.
	 * 
	 */
	public Connection()
	{
		DMLScript.rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
		
		//setup basic parameters for embedded execution
		//(parser, compiler, and runtime parameters)
		CompilerConfig cconf = new CompilerConfig();
		cconf.set(ConfigType.IGNORE_UNSPECIFIED_ARGS, true);
		cconf.set(ConfigType.IGNORE_READ_WRITE_METADATA, true);
		cconf.set(ConfigType.REJECT_READ_WRITE_UNKNOWNS, false);
		cconf.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, false);
		cconf.set(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS, false);
		cconf.set(ConfigType.PARALLEL_CP_READ_BINARYFORMATS, false);
		cconf.set(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS, false);
		cconf.set(ConfigType.PARALLEL_CP_MATRIX_OPERATIONS, false);
		cconf.set(ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR, false);
		cconf.set(ConfigType.ALLOW_DYN_RECOMPILATION, false);
		cconf.set(ConfigType.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS, false);
		cconf.set(ConfigType.ALLOW_CSE_PERSISTENT_READS, false);
		ConfigurationManager.setLocalConfig(cconf);
		
		//disable caching globally 
		CacheableData.disableCaching();
		
		//create thread-local default configuration
		_dmlconf = new DMLConfig();
		ConfigurationManager.setLocalConfig(_dmlconf);
	}
	
	/**
	 * Connection constructor, the starting point for any other JMLC API calls.
	 * This variant allows to enable a set of boolean compiler configurations.
	 * 
	 * @param configs one or many boolean compiler configurations to enable.
	 */
	public Connection(CompilerConfig.ConfigType... configs) {
		//basic constructor, which also constructs the compiler config
		this();
		
		//set optional compiler configurations in current config
		CompilerConfig cconf = ConfigurationManager.getCompilerConfig();
		for( ConfigType configType : configs )
			cconf.set(configType, true);
		ConfigurationManager.setLocalConfig(cconf);
	}
	
	/**
	 * Prepares (precompiles) a script and registers input and output variables.
	 * 
	 * @param script string representing the DML or PyDML script
	 * @param inputs string array of input variables to register
	 * @param outputs string array of output variables to register
	 * @param parsePyDML {@code true} if PyDML, {@code false} if DML
	 * @return PreparedScript object representing the precompiled script
	 * @throws DMLException if DMLException occurs
	 */
	public PreparedScript prepareScript( String script, String[] inputs, String[] outputs, boolean parsePyDML) 
		throws DMLException 
	{
		return prepareScript(script, new HashMap<String,String>(), inputs, outputs, parsePyDML);
	}
	
	/**
	 * Prepares (precompiles) a script, sets input parameter values, and registers input and output variables.
	 * 
	 * @param script string representing the DML or PyDML script
	 * @param args map of input parameters ($) and their values
	 * @param inputs string array of input variables to register
	 * @param outputs string array of output variables to register
	 * @param parsePyDML {@code true} if PyDML, {@code false} if DML
	 * @return PreparedScript object representing the precompiled script
	 * @throws DMLException if DMLException occurs
	 */
	public PreparedScript prepareScript( String script, Map<String, String> args, String[] inputs, String[] outputs, boolean parsePyDML) 
		throws DMLException 
	{
		DMLScript.SCRIPT_TYPE = parsePyDML ? ScriptType.PYDML : ScriptType.DML;

		//prepare arguments
		
		//simplified compilation chain
		Program rtprog = null;
		try
		{
			//parsing
			ParserWrapper parser = ParserFactory.createParser(parsePyDML ? ScriptType.PYDML : ScriptType.DML);
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
			rtprog = prog.getRuntimeProgram(_dmlconf);
			
			//final cleanup runtime prog
			JMLCUtils.cleanupRuntimeProgram(rtprog, outputs);
			
			//activate thread-local proxy for dynamic recompilation
			if( ConfigurationManager.isDynamicRecompilation() )
				JMLCProxy.setActive(outputs);
		}
		catch(ParseException pe) {
			// don't chain ParseException (for cleaner error output)
			throw pe;
		}
		catch(Exception ex) {
			throw new DMLException(ex);
		}
			
		//return newly create precompiled script 
		return new PreparedScript(rtprog, inputs, outputs);
	}
	
	/**
	 * Close connection to SystemML, which clears the
	 * thread-local DML and compiler configurations.
	 */
	@Override
	public void close() {
		//clear thread-local dml / compiler configs
		ConfigurationManager.clearLocalConfigs();
		if( ConfigurationManager.isDynamicRecompilation() )
			JMLCProxy.setActive(null);
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
			if(    fname.startsWith("hdfs:") 
				|| fname.startsWith("gpfs:") ) 
			{ 
				Path scriptPath = new Path(fname);
				FileSystem fs = IOUtilFunctions.getFileSystem(scriptPath);
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
			JSONObject jmtd = new DataExpression().readMetadataFile(fnamemtd, false);
			
			//parse json meta data 
			long rows = jmtd.getLong(DataExpression.READROWPARAM);
			long cols = jmtd.getLong(DataExpression.READCOLPARAM);
			int brlen = jmtd.containsKey(DataExpression.ROWBLOCKCOUNTPARAM)?
					jmtd.getInt(DataExpression.ROWBLOCKCOUNTPARAM) : -1;
			int bclen = jmtd.containsKey(DataExpression.COLUMNBLOCKCOUNTPARAM)?
					jmtd.getInt(DataExpression.COLUMNBLOCKCOUNTPARAM) : -1;
			long nnz = jmtd.containsKey(DataExpression.READNUMNONZEROPARAM)?
					jmtd.getLong(DataExpression.READNUMNONZEROPARAM) : -1;
			String format = jmtd.getString(DataExpression.FORMAT_TYPE);
			InputInfo iinfo = InputInfo.stringExternalToInputInfo(format);			
		
			//read matrix file
			return readDoubleMatrix(fname, iinfo, rows, cols, brlen, bclen, nnz);
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
	 * @param iinfo InputInfo object
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @param brlen number of rows per block
	 * @param bclen number of columns per block
	 * @param nnz number of non-zero values, -1 indicates unknown
	 * @return matrix as a two-dimensional double array
	 * @throws IOException if IOException occurs
	 */
	public double[][] readDoubleMatrix(String fname, InputInfo iinfo, long rows, long cols, int brlen, int bclen, long nnz) 
		throws IOException
	{
		try {
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(iinfo);
			MatrixBlock mb = reader.readMatrixFromHDFS(fname, rows, cols, brlen, bclen, nnz);
			return DataConverter.convertToDoubleMatrix(mb);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * Converts an input string representation of a matrix in csv or textcell format
	 * into a dense double array. The meta data string is the SystemML generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input string matrix in csv or textcell format
	 * @param meta string representing SystemML matrix metadata in JSON format
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
		return convertToDoubleMatrix(IOUtilFunctions.toInputStream(input), rows, cols);
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
		return convertToDoubleMatrix(input, rows, cols, DataExpression.FORMAT_TYPE_VALUE_TEXT);
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
	 * into a matrix block. The meta data string is the SystemML generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input string matrix in csv or textcell format
	 * @param meta string representing SystemML matrix metadata in JSON format
	 * @return matrix as a matrix block
	 * @throws IOException if IOException occurs
	 */
	public MatrixBlock convertToMatrix(String input, String meta) throws IOException {
		return convertToMatrix(IOUtilFunctions.toInputStream(input), meta);
	}
	
	/**
	 * Converts an input stream of a string matrix in csv or textcell format
	 * into a matrix block. The meta data string is the SystemML generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input InputStream to a string matrix in csv or textcell format
	 * @param meta string representing SystemML matrix metadata in JSON format
	 * @return matrix as a matrix block
	 * @throws IOException if IOException occurs
	 */
	public MatrixBlock convertToMatrix(InputStream input, String meta) 
		throws IOException
	{
		try {
			//parse json meta data 
			JSONObject jmtd = new JSONObject(meta);
			int rows = jmtd.getInt(DataExpression.READROWPARAM);
			int cols = jmtd.getInt(DataExpression.READCOLPARAM);
			String format = jmtd.getString(DataExpression.FORMAT_TYPE);
			
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
		return convertToMatrix(IOUtilFunctions.toInputStream(input), rows, cols);
	}
	
	/**
	 * Converts an input stream of a string matrix in textcell format
	 * into a matrix block. 
	 * 
	 * @param input InputStream to a string matrix in textcell format
	 * @param rows number of rows in the matrix
	 * @param cols number of columns in the matrix
	 * @return matrix as a matrix block
	 * @throws IOException if IOException occurs
	 */
	public MatrixBlock convertToMatrix(InputStream input, int rows, int cols) throws IOException {
		return convertToMatrix(input, rows, cols, DataExpression.FORMAT_TYPE_VALUE_TEXT);
	}
	
	/**
	 * Converts an input stream of a string matrix in csv or textcell format
	 * into a matrix block. 
	 * 
	 * @param input InputStream to a string matrix in csv or textcell format
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
		if(!(DataExpression.FORMAT_TYPE_VALUE_TEXT.equals(format)
			||DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET.equals(format)
			||DataExpression.FORMAT_TYPE_VALUE_CSV.equals(format)) ) {
			throw new IOException("Invalid input format (expected: csv, text or mm): "+format);
		}
		
		try {
			//read input matrix
			InputInfo iinfo = DataExpression.FORMAT_TYPE_VALUE_CSV.equals(format) ? 
					InputInfo.CSVInputInfo : InputInfo.TextCellInputInfo;
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(iinfo);
			int blksz = ConfigurationManager.getBlocksize();
			ret = reader.readMatrixFromInputStream(input, 
					rows, cols, blksz, blksz, (long)rows*cols);
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
			JSONObject jmtd = new DataExpression().readMetadataFile(fnamemtd, false);
			
			//parse json meta data 
			long rows = jmtd.getLong(DataExpression.READROWPARAM);
			long cols = jmtd.getLong(DataExpression.READCOLPARAM);
			String format = jmtd.getString(DataExpression.FORMAT_TYPE);
			InputInfo iinfo = InputInfo.stringExternalToInputInfo(format);			
		
			//read frame file
			return readStringFrame(fname, iinfo, rows, cols);
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
	 * @param iinfo InputInfo object
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @return frame as a two-dimensional string array
	 * @throws IOException if IOException occurs
	 */
	public String[][] readStringFrame(String fname, InputInfo iinfo, long rows, long cols) 
		throws IOException
	{
		try {
			FrameReader reader = FrameReaderFactory.createFrameReader(iinfo);
			FrameBlock mb = reader.readFrameFromHDFS(fname, rows, cols);
			return DataConverter.convertToStringFrame(mb);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * Converts an input string representation of a frame in csv or textcell format
	 * into a dense string array. The meta data string is the SystemML generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input string frame in csv or textcell format
	 * @param meta string representing SystemML frame metadata in JSON format
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
		return convertToStringFrame(IOUtilFunctions.toInputStream(input), rows, cols);
	}
	
	/**
	 * Converts an input stream of a string frame in textcell format
	 * into a dense string array. 
	 * 
	 * @param input InputStream to a string frame in textcell format
	 * @param rows number of rows in the frame
	 * @param cols number of columns in the frame
	 * @return frame as a two-dimensional string array
	 * @throws IOException if IOException occurs
	 */
	public String[][] convertToStringFrame(InputStream input, int rows, int cols) throws IOException {
		return convertToStringFrame(input, rows, cols, DataExpression.FORMAT_TYPE_VALUE_TEXT);
	}
	
	/**
	 * Converts an input stream of a string frame in csv or textcell format
	 * into a dense string array. 
	 * 
	 * @param input InputStream to a string frame in csv or textcell format
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
	 * into a frame block. The meta data string is the SystemML generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input string frame in csv or textcell format
	 * @param meta string representing SystemML frame metadata in JSON format
	 * @return frame as a frame block
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock convertToFrame(String input, String meta) throws IOException {
		return convertToFrame(IOUtilFunctions.toInputStream(input), meta);
	}
	
	/**
	 * Converts an input stream of a string frame in csv or textcell format
	 * into a frame block. The meta data string is the SystemML generated
	 * .mtd file including the number of rows and columns.
	 * 
	 * @param input InputStream to a string frame in csv or textcell format
	 * @param meta string representing SystemML frame metadata in JSON format
	 * @return frame as a frame block
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock convertToFrame(InputStream input, String meta) 
		throws IOException
	{
		try {
			//parse json meta data 
			JSONObject jmtd = new JSONObject(meta);
			int rows = jmtd.getInt(DataExpression.READROWPARAM);
			int cols = jmtd.getInt(DataExpression.READCOLPARAM);
			String format = jmtd.getString(DataExpression.FORMAT_TYPE);
			
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
		return convertToFrame(IOUtilFunctions.toInputStream(input), rows, cols);
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
		return convertToFrame(input, rows, cols, DataExpression.FORMAT_TYPE_VALUE_TEXT);
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
		if(!(DataExpression.FORMAT_TYPE_VALUE_TEXT.equals(format)
			||DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET.equals(format)
			||DataExpression.FORMAT_TYPE_VALUE_CSV.equals(format))) {
			throw new IOException("Invalid input format (expected: csv, text or mm): "+format);
		}
		
		try {
			//read input frame
			InputInfo iinfo = DataExpression.FORMAT_TYPE_VALUE_CSV.equals(format) ? 
					InputInfo.CSVInputInfo : InputInfo.TextCellInputInfo;
			FrameReader reader = FrameReaderFactory.createFrameReader(iinfo);
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
	 * @param metapath  hdfs file path to meta data directory
	 * @return FrameBlock object representing transform metadata
	 * @throws IOException if IOException occurs
	 */
	public FrameBlock readTransformMetaDataFromFile(String metapath) throws IOException {
		return readTransformMetaDataFromFile(null, metapath, TfUtils.TXMTD_SEP);
	}
	
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
}
