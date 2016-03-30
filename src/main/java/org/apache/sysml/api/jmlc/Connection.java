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
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteRemovePersistentReadWrite;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.AParserWrapper;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.io.MatrixReaderFactory;
import org.apache.sysml.runtime.io.ReaderTextCell;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.transform.TransformationAgent;
import org.apache.sysml.runtime.transform.TransformationAgent.TX_METHOD;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONObject;

/**
 * JMLC (Java Machine Learning Connector) API:
 * 
 * NOTES: 
 *   * Currently fused API and implementation in order to reduce complexity. 
 *   * See JUnit test cases (org.apache.sysml.test.integration.functions.jmlc) for examples. 
 */
public class Connection 
{	
	private static final Log LOG = LogFactory.getLog(Connection.class.getName());
	
	private DMLConfig _dmlconf = null;
	private CompilerConfig _cconf = null;
	
	/**
	 * Connection constructor, starting point for any other JMLC API calls.
	 * 
	 */
	public Connection()
	{
		DMLScript.rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
		
		//setup basic parameters for embedded execution
		//(parser, compiler, and runtime parameters)
		_cconf = new CompilerConfig();
		_cconf.set(ConfigType.IGNORE_UNSPECIFIED_ARGS, true);
		_cconf.set(ConfigType.IGNORE_READ_WRITE_METADATA, true);
		_cconf.set(ConfigType.REJECT_READ_WRITE_UNKNOWNS, false);
		_cconf.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, false);
		_cconf.set(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS, false);
		_cconf.set(ConfigType.PARALLEL_CP_READ_BINARYFORMATS, false);
		_cconf.set(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS, false);
		_cconf.set(ConfigType.PARALLEL_CP_MATRIX_OPERATIONS, false);
		_cconf.set(ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR, false);
		_cconf.set(ConfigType.ALLOW_DYN_RECOMPILATION, false);
		_cconf.set(ConfigType.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS, false);
		_cconf.set(ConfigType.ALLOW_CSE_PERSISTENT_READS, false);
		ConfigurationManager.setLocalConfig(_cconf);
		
		//disable caching globally 
		CacheableData.disableCaching();
		
		//create thread-local default configuration
		_dmlconf = new DMLConfig();
		ConfigurationManager.setLocalConfig(_dmlconf);
	}
	
	/**
	 * 
	 * @param script
	 * @param inputs
	 * @param outputs
	 * @return
	 * @throws DMLException
	 */
	public PreparedScript prepareScript( String script, String[] inputs, String[] outputs, boolean parsePyDML) 
		throws DMLException 
	{
		return prepareScript(script, new HashMap<String,String>(), inputs, outputs, parsePyDML);
	}
	
	/**
	 * 
	 * @param script
	 * @param args
	 * @param inputs
	 * @param outputs
	 * @return
	 * @throws DMLException
	 */
	public PreparedScript prepareScript( String script, HashMap<String, String> args, String[] inputs, String[] outputs, boolean parsePyDML) 
		throws DMLException 
	{
		//prepare arguments
		
		//simplified compilation chain
		Program rtprog = null;
		try
		{
			//parsing
			AParserWrapper parser = AParserWrapper.createParser(parsePyDML);
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
			
			//System.out.println(Explain.explain(rtprog));
		}
		catch(Exception ex)
		{
			throw new DMLException(ex);
		}
			
		//return newly create precompiled script 
		return new PreparedScript(rtprog, inputs, outputs);
	}
	
	/**
	 * 
	 */
	public void close() {
		//clear thread-local dml / compiler configs
		ConfigurationManager.clearLocalConfigs();
	}
	
	/**
	 * 
	 * @param fname
	 * @return
	 * @throws IOException
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
				FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
				Path scriptPath = new Path(fname);
				in = new BufferedReader(new InputStreamReader(fs.open(scriptPath)));
			}
			// from local file system
			else 
			{ 
				in = new BufferedReader(new FileReader(fname));
			}
			
			//core script reading
			String tmp = null;
			while ((tmp = in.readLine()) != null)
			{
				sb.append( tmp );
				sb.append( "\n" );
			}
		}
		catch (IOException ex)
		{
			throw ex;
		}
		finally 
		{
			if( in != null )
			 	in.close();
		}
		
		return sb.toString();
	}
	
	/**
	 * Converts an input string representation of a matrix in textcell format
	 * into a dense double array. The number of rows and columns need to be 
	 * specified because textcell only represents non-zero values and hence
	 * does not define the dimensions in the general case.
	 * 
	 * @param input  a string representation of an input matrix, 
	 *              in format textcell (rowindex colindex value)
	 * @param rows number of rows
	 * @param cols number of columns 
	 * @return
	 * @throws IOException 
	 */
	public double[][] convertToDoubleMatrix(String input, int rows, int cols) 
		throws IOException
	{
		double[][] ret = null;
		
		try 
		{
			//read input matrix
			InputStream is = new ByteArrayInputStream(input.getBytes("UTF-8"));
			ReaderTextCell reader = (ReaderTextCell)MatrixReaderFactory.createMatrixReader(InputInfo.TextCellInputInfo);
			MatrixBlock mb = reader.readMatrixFromInputStream(is, rows, cols, ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize(), (long)rows*cols);
		
			//convert to double array
			ret = DataConverter.convertToDoubleMatrix( mb );
		}
		catch(DMLRuntimeException rex) 
		{
			throw new IOException( rex );
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param spec
	 * @param metapath
	 * @return
	 * @throws IOException 
	 */
	public FrameBlock readTransformMetaData(String spec, String metapath) 
		throws IOException 
	{
		//read column types (for sanity check column names)
		String coltypesStr = MapReduceTool.readStringFromHDFSFile(metapath+File.separator+"coltypes.csv");
		List<String> coltypes = Arrays.asList(IOUtilFunctions.split(coltypesStr.trim(), ","));
		
		//read column names
		String colnamesStr = MapReduceTool.readStringFromHDFSFile(metapath+File.separator+"column.names");
		List<String> colnames = Arrays.asList(IOUtilFunctions.split(colnamesStr.trim(), ","));
		if( coltypes.size() != colnames.size() ) {
			LOG.warn("Number of columns names: "+colnames.size()+" (expected: "+coltypes.size()+").");
			LOG.warn("--Sample column names: "+(!colnames.isEmpty()?colnames.get(0):"null"));
		}
		
		//read meta data (currently only recode supported, without parsing spec)
		HashMap<String,String> meta = new HashMap<String,String>();
		int rows = 0;
		for( int j=0; j<colnames.size(); j++ ) {
			String colName = colnames.get(j);
			String name = metapath+File.separator+"Recode"+File.separator+colName;
			if( MapReduceTool.existsFileOnHDFS(name+".map") ) {
				meta.put(colName, MapReduceTool.readStringFromHDFSFile(name+".map"));
				String ndistinct = MapReduceTool.readStringFromHDFSFile(name+".ndistinct");
				rows = Math.max(rows, Integer.parseInt(ndistinct));
			}
			else if( coltypes.get(j).equals("2") ) {
				LOG.warn("Recode map for column '"+colName+"' does not exist.");
			}
		}
		
		//create frame block from in-memory strings
		return readTransformMetaData(spec, rows, colnames, meta);
	}
	
	/**
	 * 
	 * @param spec
	 * @param meta
	 * @return
	 * @throws IOException 
	 */
	public FrameBlock readTransformMetaData(String spec, int rows, List<String> colnames, HashMap<String,String> meta) 
		throws IOException 
	{
		//create frame block w/ pure string schema
		List<ValueType> schema = Collections.nCopies(colnames.size(), ValueType.STRING);
		FrameBlock ret = new FrameBlock(schema, colnames);
		ret.ensureAllocatedColumns(rows);
		
		try
		{
			ArrayList<Integer> specRecodeIDs = new ArrayList<Integer>();
			
			//parse json transform specification
			JSONObject jSpec = new JSONObject(spec);
			if ( jSpec.containsKey(TX_METHOD.RECODE.toString()))  {
				JSONArray attrs = null; //TODO simplify once json spec consolidated
				if( jSpec.get(TX_METHOD.RECODE.toString()) instanceof JSONObject ) {
					JSONObject obj = (JSONObject) jSpec.get(TX_METHOD.RECODE.toString());
					attrs = (JSONArray) obj.get(TransformationAgent.JSON_ATTRS);
				}
				else
					attrs = (JSONArray)jSpec.get(TX_METHOD.RECODE.toString());				
				for(int j=0; j<attrs.length(); j++) 
					specRecodeIDs.add(UtilFunctions.toInt(attrs.get(j)));
			}	
			
			//encode recode maps into frame
			for( Integer colID : specRecodeIDs ) {
				String name = colnames.get(colID-1);
				String map = meta.get(name);
				if( map == null )
					throw new IOException("Recode map for column '"+name+"' (id="+colID+") not existing.");
				
				InputStream is = new ByteArrayInputStream(map.getBytes("UTF-8"));
				BufferedReader br = new BufferedReader(new InputStreamReader(is));
				String line = null; int rpos = 0;
				while( (line = br.readLine()) != null ) {
					String parts[] = IOUtilFunctions.split(line.trim(), ",");
					String pair = parts[0] + Lop.DATATYPE_PREFIX + parts[1]; //sval.code
					ret.set(rpos++, colID-1, pair);
				}
			}
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
		
		return ret;
	}
}
