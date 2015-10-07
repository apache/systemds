/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.api.jmlc;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.rewrite.ProgramRewriter;
import com.ibm.bi.dml.hops.rewrite.RewriteRemovePersistentReadWrite;
import com.ibm.bi.dml.parser.AParserWrapper;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.io.MatrixReaderFactory;
import com.ibm.bi.dml.runtime.io.ReaderTextCell;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.DataConverter;

/**
 * JMLC (Java Machine Learning Connector) API:
 * 
 * NOTES: 
 *   * Currently fused API and implementation in order to reduce complexity. 
 *   * See SystemTMulticlassSVMScoreTest for an usage example. 
 */
public class Connection 
{
	
	private DMLConfig _conf = null;
	
	/**
	 * Connection constructor, starting point for any other JMLC API calls.
	 * 
	 */
	public Connection()
	{
		//setup basic parameters for embedded execution
		DataExpression.REJECT_READ_UNKNOWN_SIZE = false;
		DMLScript.rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
		OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS = false;
		CacheableData.disableCaching();
		
		//create default configuration
		_conf = new DMLConfig();
		ConfigurationManager.setConfig(_conf);
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
			rtprog = prog.getRuntimeProgram(_conf);
			
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
	public void close()
	{
		
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
			MatrixBlock mb = reader.readMatrixFromInputStream(is, rows, cols, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, (long)rows*cols);
		
			//convert to double array
			ret = DataConverter.convertToDoubleMatrix( mb );
		}
		catch(DMLRuntimeException rex) 
		{
			throw new IOException( rex );
		}
		
		return ret;
	}
	
}
