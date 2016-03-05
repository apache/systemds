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

package org.apache.sysml.parser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.dml.DMLParserWrapper;
import org.apache.sysml.parser.pydml.PyDMLParserWrapper;
import org.apache.sysml.runtime.util.LocalFileUtils;

/**
 * Base class for all dml parsers in order to make the various compilation chains
 * independent of the used parser.
 */
public abstract class AParserWrapper 
{
	//global parser configuration dml/pydml:
	//1) skip errors on unspecified args (modified by mlcontext / jmlc)
	public static boolean IGNORE_UNSPECIFIED_ARGS = false; 
	
	
	public abstract DMLProgram parse(String fileName, String dmlScript, HashMap<String, String> argVals) throws ParseException;

	
	/**
	 * Factory method for creating instances of AParserWrapper, for
	 * simplificy fused with the abstract class.
	 * 
	 * @param pydml true if a PyDML parser is needed
	 * @return
	 */
	public static AParserWrapper createParser(boolean pydml)
	{
		AParserWrapper ret = null;
		
		//create the parser instance
		if( pydml )
			ret = new PyDMLParserWrapper();
		else
			ret = new DMLParserWrapper();
		
		return ret;
	}
	
	/**
	 * Custom wrapper to convert statement into statement blocks. Called by doParse and in DmlSyntacticValidator for for, parfor, while, ...
	 * @param current a statement
	 * @return corresponding statement block
	 */
	public static StatementBlock getStatementBlock(Statement current) {
		StatementBlock blk = null;
		if(current instanceof ParForStatement) {
			blk = new ParForStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof ForStatement) {
			blk = new ForStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof IfStatement) {
			blk = new IfStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof WhileStatement) {
			blk = new WhileStatementBlock();
			blk.addStatement(current);
		}
		else {
			// This includes ImportStatement
			blk = new StatementBlock();
			blk.addStatement(current);
		}
		return blk;
	}
	
	
	public static String readDMLScript( String script, Log LOG) 
			throws IOException, LanguageException
	{
		String dmlScriptStr = null;
		
		//read DML script from file
		if(script == null)
			throw new LanguageException("DML script path was not specified!");
		
		StringBuilder sb = new StringBuilder();
		BufferedReader in = null;
		try 
		{
			//read from hdfs or gpfs file system
			if(    script.startsWith("hdfs:") 
				|| script.startsWith("gpfs:") ) 
			{ 
				if( !LocalFileUtils.validateExternalFilename(script, true) )
					throw new LanguageException("Invalid (non-trustworthy) hdfs filename.");
				FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
				Path scriptPath = new Path(script);
				in = new BufferedReader(new InputStreamReader(fs.open(scriptPath)));
			}
			// from local file system
			else 
			{ 
				if( !LocalFileUtils.validateExternalFilename(script, false) )
					throw new LanguageException("Invalid (non-trustworthy) local filename.");
				in = new BufferedReader(new FileReader(script));
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
			LOG.error("Failed to read the script from the file system", ex);
			throw ex;
		}
		finally 
		{
			if( in != null )
				in.close();
		}
		
		dmlScriptStr = sb.toString();
		
		return dmlScriptStr;
	}
}
