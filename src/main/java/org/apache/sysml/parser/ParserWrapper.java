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
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.apache.commons.logging.Log;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysml.parser.common.CustomErrorListener.ParseIssue;
import org.apache.sysml.runtime.io.IOUtilFunctions;

/**
 * Base class for all dml parsers in order to make the various compilation chains
 * independent of the used parser.
 */
public abstract class ParserWrapper {
	protected boolean atLeastOneError = false;
	protected boolean atLeastOneWarning = false;
	protected List<ParseIssue> parseIssues;
	
	public abstract DMLProgram parse(String fileName, String dmlScript, Map<String, String> argVals)
		throws ParseException;

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
				LOG.debug("Looking for the following file in HDFS or GPFS: " + script);
				Path scriptPath = new Path(script);
				FileSystem fs = IOUtilFunctions.getFileSystem(scriptPath);
				in = new BufferedReader(new InputStreamReader(fs.open(scriptPath)));
			}
			// from local file system
			else 
			{
				LOG.debug("Looking for the following file in the local file system: " + script);
				if (Files.exists(Paths.get(script)))
					in = new BufferedReader(new FileReader(script));
				else  // check in scripts/ directory for file (useful for tests)
					in = new BufferedReader(new FileReader("scripts/" + script));
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
			String resPath = scriptPathToResourcePath(script);
			LOG.debug("Looking for the following resource from the SystemML jar file: " + resPath);
			InputStream is = ParserWrapper.class.getResourceAsStream(resPath);
			if (is == null) {
				if (resPath.startsWith("/scripts")) {
					LOG.error("Failed to read from the file system ('" + script + "') or SystemML jar file ('" + resPath + "')");
					throw ex;
				} else {
					// for accessing script packages in the scripts directory
					String scriptsResPath = "/scripts" + resPath;
					LOG.debug("Looking for the following resource from the SystemML jar file: " + scriptsResPath);
					is = ParserWrapper.class.getResourceAsStream(scriptsResPath);
					if (is == null) {
						LOG.error("Failed to read from the file system ('" + script + "') or SystemML jar file ('" + resPath + "' or '" + scriptsResPath + "')");
						throw ex;
					}
				}
			}
			String s = IOUtils.toString(is);
			return s;
		}
		finally {
			IOUtilFunctions.closeSilently(in);
		}
		
		dmlScriptStr = sb.toString();
		
		return dmlScriptStr;
	}

	private static String scriptPathToResourcePath(String scriptPath) {
		String resPath = scriptPath;
		if (resPath.startsWith(".")) {
			resPath = resPath.substring(1);
		} else if (resPath.startsWith("\\")) {
			// do nothing
		} else if (!resPath.startsWith("/")) {
			resPath = "/" + resPath;
		}
		resPath = resPath.replace("\\", "/");
		return resPath;
	}

	public boolean isAtLeastOneError() {
		return atLeastOneError;
	}

	public boolean isAtLeastOneWarning() {
		return atLeastOneWarning;
	}

	public List<ParseIssue> getParseIssues() {
		return parseIssues;
	}
}
