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

package org.apache.sysml.parser.antlr4;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.api.DMLScript;

import java.util.Stack;

public class DmlSyntacticErrorListener {
	
	
	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());
	
	public static class CustomDmlErrorListener extends BaseErrorListener {
		
		private boolean atleastOneError = false;
		private Stack<String> currentFileName = new Stack<String>();
		
		public void pushCurrentFileName(String currentFilePath) {
			currentFileName.push(currentFilePath);
		}
		
		public String peekFileName() {
			return currentFileName.peek();
		}
		
		public String popFileName() {
			return currentFileName.pop();
		}
		
		public void validationError(int line, int charPositionInLine, String msg) {
			try {
				setAtleastOneError(true);
				// Print error messages with file name
				if(currentFileName == null || currentFileName.empty()) {
					LOG.error("line "+line+":"+charPositionInLine+" "+msg);
				}
				else {
					String fileName = currentFileName.peek();
					LOG.error(fileName + " line "+line+":"+charPositionInLine+" "+msg);
				}
			}
			catch(Exception e1) {
				LOG.error("ERROR: while customizing error message:" + e1);
			}
		}
		
		public void validationWarning(int line, int charPositionInLine, String msg) {
			try {
				//atleastOneError = true; ---> not an error, just warning
				// Print error messages with file name
				if(currentFileName == null || currentFileName.empty())
					LOG.warn("line "+line+":"+charPositionInLine+" "+msg);
				else {
					String fileName = currentFileName.peek();
					LOG.warn(fileName + " line "+line+":"+charPositionInLine+" "+msg);
				}
			}
			catch(Exception e1) {
				LOG.warn("ERROR: while customizing error message:" + e1);
			}
		}
		
		@Override
		public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
				int line, int charPositionInLine,
				String msg, RecognitionException e)
		{	
			try {
				setAtleastOneError(true);
				// Print error messages with file name
				if(currentFileName == null || currentFileName.empty())
					LOG.error("line "+line+":"+charPositionInLine+" "+msg);
				else {
					String fileName = currentFileName.peek();
					LOG.error(fileName + " line "+line+":"+charPositionInLine+" "+msg);
				}
			}
			catch(Exception e1) {
				LOG.error("ERROR: while customizing error message:" + e1);
			}
		}

		public boolean isAtleastOneError() {
			return atleastOneError;
		}

		public void setAtleastOneError(boolean atleastOneError) {
			this.atleastOneError = atleastOneError;
		}
	}
}
