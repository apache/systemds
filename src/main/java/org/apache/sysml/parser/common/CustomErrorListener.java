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

package org.apache.sysml.parser.common;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;

public class CustomErrorListener extends BaseErrorListener {

	private static final Log log = LogFactory.getLog(DMLScript.class.getName());

	private boolean atleastOneError = false;
	private String currentFileName = null;

	/**
	 * List of parse issues.
	 */
	private List<ParseIssue> parseIssues = new ArrayList<ParseIssue>();

	public void setCurrentFileName(String currentFilePath) {
		currentFileName = currentFilePath;
	}

	public String getCurrentFileName() {
		return currentFileName;
	}

	public void unsetCurrentFileName() {
		currentFileName = null;
	}

	/**
	 * Validation error occurred. Add the error to the list of parse issues.
	 * 
	 * @param line
	 *            Line number where error was detected
	 * @param charPositionInLine
	 *            Character position where error was detected
	 * @param msg
	 *            Message describing the nature of the validation error
	 */
	public void validationError(int line, int charPositionInLine, String msg) {
		parseIssues
				.add(new ParseIssue(line, charPositionInLine, msg, currentFileName, ParseIssueType.VALIDATION_ERROR));
		try {
			setAtleastOneError(true);
			// Print error messages with file name
			if (currentFileName == null) {
				log.error("line " + line + ":" + charPositionInLine + " " + msg);
			} else {
				String fileName = currentFileName;
				log.error(fileName + " line " + line + ":" + charPositionInLine + " " + msg);
			}
		} catch (Exception e1) {
			log.error("ERROR: while customizing error message:" + e1);
		}
	}

	/**
	 * Validation warning occurred. Add the warning to the list of parse issues.
	 * 
	 * @param line
	 *            Line number where warning was detected
	 * @param charPositionInLine
	 *            Character position where warning was detected
	 * @param msg
	 *            Message describing the nature of the validation warning
	 */
	public void validationWarning(int line, int charPositionInLine, String msg) {
		parseIssues.add(new ParseIssue(line, charPositionInLine, msg, currentFileName,
				ParseIssueType.VALIDATION_WARNING));
		try {
			// atleastOneError = true; ---> not an error, just warning
			// Print error messages with file name
			if (currentFileName == null)
				log.warn("line " + line + ":" + charPositionInLine + " " + msg);
			else {
				String fileName = currentFileName;
				log.warn(fileName + " line " + line + ":" + charPositionInLine + " " + msg);
			}
		} catch (Exception e1) {
			log.warn("ERROR: while customizing error message:" + e1);
		}
	}

	/**
	 * Syntax error occurred. Add the error to the list of parse issues.
	 */
	@Override
	public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol, int line, int charPositionInLine,
			String msg, RecognitionException e) {
		parseIssues.add(new ParseIssue(line, charPositionInLine, msg, currentFileName, ParseIssueType.SYNTAX_ERROR));
		try {
			setAtleastOneError(true);
			// Print error messages with file name
			if (currentFileName == null)
				log.error("line " + line + ":" + charPositionInLine + " " + msg);
			else {
				String fileName = currentFileName;
				log.error(fileName + " line " + line + ":" + charPositionInLine + " " + msg);
			}
		} catch (Exception e1) {
			log.error("ERROR: while customizing error message:" + e1);
		}
	}

	public boolean isAtleastOneError() {
		return atleastOneError;
	}

	public void setAtleastOneError(boolean atleastOneError) {
		this.atleastOneError = atleastOneError;
	}

	/**
	 * A parse issue (such as an parse error or a parse warning).
	 *
	 */
	public class ParseIssue implements Comparable<ParseIssue> {
		/**
		 * Line number of the parse issue
		 */
		int line;
		/**
		 * Character position of the parse issue
		 */
		int charPositionInLine;
		/**
		 * Message describing the parse issue
		 */
		String message;
		/**
		 * The filename (if available) of the script containing the parse issue
		 */
		String fileName;

		/**
		 * The type of parse issue.
		 */
		ParseIssueType parseIssueType;

		public ParseIssue(int line, int charPositionInLine, String message) {
			this.line = line;
			this.charPositionInLine = charPositionInLine;
			this.message = message;
		}

		public ParseIssue(int line, int charPositionInLine, String message, String fileName) {
			this.line = line;
			this.charPositionInLine = charPositionInLine;
			this.message = message;
			this.fileName = fileName;
		}

		public ParseIssue(int line, int charPositionInLine, String message, String fileName,
				ParseIssueType parseIssueType) {
			this.line = line;
			this.charPositionInLine = charPositionInLine;
			this.message = message;
			this.fileName = fileName;
			this.parseIssueType = parseIssueType;
		}

		/**
		 * Obtain line number of the parse issue.
		 * 
		 * @return Line number of the parse issue
		 */
		public int getLine() {
			return line;
		}

		public void setLine(int line) {
			this.line = line;
		}

		/**
		 * Obtain character position of the parse issue.
		 * 
		 * @return Character position of the parse issue
		 */
		public int getCharPositionInLine() {
			return charPositionInLine;
		}

		public void setCharPositionInLine(int charPositionInLine) {
			this.charPositionInLine = charPositionInLine;
		}

		/**
		 * Obtain the message describing the parse issue.
		 * 
		 * @return Message describing the parse issue
		 */
		public String getMessage() {
			return message;
		}

		public void setMessage(String message) {
			this.message = message;
		}

		/**
		 * Obtain the filename of the script containing the parse issue, if
		 * available.
		 * 
		 * @return The filename of the script contain the parse issue (if
		 *         available)
		 */
		public String getFileName() {
			return fileName;
		}

		public void setFileName(String fileName) {
			this.fileName = fileName;
		}

		/**
		 * Obtain the type of the parse issue.
		 * 
		 * @return The type of the parse issue.
		 */
		public ParseIssueType getParseIssueType() {
			return parseIssueType;
		}

		public void setParseIssueType(ParseIssueType parseIssueType) {
			this.parseIssueType = parseIssueType;
		}

		/**
		 * Order by parse issues primarily by line number, and secondarily by
		 * character position.
		 */
		@Override
		public int compareTo(ParseIssue that) {
			if (this.line == that.line) {
				return this.charPositionInLine - that.charPositionInLine;
			} else {
				return this.line - that.line;
			}
		}
	}

	/**
	 * Parse issues can be syntax errors, validation errors, and validation
	 * warnings. Include a string representation of each enum value for display
	 * to the user.
	 */
	public enum ParseIssueType {
		SYNTAX_ERROR("Syntax error"), VALIDATION_ERROR("Validation error"), VALIDATION_WARNING("Validation warning");

		ParseIssueType(String text) {
			this.text = text;
		}

		private final String text;

		/**
		 * Obtain the user-friendly string representation of the enum value.
		 * 
		 * @return User-friendly string representation of the enum value.
		 */
		public String getText() {
			return text;
		}

	}

	/**
	 * Obtain the list of parse issues.
	 * 
	 * @return The list of parse issues.
	 */
	public List<ParseIssue> getParseIssues() {
		Collections.sort(parseIssues);
		return parseIssues;
	}

	/**
	 * Set the list of parse issues.
	 * 
	 * @param parseIssues
	 *            The list of parse issues.
	 */
	public void setParseIssues(List<ParseIssue> parseIssues) {
		this.parseIssues = parseIssues;
	}

	/**
	 * Clear the list of parse issues.
	 * 
	 */
	public void clearParseIssues() {
		parseIssues.clear();
	}

}
