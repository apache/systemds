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

import java.util.List;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.parser.common.CustomErrorListener.ParseIssue;

/**
 * This exception is thrown when parse issues are encountered.
 * 
 */
public class ParseException extends DMLException 
{
	private static final long serialVersionUID = 9199966053655385928L;

	/**
	 * List of issues that happened during parsing. Typically set by the error
	 * listener.
	 */
	private List<ParseIssue> _parseIssues;

	/**
	 * The DML/PyDML script string. Used to display the original lines where the
	 * parse issues occurred.
	 */
	private String _scriptString;

	/**
	 * Standard exception message. If no list of parse issues exists, this
	 * message can be used to display a parse exception message that doesn't
	 * relate to the list of parse issues.
	 */
	private String _message;

	
	public ParseException() {
		super();
	}

	public ParseException(String message) {
		super(message);
		_message = message;
	}

	public ParseException(String message, Exception e) {
		super(message, e);
		_message = message;
	}

	/**
	 * This constructor takes a list of parse issues that were generated during
	 * script parsing and the original DML/PyDML script String.
	 * 
	 * @param parseIssues
	 *            List of parse issues (syntax errors, validation errors, and
	 *            validation warnings) generated during parsing.
	 * @param scriptString
	 *            The DML/PyDML script String.
	 */
	public ParseException(List<ParseIssue> parseIssues, String scriptString) {
		super();
		_parseIssues = parseIssues;
		_scriptString = scriptString;
	}

	/**
	 * Obtain the list of parse issues that occurred.
	 * 
	 * @return the list of parse issues
	 */
	public List<ParseIssue> getParseIssues() {
		return _parseIssues;
	}

	/**
	 * Set the list of parse issues.
	 * 
	 * @param parseIssues
	 *            the list of parse issues
	 */
	public void setParseIssues(List<ParseIssue> parseIssues) {
		_parseIssues = parseIssues;
	}

	/**
	 * Obtain the original DML/PyDML script string.
	 * 
	 * @return the original DML/PyDML script string
	 */
	public String getScriptString() {
		return _scriptString;
	}

	/**
	 * Set the original DML/PyDML script string.
	 * 
	 * @param scriptString
	 *            the original DML/PyDML script string
	 */
	public void setScriptString(String scriptString) {
		_scriptString = scriptString;
	}

	/**
	 * Does this ParseException contain a list of parse issues?
	 * 
	 * @return <code>true</code> if the list of parse issues exists and is
	 *         greater than 0, <code>false</code> otherwise
	 */
	public boolean hasParseIssues() {
		return (_parseIssues != null && _parseIssues.size() > 0);
	}

	/**
	 * Obtain the exception message. If there is a list of parse issues, these
	 * are used to generate the exception message.
	 * 
	 */
	@Override
	public String getMessage() {
		return hasParseIssues() ? generateParseIssuesMessage() :
			(_message != null) ? _message : "No parse issue message.";
	}

	/**
	 * Generate a message displaying information about the parse issues that
	 * occurred.
	 * 
	 * @return String representing the list of parse issues.
	 */
	private String generateParseIssuesMessage() {
		if (_scriptString == null)
			return "No script string available.";
		
		String[] scriptLines = _scriptString.split("\\n");
		StringBuilder sb = new StringBuilder();
		sb.append("\n--------------------------------------------------------------");
		if (_parseIssues.size() == 1)
			sb.append("\nThe following parse issue was encountered:\n");
		else
			sb.append("\nThe following " + _parseIssues.size() + " parse issues were encountered:\n");
		int count = 1;
		for (ParseIssue parseIssue : _parseIssues) {
			if (_parseIssues.size() > 1) {
				sb.append("#");
				sb.append(count++);
				sb.append(" ");
			}

			int issueLineNum = parseIssue.getLine();
			boolean displayScriptLine = false;
			String scriptLine = null;
			if ((issueLineNum > 0) && (issueLineNum <= scriptLines.length)) {
				displayScriptLine = true;
				scriptLine = scriptLines[issueLineNum - 1];
			}

			String name = parseIssue.getFileName();
			if (name != null) {
				sb.append(name);
				sb.append(" ");
			}
			sb.append("[line ");
			sb.append(issueLineNum);
			sb.append(":");
			sb.append(parseIssue.getCharPositionInLine());
			sb.append("] [");
			sb.append(parseIssue.getParseIssueType().getText());
			sb.append("]");
			if (displayScriptLine) {
				sb.append(" -> ");
				sb.append(scriptLine);
			}
			sb.append("\n   ");
			sb.append(parseIssue.getMessage());
			sb.append("\n");
		}
		sb.append("--------------------------------------------------------------");
		return sb.toString();
	}
}
