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

import org.apache.sysml.parser.common.CustomErrorListener.ParseIssue;

/**
 * This exception is thrown when parse issues are encountered.
 * 
 * NOTE: MB: Originally, this exception was generated with javacc.
 * Unfortunately, many classes directly use this exception. Hence, on removing
 * javacc we kept a condensed version of it. TODO: eventually we should remove
 * the token class and create our own exception or just use LanguageException.
 */
public class ParseException extends Exception {

	/**
	 * The version identifier for this Serializable class. Increment only if the
	 * <i>serialized</i> form of the class changes.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * The following constructors are for use by you for whatever purpose you
	 * can think of. Constructing the exception in this manner makes the
	 * exception behave in the normal way - i.e., as documented in the class
	 * "Throwable". The fields "errorToken", "expectedTokenSequences", and
	 * "tokenImage" do not contain relevant information. The JavaCC generated
	 * code does not use these constructors.
	 */

	public ParseException() {
		super();
	}

	/** Constructor with message. */
	public ParseException(String message) {
		super(message);
		this.message = message;
	}

	public ParseException(String message, Exception e) {
		super(message, e);
		this.message = message;
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
		this.parseIssues = parseIssues;
		this.scriptString = scriptString;
	}

	/**
	 * This is the last token that has been consumed successfully. If this
	 * object has been created due to a parse issue, the token following this
	 * token will (therefore) be the first error token.
	 */
	public Token currentToken;

	/**
	 * Each entry in this array is an array of integers. Each array of integers
	 * represents a sequence of tokens (by their ordinal values) that is
	 * expected at this point of the parse.
	 */
	public int[][] expectedTokenSequences;

	/**
	 * This is a reference to the "tokenImage" array of the generated parser
	 * within which the parse issue occurred. This array is defined in the
	 * generated ...Constants interface.
	 */
	public String[] tokenImage;

	/**
	 * List of issues that happened during parsing. Typically set by the error
	 * listener.
	 */
	private List<ParseIssue> parseIssues;

	/**
	 * The DML/PyDML script string. Used to display the original lines where the
	 * parse issues occurred.
	 */
	private String scriptString;

	/**
	 * Standard exception message. If no list of parse issues exists, this
	 * message can be used to display a parse exception message that doesn't
	 * relate to the list of parse issues.
	 */
	private String message;

	/**
	 * Obtain the list of parse issues that occurred.
	 * 
	 * @return the list of parse issues
	 */
	public List<ParseIssue> getParseIssues() {
		return parseIssues;
	}

	/**
	 * Set the list of parse issues.
	 * 
	 * @param parseIssues
	 *            the list of parse issues
	 */
	public void setParseIssues(List<ParseIssue> parseIssues) {
		this.parseIssues = parseIssues;
	}

	/**
	 * Obtain the original DML/PyDML script string.
	 * 
	 * @return the original DML/PyDML script string
	 */
	public String getScriptString() {
		return scriptString;
	}

	/**
	 * Set the original DML/PyDML script string.
	 * 
	 * @param scriptString
	 *            the original DML/PyDML script string
	 */
	public void setScriptString(String scriptString) {
		this.scriptString = scriptString;
	}

	/**
	 * Does this ParseException contain a list of parse issues?
	 * 
	 * @return <code>true</code> if the list of parse issues exists and is
	 *         greater than 0, <code>false</code> otherwise
	 */
	public boolean hasParseIssues() {
		if ((parseIssues != null) && (parseIssues.size() > 0)) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * Obtain the exception message. If there is a list of parse issues, these
	 * are used to generate the exception message.
	 * 
	 */
	@Override
	public String getMessage() {
		if (!hasParseIssues()) {
			if (message != null) {
				return message;
			} else {
				return "No parse issue message.";
			}
		} else {
			return generateParseIssuesMessage();
		}
	}

	/**
	 * Generate a message displaying information about the parse issues that
	 * occurred.
	 * 
	 * @return String representing the list of parse issues.
	 */
	private String generateParseIssuesMessage() {
		if (scriptString == null) {
			return "No script string available.";
		}
		String[] scriptLines = scriptString.split("\\n");
		StringBuilder sb = new StringBuilder();
		sb.append("\n--------------------------------------------------------------");
		if (parseIssues.size() == 1) {
			sb.append("\nThe following parse issue was encountered:\n");
		} else {
			sb.append("\nThe following " + parseIssues.size() + " parse issues were encountered:\n");
		}
		int count = 1;
		for (ParseIssue parseIssue : parseIssues) {
			if (parseIssues.size() > 1) {
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
