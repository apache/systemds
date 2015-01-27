/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser.python;

import java.util.Stack;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;

public class PydmlSyntacticErrorListener {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static boolean atleastOneError = false;
	public static Stack<String> currentFileName = new Stack<String>();
	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());
	
	public static class CustomDmlErrorListener extends BaseErrorListener {
		
		public void validationError(int line, int charPositionInLine, String msg) {
			try {
				atleastOneError = true;
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
				atleastOneError = true;
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
	}
}
