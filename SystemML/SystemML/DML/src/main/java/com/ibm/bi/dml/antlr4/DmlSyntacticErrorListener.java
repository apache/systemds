/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.antlr4;

import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;

import java.util.Stack;

public class DmlSyntacticErrorListener {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static boolean atleastOneError = false;
	public static Stack<String> currentFileName = new Stack<String>();
	
	public static class CustomDmlErrorListener extends BaseErrorListener {
		
		public void validationError(int line, int charPositionInLine, String msg) {
			String type = "ERROR: ";
			try {
				atleastOneError = true;
				// Print error messages with file name
				if(currentFileName == null || currentFileName.empty())
					System.err.println(type + "line "+line+":"+charPositionInLine+" "+msg);
				else {
					String fileName = currentFileName.peek();
					System.err.println(type + fileName + " line "+line+":"+charPositionInLine+" "+msg);
				}
			}
			catch(Exception e1) {
				System.err.println("ERROR: while customizing error message:" + e1);
			}
		}
		
		public void validationWarning(int line, int charPositionInLine, String msg) {
			String type = "WARNING: ";
			try {
				//atleastOneError = true;
				// Print error messages with file name
				if(currentFileName == null || currentFileName.empty())
					System.err.println(type + "line "+line+":"+charPositionInLine+" "+msg);
				else {
					String fileName = currentFileName.peek();
					System.err.println(type + fileName + " line "+line+":"+charPositionInLine+" "+msg);
				}
			}
			catch(Exception e1) {
				System.err.println("ERROR: while customizing error message:" + e1);
			}
		}
		
		@Override
		public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
				int line, int charPositionInLine,
				String msg, RecognitionException e)
		{	
			String type = "ERROR: ";
			try {
				atleastOneError = true;
				// Print error messages with file name
				if(currentFileName == null || currentFileName.empty())
					System.err.println(type + "line "+line+":"+charPositionInLine+" "+msg);
				else {
					String fileName = currentFileName.peek();
					System.err.println(type + fileName + " line "+line+":"+charPositionInLine+" "+msg);
				}
			}
			catch(Exception e1) {
				System.err.println("ERROR: while customizing error message:" + e1);
			}
		}
	}
}
