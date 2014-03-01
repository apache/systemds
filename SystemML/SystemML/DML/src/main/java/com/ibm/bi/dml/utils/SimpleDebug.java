/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils;

import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * This class provides support for debugging on machines where runtime-debugging
 * is not accessible.
 * 
 */
public class SimpleDebug 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * This method is supposed to help you debugging. It appends date signature
	 * and msg to a logfile logger.txt in the directory of execution.
	 * 
	 * @param msg
	 */
	public static void log(String msg) {
		try {
			PrintWriter pw = new PrintWriter("logger.txt");
			pw.append(new SimpleDateFormat("yyyy.MM.dd hh:mm:ss-").format(new Date()) + msg);
			pw.close();
		} catch (Exception e) {
		}
	}
}
