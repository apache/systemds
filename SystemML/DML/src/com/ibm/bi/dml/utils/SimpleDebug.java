package com.ibm.bi.dml.utils;

import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * This class provides support for debugging on machines where runtime-debugging
 * is not accessible.
 * 
 * @author Felix Hamborg
 * 
 */
public class SimpleDebug {
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
