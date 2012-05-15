package com.ibm.bi.dml.utils;

/**
 * This class provides helper methods for external built-in functions.
 * 
 * @author Felix Hamborg
 * 
 */
public class ExtBuiltinHelper {
	/**
	 * Returns the name of the x'th unnamed parameter
	 * 
	 * @param i
	 * @return
	 */
	public static String getUnnamedParamName(int i) {
		return "arg" + i;
	}
}
