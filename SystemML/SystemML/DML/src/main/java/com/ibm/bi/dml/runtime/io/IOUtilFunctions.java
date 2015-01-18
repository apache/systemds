/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class IOUtilFunctions 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(UtilFunctions.class.getName());


	/**
	 * 
	 * @param is
	 */
	public static void closeSilently( InputStream is ) 
	{
		try {
			if( is != null )
				is.close();
        } 
		catch (Exception ex) {
           LOG.error("Failed to close input stream.", ex);
		}
	}
	
	/**
	 * 
	 * @param is
	 */
	public static void closeSilently( OutputStream os ) 
	{
		try {
			if( os != null )
				os.close();
        } 
		catch (Exception ex) {
           LOG.error("Failed to close output stream.", ex);
		}
	}
	
	/**
	 * 
	 * @param br
	 */
	public static void closeSilently( BufferedReader br ) 
	{
		try {
			if( br != null )
				br.close();
        } 
		catch (Exception ex) {
           LOG.error("Failed to buffered reader.", ex);
		}
	}
	
	/**
	 * 
	 * @param br
	 */
	public static void closeSilently( BufferedWriter bw ) 
	{
		try {
			if( bw != null )
				bw.close();
        } 
		catch (Exception ex) {
           LOG.error("Failed to buffered writer.", ex);
		}
	}
}
