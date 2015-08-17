/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.SequenceFile;

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
           LOG.error("Failed to close buffered reader.", ex);
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
	
	/**
	 * 
	 * @param br
	 */
	public static void closeSilently( SequenceFile.Reader br ) 
	{
		try {
			if( br != null )
				br.close();
        } 
		catch (Exception ex) {
           LOG.error("Failed to close reader.", ex);
		}
	}
	
	/**
	 * 
	 * @param br
	 */
	public static void closeSilently( SequenceFile.Writer bw ) 
	{
		try {
			if( bw != null )
				bw.close();
        } 
		catch (Exception ex) {
           LOG.error("Failed to writer.", ex);
		}
	}

	
	/**
	 * 
	 * @param br
	 */
	public static double parseDoubleParallel( String str ) 
	{
		//return FloatingDecimal.parseDouble(str);
		return Double.parseDouble(str);
	}

}
