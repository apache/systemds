/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;

/**
 * 
 * 
 */
public class MatrixWriterFactory 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	
	/**
	 * 
	 * @param oinfo
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixWriter createMatrixWriter( OutputInfo oinfo ) 
			throws DMLRuntimeException
	{
		return createMatrixWriter(oinfo, -1, null);
	}
	
	/**
	 * 
	 * @param oinfo
	 * @param props 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixWriter createMatrixWriter( OutputInfo oinfo, int replication, FileFormatProperties props ) 
		throws DMLRuntimeException
	{
		MatrixWriter writer = null;
		
		if( oinfo == OutputInfo.TextCellOutputInfo ) {
			writer = new WriterTextCell();
		}
		else if( oinfo == OutputInfo.MatrixMarketOutputInfo ) {
			writer = new WriterMatrixMarket();
		}
		else if( oinfo == OutputInfo.CSVOutputInfo ) {
			if( props!=null && !(props instanceof CSVFileFormatProperties) )
				throw new DMLRuntimeException("Wrong type of file format properties for CSV writer.");
			writer = new WriterTextCSV((CSVFileFormatProperties)props);
		}
		else if( oinfo == OutputInfo.BinaryCellOutputInfo ) {
			writer = new WriterBinaryCell();
		}
		else if( oinfo == OutputInfo.BinaryBlockOutputInfo ) {
			writer = new WriterBinaryBlock(replication);
		}
		else {
			throw new DMLRuntimeException("Failed to create matrix writer for unknown output info: "
		                                   + OutputInfo.outputInfoToString(oinfo));
		}
		
		return writer;
	}
	
}
