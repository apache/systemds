/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;

/**
 * 
 * 
 */
public class MatrixReaderFactory 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	/**
	 * 
	 * @param iinfo
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixReader createMatrixReader( InputInfo iinfo ) 
		throws DMLRuntimeException
	{
		MatrixReader reader = null;
		
		if( iinfo == InputInfo.TextCellInputInfo || iinfo == InputInfo.MatrixMarketInputInfo )
		{
			if( OptimizerUtils.PARALLEL_READ_TEXTFORMATS )
				reader = new ReaderTextCellParallel( iinfo );
			else
				reader = new ReaderTextCell( iinfo );	
		}
		else if( iinfo == InputInfo.CSVInputInfo )
		{
			if( OptimizerUtils.PARALLEL_READ_TEXTFORMATS )
				reader = new ReaderTextCSVParallel(new CSVFileFormatProperties());
			else
				reader = new ReaderTextCSV(new CSVFileFormatProperties());
		}
		else if( iinfo == InputInfo.BinaryCellInputInfo ) 
			reader = new ReaderBinaryCell();
		else if( iinfo == InputInfo.BinaryBlockInputInfo )
			reader = new ReaderBinaryBlock( false );
		else {
			throw new DMLRuntimeException("Failed to create matrix reader for unknown input info: "
		                                   + InputInfo.inputInfoToString(iinfo));
		}
		
		return reader;
	}
	
	/**
	 * 
	 * @param props
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixReader createMatrixReader( ReadProperties props ) 
		throws DMLRuntimeException
	{
		//check valid read properties
		if( props == null )
			throw new DMLRuntimeException("Failed to create matrix reader with empty properties.");
		
		MatrixReader reader = null;
		InputInfo iinfo = props.inputInfo;

		if( iinfo == InputInfo.TextCellInputInfo || iinfo == InputInfo.MatrixMarketInputInfo ) {
			if( OptimizerUtils.PARALLEL_READ_TEXTFORMATS )
				reader = new ReaderTextCellParallel( iinfo );
			else
				reader = new ReaderTextCell( iinfo );
		}
		else if( iinfo == InputInfo.CSVInputInfo ) {
			if( OptimizerUtils.PARALLEL_READ_TEXTFORMATS )
				reader = new ReaderTextCSVParallel( props.formatProperties!=null ? (CSVFileFormatProperties)props.formatProperties : new CSVFileFormatProperties());
			else
				reader = new ReaderTextCSV( props.formatProperties!=null ? (CSVFileFormatProperties)props.formatProperties : new CSVFileFormatProperties());
		}
		else if( iinfo == InputInfo.BinaryCellInputInfo ) 
			reader = new ReaderBinaryCell();
		else if( iinfo == InputInfo.BinaryBlockInputInfo )
			reader = new ReaderBinaryBlock( props.localFS );
		else {
			throw new DMLRuntimeException("Failed to create matrix reader for unknown input info: "
		                                   + InputInfo.inputInfoToString(iinfo));
		}
		
		return reader;
	}
}
