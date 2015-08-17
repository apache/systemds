/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;


public class JobReturn 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public boolean successful;
	// public MatrixCharacteristics[] stats;
	// public MetaData[] otherMetadata=null;
	public MetaData[] metadata = null;

	public JobReturn() {
		successful = false;
		metadata = null;
	}

	public JobReturn(MatrixCharacteristics[] sts, boolean success) {
		successful = success;
		metadata = new MatrixDimensionsMetaData[sts.length];
		for (int i = 0; i < sts.length; i++) {
			metadata[i] = new MatrixDimensionsMetaData(sts[i]);
		}
	}

	public JobReturn(MatrixCharacteristics[] sts, OutputInfo[] infos,
			boolean success) throws DMLRuntimeException {
		successful = success;
		metadata = new MatrixFormatMetaData[sts.length];
		for (int i = 0; i < sts.length; i++) {
			metadata[i] = new MatrixFormatMetaData(sts[i], infos[i], OutputInfo.getMatchingInputInfo(infos[i]));
		}
	}

	public JobReturn(MatrixCharacteristics sts, OutputInfo info, boolean success) throws DMLRuntimeException {
		successful = success;
		metadata = new MatrixFormatMetaData[1];
		metadata[0] = new MatrixFormatMetaData(sts, info, OutputInfo.getMatchingInputInfo(info));
	}
	
	public JobReturn(MatrixCharacteristics mc, long[] items, int partition0, long number0s, boolean success) {
		successful = success;
		metadata = new NumItemsByEachReducerMetaData[1];
		metadata[0] = new NumItemsByEachReducerMetaData(mc, items, partition0, number0s);
	}

	public boolean checkReturnStatus() throws DMLRuntimeException {
		if( !successful )
			throw new DMLRuntimeException("Error in executing the DML program.");
		return successful;
	}

	public MetaData[] getMetaData() {
		return metadata;
	}

	public MetaData getMetaData(int i) {
		return metadata[i];
	}

	/*
	 * Since MatrixCharacteristics is the most common type of metadata, we
	 * define a method to extract this information for a given output index.
	 */
	public MatrixCharacteristics getMatrixCharacteristics(int i) {
		return ((MatrixDimensionsMetaData) metadata[i])
				.getMatrixCharacteristics();
	}

}
