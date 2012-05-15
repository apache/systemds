package com.ibm.bi.dml.runtime.matrix;

import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class JobReturn {
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
		if (successful == false)
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
