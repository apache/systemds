package dml.runtime.matrix;

import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import dml.utils.DMLRuntimeException;

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

	public JobReturn(MatrixCharacteristics[] sts, InputInfo[] infos,
			boolean success) {
		successful = success;
		metadata = new InputInfoMetaData[sts.length];
		for (int i = 0; i < sts.length; i++) {
			metadata[i] = new InputInfoMetaData(sts[i], infos[i]);
		}
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
