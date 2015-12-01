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


package org.apache.sysml.runtime.matrix;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.NumItemsByEachReducerMetaData;
import org.apache.sysml.runtime.matrix.data.OutputInfo;


public class JobReturn 
{
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
