/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.lops;

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.meta.DataCharacteristics;

/**
 * class to maintain output parameters for a lop.
 * 
 */

public class OutputParameters 
{
	private boolean _blocked = true;
	private long _num_rows = -1;
	private long _num_cols = -1;
	private long _nnz = -1;
	private UpdateType _updateType = UpdateType.COPY;
	private long _blocksize = -1;
	private String _file_name = null;
	private String _file_label = null;
	private boolean _linCacheCandidate = true;

	FileFormat matrix_format = FileFormat.BINARY;
	
	public String getFile_name() {
		return _file_name;
	}

	public void setFile_name(String fileName) {
		_file_name = fileName;
	}
	
	public String getLabel() {
		return _file_label;
	}

	public void setLabel(String label) {
		_file_label = label;
	}
	
	public void setDimensions(DataCharacteristics dc) {
		setDimensions(dc.getRows(), dc.getCols(), dc.getBlocksize(), dc.getNonZeros());
	}

	public void setDimensions(long rows, long cols, long blen, long nnz) {
		_num_rows = rows;
		_num_cols = cols;
		_nnz = nnz;
		_blocksize = blen;

		if ( _blocksize == 0 || _blocksize == -1) {
			_blocked = false;
 		}
		else if ( _blocksize > 0 ) {
			_blocked = true;
		}
		else {
			throw new HopsException("In OutputParameters Lop, Invalid values for blocking dimensions: [" + _blocksize + "," + _blocksize +"].");
		}
	}

	public void setDimensions(long rows, long cols, long blen, long nnz, UpdateType update) {
		_updateType = update;
		setDimensions(rows, cols, blen, nnz);
	}

	public void setDimensions(long rows, long cols, long blen, long nnz, boolean linCacheCand) {
		_linCacheCandidate = linCacheCand;
		setDimensions(rows, cols, blen, nnz);
	}
	
	public void setDimensions(OutputParameters input) {
		_num_rows = input._num_rows;
		_num_cols = input._num_cols;
		_blocksize = input._blocksize;
	}
	
	public FileFormat getFormat() {
		return matrix_format;
	}

	public void setFormat(FileFormat fmt) {
		matrix_format = fmt;
	}

	public boolean isBlocked() {
		return _blocked;
	}

	public void setBlocked(boolean blocked) {
		_blocked = blocked;
	}
	
	public long getNumRows() {
		return _num_rows;
	}
	
	public void setNumRows(long rows) {
		_num_rows = rows;
	}
	
	public long getNumCols() {
		return _num_cols;
	}
	
	public void setNumCols(long cols) {
		_num_cols = cols;
	}
	
	public Long getNnz() {
		return _nnz;
	}
	
	public void setNnz(long nnz) {
		_nnz = nnz;
	}
	
	public UpdateType getUpdateType() {
		return _updateType;
	}
	
	public void setUpdateType(UpdateType update) {
		_updateType = update;
	}
	
	public boolean getLinCacheMarking() {
		return _linCacheCandidate;
	}

	public long getBlocksize() {
		return _blocksize;
	}
	
	public void setBlocksize(long blen) {
		_blocksize = blen;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("rows=" + getNumRows() + Lop.VALUETYPE_PREFIX);
		sb.append("cols=" + getNumCols() + Lop.VALUETYPE_PREFIX);
		sb.append("nnz=" + getNnz() + Lop.VALUETYPE_PREFIX);
		sb.append("updateInPlace=" + getUpdateType().toString() + Lop.VALUETYPE_PREFIX);
		sb.append("blocksize=" + getBlocksize() + Lop.VALUETYPE_PREFIX);
		sb.append("isBlockedRepresentation=" + isBlocked() + Lop.VALUETYPE_PREFIX);
		sb.append("format=" + getFormat() + Lop.VALUETYPE_PREFIX);
		sb.append("label=" + getLabel() + Lop.VALUETYPE_PREFIX);
		sb.append("filename=" + getFile_name());
		return sb.toString();
	}
}
