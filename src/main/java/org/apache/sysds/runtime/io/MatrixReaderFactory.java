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

package org.apache.sysds.runtime.io;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.InputInfo;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class MatrixReaderFactory 
{
	public static MatrixReader createMatrixReader(InputInfo iinfo)
	{
		MatrixReader reader = null;
		boolean par = ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_TEXTFORMATS);
		boolean mcsr = MatrixBlock.DEFAULT_SPARSEBLOCK == SparseBlock.Type.MCSR;
		
		if( iinfo == InputInfo.TextCellInputInfo || iinfo == InputInfo.MatrixMarketInputInfo ) {
			reader = (par & mcsr) ? 
				new ReaderTextCellParallel(iinfo) : new ReaderTextCell(iinfo);
		}
		else if( iinfo == InputInfo.CSVInputInfo ) {
			reader = (par & mcsr) ? 
				new ReaderTextCSVParallel(new FileFormatPropertiesCSV()) :
				new ReaderTextCSV(new FileFormatPropertiesCSV());
		}
		else if( iinfo == InputInfo.LIBSVMInputInfo) {
			reader = (par & mcsr) ? 
				new ReaderTextLIBSVMParallel() : new ReaderTextLIBSVM();
		}
		else if( iinfo == InputInfo.BinaryCellInputInfo ) 
			reader = new ReaderBinaryCell();
		else if( iinfo == InputInfo.BinaryBlockInputInfo ) {
			reader = (par & mcsr) ? 
				new ReaderBinaryBlockParallel(false) : new ReaderBinaryBlock(false);
		}
		else {
			throw new DMLRuntimeException("Failed to create matrix reader for unknown input info: "
				+ InputInfo.inputInfoToString(iinfo));
		}
		
		return reader;
	}

	public static MatrixReader createMatrixReader( ReadProperties props ) 
	{
		//check valid read properties
		if( props == null )
			throw new DMLRuntimeException("Failed to create matrix reader with empty properties.");
		
		MatrixReader reader = null;
		InputInfo iinfo = props.inputInfo;
		boolean par = ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_TEXTFORMATS);
		boolean mcsr = MatrixBlock.DEFAULT_SPARSEBLOCK == SparseBlock.Type.MCSR;
		
		if( iinfo == InputInfo.TextCellInputInfo || iinfo == InputInfo.MatrixMarketInputInfo ) {
			reader = (par & mcsr) ? 
				new ReaderTextCellParallel(iinfo) : new ReaderTextCell(iinfo);
		}
		else if( iinfo == InputInfo.CSVInputInfo ) {
			reader = (par & mcsr) ?
				new ReaderTextCSVParallel( props.formatProperties!=null ?
					(FileFormatPropertiesCSV)props.formatProperties : new FileFormatPropertiesCSV()) :
				new ReaderTextCSV( props.formatProperties!=null ? 
					(FileFormatPropertiesCSV)props.formatProperties : new FileFormatPropertiesCSV());
		}
		else if( iinfo == InputInfo.LIBSVMInputInfo) {
			reader = (par & mcsr) ? 
				new ReaderTextLIBSVMParallel() : new ReaderTextLIBSVM();
		}
		else if( iinfo == InputInfo.BinaryCellInputInfo ) 
			reader = new ReaderBinaryCell();
		else if( iinfo == InputInfo.BinaryBlockInputInfo ) {
			reader = (par & mcsr) ? 
				new ReaderBinaryBlockParallel(props.localFS) : new ReaderBinaryBlock(props.localFS);
		}
		else {
			throw new DMLRuntimeException("Failed to create matrix reader for unknown input info: "
				+ InputInfo.inputInfoToString(iinfo));
		}
		
		return reader;
	}
}
