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

package org.apache.sysml.runtime.io;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.OutputInfo;

/**
 * 
 * 
 */
public class MatrixWriterFactory 
{

	
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
			if( OptimizerUtils.PARALLEL_CP_WRITE_TEXTFORMATS )
				writer = new WriterTextCellParallel();
			else
				writer = new WriterTextCell();
		}
		else if( oinfo == OutputInfo.MatrixMarketOutputInfo ) {
			//note: disabled parallel cp write of matrix market in order to ensure the
			//requirement of writing out a single file
			
			//if( OptimizerUtils.PARALLEL_CP_WRITE_TEXTFORMATS )
			//	writer = new WriterMatrixMarketParallel();
			
			writer = new WriterMatrixMarket();
		}
		else if( oinfo == OutputInfo.CSVOutputInfo ) {
			if( props!=null && !(props instanceof CSVFileFormatProperties) )
				throw new DMLRuntimeException("Wrong type of file format properties for CSV writer.");
			if( OptimizerUtils.PARALLEL_CP_WRITE_TEXTFORMATS )
				writer = new WriterTextCSVParallel((CSVFileFormatProperties)props);
			else
				writer = new WriterTextCSV((CSVFileFormatProperties)props);
		}
		else if( oinfo == OutputInfo.BinaryCellOutputInfo ) {
			writer = new WriterBinaryCell();
		}
		else if( oinfo == OutputInfo.BinaryBlockOutputInfo ) {
			if( OptimizerUtils.PARALLEL_CP_WRITE_BINARYFORMATS )
				writer = new WriterBinaryBlockParallel(replication);
			else
				writer = new WriterBinaryBlock(replication);
		}
		else {
			throw new DMLRuntimeException("Failed to create matrix writer for unknown output info: "
		                                   + OutputInfo.outputInfoToString(oinfo));
		}
		
		return writer;
	}
	
}
