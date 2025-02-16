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

package org.apache.sysds.test.functions.io;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FileFormatPropertiesHDF5;
import org.apache.sysds.runtime.io.FileFormatPropertiesLIBSVM;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderBinaryBlock;
import org.apache.sysds.runtime.io.FrameReaderBinaryBlockParallel;
import org.apache.sysds.runtime.io.FrameReaderTextCSV;
import org.apache.sysds.runtime.io.FrameReaderTextCSVParallel;
import org.apache.sysds.runtime.io.FrameReaderTextCell;
import org.apache.sysds.runtime.io.FrameReaderTextCellParallel;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterBinaryBlock;
import org.apache.sysds.runtime.io.FrameWriterBinaryBlockParallel;
import org.apache.sysds.runtime.io.FrameWriterTextCSV;
import org.apache.sysds.runtime.io.FrameWriterTextCSVParallel;
import org.apache.sysds.runtime.io.FrameWriterTextCell;
import org.apache.sysds.runtime.io.FrameWriterTextCellParallel;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.ReaderBinaryBlock;
import org.apache.sysds.runtime.io.ReaderBinaryBlockParallel;
import org.apache.sysds.runtime.io.ReaderHDF5;
import org.apache.sysds.runtime.io.ReaderHDF5Parallel;
import org.apache.sysds.runtime.io.ReaderTextCSV;
import org.apache.sysds.runtime.io.ReaderTextCSVParallel;
import org.apache.sysds.runtime.io.ReaderTextCell;
import org.apache.sysds.runtime.io.ReaderTextCellParallel;
import org.apache.sysds.runtime.io.ReaderTextLIBSVM;
import org.apache.sysds.runtime.io.ReaderTextLIBSVMParallel;
import org.apache.sysds.runtime.io.WriterBinaryBlock;
import org.apache.sysds.runtime.io.WriterBinaryBlockParallel;
import org.apache.sysds.runtime.io.WriterHDF5;
import org.apache.sysds.runtime.io.WriterHDF5Parallel;
import org.apache.sysds.runtime.io.WriterMatrixMarket;
import org.apache.sysds.runtime.io.WriterMatrixMarketParallel;
import org.apache.sysds.runtime.io.WriterTextCSV;
import org.apache.sysds.runtime.io.WriterTextCSVParallel;
import org.apache.sysds.runtime.io.WriterTextCell;
import org.apache.sysds.runtime.io.WriterTextCellParallel;
import org.apache.sysds.runtime.io.WriterTextLIBSVM;
import org.apache.sysds.runtime.io.WriterTextLIBSVMParallel;
import org.apache.sysds.runtime.io.FrameReaderParquet;
import org.apache.sysds.runtime.io.FrameReaderParquetParallel;
import org.apache.sysds.runtime.io.FrameWriterParquet;
import org.apache.sysds.runtime.io.FrameWriterParquetParallel;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class SeqParReadTest2 extends AutomatedTestBase {
	
	private final static String TEST_NAME = "SeqParReadTest";
	private final static String TEST_DIR = "functions/io/";
	private final static String TEST_CLASS_DIR = TEST_DIR + SeqParReadTest2.class.getSimpleName() + "/";
	private static final String PACKAGE = "org.apache.sysds.runtime.io";
	private static Level _oldLevel = null;
	
	private final static int rows = 1200;
	private final static int cols = 300;
	private final static ValueType[] schema = UtilFunctions.nCopies(cols, ValueType.FP64);
	private final static double eps = 1e-9;
	private final boolean _matrix;
	private final String _format;
	private final boolean _par;
	private final double _sparsity;
	
	public SeqParReadTest2(boolean matrix, String format, boolean par, double sparsity) {
		_matrix = matrix;
		_format = format;
		_par = par;
		_sparsity = sparsity;
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) ); 
		_oldLevel = Logger.getLogger(PACKAGE).getLevel();
		Logger.getLogger(PACKAGE).setLevel( Level.TRACE );
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { 
			//matrix/frame, format, par, sparsity
			{true, "text", false, 0.7},
			{true, "text", false, 0.1},
			{true, "text", true, 0.7},
			{true, "text", true, 0.1},
			{false, "text", false, 0.7},
			{false, "text", false, 0.1},
			{false, "text", true, 0.7},
			{false, "text", true, 0.1},
			{true, "mm", false, 0.7},
			{true, "mm", false, 0.1},
			{true, "mm", true, 0.7},
			{true, "mm", true, 0.1},
			{false, "mm", false, 0.7},
			{false, "mm", false, 0.1},
			{false, "mm", true, 0.7},
			{false, "mm", true, 0.1},
			{true, "csv", false, 0.7},
			{true, "csv", false, 0.1},
			{true, "csv", true, 0.7},
			{true, "csv", true, 0.1},
			{false, "csv", false, 0.7},
			{false, "csv", false, 0.1},
			{false, "csv", true, 0.7},
			{false, "csv", true, 0.1},
			{true, "binary", false, 0.7},
			{true, "binary", false, 0.1},
			{true, "binary", true, 0.7},
			{true, "binary", true, 0.1},
			{false, "binary", false, 0.7},
			{false, "binary", false, 0.1},
			{false, "binary", true, 0.7},
			{false, "binary", true, 0.1},
			{true, "hdf5", false, 0.7},
			{true, "hdf5", false, 0.1},
			{true, "hdf5", true, 0.7},
			{true, "hdf5", true, 0.1},
			{true, "libsvm", false, 0.7},
			{true, "libsvm", false, 0.1},
			{true, "libsvm", true, 0.7},
			{true, "libsvm", true, 0.1},
			{false, "parquet", false, 0.7},
			{false, "parquet", true, 0.7},
		};
		return Arrays.asList(data);
	}
	
	@Override
	public void tearDown() {
		super.tearDown();
		Logger.getLogger(PACKAGE).setLevel( _oldLevel );
	}
	
	@Test
	public void textWriteRead() {
		getAndLoadTestConfiguration(TEST_NAME);
		setOutputBuffering(false);
		String fname = output("Rout");
		
		MatrixBlock data = MatrixBlock.randOperations(rows, cols, _sparsity, 0, 1, "uniform", 7);
		MatrixBlock data2 = null;
		
		try {
			if( _matrix ) {
				MatrixWriter writer = createMatrixWriter(FileFormat.safeValueOf(_format), _par);
				writer.writeMatrixToHDFS(data, fname, rows, cols, 100, data.getNonZeros());
				writer.setForcedParallel(_par);
				MatrixReader reader = createMatrixReader(FileFormat.safeValueOf(_format), _par);
				data2 = reader.readMatrixFromHDFS(fname, rows, cols, 100, data.getNonZeros());
			}
			else {
				FrameBlock fdata = DataConverter.convertToFrameBlock(data);
				FrameWriter writer = createFrameWriter(FileFormat.safeValueOf(_format), _par);
				writer.setForcedParallel(_par);
				writer.writeFrameToHDFS(fdata, fname, rows, cols);
				FrameReader reader = createFrameReader(FileFormat.safeValueOf(_format), _par);
				FrameBlock fdata2 = reader.readFrameFromHDFS(fname, schema, rows, cols);
				data2 = DataConverter.convertToMatrixBlock(fdata2);
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			Assert.fail();
		}
		
		//compare read content is equivalent to original
		if( data2 != null ) {
			Assert.assertEquals(data.getNonZeros(), data2.getNonZeros());
			TestUtils.compareMatrices(data, data2, eps);
		}
	}
	
	@SuppressWarnings("incomplete-switch")
	public static MatrixWriter createMatrixWriter(FileFormat fmt, boolean par) {
		switch(fmt) {
			case TEXT: return par? new WriterTextCellParallel() : new WriterTextCell();
			case MM:   return par? new WriterMatrixMarketParallel() : new WriterMatrixMarket();
			case CSV:  return par ? 
							new WriterTextCSVParallel(new FileFormatPropertiesCSV()) :
							new WriterTextCSV(new FileFormatPropertiesCSV());
			case LIBSVM: return par ? 
							new WriterTextLIBSVMParallel(new FileFormatPropertiesLIBSVM()) :
							new WriterTextLIBSVM(new FileFormatPropertiesLIBSVM());
			case BINARY: return par ? new WriterBinaryBlockParallel(3) : new WriterBinaryBlock(3);
			case HDF5: return par ?
							new WriterHDF5Parallel(new FileFormatPropertiesHDF5()) :
							new WriterHDF5(new FileFormatPropertiesHDF5());
		}
		return null;
	}
	
	@SuppressWarnings("incomplete-switch")
	public static MatrixReader createMatrixReader(FileFormat fmt, boolean par) {
		switch(fmt) {
			case TEXT: return par? new ReaderTextCellParallel(fmt) : new ReaderTextCell(fmt);
			case MM:   return par? new ReaderTextCell(fmt) : new ReaderTextCell(fmt);
			case CSV:  return par ? 
							new ReaderTextCSVParallel(new FileFormatPropertiesCSV()) :
							new ReaderTextCSV(new FileFormatPropertiesCSV());
			case LIBSVM: return par ? 
							new ReaderTextLIBSVMParallel(new FileFormatPropertiesLIBSVM()) :
							new ReaderTextLIBSVM(new FileFormatPropertiesLIBSVM());
			case BINARY: return par ? new ReaderBinaryBlockParallel(false) : new ReaderBinaryBlock(false);
			case HDF5: return par ?
							new ReaderHDF5Parallel(new FileFormatPropertiesHDF5()) :
							new ReaderHDF5(new FileFormatPropertiesHDF5());
		}
		return null;
	}
	
	@SuppressWarnings("incomplete-switch")
	public static FrameWriter createFrameWriter(FileFormat fmt, boolean par) {
		switch(fmt) {
			case TEXT: return par? new FrameWriterTextCellParallel() : new FrameWriterTextCell();
			case MM:   return par? new FrameWriterTextCellParallel() : new FrameWriterTextCell();
			case CSV:  return par ? 
							new FrameWriterTextCSVParallel(new FileFormatPropertiesCSV()) :
							new FrameWriterTextCSV(new FileFormatPropertiesCSV());
			case BINARY: return par ? new FrameWriterBinaryBlockParallel() : new FrameWriterBinaryBlock();
			case PARQUET: return par ? new FrameWriterParquetParallel() : new FrameWriterParquet();
		}
		return null;
	}
	
	@SuppressWarnings("incomplete-switch")
	public static FrameReader createFrameReader(FileFormat fmt, boolean par) {
		switch(fmt) {
			case TEXT: return par? new FrameReaderTextCellParallel() : new FrameReaderTextCell();
			case MM:   return par? new FrameReaderTextCell() : new FrameReaderTextCell();
			case CSV:  return par ? 
							new FrameReaderTextCSVParallel(new FileFormatPropertiesCSV()) :
							new FrameReaderTextCSV(new FileFormatPropertiesCSV());
			case BINARY: return par ? new FrameReaderBinaryBlockParallel() : new FrameReaderBinaryBlock();
			case PARQUET: return par ? new FrameReaderParquetParallel() : new FrameReaderParquet();
		}
		return null;
	}
}
