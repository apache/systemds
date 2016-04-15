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

package org.apache.sysml.runtime.instructions.flink;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.io.TextOutputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.FlinkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.flink.functions.ConvertMatrixBlockToIJVLines;
import org.apache.sysml.runtime.instructions.flink.utils.DataSetAggregateUtils;
import org.apache.sysml.runtime.instructions.flink.utils.DataSetConverterUtils;
import org.apache.sysml.runtime.instructions.flink.utils.IOUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class WriteFLInstruction extends FLInstruction {
	private CPOperand input1 = null;
	private CPOperand input2 = null;
	private CPOperand input3 = null;
	private FileFormatProperties formatProperties;

	//scalars might occur for transform
	private boolean isInputMatrixBlock = true;

	public WriteFLInstruction(String opcode, String istr) {
		super(opcode, istr);
	}

	public WriteFLInstruction(CPOperand in1, CPOperand in2, CPOperand in3, String opcode, String str) {
		super(opcode, str);
		input1 = in1;
		input2 = in2;
		input3 = in3;

		formatProperties = null; // set in case of csv
	}

	public static WriteFLInstruction parseInstruction(String str)
			throws DMLRuntimeException {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if (!opcode.equals("write")) {
			throw new DMLRuntimeException("Unsupported opcode");
		}

		// All write instructions have 3 parameters, except in case of delimited/csv file.
		// Write instructions for csv files also include three additional parameters (hasHeader, delimiter, sparse)
		if (parts.length != 4 && parts.length != 8) {
			throw new DMLRuntimeException("Invalid number of operands in write instruction: " + str);
		}

		//FLINK°write°_mVar2·MATRIX·DOUBLE°./src/test/scripts/functions/data/out/B·SCALAR·STRING·true°matrixmarket
		// ·SCALAR·STRING·true
		// _mVar2·MATRIX·DOUBLE
		CPOperand in1 = null, in2 = null, in3 = null;
		in1 = new CPOperand(parts[1]);
		in2 = new CPOperand(parts[2]);
		in3 = new CPOperand(parts[3]);

		WriteFLInstruction inst = new WriteFLInstruction(in1, in2, in3, opcode, str);

		if (in3.getName().equalsIgnoreCase("csv")) {
			boolean hasHeader = Boolean.parseBoolean(parts[4]);
			String delim = parts[5];
			boolean sparse = Boolean.parseBoolean(parts[6]);
			FileFormatProperties formatProperties = new CSVFileFormatProperties(hasHeader, delim, sparse);
			inst.setFormatProperties(formatProperties);

			boolean isInputMB = Boolean.parseBoolean(parts[7]);
			inst.setInputMatrixBlock(isInputMB);
		}
		return inst;
	}


	public FileFormatProperties getFormatProperties() {
		return formatProperties;
	}

	public void setFormatProperties(FileFormatProperties prop) {
		formatProperties = prop;
	}

	public void setInputMatrixBlock(boolean isMB) {
		isInputMatrixBlock = isMB;
	}

	public boolean isInputMatrixBlock() {
		return isInputMatrixBlock;
	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException {
		FlinkExecutionContext flec = (FlinkExecutionContext) ec;

		//get filename (literal or variable expression)
		String fname = ec.getScalarInput(input2.getName(), Expression.ValueType.STRING, input2.isLiteral())
				.getStringValue();

		try {
			//if the file already exists on HDFS, remove it.
			MapReduceTool.deleteFileIfExistOnHDFS(fname);
		} catch (IOException ioe) {
			throw new DMLRuntimeException("Could not delete file on hdfs: " + fname);
		}

		//prepare output info according to meta data
		String outFmt = input3.getName();
		OutputInfo oi = OutputInfo.stringToOutputInfo(outFmt);

		//get dataset
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> in1 = flec.getBinaryBlockDataSetHandleForVariable(input1.getName
				());
		MatrixCharacteristics mc = flec.getMatrixCharacteristics(input1.getName());

		if (oi == OutputInfo.MatrixMarketOutputInfo || oi == OutputInfo.TextCellOutputInfo) {
			//recompute nnz if necessary (required for header if matrix market)
			if (isInputMatrixBlock && !mc.nnzKnown()) {
				mc.setNonZeros(DataSetAggregateUtils.computeNNZFromBlocks(in1));
			}

			DataSet<String> header = null;
			if (outFmt.equalsIgnoreCase("matrixmarket")) {
				ArrayList<String> headerContainer = new ArrayList<String>(1);
				// First output MM header
				String headerStr = "%%MatrixMarket matrix coordinate real general\n" +
						// output number of rows, number of columns and number of nnz
						mc.getRows() + " " + mc.getCols() + " " + mc.getNonZeros();
				headerContainer.add(headerStr);
				header = flec.getFlinkContext().fromCollection(headerContainer);
			}

			DataSet<String> ijv = in1.flatMap(new ConvertMatrixBlockToIJVLines(mc.getRowsPerBlock(), mc
					.getColsPerBlock()));
			if (header != null)
				customSaveTextFile(header.union(ijv), fname, true);
			else
				customSaveTextFile(ijv, fname, false);
		} else if (oi == OutputInfo.CSVOutputInfo) {
			DataSet<String> out = null;
			long nnz = 0;

			if (isInputMatrixBlock) {
				if (!mc.nnzKnown()) {
					nnz = DataSetAggregateUtils.computeNNZFromBlocks(in1);
				}

				out = DataSetConverterUtils.binaryBlockToCsv(in1, mc, (CSVFileFormatProperties) formatProperties,
						true);
			} else {
				/// This case is applicable when the CSV output from transform() is written out
				@SuppressWarnings("unchecked")
				DataSet<Tuple2<Long, String>> dataset = (DataSet<Tuple2<Long, String>>) ((MatrixObject) flec.getVariable(
						input1.getName())).getDataSetHandle().getDataSet();
				out = dataset.map(new DataSetConverterUtils.ExtractElement<Tuple2<Long, String>, String>(1)).returns(
						String.class);

				String sep = ",";
				boolean hasHeader = false;
				if (formatProperties != null) {
					sep = ((CSVFileFormatProperties) formatProperties).getDelim();
					hasHeader = ((CSVFileFormatProperties) formatProperties).hasHeader();
				}

				if (hasHeader) {
					StringBuffer buf = new StringBuffer();
					for (int j = 1; j < mc.getCols(); j++) {
						if (j != 1) {
							buf.append(sep);
						}
						buf.append("C" + j);
					}
					ArrayList<String> headerContainer = new ArrayList<String>(1);
					headerContainer.add(0, buf.toString());
					DataSet<String> header = flec.getFlinkContext().fromCollection(headerContainer);
					out = header.union(out);
				}
			}
			out.writeAsText(fname);

			if (isInputMatrixBlock && !mc.nnzKnown())
				mc.setNonZeros(nnz);
		} else if (oi == OutputInfo.BinaryBlockOutputInfo) {
			//piggyback nnz computation on actual write
			long nnz = 0;
			if (!mc.nnzKnown()) {
				nnz = DataSetAggregateUtils.computeNNZFromBlocks(in1);
			}

			//save binary block dataset on hdfs
			IOUtils.saveAsHadoopFile(in1, fname, MatrixIndexes.class, MatrixBlock.class,
					SequenceFileOutputFormat.class);

			if (!mc.nnzKnown())
				mc.setNonZeros(nnz);
		} else {
			//unsupported formats: binarycell (not externalized)
			throw new DMLRuntimeException("Unexpected data format: " + outFmt);
		}
		// write meta data file
		try {
			MapReduceTool.writeMetaDataFile(fname + ".mtd", Expression.ValueType.DOUBLE, mc, oi, formatProperties);
		} catch (IOException ioe) {
			throw new DMLRuntimeException("Could not write metadata-file for output " + fname, ioe);
		}
	}

	/**
	 * @param dataset
	 * @param fname
	 * @param inSingleFile
	 * @throws DMLRuntimeException
	 */
	private void customSaveTextFile(DataSet<String> dataset, String fname, boolean inSingleFile)
			throws DMLRuntimeException {
		if (inSingleFile) {
			Random rand = new Random();
			String randFName = fname + "_" + rand.nextLong() + "_" + rand.nextLong();
			try {
				while (MapReduceTool.existsFileOnHDFS(randFName)) {
					randFName = fname + "_" + rand.nextLong() + "_" + rand.nextLong();
				}

				dataset.output(new TextOutputFormat<String>(new org.apache.flink.core.fs.Path(fname)));
				//MapReduceTool.mergeIntoSingleFile(randFName, fname); // Faster version :)
				//TODO: how do we merge the files??

				// dataset.coalesce(1, true).saveAsTextFile(randFName);
				// MapReduceTool.copyFileOnHDFS(randFName + "/part-00000", fname);
			} catch (Exception e) {
				throw new DMLRuntimeException("Cannot merge the output into single file: " + e.getMessage());
			} finally {
				try {
					// This is to make sure that we donot create random files on HDFS
					MapReduceTool.deleteFileIfExistOnHDFS(randFName);
				} catch (IOException e) {
					throw new DMLRuntimeException("Cannot merge the output into single file: " + e.getMessage());
				}
			}
		} else {
			dataset.output(new TextOutputFormat<String>(new org.apache.flink.core.fs.Path(fname)));
		}
	}
}
