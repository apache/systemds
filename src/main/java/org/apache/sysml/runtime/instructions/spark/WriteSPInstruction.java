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

package org.apache.sysml.runtime.instructions.spark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.functions.ComputeBinaryBlockNnzFunction;
import org.apache.sysml.runtime.instructions.spark.functions.ConvertMatrixBlockToIJVLines;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FileFormatProperties;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;

public class WriteSPInstruction extends SPInstruction 
{	
	private CPOperand input1 = null; 
	private CPOperand input2 = null;
	private CPOperand input3 = null;
	private FileFormatProperties formatProperties;
	
	//scalars might occur for transform
	private boolean isInputMatrixBlock = true; 
	
	public WriteSPInstruction(String opcode, String istr) {
		super(opcode, istr);
	}

	public WriteSPInstruction(CPOperand in1, CPOperand in2, CPOperand in3, String opcode, String str) {
		super(opcode, str);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		
		formatProperties = null; // set in case of csv
	}

	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		String opcode = parts[0];
		
		if( !opcode.equals("write") ) {
			throw new DMLRuntimeException("Unsupported opcode");
		}
		
		// All write instructions have 3 parameters, except in case of delimited/csv file.
		// Write instructions for csv files also include three additional parameters (hasHeader, delimiter, sparse)
		if ( parts.length != 4 && parts.length != 8 ) {
			throw new DMLRuntimeException("Invalid number of operands in write instruction: " + str);
		}
		
		//SPARK°write°_mVar2·MATRIX·DOUBLE°./src/test/scripts/functions/data/out/B·SCALAR·STRING·true°matrixmarket·SCALAR·STRING·true
		// _mVar2·MATRIX·DOUBLE
		CPOperand in1=null, in2=null, in3=null;
		in1 = new CPOperand(parts[1]);
		in2 = new CPOperand(parts[2]);
		in3 = new CPOperand(parts[3]);
		
		WriteSPInstruction inst = new WriteSPInstruction(in1, in2, in3, opcode, str); 
		
		if ( in3.getName().equalsIgnoreCase("csv") ) {
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
			throws DMLRuntimeException, DMLUnsupportedOperationException 
	{			
		SparkExecutionContext sec = (SparkExecutionContext) ec;

		//get filename (literal or variable expression)
		String fname = ec.getScalarInput(input2.getName(), ValueType.STRING, input2.isLiteral()).getStringValue();
		
		try
		{
			//if the file already exists on HDFS, remove it.
			MapReduceTool.deleteFileIfExistOnHDFS( fname );

			//prepare output info according to meta data
			String outFmt = input3.getName();
			OutputInfo oi = OutputInfo.stringToOutputInfo(outFmt);
				
			//get input rdd
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
			
			if(    oi == OutputInfo.MatrixMarketOutputInfo
				|| oi == OutputInfo.TextCellOutputInfo     ) 
			{
				//recompute nnz if necessary (required for header if matrix market)
				if ( isInputMatrixBlock && !mc.nnzKnown() )
					mc.setNonZeros( SparkUtils.computeNNZFromBlocks(in1) );
				
				JavaRDD<String> header = null;				
				if(outFmt.equalsIgnoreCase("matrixmarket")) {
					ArrayList<String> headerContainer = new ArrayList<String>(1);
					// First output MM header
					String headerStr = "%%MatrixMarket matrix coordinate real general\n" +
							// output number of rows, number of columns and number of nnz
							mc.getRows() + " " + mc.getCols() + " " + mc.getNonZeros();
					headerContainer.add(headerStr);
					header = sec.getSparkContext().parallelize(headerContainer);
				}
				
				JavaRDD<String> ijv = in1.flatMap(new ConvertMatrixBlockToIJVLines(mc.getRowsPerBlock(), mc.getColsPerBlock()));
				if(header != null)
					customSaveTextFile(header.union(ijv), fname, true);
				else
					customSaveTextFile(ijv, fname, false);
			}
			else if( oi == OutputInfo.CSVOutputInfo ) 
			{
				JavaRDD<String> out = null;
				Accumulator<Double> aNnz = null;
				
				if ( isInputMatrixBlock ) {
					//piggyback nnz computation on actual write
					if( !mc.nnzKnown() ) {
						aNnz = sec.getSparkContext().accumulator(0L);
						in1 = in1.mapValues(new ComputeBinaryBlockNnzFunction(aNnz));
					}	
					
					out = RDDConverterUtils.binaryBlockToCsv(in1, mc, 
							(CSVFileFormatProperties) formatProperties, true);
				}
				else 
				{
					// This case is applicable when the CSV output from transform() is written out
					@SuppressWarnings("unchecked")
					JavaPairRDD<Long,String> rdd = (JavaPairRDD<Long, String>) ((MatrixObject) sec.getVariable(input1.getName())).getRDDHandle().getRDD();
					out = rdd.values(); 

					String sep = ",";
					boolean hasHeader = false;
					if(formatProperties != null) {
						sep = ((CSVFileFormatProperties) formatProperties).getDelim();
						hasHeader = ((CSVFileFormatProperties) formatProperties).hasHeader();
					}
					
					if(hasHeader) {
						StringBuffer buf = new StringBuffer();
			    		for(int j = 1; j < mc.getCols(); j++) {
			    			if(j != 1) {
			    				buf.append(sep);
			    			}
			    			buf.append("C" + j);
			    		}
			    		ArrayList<String> headerContainer = new ArrayList<String>(1);
			    		headerContainer.add(0, buf.toString());
			    		JavaRDD<String> header = sec.getSparkContext().parallelize(headerContainer);
			    		out = header.union(out);
					}
				}
				
				customSaveTextFile(out, fname, false);
				
				if( isInputMatrixBlock && !mc.nnzKnown() )
					mc.setNonZeros((long)aNnz.value().longValue());
			}
			else if( oi == OutputInfo.BinaryBlockOutputInfo ) {
				//piggyback nnz computation on actual write
				Accumulator<Double> aNnz = null;
				if( !mc.nnzKnown() ) {
					aNnz = sec.getSparkContext().accumulator(0L);
					in1 = in1.mapValues(new ComputeBinaryBlockNnzFunction(aNnz));
				}
				
				//save binary block rdd on hdfs
				in1.saveAsHadoopFile(fname, MatrixIndexes.class, MatrixBlock.class, SequenceFileOutputFormat.class);
				
				if( !mc.nnzKnown() )
					mc.setNonZeros((long)aNnz.value().longValue());
			}
			else {
				//unsupported formats: binarycell (not externalized)
				throw new DMLRuntimeException("Unexpected data format: " + outFmt);
			}
			
			// write meta data file
			MapReduceTool.writeMetaDataFile (fname + ".mtd", ValueType.DOUBLE, mc, oi, formatProperties);	
		}
		catch(IOException ex)
		{
			throw new DMLRuntimeException("Failed to process write instruction", ex);
		}
	}
	
	/**
	 * 
	 * @param rdd
	 * @param fname
	 * @param inSingleFile
	 * @throws DMLRuntimeException
	 */
	private void customSaveTextFile(JavaRDD<String> rdd, String fname, boolean inSingleFile) 
		throws DMLRuntimeException 
	{
		if(inSingleFile) {
			Random rand = new Random();
			String randFName = fname + "_" + rand.nextLong() + "_" + rand.nextLong();
			try {
				while(MapReduceTool.existsFileOnHDFS(randFName)) {
					randFName = fname + "_" + rand.nextLong() + "_" + rand.nextLong();
				}
				
				rdd.saveAsTextFile(randFName);
				MapReduceTool.mergeIntoSingleFile(randFName, fname); // Faster version :)
				
				// rdd.coalesce(1, true).saveAsTextFile(randFName);
				// MapReduceTool.copyFileOnHDFS(randFName + "/part-00000", fname);
			} catch (IOException e) {
				throw new DMLRuntimeException("Cannot merge the output into single file: " + e.getMessage());
			}
			finally {
				try {
					// This is to make sure that we donot create random files on HDFS
					MapReduceTool.deleteFileIfExistOnHDFS( randFName );
				} catch (IOException e) {
					throw new DMLRuntimeException("Cannot merge the output into single file: " + e.getMessage());
				}
			}
		}
		else {
			rdd.saveAsTextFile(fname);
		}
	}
}
