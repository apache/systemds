/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertMatrixBlockToIJVLines;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

import org.apache.hadoop.mapred.SequenceFileOutputFormat;

import scala.Tuple2;

public class WriteSPInstruction extends SPInstruction {

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CPOperand input1 = null; 
	private CPOperand input2 = null;
	private CPOperand input3 = null;
	private FileFormatProperties formatProperties;
	
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

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		String opcode = InstructionUtils.getOpCode(str);
		if(opcode.compareTo("write") != 0) {
			throw new DMLRuntimeException("Unsupported opcode");
		}
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		
		// All write instructions have 3 parameters, except in case of delimited/csv file.
		// Write instructions for csv files also include three additional parameters (hasHeader, delimiter, sparse)
		if ( parts.length != 4 && parts.length != 7 ) {
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
		}
		return inst;
		
	}
	
	
	public FileFormatProperties getFormatProperties() {
		return formatProperties;
	}
	
	public void setFormatProperties(FileFormatProperties prop) {
		formatProperties = prop;
	}
	
	private void customSaveTextFile(JavaRDD<String> rdd, String fname, boolean inSingleFile) throws DMLRuntimeException {
		//if the file already exists on HDFS, remove it.
		try {
			MapReduceTool.deleteFileIfExistOnHDFS( fname );
		} catch (IOException e) {
			throw new DMLRuntimeException("Error: While deleting file on HDFS");
		}
		
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

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		//get filename (literal or variable expression)
		String fname = ec.getScalarInput(input2.getName(), ValueType.STRING, input2.isLiteral()).getStringValue();
		
		SparkExecutionContext sec = (SparkExecutionContext) ec;
		if ( input1.getDataType() == DataType.SCALAR ) {
			writeScalarToHDFS(sec, fname);
		}
		else 
		{
			String outFmt = input3.getName();
			
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
			
			
			MatrixObject mo = (MatrixObject)sec.getVariable(input1.getName());
			OutputInfo oi = ((MatrixFormatMetaData)mo.getMetaData()).getOutputInfo();
						
			if (outFmt.equalsIgnoreCase("matrixmarket") || outFmt.equalsIgnoreCase("textcell")) {
				JavaRDD<String> header = null;
				
				if(outFmt.equalsIgnoreCase("matrixmarket")) {
					ArrayList<String> headerContainer = new ArrayList<String>(1);
					// First output MM header
					String headerStr = "%%MatrixMarket matrix coordinate real general\n"
							+
							// output number of rows, number of columns and number of nnz
							mc.getRows() + " " + mc.getCols() + " " + mc.getNonZeros() ;
					headerContainer.add(headerStr);
					header = sec.getSparkContext().parallelize(headerContainer);
				}
				
				JavaRDD<String> ijv = in1.flatMap(new ConvertMatrixBlockToIJVLines(mc.getRowsPerBlock(), mc.getColsPerBlock()));
				if(header != null) {
					customSaveTextFile(header.union(ijv), fname, true);
				}
				else {
					customSaveTextFile(ijv, fname, false);
				}
			}
			else if (outFmt.equalsIgnoreCase("csv") ) {
				String sep = ",";
				boolean sparse = false;
				boolean hasHeader = false;
				if(formatProperties != null) {
					sep = ((CSVFileFormatProperties) formatProperties).getDelim();
					sparse = ((CSVFileFormatProperties) formatProperties).isSparse();
					hasHeader = ((CSVFileFormatProperties) formatProperties).hasHeader();
				}
				JavaRDD<String> out = in1.flatMapToPair(new ExtractRows(mc.getRows(), mc.getCols(), mc.getRowsPerBlock(), mc.getColsPerBlock(), sep, sparse)).groupByKey()
										.mapToPair(new ConcatenateColumnsInRow(mc.getCols(), mc.getColsPerBlock(), sep)).sortByKey(true).values();
				if(hasHeader) {
					StringBuffer buf = new StringBuffer();
		    		for(int j = 1; j < mc.getCols(); j++) {
		    			if(j != 1) {
		    				buf.append(sep);
		    			}
		    			buf.append("C" + j);
		    		}
		    		ArrayList<String> headerContainer = new ArrayList<String>(1);
		    		JavaRDD<String> header = sec.getSparkContext().parallelize(headerContainer);
		    		out = header.union(out);
				}
				
				customSaveTextFile(out, fname, false);
			}
			else if (outFmt.equalsIgnoreCase("binaryblock")) {
			// else if(oi == OutputInfo.BinaryBlockOutputInfo) {
				in1.saveAsHadoopFile(fname, MatrixIndexes.class, MatrixBlock.class, SequenceFileOutputFormat.class);
			}
			else if (outFmt.equalsIgnoreCase("binarycell")) {
			// else if(oi == OutputInfo.BinaryCellOutputInfo) {
				throw new DMLRuntimeException("Writing using binary cell format is not implemented in WriteSPInstruction");
			}
			else {
				throw new DMLRuntimeException("Unexpected data format: " + outFmt);
			}
			
			// Write Metadata file
			try {
				MapReduceTool.writeMetaDataFile (fname + ".mtd", mo.getValueType(), mc, oi, formatProperties);
			} catch (IOException e) {
				throw new DMLRuntimeException("Unable to write metadata file in WriteSPInstruction");
			}
		}
	}
	
	// Returns rowCellIndex, <columnBlockIndex, csv string>
	public static class ExtractRows implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, Long, Tuple2<Long, String>> {

		private static final long serialVersionUID = 5185943302519860487L;
		
		long rlen; int brlen;
		long clen; int bclen;
		String sep; boolean sparse;
		public ExtractRows(long rlen, long clen, int brlen, int bclen, String sep, boolean sparse) {
			this.rlen = rlen;
			this.brlen = brlen;
			this.clen = clen;
			this.bclen = bclen;
			this.sep = sep;
			this.sparse = sparse;
		}

		@Override
		public Iterable<Tuple2<Long, Tuple2<Long, String>>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			long columnBlockIndex = kv._1.getColumnIndex();
			
			int lrlen = UtilFunctions.computeBlockSize(rlen, kv._1.getRowIndex(), brlen);
    		int lclen = UtilFunctions.computeBlockSize(clen, kv._1.getColumnIndex(), bclen);
    		long cellIndexTopRow = UtilFunctions.cellIndexCalculation(kv._1.getRowIndex(), brlen, 0);
    		
    		ArrayList<Tuple2<Long, Tuple2<Long, String>>> retVal = new ArrayList<Tuple2<Long,Tuple2<Long,String>>>(lrlen);
    		for(int i = 0; i < lrlen; i++) {
    			StringBuffer buf = new StringBuffer();
	    		for(int j = 0; j < lclen; j++) {
	    			if(j != 0) {
	    				buf.append(sep);
	    			}
	    			double val = kv._2.getValue(i, j);
	    			if(!(sparse && val == 0))
	    				buf.append(val);
				}
	    		retVal.add(new Tuple2<Long, Tuple2<Long,String>>(cellIndexTopRow, new Tuple2<Long,String>(columnBlockIndex, buf.toString())));
	    		cellIndexTopRow++;
    		}
    		
			return retVal;
		}
		
	}
	
	// Returns rowCellIndex, csv string
	public static class ConcatenateColumnsInRow implements PairFunction<Tuple2<Long,Iterable<Tuple2<Long,String>>>, Long, String> {

		private static final long serialVersionUID = -8529245417692255289L;
		
		long numColBlocks = -1;
		String sep;
		
		public ConcatenateColumnsInRow(long clen, int bclen, String sep) {
			numColBlocks = (long) Math.ceil((double)clen / (double)bclen);
			this.sep = sep;
		}
		
		public String getValue(Iterable<Tuple2<Long, String>> collection, Long key) throws Exception {
			for(Tuple2<Long, String> entry : collection) {
				if(entry._1== key) {
					return entry._2;
				}
			}
			throw new Exception("No value found for the key:" + key);
		}

		@Override
		public Tuple2<Long, String> call(Tuple2<Long, Iterable<Tuple2<Long, String>>> kv) throws Exception {
			StringBuffer buf = new StringBuffer();
			for(long i = 1; i <= numColBlocks; i++) {
				if(i != 1) {
					buf.append(sep);
				}
				buf.append(getValue(kv._2, i));
			}
			return new Tuple2<Long, String>(kv._1, buf.toString());
		}
		
	}
	
	/**
	 * Helper function to write scalars to HDFS based on its value type.
	 * @throws DMLRuntimeException 
	 */
	private void writeScalarToHDFS(ExecutionContext ec, String fname) 
		throws DMLRuntimeException 
	{
		ScalarObject scalar = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
		try {
			switch ( input1.getValueType() ) {
			case DOUBLE:
				MapReduceTool.writeDoubleToHDFS(scalar.getDoubleValue(), fname);
				break;
			case INT:
				MapReduceTool.writeIntToHDFS(scalar.getLongValue(), fname);
				break;
			case BOOLEAN:
				MapReduceTool.writeBooleanToHDFS(scalar.getBooleanValue(), fname);
				break;
			case STRING:
				MapReduceTool.writeStringToHDFS(scalar.getStringValue(), fname);
				break;
			default:
				throw new DMLRuntimeException("Invalid value type (" + input1.getValueType() + ") in write instruction while writing scalar: " + instString);
			}
		  // write out .mtd file
		  MapReduceTool.writeScalarMetaDataFile(fname +".mtd", input1.getValueType());
		} catch ( IOException e ) {
			throw new DMLRuntimeException(e);
		}
	}
	
}
