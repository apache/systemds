package com.ibm.bi.dml.runtime.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;

import com.ibm.bi.dml.runtime.matrix.io.BinaryBlockToBinaryCellConverter;
import com.ibm.bi.dml.runtime.matrix.io.BinaryBlockToTextCellConverter;
import com.ibm.bi.dml.runtime.matrix.io.Converter;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock1D;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.io.TextToBinaryCellConverter;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.utils.DMLRuntimeException;


/**
 * This class provides methods to convert data from SystemML's internal and
 * external data representations to representations needed external functions,
 * such as those in JLapack, by and vice versa.
 * 
 * @author Felix Hamborg
 * 
 */
public class DataConverter {
	private static Configuration conf = new Configuration();
	
	public static void writeMatrixToHDFS(MatrixBlock mat, 
										 String dir, 
										 OutputInfo outputinfo, 
										 long rlen, 
										 long clen, 
										 int brlen, 
										 int bclen)
		throws IOException{
		JobConf job = new JobConf();
		FileOutputFormat.setOutputPath(job, new Path(dir));
		
		try{
			OutputFormat informat = outputinfo.outputFormatClass.newInstance();
			
			if ( outputinfo == OutputInfo.TextCellOutputInfo ) {
		        Path pt=new Path(dir);
		        FileSystem fs = FileSystem.get(new Configuration());
		        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		
		        Converter outputConverter = new BinaryBlockToTextCellConverter();

				outputConverter.setBlockSize((int)rlen, (int)clen);
				
				outputConverter.convert(new MatrixIndexes(1, 1), mat);
				while(outputConverter.hasNext()){
					br.write(outputConverter.next().getValue().toString() + "\n");
				}
				br.close();
			}
			else if ( outputinfo == OutputInfo.BinaryCellOutputInfo ) {
				SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(job), job, new Path(dir), outputinfo.outputKeyClass, outputinfo.outputValueClass);
				Converter outputConverter = new BinaryBlockToBinaryCellConverter();
				
				outputConverter.setBlockSize((int)rlen, (int)clen);
				
				outputConverter.convert(new MatrixIndexes(1, 1), mat);
				while(outputConverter.hasNext()){
					Pair pair = outputConverter.next();
					Writable index = (Writable) pair.getKey();
					Writable cell = (Writable) pair.getValue();
					
					writer.append(index, cell);
				}
				writer.close();
			}
			else{
				SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(job), job, new Path(dir), outputinfo.outputKeyClass, outputinfo.outputValueClass);
				//reblock
				MatrixBlock fullBlock = new MatrixBlock(brlen, bclen, false);
				
				MatrixBlock block;
				for(int blockRow = 0; blockRow < (int)Math.ceil(mat.getNumRows()/(double)brlen); blockRow++){
					for(int blockCol = 0; blockCol < (int)Math.ceil(mat.getNumColumns()/(double)bclen); blockCol++){
						int maxRow = (blockRow*brlen + brlen < mat.getNumRows()) ? brlen : mat.getNumRows() - blockRow*brlen;
						int maxCol = (blockCol*bclen + bclen < mat.getNumColumns()) ? bclen : mat.getNumColumns() - blockCol*bclen;
						
						if(maxRow < brlen || maxCol < bclen)
							block = new MatrixBlock(maxRow, maxCol, false);
						else block = fullBlock;
						
						for(int row = 0; row < maxRow; row++) {
							for(int col = 0; col < maxCol; col++){
								double value = mat.getValue(row + blockRow*brlen, col + blockCol*bclen);
								block.setValue(row, col, value);
							}
						}
						if ( blockRow == 0 && blockCol == 0 & block.getNonZeros() == 0 )
							block.addDummyZeroValue();
						writer.append(new MatrixIndexes(blockRow+1, blockCol+1), block);
						block.reset();
					}
				}
				
				writer.close();
			}
		}catch(Exception e){
			throw new IOException(e);
		}
	}
	
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen) 
	throws IOException
	{	
		MatrixBlock ret = new MatrixBlock((int)rlen, (int)clen, false);
		
	//	String filename = getSubDirsIgnoreLogs(dir);
		JobConf job = new JobConf();
		
		if(!FileSystem.get(job).exists(new Path(dir)))	
			return null;
		
		FileInputFormat.addInputPath(job, new Path(dir));
		
		try {

			InputFormat informat=inputinfo.inputFormatClass.newInstance();
			if(informat instanceof TextInputFormat)
				((TextInputFormat)informat).configure(job);
			InputSplit[] splits= informat.getSplits(job, 1);
			
			Converter inputConverter=MRJobConfiguration.getConverterClass(inputinfo, false, brlen, bclen).newInstance();
			inputConverter.setBlockSize(brlen, bclen);
    		
			Writable key=inputinfo.inputKeyClass.newInstance();
			Writable value=inputinfo.inputValueClass.newInstance();
			
			for(InputSplit split: splits)
			{
				RecordReader reader=informat.getRecordReader(split, job, Reporter.NULL);
				while(reader.next(key, value))
				{
					inputConverter.convert(key, value);
					while(inputConverter.hasNext())
					{
						Pair pair=inputConverter.next();
						MatrixIndexes index=(MatrixIndexes) pair.getKey();
						MatrixCell cell=(MatrixCell) pair.getValue();
						ret.setValue((int)index.getRowIndex()-1, (int)index.getColumnIndex()-1, cell.getValue());
					}
				}
				reader.close();
			}
			
			ret.examSparsity();
		} catch (Exception e) {
			throw new IOException(e);
		}
		
		return ret;
	}
	
	/**
	 * Reads a matrix from HDFS (in block format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix should have
	 * @param numCols
	 *            Number of columns which the matrix should have
	 * @param blockSizeRows
	 *            Number of rows in normal blocks
	 * @param blockSizeCols
	 *            Number of cols in normal blocks
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public static void readDouble1DArrayMatrixFromHDFSBlock(String dir, 
															MatrixBlock1D mat,
															int blockSizeRows, 
															int blockSizeCols) 
			throws DMLRuntimeException {		
		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixBlock value = new MatrixBlock();

			for (Path path : subpaths) {
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
				BinaryBlockToBinaryCellConverter conv = new BinaryBlockToBinaryCellConverter();
				conv.setBlockSize(blockSizeRows, blockSizeCols);

				while (reader.next(indexes, value)) {
					conv.convert(indexes, value);
					while (conv.hasNext()) {
						Pair<MatrixIndexes, MatrixCell> pair = conv.next();
						int row = (int)(pair.getKey().getRowIndex()) - 1;
						int col = (int)(pair.getKey().getColumnIndex()) - 1;
						
						if(row >= mat.getNumRows() || col >= mat.getNumColumns())
							throw new DMLRuntimeException("matrix on disk "
														  + dir
														  + " does not match size of matrix object");
						
						mat.setValue(row,
									 col,
									 pair.getValue().getValue());
					}
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	/**
	 * Reads a matrix from HDFS (in block format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix should have
	 * @param numCols
	 *            Number of columns which the matrix should have
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public static void readDouble1DArrayMatrixFromHDFSText(String dir, MatrixBlock1D mat)
			throws DMLRuntimeException {
		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			if (!fs.isDirectory(new Path(dir))) {
				subpaths = new Path[] { new Path(dir) };
			}

			LongWritable indexes = new LongWritable();
			Text value = new Text();

			for (Path path : subpaths) {
				// SequenceFile.Reader reader = new SequenceFile.Reader(fs,
				// path, conf);
				TextToBinaryCellConverter conv = new TextToBinaryCellConverter();
				FSDataInputStream fi = fs.open(path);
				BufferedReader br = new BufferedReader(new InputStreamReader(fi));
				String line = null;
				while ((line = br.readLine()) != null) {
					value = new Text(line);

					conv.convert(indexes, value);
					while (conv.hasNext()) {
						Pair<MatrixIndexes, MatrixCell> pair = conv.next();
						
						int row = (int)(pair.getKey().getRowIndex() - 1);
						int col = (int)(pair.getKey().getColumnIndex() - 1);
						
						if(row >= mat.getNumRows() || col >= mat.getNumColumns())
							throw new DMLRuntimeException("matrix on disk "
														  + dir
														  + " does not match size of matrix object");
						
						mat.setValue(row, 
									  col, 
									  pair.getValue().getValue());
					}
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	/**
	 * Reads a matrix from HDFS (in cell format) and returns its values in a 1D
	 * array containing double values.
	 * 
	 * @param dir
	 * @param numRows
	 *            Number of rows which the matrix is expected to have
	 * @param numCols
	 *            Number of columns which the matrix is expected to have
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public static void readDouble1DArrayMatrixFromHDFSCell(String dir, MatrixBlock1D mat)
			throws DMLRuntimeException {
		try {
			Path[] subpaths = getSubDirs(dir);
			FileSystem fs = FileSystem.get(conf);
			MatrixIndexes indexes = new MatrixIndexes();
			MatrixCell value = new MatrixCell();

			for (Path path : subpaths) {
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

				while (reader.next(indexes, value)) {
					long i = indexes.getRowIndex() - 1;
					long j = indexes.getColumnIndex() - 1;
					long p = i * mat.getNumColumns() + j;
					if (p > (int) p)
						throw new DMLRuntimeException("Matrix is too large");

					int row = (int)i;
					int col = (int)j;
					
					if(row >= mat.getNumRows() || col >= mat.getNumColumns())
						throw new DMLRuntimeException("matrix on disk "
													  + dir
													  + " does not match size of matrix object");
					
					mat.setValue(row, col, value.getValue());
				}
			}
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	public static void writeDoubleMatrixToHDFSBlock(String dir, 
													int blockSizeRows, 
													int blockSizeCols,
													MatrixBlock1D mat) 
			throws DMLRuntimeException {
		int numRows = mat.getNumRows();
		int numCols = mat.getNumColumns();
		try {
			SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(conf), conf, new Path(dir),
					MatrixIndexes.class, MatrixBlock.class);

			MatrixIndexes index = new MatrixIndexes();
			MatrixBlock value = new MatrixBlock();
			for (int i = 0; i < mat.getNumRows(); i += numRows) {
				int rows = Math.min(numRows, (mat.getNumRows() - i));
				for (int j = 0; j < mat.getNumColumns(); j += numCols) {
					int cols = Math.min(numCols, (mat.getNumColumns() - j));
					index.setIndexes(((i / numRows) + 1), ((j / numCols) + 1));
					value = new MatrixBlock(rows, cols, true);
					for (int k = 0; k < rows; k++) {
						for (int l = 0; l < cols; l++) 
							value.setValue(k, l, mat.getValue(i+k, j+l));
					}
					writer.append(index, value);
				}
			}

			writer.close();
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	//writes to one file only
	public static void writeDoubleMatrixToHDFSText(String dir, MatrixBlock1D mat)
		throws DMLRuntimeException{
		try{
			Path path = new Path(dir);
			FileSystem fs = FileSystem.get(conf);
			PrintWriter writer;
			if(fs.exists(path))
				System.err.println(path.toString()+" already exists");
			writer = new PrintWriter(fs.create(path, true));
			
			if(mat.isInSparseFormat()){
				HashMap<CellIndex, Double> map = mat.getSparseMap();
				Iterator<Map.Entry<CellIndex, Double>> it = map.entrySet().iterator();
				while(it.hasNext()){
					Map.Entry<CellIndex, Double> elt = it.next();
					int row = elt.getKey().row;
					int col = elt.getKey().column;
					double v = elt.getValue().doubleValue();
					writer.println((row+1)+" "+(col+1)+" "+v);
				}
			}else
				for(int i=0; i<mat.getNumRows(); i++)
					for(int j=0; j<mat.getNumColumns(); j++)
						writer.println((i+1)+" "+(j+1)+" "+mat.getValue(i, j));
			
			writer.close();
		}catch(IOException e){
			throw new DMLRuntimeException(e);
		}
	}

	/**
	 * Creates a two-dimensional double matrix of the input matrix block. 
	 * 
	 * @param mb
	 * @return
	 */
	public static double[][] convertToDoubleMatrix( MatrixBlock mb )
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[][] ret = new double[rows][cols];
		
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
				ret[i][j] = mb.getValue(i, j);
				
		return ret;
	}
	
	/**
	 * Creates a dense Matrix Block and copies the given double matrix into it.
	 * 
	 * @param data
	 * @return
	 */
	public static MatrixBlock convertToMatrixBlock( double[][] data )
	{
		int rows = data.length;
		int cols = (rows > 0)? data[0].length : 0;
		
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
				if( data[i][j] != 0 )
					mb.setValue(i, j, data[i][j]);
		
		return mb;
	}
	
	private static Path[] getSubDirs(String dir) throws IOException {
		FileSystem fs = FileSystem.get(new Configuration());
		ArrayList<Path> paths = new ArrayList<Path>();
		for (FileStatus cur : fs.listStatus(new Path(dir))) {
			paths.add(cur.getPath());
		}
		return paths.toArray(new Path[paths.size()]);
	}

}
