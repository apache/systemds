/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPFileInstructions;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map.Entry;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.Cell;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.StagingFileUtils;
import com.ibm.bi.dml.runtime.functionobjects.ParameterizedBuiltin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPOperand;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ParameterizedBuiltinCPInstruction;
import com.ibm.bi.dml.runtime.io.MatrixReader;
import com.ibm.bi.dml.runtime.io.MatrixWriter;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.util.FastStringTokenizer;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * File-based (out-of-core) realization of remove empty for robustness because there is no
 * parallel version due to data-dependent row- and column dependencies.
 * 
 */
public class ParameterizedBuiltinCPFileInstruction extends ParameterizedBuiltinCPInstruction 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ParameterizedBuiltinCPFileInstruction(Operator op, HashMap<String, String> paramsMap, CPOperand out, String istr) 
	{
		super(op, paramsMap, out, istr);
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static Instruction parseInstruction( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand( parts[parts.length-1] ); 

		// process remaining parts and build a hash map
		HashMap<String,String> paramsMap = constructParameterMap(parts);

		// determine the appropriate value function
		ValueFunction func = null;
		if ( opcode.equalsIgnoreCase("rmempty") ) {
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinCPFileInstruction(new SimpleOperator(func), paramsMap, out, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
		}
	}
	
	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String opcode = InstructionUtils.getOpCode(instString);
		
		if ( opcode.equalsIgnoreCase("rmempty") ) 
		{
			// get inputs
			MatrixObject src = (MatrixObject)ec.getVariable( params.get("target") );
			MatrixObject out = (MatrixObject)ec.getVariable( output.get_name() );
			String margin = params.get("margin");
			
			// export input matrix (if necessary)
			src.exportData();
			
			//core execution
			RemoveEmpty rm = new RemoveEmpty( margin, src, out );
			out = rm.execute();
		
			//put output
			ec.setVariable(output.get_name(), out);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}
	}

	/**
	 * Remove empty rows as a inner class in order to allow testing independent of the
	 * overall SystemML instruction framework.
	 * 
	 */
	public class RemoveEmpty
	{
		private String _margin = null;
		private MatrixObject _src = null;
		private MatrixObject _out = null;
		
		public RemoveEmpty( String margin, MatrixObject src, MatrixObject out )
		{
			_margin = margin;
			_src = src;
			_out = out;
		}
		
		/**
		 * 
		 * @return
		 * @throws DMLRuntimeException
		 */
		public MatrixObject execute() 
			throws DMLRuntimeException 
		{
			//Timing time = new Timing();
			//time.start();
			
			//initial setup
			String fnameOld = _src.getFileName();
			String fnameNew = _out.getFileName();
			InputInfo ii = ((MatrixFormatMetaData)_src.getMetaData()).getInputInfo();
			MatrixCharacteristics mc = ((MatrixFormatMetaData)_src.getMetaData()).getMatrixCharacteristics();
			
			String stagingDir = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_WORK);
			LocalFileUtils.createLocalFileIfNotExist(stagingDir);
			
			long ret = -1;
			try
			{
				boolean diagBlocks = false;
				
				//Phase 1: write file to staging 
				if( ii == InputInfo.TextCellInputInfo )
					createTextCellStagingFile( fnameOld, stagingDir );
				else if( ii == InputInfo.BinaryCellInputInfo )
					createBinaryCellStagingFile( fnameOld, stagingDir );
				else if( ii == InputInfo.BinaryBlockInputInfo )
					diagBlocks = createBinaryBlockStagingFile( fnameOld, stagingDir );
				
				//System.out.println("Executed phase 1 in "+time.stop());
				
				//Phase 2: scan empty rows/cols
				if( diagBlocks )
					ret = createKeyMappingDiag(stagingDir, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block(), ii);
				else
					ret = createKeyMapping(stagingDir, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block(), ii);
				
				//System.out.println("Executed phase 2 in "+time.stop());
				
				//Phase 3: create output files
				MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
				if(   ii == InputInfo.TextCellInputInfo 
				   || ii == InputInfo.BinaryCellInputInfo )
				{
					createCellResultFile( fnameNew, stagingDir, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block(), ii );
				}
				else if( ii == InputInfo.BinaryBlockInputInfo )
				{
					if( diagBlocks )
						createBlockResultFileDiag( fnameNew, stagingDir, mc.get_rows(), mc.get_cols(), ret, mc.getNonZeros(), mc.get_rows_per_block(), mc.get_cols_per_block(), ii );
					else
						createBlockResultFile( fnameNew, stagingDir, mc.get_rows(), mc.get_cols(), ret, mc.getNonZeros(), mc.get_rows_per_block(), mc.get_cols_per_block(), ii );
				}
				
				//System.out.println("Executed phase 3 in "+time.stop());
			}
			catch( IOException ioe )
			{
				throw new DMLRuntimeException( ioe );
			}
			
			//final cleanup
			LocalFileUtils.cleanupWorkingDirectory(stagingDir);
			
			//create and return new output object
			if( _margin.equals("rows") )
				return createNewOutputObject(_src, _out, ret, mc.get_cols());
			else
				return createNewOutputObject(_src, _out, mc.get_rows(), ret );
		}
		
		/**
		 * 
		 * @param src
		 * @param out
		 * @param rows
		 * @param cols
		 * @return
		 */
		private MatrixObject createNewOutputObject( MatrixObject src, MatrixObject out, long rows, long cols )
		{
			String varName = out.getVarName();
			String fName = out.getFileName();
			ValueType vt = src.getValueType();
			MatrixFormatMetaData metadata = (MatrixFormatMetaData) src.getMetaData();
			
			MatrixObject moNew = new MatrixObject( vt, fName );
			moNew.setVarName( varName );
			moNew.setDataType( DataType.MATRIX );
			
			//create deep copy of metadata obj
			MatrixCharacteristics mcOld = metadata.getMatrixCharacteristics();
			OutputInfo oiOld = metadata.getOutputInfo();
			InputInfo iiOld = metadata.getInputInfo();
			MatrixCharacteristics mc = new MatrixCharacteristics( rows, cols, mcOld.get_rows_per_block(),
					                                              mcOld.get_cols_per_block(), mcOld.getNonZeros());
			MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,oiOld,iiOld);
			moNew.setMetaData( meta );

			return moNew;
		}

		/**
		 * 
		 * @param fnameOld
		 * @param stagingDir
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		public void createTextCellStagingFile( String fnameOld, String stagingDir ) 
			throws IOException, DMLRuntimeException
		{	
			//prepare input
			JobConf job = new JobConf();	
			Path path = new Path(fnameOld);
			FileSystem fs = FileSystem.get(job);
			if( !fs.exists(path) )	
				throw new IOException("File "+fnameOld+" does not exist on HDFS.");
			FileInputFormat.addInputPath(job, path); 
			TextInputFormat informat = new TextInputFormat();
			informat.configure(job);
			InputSplit[] splits = informat.getSplits(job, 1);
		
			LinkedList<Cell> buffer = new LinkedList<Cell>();
			
			LongWritable key = new LongWritable();
			Text value = new Text();
			FastStringTokenizer st = new FastStringTokenizer(' ');		
			
			for(InputSplit split: splits)
			{
				RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);				
				try
				{
					while( reader.next(key, value) )
					{
						st.reset( value.toString() ); //reset tokenizer
						long row = st.nextLong();
						long col = st.nextLong();
						double lvalue = st.nextDouble();
						
						buffer.add(new Cell(row,col,lvalue));
						
						if( buffer.size() > StagingFileUtils.CELL_BUFFER_SIZE )
						{
							appendCellBufferToStagingArea(stagingDir, buffer, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
							buffer.clear();
						}
					}
					
					if( buffer.size() > 0 )
					{
						appendCellBufferToStagingArea(stagingDir, buffer, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
						buffer.clear();
					}
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}
		}		

		/**
		 * 
		 * @param fnameOld
		 * @param stagingDir
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		@SuppressWarnings("deprecation")
		public void createBinaryCellStagingFile( String fnameOld, String stagingDir ) 
			throws IOException, DMLRuntimeException
		{
			//prepare input
			JobConf job = new JobConf();	
			Path path = new Path(fnameOld);
			FileSystem fs = FileSystem.get(job);
			if( !fs.exists(path) )	
				throw new IOException("File "+fnameOld+" does not exist on HDFS.");
			
			LinkedList<Cell> buffer = new LinkedList<Cell>();
			
			MatrixIndexes key = new MatrixIndexes();
			MatrixCell value = new MatrixCell();

			for(Path lpath: MatrixReader.getSequenceFilePaths(fs, path))
			{
				SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
				try
				{
					while(reader.next(key, value))
					{
						long row = (int)key.getRowIndex();
						long col = (int)key.getColumnIndex();
						double lvalue = value.getValue();
						
						buffer.add(new Cell(row,col,lvalue));
						
						if( buffer.size() > StagingFileUtils.CELL_BUFFER_SIZE )
						{
							appendCellBufferToStagingArea(stagingDir, buffer, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
							buffer.clear();
						}
					}
					
					if( buffer.size() > 0 )
					{
						appendCellBufferToStagingArea(stagingDir, buffer, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
						buffer.clear();
					}
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}
		}

		/**
		 * Creates a binary block staging file and returns if the input matrix is a diag,
		 * because diag is the primary usecase and there is lots of optimization potential.
		 * 
		 * @param fnameOld
		 * @param stagingDir
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		@SuppressWarnings("deprecation")
		public boolean createBinaryBlockStagingFile( String fnameOld, String stagingDir ) 
			throws IOException, DMLRuntimeException
		{
			//prepare input
			JobConf job = new JobConf();	
			Path path = new Path(fnameOld);
			FileSystem fs = FileSystem.get(job);
			if( !fs.exists(path) )	
				throw new IOException("File "+fnameOld+" does not exist on HDFS.");
			
			MatrixIndexes key = new MatrixIndexes(); 
			MatrixBlock value = new MatrixBlock();
			boolean diagBlocks = true;
			
			for(Path lpath : MatrixReader.getSequenceFilePaths(fs, path))
			{
				SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
				
				try
				{
					while( reader.next(key, value) )
					{
						if( !value.isEmptyBlock() ) //skip empty blocks (important for diag)
						{
							String fname = stagingDir +"/"+key.getRowIndex()+"_"+key.getColumnIndex();
							LocalFileUtils.writeMatrixBlockToLocal(fname, value);							
							diagBlocks &= (key.getRowIndex()==key.getColumnIndex());
						}
					}	
				}
				finally
				{
					if( reader != null )
						reader.close();
				}
			}
			
			return diagBlocks;
		}
		
		/**
		 * 
		 * @param dir
		 * @param buffer
		 * @param brlen
		 * @param bclen
		 * @throws DMLRuntimeException
		 * @throws IOException
		 */
		private void appendCellBufferToStagingArea( String dir, LinkedList<Cell> buffer, int brlen, int bclen ) 
			throws DMLRuntimeException, IOException
		{
			HashMap<String,LinkedList<Cell>> sortedBuffer = new HashMap<String,LinkedList<Cell>>();
			
			//sort cells in buffer wrt key
			String key = null;
			for( Cell c : buffer )
			{
				key = (c.getRow()/brlen+1) +"_"+(c.getCol()/bclen+1);
				
				if( !sortedBuffer.containsKey(key) )
					sortedBuffer.put(key, new LinkedList<Cell>());
				sortedBuffer.get(key).addLast(c);
			}	
			
			//write lists of cells to local files
			for( Entry<String,LinkedList<Cell>> e : sortedBuffer.entrySet() )
			{
				
				String pfname = dir + "/" + e.getKey();
				StagingFileUtils.writeCellListToLocal(pfname, e.getValue());
			}
		}	

		/**
		 * 
		 * @param stagingDir
		 * @param rlen
		 * @param clen
		 * @param brlen
		 * @param bclen
		 * @param ii
		 * @return
		 * @throws FileNotFoundException
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		private long createKeyMapping( String stagingDir, long rlen, long clen, int brlen, int bclen, InputInfo ii) 
			throws FileNotFoundException, IOException, DMLRuntimeException 
		{
			String metaOut = stagingDir+"/meta";
			
			long len = 0;
			long lastKey = 0;
			
			if(_margin.equals("rows"))
			{
				for(int blockRow = 0; blockRow < (int)Math.ceil(rlen/(double)brlen); blockRow++)
				{	
					boolean[] flags = new boolean[brlen];
					for( int k=0; k<brlen; k++ )
						flags[k] = true;
					
					//scan for empty rows
					for(int blockCol = 0; blockCol < (int)Math.ceil(clen/(double)bclen); blockCol++)
					{
						String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockCol+1);
						if( ii == InputInfo.BinaryBlockInputInfo ){
							if( !LocalFileUtils.isExisting(fname) )
								continue;
							MatrixBlock buffer = LocalFileUtils.readMatrixBlockFromLocal(fname);
							for( int i=0; i<buffer.getNumRows(); i++ )
								for( int j=0; j<buffer.getNumColumns(); j++ )
								{
									double lvalue = buffer.quickGetValue(i, j);
									if( lvalue != 0 )
										flags[ i ] = false;
								}
						}
						else{
							LinkedList<Cell> buffer = StagingFileUtils.readCellListFromLocal(fname);
							for( Cell c : buffer )
								flags[ (int)c.getRow()-blockRow*brlen-1 ] = false;
						}
					} 
			
					//create and append key mapping
					LinkedList<long[]> keyMapping = new LinkedList<long[]>();
					for( int i = 0; i<flags.length; i++ )
						if( !flags[i] )
							keyMapping.add(new long[]{blockRow*brlen+i, lastKey++});
					len += keyMapping.size();
					StagingFileUtils.writeKeyMappingToLocal(metaOut, keyMapping.toArray(new long[0][0]));
				}
			}
			else
			{
				for(int blockCol = 0; blockCol < (int)Math.ceil(clen/(double)bclen); blockCol++)
				{	
					boolean[] flags = new boolean[bclen];
					for( int k=0; k<bclen; k++ )
						flags[k] = true;
					
					//scan for empty rows
					for(int blockRow = 0; blockRow < (int)Math.ceil(rlen/(double)brlen); blockRow++)
					{
						String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockCol+1);
						if( ii == InputInfo.BinaryBlockInputInfo ){
							if( !LocalFileUtils.isExisting(fname) )
								continue;
							MatrixBlock buffer = LocalFileUtils.readMatrixBlockFromLocal(fname);
							for( int i=0; i<buffer.getNumRows(); i++ )
								for( int j=0; j<buffer.getNumColumns(); j++ )
								{
									double lvalue = buffer.quickGetValue(i, j);
									if( lvalue != 0 )
										flags[ j ] = false;
								}
						}
						else{
							LinkedList<Cell> buffer = StagingFileUtils.readCellListFromLocal(fname);
							for( Cell c : buffer )
								flags[ (int)c.getCol()-blockCol*bclen-1 ] = false;
						}
					} 
			
					//create and append key mapping
					LinkedList<long[]> keyMapping = new LinkedList<long[]>();
					for( int i = 0; i<flags.length; i++ )
						if( !flags[i] )
							keyMapping.add(new long[]{blockCol*bclen+i, lastKey++});
					len += keyMapping.size();
					StagingFileUtils.writeKeyMappingToLocal(metaOut, keyMapping.toArray(new long[0][0]));
				}
			}
			
			//final validation (matrices with dimensions 0x0 not allowed)
			if( len <= 0 )
				throw new DMLRuntimeException("Matrices with dimensions [0,0] not supported.");
			
			return len;
		}

		/**
		 * 
		 * @param stagingDir
		 * @param rlen
		 * @param clen
		 * @param brlen
		 * @param bclen
		 * @param ii
		 * @return
		 * @throws FileNotFoundException
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		private long createKeyMappingDiag( String stagingDir, long rlen, long clen, int brlen, int bclen, InputInfo ii) 
			throws FileNotFoundException, IOException, DMLRuntimeException 
		{
			String metaOut = stagingDir+"/meta";
			
			long len = 0;
			long lastKey = 0;
			
			if(_margin.equals("rows"))
			{
				for(int blockRow = 0; blockRow < (int)Math.ceil(rlen/(double)brlen); blockRow++)
				{	
					boolean[] flags = new boolean[brlen];
					for( int k=0; k<brlen; k++ )
						flags[k] = true;
					
					//scan for empty rows
					String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockRow+1);
					if( ii == InputInfo.BinaryBlockInputInfo ){
						if( !LocalFileUtils.isExisting(fname) )
							continue;
						MatrixBlock buffer = LocalFileUtils.readMatrixBlockFromLocal(fname);
						for( int i=0; i<buffer.getNumRows(); i++ )
							for( int j=0; j<buffer.getNumColumns(); j++ )
							{
								double lvalue = buffer.quickGetValue(i, j);
								if( lvalue != 0 )
									flags[ i ] = false;
							}
					}
					else{
						LinkedList<Cell> buffer = StagingFileUtils.readCellListFromLocal(fname);
						for( Cell c : buffer )
							flags[ (int)c.getRow()-blockRow*brlen-1 ] = false;
					}
					 
			
					//create and append key mapping
					LinkedList<long[]> keyMapping = new LinkedList<long[]>();
					for( int i = 0; i<flags.length; i++ )
						if( !flags[i] )
							keyMapping.add(new long[]{blockRow*brlen+i, lastKey++});
					len += keyMapping.size();
					StagingFileUtils.writeKeyMappingToLocal(metaOut, keyMapping.toArray(new long[0][0]));
				}
			}
			else
			{
				for(int blockCol = 0; blockCol < (int)Math.ceil(clen/(double)bclen); blockCol++)
				{	
					boolean[] flags = new boolean[bclen];
					for( int k=0; k<bclen; k++ )
						flags[k] = true;
					
					//scan for empty rows
					String fname = stagingDir+"/"+(blockCol+1)+"_"+(blockCol+1);
					if( ii == InputInfo.BinaryBlockInputInfo ){
						if( !LocalFileUtils.isExisting(fname) )
							continue;
						MatrixBlock buffer = LocalFileUtils.readMatrixBlockFromLocal(fname);
						for( int i=0; i<buffer.getNumRows(); i++ )
							for( int j=0; j<buffer.getNumColumns(); j++ )
							{
								double lvalue = buffer.quickGetValue(i, j);
								if( lvalue != 0 )
									flags[ j ] = false;
							}
					}
					else{
						LinkedList<Cell> buffer = StagingFileUtils.readCellListFromLocal(fname);
						for( Cell c : buffer )
							flags[ (int)c.getCol()-blockCol*bclen-1 ] = false;
					}
					 
			
					//create and append key mapping
					LinkedList<long[]> keyMapping = new LinkedList<long[]>();
					for( int i = 0; i<flags.length; i++ )
						if( !flags[i] )
							keyMapping.add(new long[]{blockCol*bclen+i, lastKey++});
					len += keyMapping.size();
					StagingFileUtils.writeKeyMappingToLocal(metaOut, keyMapping.toArray(new long[0][0]));
				}
			}
			
			//final validation (matrices with dimensions 0x0 not allowed)
			if( len <= 0 )
				throw new DMLRuntimeException("Matrices with dimensions [0,0] not supported.");
			
			return len;
		}
		
		/**
		 * 
		 * @param fnameNew
		 * @param stagingDir
		 * @param rlen
		 * @param clen
		 * @param brlen
		 * @param bclen
		 * @param ii
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		@SuppressWarnings("deprecation")
		public void createCellResultFile( String fnameNew, String stagingDir, long rlen, long clen, int brlen, int bclen, InputInfo ii ) 
			throws IOException, DMLRuntimeException
		{
			//prepare input
			JobConf job = new JobConf();	
			Path path = new Path(fnameNew);
			FileSystem fs = FileSystem.get(job);
			String metaOut = stagingDir+"/meta";

			//prepare output
			BufferedWriter twriter = null;			
			SequenceFile.Writer bwriter = null; 
			if( ii == InputInfo.TextCellInputInfo )
				twriter = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));	
			else if( ii == InputInfo.BinaryCellInputInfo )
				bwriter = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixCell.class);
			else
				throw new DMLRuntimeException("Unsupported cell input info: "+InputInfo.inputInfoToString(ii));
			
			StringBuilder sb = new StringBuilder();
			MatrixIndexes key = new MatrixIndexes();
			MatrixCell value = new MatrixCell();

			HashMap<Integer,HashMap<Long,Long>> keyMap = new HashMap<Integer, HashMap<Long,Long>>();
			BufferedReader fkeyMap = StagingFileUtils.openKeyMap(metaOut);
			try
			{
				if( _margin.equals("rows") )
				{
					for(int blockRow = 0; blockRow < (int)Math.ceil(rlen/(double)brlen); blockRow++)
					{
						StagingFileUtils.nextKeyMap(fkeyMap, keyMap, blockRow, brlen);		
						for(int blockCol = 0; blockCol < (int)Math.ceil(clen/(double)bclen); blockCol++)
						{
							String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockCol+1);
							LinkedList<Cell> buffer = StagingFileUtils.readCellListFromLocal(fname);
							if( ii == InputInfo.TextCellInputInfo )
								for( Cell c : buffer )
								{
									sb.append(keyMap.get(blockRow).get(c.getRow()-1)+1);
									sb.append(' ');
									sb.append(c.getCol());
									sb.append(' ');
									sb.append(c.getValue());
									sb.append('\n');
									twriter.write( sb.toString() );	
									sb.setLength(0);
								}
							else if( ii == InputInfo.BinaryCellInputInfo )
								for( Cell c : buffer )
								{
									key.setIndexes(keyMap.get(blockRow).get(c.getRow()-1)+1, c.getCol());
									value.setValue(c.getValue());
									bwriter.append(key, value);	
								}
						}
						keyMap.remove(blockRow);
					}
				}
				else
				{
					for(int blockCol = 0; blockCol < (int)Math.ceil(clen/(double)bclen); blockCol++)
					{
						StagingFileUtils.nextKeyMap(fkeyMap, keyMap, blockCol, bclen);		
						for(int blockRow = 0; blockRow < (int)Math.ceil(rlen/(double)brlen); blockRow++)
						{
							String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockCol+1);
							LinkedList<Cell> buffer = StagingFileUtils.readCellListFromLocal(fname);
							if( ii == InputInfo.TextCellInputInfo )
								for( Cell c : buffer )
								{
									sb.append(c.getRow());
									sb.append(' ');
									sb.append(keyMap.get(blockCol).get(c.getCol()-1)+1);
									sb.append(' ');
									sb.append(c.getValue());
									sb.append('\n');
									twriter.write( sb.toString() );	
									sb.setLength(0);
								}
							else if( ii == InputInfo.BinaryCellInputInfo )
								for( Cell c : buffer )
								{
									key.setIndexes(c.getRow(), keyMap.get(blockCol).get(c.getCol()-1)+1);
									value.setValue(c.getValue());
									bwriter.append(key, value);	
								}
						}
						keyMap.remove(blockCol);
					}
				}

				//Note: no need to handle empty result
			}
			finally
			{
				if( twriter != null )
					twriter.close();	
				if( bwriter != null )
					bwriter.close();	
			}
		}
	
		/**
		 * 
		 * @param fnameNew
		 * @param stagingDir
		 * @param rlen
		 * @param clen
		 * @param newlen
		 * @param nnz
		 * @param brlen
		 * @param bclen
		 * @param ii
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		@SuppressWarnings("deprecation")
		public void createBlockResultFile( String fnameNew, String stagingDir, long rlen, long clen, long newlen, long nnz, int brlen, int bclen, InputInfo ii ) 
			throws IOException, DMLRuntimeException
		{
			//prepare input
			JobConf job = new JobConf();	
			Path path = new Path(fnameNew);
			FileSystem fs = FileSystem.get(job);
			String metaOut = stagingDir+"/meta";
	
			//prepare output
			SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class);
			
			MatrixIndexes key = new MatrixIndexes(); 
			
			try
			{
				if( _margin.equals("rows") ) 
				{
					MatrixBlock[] blocks = MatrixWriter.createMatrixBlocksForReuse(newlen, clen, brlen, bclen, 
							                     MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz), nnz);  
					
					for(int blockCol = 0; blockCol < (int)Math.ceil(clen/(double)bclen); blockCol++)
					{
						HashMap<Integer,HashMap<Long,Long>> keyMap = new HashMap<Integer, HashMap<Long,Long>>();
						BufferedReader fkeyMap = StagingFileUtils.openKeyMap(metaOut);
						int maxCol = (int)((blockCol*bclen + bclen < clen) ? bclen : clen - blockCol*bclen);
						
						int blockRowOut = 0;
						int currentSize = -1;
						while( (currentSize = StagingFileUtils.nextSizedKeyMap(fkeyMap, keyMap, brlen, brlen)) > 0  )
						{
							int maxRow = currentSize;
							
							//get reuse matrix block
							MatrixBlock block = MatrixWriter.getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
							block.reset(maxRow, maxCol);
							
							int rowPos = 0;
							int blockRow = Collections.min(keyMap.keySet());
							
							for( ; blockRow < (int)Math.ceil(rlen/(double)brlen) && rowPos<brlen ; blockRow++)
							{
								if( keyMap.containsKey(blockRow) )
								{
									String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockCol+1);
									
									if( LocalFileUtils.isExisting(fname) ) 
									{	
										MatrixBlock tmp = LocalFileUtils.readMatrixBlockFromLocal(fname);
										
										HashMap<Long,Long> lkeyMap = keyMap.get(blockRow);
										long row_offset = blockRow*brlen;
										for( int i=0; i<tmp.getNumRows(); i++ )
											if( lkeyMap.containsKey(row_offset+i) ) {	
												//copy row
												for( int j=0; j<tmp.getNumColumns(); j++ ) {
													double lvalue = tmp.quickGetValue(i, j);
													if( lvalue != 0 )
														block.quickSetValue(rowPos, j, lvalue);
												}
												rowPos++;
											}
									}
									else
									{
										HashMap<Long,Long> lkeyMap = keyMap.get(blockRow);
										rowPos+=lkeyMap.size();
									}
								}				
								keyMap.remove(blockRow);
							}
							
							key.setIndexes(blockRowOut+1, blockCol+1);
							writer.append(key, block);
							blockRowOut++;
						}
						
						if( fkeyMap != null )
							StagingFileUtils.closeKeyMap(fkeyMap);
					}
				}
				else
				{
					MatrixBlock[] blocks = MatrixWriter.createMatrixBlocksForReuse(rlen, newlen, brlen, bclen, 
							                    MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz), nnz);  
					
					for(int blockRow = 0; blockRow < (int)Math.ceil(rlen/(double)brlen); blockRow++)
					{
						HashMap<Integer,HashMap<Long,Long>> keyMap = new HashMap<Integer, HashMap<Long,Long>>();
						BufferedReader fkeyMap = StagingFileUtils.openKeyMap(metaOut);
						int maxRow = (int)((blockRow*brlen + brlen < rlen) ? brlen : rlen - blockRow*brlen);
						
						int blockColOut = 0;
						int currentSize = -1;
						while( (currentSize = StagingFileUtils.nextSizedKeyMap(fkeyMap, keyMap, bclen, bclen)) > 0  )
						{
							int maxCol = currentSize;
							
							//get reuse matrix block
							MatrixBlock block = MatrixWriter.getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
							block.reset(maxRow, maxCol);
							int colPos = 0;
							
							int blockCol = Collections.min(keyMap.keySet());
							for( ; blockCol < (int)Math.ceil(clen/(double)bclen) && colPos<bclen ; blockCol++)
							{
								if( keyMap.containsKey(blockCol) )
								{
									String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockCol+1);
									
									if( LocalFileUtils.isExisting(fname) ) 
									{
										MatrixBlock tmp = LocalFileUtils.readMatrixBlockFromLocal(fname);
										
										HashMap<Long,Long> lkeyMap = keyMap.get(blockCol);
										long col_offset = blockCol*bclen;
										for( int j=0; j<tmp.getNumColumns(); j++ )
											if( lkeyMap.containsKey(col_offset+j) ) {	
												//copy column
												for( int i=0; i<tmp.getNumRows(); i++ ){
													double lvalue = tmp.quickGetValue(i, j);
													if( lvalue != 0 )
														block.quickSetValue(i, colPos, lvalue);
												}
												colPos++;
											}
									}
									else
									{
										HashMap<Long,Long> lkeyMap = keyMap.get(blockCol);
										colPos+=lkeyMap.size();
									}
								}							
								keyMap.remove(blockCol);
							}
							
							key.setIndexes(blockRow+1, blockColOut+1);
							writer.append(key, block);
							blockColOut++;
						}
						
						if( fkeyMap != null )
							StagingFileUtils.closeKeyMap(fkeyMap);
					}
				}
				
				//Note: no handling of empty matrices necessary
			}
			finally
			{
				if( writer != null )
					writer.close();
			}
		}
		
		/**
		 * 
		 * @param fnameNew
		 * @param stagingDir
		 * @param rlen
		 * @param clen
		 * @param newlen
		 * @param nnz
		 * @param brlen
		 * @param bclen
		 * @param ii
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		@SuppressWarnings("deprecation")
		public void createBlockResultFileDiag( String fnameNew, String stagingDir, long rlen, long clen, long newlen, long nnz, int brlen, int bclen, InputInfo ii ) 
			throws IOException, DMLRuntimeException
		{
			//prepare input
			JobConf job = new JobConf();	
			Path path = new Path(fnameNew);
			FileSystem fs = FileSystem.get(job);
			String metaOut = stagingDir+"/meta";
	
			//prepare output
			SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class);
			MatrixIndexes key = new MatrixIndexes(); 
			HashSet<Long> writtenBlocks = new HashSet<Long>();
			
			try
			{
				if( _margin.equals("rows") ) 
				{
					MatrixBlock[] blocks = MatrixWriter.createMatrixBlocksForReuse(newlen, clen, brlen, bclen, 
							                       MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz), nnz);  
					HashMap<Integer,HashMap<Long,Long>> keyMap = new HashMap<Integer, HashMap<Long,Long>>();
					BufferedReader fkeyMap = StagingFileUtils.openKeyMap(metaOut);
					int currentSize = -1;
					int blockRowOut = 0;
					
					while( (currentSize = StagingFileUtils.nextSizedKeyMap(fkeyMap, keyMap, brlen, brlen)) > 0  )
					{
						int rowPos = 0;
						int blockRow = Collections.min(keyMap.keySet()); 
						int maxRow = currentSize;
						for( ; blockRow < (int)Math.ceil(rlen/(double)brlen); blockRow++)
						{
							int blockCol = blockRow; // for diag known to be equivalent
							int maxCol = (int)((blockCol*bclen + bclen < clen) ? bclen : clen - blockCol*bclen);
							
							//get reuse matrix block
							MatrixBlock block = MatrixWriter.getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
							block.reset(maxRow, maxCol);
							
							if( keyMap.containsKey(blockRow) )
							{
								String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockCol+1);
								MatrixBlock tmp = LocalFileUtils.readMatrixBlockFromLocal(fname);
								
								HashMap<Long,Long> lkeyMap = keyMap.get(blockRow);
								long row_offset = blockRow*brlen;
								for( int i=0; i<tmp.getNumRows(); i++ )
									if( lkeyMap.containsKey(row_offset+i) ) {	
										//copy row
										for( int j=0; j<tmp.getNumColumns(); j++ ) {
											double lvalue = tmp.quickGetValue(i, j);
											if( lvalue != 0 )
												block.quickSetValue(rowPos, j, lvalue);
										}
										rowPos++;
									}
							}
							
							//output current block (by def of diagBlocks, no additional rows)
							key.setIndexes(blockRowOut+1, blockCol+1);
							writer.append(key, block);
							writtenBlocks.add(IDHandler.concatIntIDsToLong(blockRowOut+1, blockCol+1));
							
							//finished block
							if( rowPos == maxRow )
							{
								keyMap.remove(blockRow); 	
								blockRowOut++;
								break;
							}
						}
					}
					if( fkeyMap != null )
						StagingFileUtils.closeKeyMap(fkeyMap);
					
				}
				else //cols
				{
					MatrixBlock[] blocks = MatrixWriter.createMatrixBlocksForReuse(rlen, newlen, brlen, bclen, 
							                     MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz), nnz);  
					HashMap<Integer,HashMap<Long,Long>> keyMap = new HashMap<Integer, HashMap<Long,Long>>();
					BufferedReader fkeyMap = StagingFileUtils.openKeyMap(metaOut);
					int currentSize = -1;
					int blockColOut = 0;
					
					while( (currentSize = StagingFileUtils.nextSizedKeyMap(fkeyMap, keyMap, bclen, bclen)) > 0  )
					{
						int colPos = 0;
						int blockCol = Collections.min(keyMap.keySet()); 
						int maxCol = currentSize;
						for( ; blockCol < (int)Math.ceil(clen/(double)bclen); blockCol++)
						{
							int blockRow = blockCol; // for diag known to be equivalent
							int maxRow = (int)((blockRow*brlen + brlen < rlen) ? brlen : rlen - blockRow*brlen);
							
							//get reuse matrix block
							MatrixBlock block = MatrixWriter.getMatrixBlockForReuse(blocks, maxRow, maxCol, brlen, bclen);
							block.reset(maxRow, maxCol);
						
							if( keyMap.containsKey(blockCol) )
							{
								String fname = stagingDir+"/"+(blockRow+1)+"_"+(blockCol+1);
								MatrixBlock tmp = LocalFileUtils.readMatrixBlockFromLocal(fname);
								
								HashMap<Long,Long> lkeyMap = keyMap.get(blockCol);
								long col_offset = blockCol*bclen;
								for( int j=0; j<tmp.getNumColumns(); j++ )
									if( lkeyMap.containsKey(col_offset+j) ) {	
										//copy column
										for( int i=0; i<tmp.getNumRows(); i++ ){
											double lvalue = tmp.quickGetValue(i, j);
											if( lvalue != 0 )
												block.quickSetValue(i, colPos, lvalue);
										}
										colPos++;
									}
							}
								
							//output current block (by def of diagBlocks, no additional cols)
							key.setIndexes(blockRow+1, blockColOut+1);
							writer.append(key, block);
							writtenBlocks.add(IDHandler.concatIntIDsToLong(blockRow+1, blockColOut+1));
							
							//finished block
							if( colPos == maxCol )
							{
								keyMap.remove(blockCol); 	
								blockColOut++;
								break;
							}
						}
					}
					if( fkeyMap != null )
						StagingFileUtils.closeKeyMap(fkeyMap);
				}
				
				//write remaining empty blocks
				MatrixBlock empty = new MatrixBlock(1,1,true);
				long rows = _margin.equals("rows") ? newlen : rlen;
				long cols = _margin.equals("cols") ? newlen : clen;
				int countBlk1 = (int)Math.ceil(rows/(double)brlen)*(int)Math.ceil(cols/(double)bclen);
				int countBlk2 = writtenBlocks.size();
				for( int i=0; i<(int)Math.ceil(rows/(double)brlen); i++)
					for(int j=0; j<(int)Math.ceil(cols/(double)bclen); j++ )
						if( !writtenBlocks.contains(IDHandler.concatIntIDsToLong(i+1, j+1)) )
						{
							int maxRow = (int)((i*brlen + brlen < rows) ? brlen : rows - i*brlen);
							int maxCol = (int)((j*bclen + bclen < cols) ? bclen : cols - j*bclen);
							empty.reset(maxRow, maxCol);
							key.setIndexes(i+1, j+1);
							writer.append(key, empty);
							countBlk2++;
						}
				
				if( countBlk1 != countBlk2 )
					throw new DMLRuntimeException("Wrong number of written result blocks: "+countBlk1+" vs "+countBlk2+".");
			}
			finally
			{
				if( writer != null )
					writer.close();
			}
		}
	}
}
