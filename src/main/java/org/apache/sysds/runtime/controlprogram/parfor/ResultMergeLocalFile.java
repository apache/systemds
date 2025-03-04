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

package org.apache.sysds.runtime.controlprogram.parfor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.LocalFileUtils;

public class ResultMergeLocalFile extends ResultMergeMatrix
{
	private static final long serialVersionUID = -6905893742840020489L;

	//NOTE: if we allow simple copies, this might result in a scattered file and many MR tasks for subsequent jobs
	public static final boolean ALLOW_COPY_CELLFILES = false;
	
	//internal comparison matrix
	private IDSequence _seq = null;
	
	public ResultMergeLocalFile( MatrixObject out, MatrixObject[] in, String outputFilename, boolean accum ) {
		super( out, in, outputFilename, accum );
		_seq = new IDSequence();
	}

	@Override
	public MatrixObject executeSerialMerge() {
		MatrixObject moNew = null; //always create new matrix object (required for nested parallelism)
		
		if( LOG.isTraceEnabled() )
		LOG.trace("ResultMerge (local, file): Execute serial merge for output "
			+_output.hashCode()+" (fname="+_output.getFileName()+")");
		
		try
		{
			//collect all relevant inputs
			ArrayList<MatrixObject> inMO = new ArrayList<>();
			for( MatrixObject in : _inputs ) {
				//check for empty inputs (no iterations executed)
				if( in !=null && in != _output ) {
					//ensure that input file resides on disk
					in.exportData();
					
					//add to merge list
					inMO.add( in );
				}
			}

			if( !inMO.isEmpty() ) {
				//ensure that outputfile (for comparison) resides on disk
				_output.exportData();
				
				//actual merge
				merge( _outputFName, _output, inMO );
				
				//create new output matrix (e.g., to prevent potential export<->read file access conflict
				moNew = createNewMatrixObject( _output, inMO );
			}
			else {
				moNew = _output; //return old matrix, to prevent copy
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}

		//LOG.trace("ResultMerge (local, file): Executed serial merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+") in "+time.stop()+"ms");
		
		return moNew;
	}
	
	@Override
	public MatrixObject executeParallelMerge(int par) {
		//graceful degradation to serial merge
		return executeSerialMerge();
	}

	private MatrixObject createNewMatrixObject(MatrixObject output, ArrayList<MatrixObject> inMO) {
		MetaDataFormat metadata = (MetaDataFormat) _output.getMetaData();
		MatrixObject moNew = new MatrixObject( _output.getValueType(), _outputFName );
		
		//create deep copy of metadata obj
		DataCharacteristics mcOld = metadata.getDataCharacteristics();
		MatrixCharacteristics mc = new MatrixCharacteristics(mcOld);
		mc.setNonZeros(_isAccum ? -1 : computeNonZeros(output, inMO));
		moNew.setMetaData(new MetaDataFormat(mc, metadata.getFileFormat()));
		
		return moNew;
	}

	private void merge( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
	{
		FileFormat fmt = ((MetaDataFormat)outMo.getMetaData()).getFileFormat();
		boolean withCompare = ( outMo.getNnz() != 0 ); //if nnz exist or unknown (-1)
		
		if( fmt == FileFormat.BINARY ) {
			if(withCompare)
				mergeBinaryBlockWithComp( fnameNew, outMo, inMO );
			else
				mergeBinaryBlockWithoutComp( fnameNew, outMo, inMO );
		}
	}

	private void mergeBinaryBlockWithoutComp( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
	{
		String fnameStaging = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		
		try {
			//delete target file if already exists
			HDFSTool.deleteFileIfExistOnHDFS(fnameNew);
			
			//Step 1) read and write blocks to staging area
			for( MatrixObject in : inMO ) {
				if( LOG.isTraceEnabled() )
					LOG.trace("ResultMerge (local, file): Merge input "+in.hashCode()+" (fname="+in.getFileName()+")");
				
				createBinaryBlockStagingFile( fnameStaging, in );
			}
	
			//Step 2) read blocks, consolidate, and write to HDFS
			createBinaryBlockResultFile(fnameStaging, null, fnameNew, (MetaDataFormat)outMo.getMetaData(), false);
		}	
		catch(Exception ex) {
			throw new DMLRuntimeException("Unable to merge binary block results.", ex);
		}
		
		LocalFileUtils.cleanupWorkingDirectory(fnameStaging);
	}

	private void mergeBinaryBlockWithComp( String fnameNew, MatrixObject outMo, ArrayList<MatrixObject> inMO ) 
	{
		String fnameStaging = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		String fnameStagingCompare = LocalFileUtils.getUniqueWorkingDir(LocalFileUtils.CATEGORY_RESULTMERGE);
		
		try
		{
			//delete target file if already exists
			HDFSTool.deleteFileIfExistOnHDFS(fnameNew);
			
			//Step 0) write compare blocks to staging area (if necessary)
			if( LOG.isTraceEnabled() )
				LOG.trace("ResultMerge (local, file): Create merge compare matrix for output "
					+outMo.hashCode()+" (fname="+outMo.getFileName()+")");
			
			createBinaryBlockStagingFile(fnameStagingCompare, outMo);
			
			//Step 1) read and write blocks to staging area
			for( MatrixObject in : inMO ) {
				if( LOG.isTraceEnabled() )
					LOG.trace("ResultMerge (local, file): Merge input "+in.hashCode()+" (fname="+in.getFileName()+")");
				
				createBinaryBlockStagingFile( fnameStaging, in );
			}
			
			//Step 2) read blocks, consolidate, and write to HDFS
			createBinaryBlockResultFile(fnameStaging, fnameStagingCompare, fnameNew, (MetaDataFormat)outMo.getMetaData(), true);
		}	
		catch(Exception ex) {
			throw new DMLRuntimeException("Unable to merge binary block results.", ex);
		}
		
		LocalFileUtils.cleanupWorkingDirectory(fnameStaging);
		LocalFileUtils.cleanupWorkingDirectory(fnameStagingCompare);
	}

	@SuppressWarnings("deprecation")
	private void createBinaryBlockStagingFile( String fnameStaging, MatrixObject mo ) 
		throws IOException
	{
		MatrixIndexes key = new MatrixIndexes(); 
		MatrixBlock value = new MatrixBlock();
		
		JobConf tmpJob = new JobConf(ConfigurationManager.getCachedJobConf());
		Path tmpPath = new Path(mo.getFileName());
		FileSystem fs = IOUtilFunctions.getFileSystem(tmpPath, tmpJob);
		
		for(Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, tmpPath)) {
			try( SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,tmpJob) ) {
				while(reader.next(key, value)) { //for each block
					String lname = key.getRowIndex()+"_"+key.getColumnIndex();
					String dir = fnameStaging+"/"+lname;
					
					if( value.getNonZeros()>0 ) { //write only non-empty blocks
						LocalFileUtils.checkAndCreateStagingDir( dir );
						LocalFileUtils.writeMatrixBlockToLocal(dir+"/"+_seq.getNextID(), value);
					}
				}
			}
		}
	}

	private void createBinaryBlockResultFile( String fnameStaging, String fnameStagingCompare, String fnameNew, MetaDataFormat metadata, boolean withCompare ) 
		throws IOException, DMLRuntimeException
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fnameNew );
		
		DataCharacteristics mc = metadata.getDataCharacteristics();
		long rlen = mc.getRows();
		long clen = mc.getCols();
		int blen = mc.getBlocksize();
		
		Writer writer = IOUtilFunctions.getSeqWriter(path, job, 1);

		try {
			MatrixIndexes indexes = new MatrixIndexes();
			for(long brow = 1; brow <= (long)Math.ceil(rlen/(double)blen); brow++) {
				for(long bcol = 1; bcol <= (long)Math.ceil(clen/(double)blen); bcol++) {
					File dir = new File(fnameStaging+"/"+brow+"_"+bcol);
					File dir2 = new File(fnameStagingCompare+"/"+brow+"_"+bcol);
					MatrixBlock mb = null;
					
					if( dir.exists() ) {
						if( withCompare && dir2.exists() ) //WITH COMPARE BLOCK
						{
							//copy only values that are different from the original
							String[] lnames2 = dir2.list();
							if( lnames2.length != 1 ) //there should be exactly 1 compare block
								throw new DMLRuntimeException("Unable to merge results because multiple compare blocks found.");
							mb = LocalFileUtils.readMatrixBlockFromLocal( dir2+"/"+lnames2[0] );
							boolean appendOnly = mb.isInSparseFormat();
							DenseBlock compare = DataConverter.convertToDenseBlock(mb, true);
							for( String lname : dir.list() ) {
								MatrixBlock tmp = LocalFileUtils.readMatrixBlockFromLocal( dir+"/"+lname );
								if( _isAccum )
									mergeWithoutComp(mb, tmp, compare, appendOnly);
								else
									mergeWithComp(mb, tmp, compare);
							}
							
							//sort sparse due to append-only
							if( appendOnly && !_isAccum )
								mb.sortSparseRows();
							
							//change sparsity if required after 
							mb.examSparsity(); 
						}
						else //WITHOUT COMPARE BLOCK
						{
							//copy all non-zeros from all workers
							boolean appendOnly = false;
							for( String lname : dir.list() ) {
								if( mb == null ) {
									mb = LocalFileUtils.readMatrixBlockFromLocal( dir+"/"+lname );
									appendOnly = mb.isInSparseFormat();
								}
								else {
									MatrixBlock tmp = LocalFileUtils.readMatrixBlockFromLocal( dir+"/"+lname );
									mergeWithoutComp(mb, tmp, null, appendOnly);
								}
							}
							
							//sort sparse due to append-only
							if( appendOnly && !_isAccum )
								mb.sortSparseRows();
							
							//change sparsity if required after 
							mb.examSparsity(); 
						}
					}
					else {
						//NOTE: whenever runtime does not need all blocks anymore, this can be removed
						int maxRow = (int)(((brow-1)*blen + blen < rlen) ? blen : rlen - (brow-1)*blen);
						int maxCol = (int)(((bcol-1)*blen + blen < clen) ? blen : clen - (bcol-1)*blen);
						mb = new MatrixBlock(maxRow, maxCol, true);
					}
					
					//mb.examSparsity(); //done on write anyway and mb not reused
					indexes.setIndexes(brow, bcol);
					writer.append(indexes, mb);
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}
}
