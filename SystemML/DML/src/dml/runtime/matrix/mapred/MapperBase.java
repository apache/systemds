package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.instructions.MRInstructions.MRInstruction;
import dml.runtime.instructions.MRInstructions.RandInstruction;
import dml.runtime.instructions.MRInstructions.ReblockInstruction;
import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.Pair;

import dml.runtime.matrix.io.TaggedMatrixValue;
import dml.runtime.matrix.io.TaggedPartialBlock;
import dml.runtime.util.UtilFunctions;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public abstract class MapperBase extends MRBaseForCommonInstructions{
	protected static final Log LOG = LogFactory.getLog(MapperBase.class);
	
	//the indexes that this particular input matrix file represents
	protected Vector<Byte> representativeMatrixes=null;
	
	//the dimension for all the representative matrices 
	//(they are all the same, since coming from the same files)
	protected long[] rlens=null;
	protected long[] clens=null;
	
	//the block sizes for the representative matrices
	protected int[] brlens=null;
	protected int[] bclens=null;
	
	//upper boundaries to check
	protected long[] rbounds=null;
	protected long[] cbounds=null;
	
	//rand instructions that need to be performed in mapper
	protected Vector<RandInstruction> rand_instructions=new Vector<RandInstruction>();
	
	//instructions that need to be performed in mapper
	protected Vector<Vector<MRInstruction>> mapper_instructions=new Vector<Vector<MRInstruction>>();
	
	//block instructions that need to be performed in part by mapper
	protected Vector<Vector<ReblockInstruction>> reblock_instructions=new Vector<Vector<ReblockInstruction>>();
	
	//the indexes of the matrices that needed to be outputted
	protected Vector<Vector<Byte>> outputIndexes=new Vector<Vector<Byte>>();
	
	//converter to convert the input record into indexes and matrix value (can be a cell or a block)
	protected Converter inputConverter=null;
	
	//a counter to measure the time spent in a mapper
	protected static enum Counters {MAP_TIME };
	
	protected void commonMap(Writable rawKey, Writable rawValue, OutputCollector<Writable, Writable> out, 
	Reporter reporter) throws IOException 
	{
		long start=System.currentTimeMillis();
		
		//System.out.println("read in Mapper: "+rawKey+": "+rawValue);
		
		//for each represenattive matrix, read the record and apply instructions
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			//convert the record into the right format for the representatice matrix
			inputConverter.setBlockSize(brlens[i], bclens[i]);
			inputConverter.convert(rawKey, rawValue);
			
			byte thisMatrix=representativeMatrixes.get(i);
			
			//apply unary instructions on the converted indexes and values
			while(inputConverter.hasNext())
			{
				Pair<MatrixIndexes, MatrixValue> pair=inputConverter.next();
			//	System.out.println("convert to: "+pair);
				MatrixIndexes indexes=pair.getKey();
				
				MatrixValue value=pair.getValue();
				
			//	System.out.println("after converter: "+indexes+" -- "+value);
				
				checkValidIndex(indexes, i);
				
				//put the input in the cache
				cachedValues.reset();
				cachedValues.set(thisMatrix, indexes, value);
				
				//special operations for individual mapp type
				specialOperationsForActualMap(i, out, reporter);
			}
		}
		reporter.incrCounter(Counters.MAP_TIME, System.currentTimeMillis()-start);
	}
	
	protected abstract void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)throws IOException;

	protected void checkValidIndex(MatrixIndexes indexes, int rep) throws IOException
	{
		if(indexes.getRowIndex()<=0 || indexes.getColumnIndex()<=0 
		|| indexes.getRowIndex()>rbounds[rep] || indexes.getColumnIndex()>cbounds[rep]){
			
			//System.out.println("key: "+indexes+" is out of range: [1, "+rbounds[rep]+"] and [1, "+cbounds[rep]+"]!");
			throw new IOException("key: "+indexes+" is out of range: [1, "+rbounds[rep]+"] and [1, "+cbounds[rep]+"]!");
		}
	}
	public void configure(JobConf job)
	{
		super.configure(job);
		
		//get the indexes that this matrix file represents, 
		//since one matrix file can occur multiple times in a statement
		try {
			representativeMatrixes=MRJobConfiguration.getInputMatrixIndexesInMapper(job);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
		//get input converter information
		inputConverter=MRJobConfiguration.getInputConverter(job, representativeMatrixes.get(0));
		
		RandInstruction[] allRandIns;
		MRInstruction[] allMapperIns;
		ReblockInstruction[] allReblockIns;
		try {
			allRandIns = MRJobConfiguration.getRandInstructions(job);
			
			//parse the instructions on the matrices that this file represent
			allMapperIns=MRJobConfiguration.getInstructionsInMapper(job);
			
			//parse the reblock instructions on the matrices that this file represent
			allReblockIns=MRJobConfiguration.getReblockInstructions(job);
			
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		
		//get all the output indexes
		byte[] outputs=MRJobConfiguration.getOutputIndexesInMapper(job);
		
		//get the dimension of all the representative matrices
		rlens=new long[representativeMatrixes.size()];
		clens=new long[representativeMatrixes.size()];
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			rlens[i]=MRJobConfiguration.getNumRows(job, representativeMatrixes.get(i));
			clens[i]=MRJobConfiguration.getNumColumns(job, representativeMatrixes.get(i));
		//	System.out.println("get dimension for "+representativeMatrixes.get(i)+": "+rlens[i]+", "+clens[i]);
		}
		
		//get the block sizes of the representative matrices
		brlens=new int[representativeMatrixes.size()];
		bclens=new int[representativeMatrixes.size()];
		
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			brlens[i]=MRJobConfiguration.getNumRowsPerBlock(job, representativeMatrixes.get(i));
			bclens[i]=MRJobConfiguration.getNumColumnsPerBlock(job, representativeMatrixes.get(i));
		//	System.out.println("get blocksize for "+representativeMatrixes.get(i)+": "+brlens[i]+", "+bclens[i]);
		}
		
		rbounds=new long[representativeMatrixes.size()];
		cbounds=new long[representativeMatrixes.size()];
		//calculate upper boundaries for key value pairs
		if(valueClass.equals(MatrixCell.class))
		{
			for(int i=0; i<representativeMatrixes.size(); i++)
			{
				rbounds[i]=rlens[i];
				cbounds[i]=clens[i];
			//	System.out.println("get bound for "+representativeMatrixes.get(i)+": "+rbounds[i]+", "+cbounds[i]);
			}
		}else
		{
			for(int i=0; i<representativeMatrixes.size(); i++)
			{
				rbounds[i]=(long)Math.ceil((double)rlens[i]/(double)brlens[i]);
				cbounds[i]=(long)Math.ceil((double)clens[i]/(double)bclens[i]);
	
				// DRB: the row indexes need to be fixed 
				rbounds[i] = rlens[i];
			}
		}
		
		//collect unary instructions for each representative matrix
		HashSet<Byte> set=new HashSet<Byte>();
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			set.clear();
			set.add(representativeMatrixes.get(i));
			
			//collect the relavent rand instructions for this representative matrix
			Vector<RandInstruction> randsForThisMatrix=new Vector<RandInstruction>();
			if(allRandIns!=null)
			{
				for(RandInstruction ins:allRandIns)
				{
					if(set.contains(ins.input))
					{
						randsForThisMatrix.add(ins);
						set.add(ins.output);
					}
				}
			}
			if(randsForThisMatrix.size()>1)
				throw new RuntimeException("only expects at most one rand instruction per input");
			if(randsForThisMatrix.isEmpty())
				rand_instructions.add(null);
			else
				rand_instructions.add(randsForThisMatrix.firstElement());
						
			//collect the relavent instructions for this representative matrix
			Vector<MRInstruction> opsForThisMatrix=new Vector<MRInstruction>();
			
			if(allMapperIns!=null)
			{
				for(MRInstruction ins: allMapperIns)
				{
					try {
						/*
						boolean toAdd=true;
						for(byte input: ins.getInputIndexes())
							if(!set.contains(input))
							{
								toAdd=false;
								break;
							}
							*/
						boolean toAdd=false;
						for(byte input : ins.getInputIndexes())
							if(set.contains(input))
							{
								toAdd=true;
								break;
							}
						
						if(toAdd)
						{
							opsForThisMatrix.add(ins);
							set.add(ins.output);
						}
					} catch (DMLRuntimeException e) {
						throw new RuntimeException(e);
					}	
				}
			}
			
			mapper_instructions.add(opsForThisMatrix);
			
			//collect the relavent reblock instructions for this representative matrix
			Vector<ReblockInstruction> reblocksForThisMatrix=new Vector<ReblockInstruction>();
			if(allReblockIns!=null)
			{
				for(ReblockInstruction ins:allReblockIns)
				{
					if(set.contains(ins.input))
					{
						reblocksForThisMatrix.add(ins);
						set.add(ins.output);
					}
				}
			}
		
			reblock_instructions.add(reblocksForThisMatrix);
			
			//collect the output indexes for this representative matrix
			Vector<Byte> outsForThisMatrix=new Vector<Byte>();
			for(byte output: outputs)
			{
				if(set.contains(output))
					outsForThisMatrix.add(output);
			}
			outputIndexes.add(outsForThisMatrix);
		}
	}

	protected void processMapperInstructionsForMatrix(int index) 
	throws IOException
	{
		//apply all mapper instructions
		try {
			processMixedInstructions(mapper_instructions.get(index));
		} catch (Exception e) {
			throw new IOException(e);
		}
	}
	
	protected void processReblockInMapperAndOutput(int index, MatrixIndexes indexBuffer,
			TaggedPartialBlock partialBuffer, OutputCollector<Writable, Writable> out) 
	throws IOException
	{
		//apply reblock instructions
		for(ReblockInstruction ins: reblock_instructions.get(index))
		{
			IndexedMatrixValue inValue=cachedValues.getFirst(ins.input);
			if(inValue==null)
				continue;
			long bi=UtilFunctions.blockIndexCalculation(inValue.getIndexes().getRowIndex(),ins.brlen);
			long bj=UtilFunctions.blockIndexCalculation(inValue.getIndexes().getColumnIndex(),ins.bclen);
			indexBuffer.setIndexes(bi, bj);
			
			int ci=UtilFunctions.cellInBlockCalculation(inValue.getIndexes().getRowIndex(), ins.brlen);
			int cj=UtilFunctions.cellInBlockCalculation(inValue.getIndexes().getColumnIndex(),ins.bclen);
			partialBuffer.getBaseObject().set(ci, cj, ((MatrixCell)inValue.getValue()).getValue());
			partialBuffer.setTag(ins.output);
			out.collect(indexBuffer, partialBuffer);
		//	System.out.println("in Mapper, "+inValue+" --> "+indexBuffer+": "+partialBuffer);
		}
	}
	
	protected void processMapOutputToReducer(int index, MatrixIndexes indexBuffer, 
			TaggedMatrixValue taggedValueBuffer, OutputCollector<Writable, Writable> out) throws IOException
	{
			
		for(byte output: outputIndexes.get(index))
		{
			ArrayList<IndexedMatrixValue> results= cachedValues.get(output);
			if(results==null)
				continue;
			for(IndexedMatrixValue result: results)
			{
				if(result==null)
					continue;
				indexBuffer.setIndexes(result.getIndexes());
				////////////////////////////////////////
			//	taggedValueBuffer.getBaseObject().copy(result.getValue());
				taggedValueBuffer.setBaseObject(result.getValue());
				////////////////////////////////////////
				taggedValueBuffer.setTag(output);
				out.collect(indexBuffer, taggedValueBuffer);
			//	System.out.println("map output: "+indexBuffer+"\n"+taggedValueBuffer);
			}
			
		}	
	}
	
	protected void processMapFinalOutput(int index, MatrixIndexes indexBuffer,
			TaggedMatrixValue taggedValueBuffer, CollectMultipleConvertedOutputs collectFinalMultipleOutputs,
			Reporter reporter, HashMap<Byte, Vector<Integer>> tagMapping) throws IOException
	{
		for(byte output: outputIndexes.get(index))
		{
			IndexedMatrixValue result=cachedValues.getFirst(output);
			if(result==null)
				continue;
			indexBuffer.setIndexes(result.getIndexes());
			////////////////////////////////////////
			//	taggedValueBuffer.getBaseObject().copy(result.getValue());
			taggedValueBuffer.setBaseObject(result.getValue());
			////////////////////////////////////////
			taggedValueBuffer.setTag(output);
			for(int outputIndex: tagMapping.get(output))
			{
				collectFinalMultipleOutputs.collectOutput(indexBuffer, taggedValueBuffer.getBaseObject(), outputIndex, 
					reporter);
			}
		}	
	}
}
