package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;

import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.operators.AggregateBinaryOperator;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.util.MapReduceTool;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class PartialAggregator {

	private int fileCursor=-1;
	private int bufferCapacity=0;
	private int currentBufferSize=0;
	private Path[] files=null;
	private Pair<MatrixIndexes,MatrixValue>[] buffer=null;
	private SequenceFile.Reader reader=null;
	private SequenceFile.Writer writer=null;
	private boolean rowMajor=true;
	private long rlen=0;
	private long clen=0;
	private int brlen=0;
	private int bclen=0;
	private FileSystem fs=null;
	private JobConf job=null;
	private LinkedList<Integer> orderedList=new LinkedList<Integer>();
	private int listCursor=-1;
	private long numBlocksInRow=0;
	private long numBlocksInColumn=0;
	private AggregateBinaryOperator operation;
	private Class<? extends MatrixValue> valueClass;
	private long elementSize;
//	protected static final Log LOG = LogFactory.getLog(PartialAggregator.class);
	
	public PartialAggregator(JobConf conf, long memSize, long resultRlen, long resultClen, 
			int blockRlen, int blockClen, String filePrefix, boolean inRowMajor, 
			AggregateBinaryOperator op, Class<? extends MatrixValue> vCls) 
	throws InstantiationException, IllegalAccessException, IOException
	{
		job=conf;
		try {
			fs=FileSystem.getLocal(job);
		} catch (IOException e) {
			throw new RuntimeException(e);
		};
		rlen=resultRlen;
		clen=resultClen;
		brlen=blockRlen;
		bclen=blockClen;
		numBlocksInRow=(long)Math.ceil((double)clen/(double)bclen);
		numBlocksInColumn=(long)Math.ceil((double)rlen/(double)brlen);
		
		rowMajor=inRowMajor;
		operation=op;
		valueClass=vCls;
		
		System.gc();
		memSize=Math.min(memSize, Runtime.getRuntime().freeMemory());
		memSize=(long)((double)memSize*0.8);
		
		//allocate space for buffer
		//if the buffer space is already larger than the result size, don't need extra space
		elementSize=77+8*blockRlen*blockClen+20+12+12+4;//matrix block, matrix index, pair, integer in the linked list
		bufferCapacity=(int)Math.min((memSize/elementSize), (numBlocksInRow*numBlocksInColumn));
		if(bufferCapacity==0)
			bufferCapacity=1;
/*		LOG.info("bufferSize = "+memSize);
		LOG.info("bufferCapacity = "+bufferCapacity);
		LOG.info("before alloc array: ");
		LOG.info("Memory stats: "+Runtime.getRuntime().totalMemory()+", "+Runtime.getRuntime().freeMemory());
		*/
		buffer=new Pair[bufferCapacity];
		for(int i=0; i<bufferCapacity; i++)
			buffer[i]=new Pair<MatrixIndexes, MatrixValue>(new MatrixIndexes(), valueClass.newInstance());
		
	/*	System.out.println("memSize: "+memSize);
		System.out.println("elementSize: "+elementSize);
		System.out.println("blockRlen: "+blockRlen);
		System.out.println("blockClen: "+blockClen);
		System.out.println("bufferCapacity: "+bufferCapacity);
		System.out.println("resultRlen: "+resultRlen);
		System.out.println("resultClen: "+resultClen);*/
		
		//the list of files
		//int n=(int)Math.ceil((double)(resultRlen*resultClen)/(double)blockRlen/(double)blockClen/(double)bufferCapacity);
		int n=(int)Math.ceil((double)(numBlocksInRow*numBlocksInColumn)/(double)bufferCapacity);
				
		files=new Path[n];
	//	LOG.info("number of files: "+n);
		String hadoopLocalDir=job.get("mapred.local.dir").split(",")[0];
		for(int i=0; i<n; i++)
		{
			files[i]=new Path(hadoopLocalDir, filePrefix+"_partial_aggregator_"+i);
			MapReduceTool.deleteFileIfExistOnLFS(files[i], job);
		//	System.out.println(files[i]);
		}
		
	}
	
	public void startOver()
	{
	//	LOG.info("#### reset the listCursor");
		listCursor=0;
	}
	
	private void loadBuffer() throws IOException
	{
		currentBufferSize=0;
		orderedList.clear();
		listCursor=0;
		long nzs=0;
	//	System.out.println("** load fileCursor: "+fileCursor);
		if(fs.exists(files[fileCursor]))
		{
			reader=new SequenceFile.Reader(fs, files[fileCursor], job);
			if(reader==null)
				throw new IOException("reader is null: "+files[fileCursor]);
			
			while(currentBufferSize<bufferCapacity && reader.next(buffer[currentBufferSize].getKey(), buffer[currentBufferSize].getValue()))
			{
				nzs+=buffer[currentBufferSize].getValue().getNonZeros();
				orderedList.add(currentBufferSize);
				currentBufferSize++;
			}
			reader.close();
		}
		//LOG.info("~~~~ loaded file: "+files[fileCursor]+" with number of elements: "+currentBufferSize+", nzs= "+nzs);
	/*	System.out.println("current ordered list: "+orderedList);
		System.out.println("current buffer: "+getBufferString());*/
	}
	
	private void writeBuffer() throws IOException {
		if(fileCursor<0 || currentBufferSize<=0)
			return;
		//the old file will be overwritten
		writer=new SequenceFile.Writer(fs, job, files[fileCursor], MatrixIndexes.class, valueClass);
		Iterator<Integer> p=orderedList.iterator();
		long nzs=0;
		while(p.hasNext())
		{
			int index=p.next();
			writer.append(buffer[index].getKey(), buffer[index].getValue());
			nzs+=buffer[index].getValue().getNonZeros();
		}
		writer.close();
		//LOG.info("~~~~ wrote file: "+files[fileCursor]+" with number of elements: "+currentBufferSize+", nzs= "+nzs);
	}
	
	public void aggregateToBuffer(MatrixIndexes indexes, MatrixValue value, boolean leftcached) 
	throws IOException, DMLUnsupportedOperationException, DMLRuntimeException
	{
		int newFileCursor=getFileCursor(indexes);
		if(newFileCursor>=files.length)
		{
			throw new IOException("indexes: "+indexes+" needs to be put in file #"+newFileCursor+" which exceeds the limit: "+files.length);
		}
		if(fileCursor!=newFileCursor)
		{	
			writeBuffer();
			fileCursor=newFileCursor;
			loadBuffer();
		}
		aggregateToBufferHelp(indexes, value, leftcached);
	}

	private String getBufferString()
	{
		String str="";
		for(int i=0; i<currentBufferSize; i++)
			str+="\n"+buffer[orderedList.get(i)].getKey();//+"-->"+buffer[i].getValue();
		return str;
	}
	
	private void aggregateToBufferHelp(MatrixIndexes indexes, MatrixValue value, boolean leftcached) 
	throws DMLUnsupportedOperationException, DMLRuntimeException {
	
		/*System.out.println("** in aggregateToBufferHelp");
		System.out.println("previous ordered list: "+orderedList);
		System.out.println("previous buffer: "+getBufferString());
		System.out.println("add for index: "+indexes);
		System.out.println("value:"+value);*/
//		LOG.info("listCursor= "+listCursor);
//		LOG.info("currentBufferSize: "+currentBufferSize);
//		LOG.info("orderedList.size(): "+orderedList.size());
		int cmp=1;
		while(listCursor<currentBufferSize)
		{
			cmp=indexes.compareWithOrder(buffer[orderedList.get(listCursor)].getKey(), leftcached);
		//	LOG.info("comparing to "+buffer[orderedList.get(listCursor)].getKey());
	//	LOG.info("result is :"+cmp);
			if(cmp<=0)
				break;
			listCursor++;
		}
	//	LOG.info("after find: listCursor= "+listCursor+" cmp="+cmp);
		if(cmp==0)//aggregate to the position listCursor
			buffer[listCursor].getValue().binaryOperationsInPlace(operation.aggOp.increOp, value);
		else
			addToBuffer(indexes, value);
	/*	System.out.println("current ordered list: "+orderedList);
		System.out.println("current buffer: "+getBufferString());*/
	}
	
	private void addToBuffer(MatrixIndexes indexes, MatrixValue value)
	{
		if(currentBufferSize>=buffer.length)
		{
			throw new RuntimeException("indexes: "+indexes+" needed to be put in postition: "+currentBufferSize+" which exceeds the buffer size: "+buffer.length);
		}
		
		if(Runtime.getRuntime().freeMemory()<10*elementSize)
			System.gc();
		
	//	LOG.info("currentBufferSize : "+currentBufferSize);
	//	LOG.info("Memory stats: "+Runtime.getRuntime().totalMemory()+", "+Runtime.getRuntime().freeMemory());
		//add to the end
		buffer[currentBufferSize].getKey().setIndexes(indexes);
		buffer[currentBufferSize].getValue().copy(value);
		if(listCursor>=orderedList.size())
			orderedList.addLast(currentBufferSize);
		else
			orderedList.add(listCursor, currentBufferSize);
		currentBufferSize++;
//		LOG.info("after add: "+getBufferString());
//		LOG.info("after : "+currentBufferSize);
//		LOG.info("Memory stats: "+Runtime.getRuntime().totalMemory()+", "+Runtime.getRuntime().freeMemory());
		
	}

	private int getFileCursor(MatrixIndexes indexes) {
		if(rowMajor)
			return (int)(((indexes.getRowIndex()-1)*numBlocksInRow+indexes.getColumnIndex()-1)/bufferCapacity);
		else
			return (int)(((indexes.getColumnIndex()-1)*numBlocksInColumn+indexes.getRowIndex()-1)/bufferCapacity);
	}
	
	public long outputToHadoop(CollectMultipleConvertedOutputs outputs, int j, Reporter reporter) throws IOException
	{
		long nonZeros=0;
		if(fileCursor>=0)
		{
			//write the currentBufferSize if it is in memory
			Iterator<Integer> p=orderedList.iterator();
			while(p.hasNext())
			{
				int index=p.next();
				outputs.collectOutput(buffer[index].getKey(), buffer[index].getValue(), j, reporter);
				nonZeros+=buffer[index].getValue().getNonZeros();
				//System.out.println("MMCJ output: "+buffer[index].getKey()+" -- "+buffer[index].getValue()+" ~~ tag: "+resultIndex);
			}
			MapReduceTool.deleteFileIfExistOnHDFS(files[fileCursor], job);
		}
		
		for(int i=fileCursor+1; i<files.length; i++)
			nonZeros+=copyFileContentAndDelete(files[i], outputs, j, reporter);
		
		for(int i=0; i<fileCursor; i++)
			nonZeros+=copyFileContentAndDelete(files[i], outputs, j, reporter);
		
		return nonZeros;
	}

	private long copyFileContentAndDelete(Path path,
			CollectMultipleConvertedOutputs outputs, int j, Reporter reporter) throws IOException {
		long nonZeros=0;
		if(fs.exists(path))
		{
			reader=new SequenceFile.Reader(fs, path, job);
			if(reader==null)
				throw new IOException("reader is null: "+files[fileCursor]);
			
			while(reader.next(buffer[0].getKey(), buffer[0].getValue()))
			{
			//	System.out.println(buffer[0].getKey()+" -- "+buffer[0].getValue());
				outputs.collectOutput(buffer[0].getKey(), buffer[0].getValue(), j, reporter);
				nonZeros+=buffer[0].getValue().getNonZeros();
			}
			reader.close();
			MapReduceTool.deleteFileIfExistOnHDFS(path, job);
		}
		return nonZeros;
	}
	
	public void close() throws IOException
	{
		for(Path file: files)
		{
			MapReduceTool.deleteFileIfExistOnLFS(file, job);
		}
	}
}
