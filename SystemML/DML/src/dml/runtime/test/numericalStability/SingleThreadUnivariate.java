package dml.runtime.test.numericalStability;

import java.io.IOException;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;

import dml.runtime.functionobjects.CM;
import dml.runtime.functionobjects.KahanPlus;
import dml.runtime.instructions.CPInstructions.CM_COV_Object;
import dml.runtime.instructions.CPInstructions.KahanObject;
import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.util.MapReduceTool;

public class SingleThreadUnivariate {

	public static void main(String[] args) throws Exception {
	
		if(args.length<5)
		{
			System.out.println("SingleThreadUnivariate <input> <output> <numRows> <numReducers> <block size>");
			System.exit(-1);
		}
		String dir=args[0];
		long n=Long.parseLong(args[2]);
		String outdir=args[1];
		int npb=Integer.parseInt(args[4]);
		InputInfo inputinfo=InputInfo.TextCellInputInfo;
		if(npb>1)
		{
			inputinfo=InputInfo.BinaryBlockInputInfo;
			
		}
		JobConf job = new JobConf();
		FileInputFormat.addInputPath(job, new Path(dir));
		
		CM cmFn=CM.getCMFnObject();
		KahanPlus kahanFn=KahanPlus.getKahanPlusFnObject();
		KahanObject kahan=new KahanObject(0, 0);
		CM_COV_Object cm=new CM_COV_Object();
		
		try {

			InputFormat informat=inputinfo.inputFormatClass.newInstance();
			if(informat instanceof TextInputFormat)
				((TextInputFormat)informat).configure(job);
			InputSplit[] splits= informat.getSplits(job, 1);
			
			Converter inputConverter=MRJobConfiguration.getConverterClass(inputinfo, false, npb, 1).newInstance();
			inputConverter.setBlockSize(npb, 1);
    		
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
						kahanFn.execute(kahan, cell.getValue());
						cmFn.execute(cm, cell.getValue(), 1);
					}
				}
			}
			
		} catch (Exception e) {
			throw new IOException(e);
		}
		
		double sum=kahan._sum;
		double cm2=cm.m2._sum/n;
		double cm3=cm.m3._sum/n;
		double cm4=cm.m4._sum/n;
		double mean=sum/n;
		double var = n/(n-1.0)*cm2;
		double std = Math.sqrt(var);
		double g1 = cm3/var/std;
		double g2 = cm4/var/var - 3;
	
		String outStr="Summation: "+sum;
		outStr+="\nMean: "+mean+"\nVariance: ";
		outStr+=var+"\nStd: "+std;
		outStr+="\nSkewness: "+g1;
		outStr+="\nKurtosis: "+g2;		
		
		FileSystem fs=FileSystem.get(job);
		MapReduceTool.deleteFileIfExistOnHDFS(outdir);
		FSDataOutputStream out = fs.create(new Path(outdir), true);
		//out.writeChars(outStr);
		//out.writeUTF(outStr);
		out.write(outStr.getBytes());
		//System.out.println(outStr);
		out.close();
	}
}
