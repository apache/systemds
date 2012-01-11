package dml.runtime.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import dml.runtime.matrix.GMR;
import dml.runtime.matrix.MMCJMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.mapred.tentative.ABMR;

public class PageRank extends Configured implements Tool{

	public static InputInfo binaryinputinfo=InputInfo.BinaryBlockInputInfo;
	public static OutputInfo binaryoutputinfo=OutputInfo.BinaryBlockOutputInfo;
	String G;
	String u;
	String p;
	String e;
	long n;
	long d;
	int bn;
	int bd;
	double tol;
	private int maxIter;
	private double alpha;
	private String out="out";
	
	@Override
	public int run(String[] args) throws Exception {
		if(args.length<13)
		{
			System.err.println("[G] [u] [p] [e] [n] [d] [bn] [bd] [alpha] [tol] [maxIter] [# mappers] [# reducers] [aggregator buffer]");
			System.exit(-1);
		}
		
		G=args[0];
		u=args[1];
		p=args[2];
		e=args[3];
		n=Long.parseLong(args[4]);
		d=Long.parseLong(args[5]);
		bn=Integer.parseInt(args[6]);
		bd=Integer.parseInt(args[7]);
		alpha=Double.parseDouble(args[8]);
		tol=Double.parseDouble(args[9]);
		maxIter=Integer.parseInt(args[10]);
		int numMappers=Integer.parseInt(args[11]);
		int numReducers=Integer.parseInt(args[12]);
		int partialAggCacheSize=Integer.parseInt(args[13]);
		int replication=1;
		
		String[] inputs;
		InputInfo[] inputInfos;
		long[] rlens;
		long[] clens;
		int[] brlens;
		int[] bclens;
		String instructionsInMapper;
		String aggInstructionsInReducer;
		String otherInstructionsInReducer;
		byte[] resultIndexes;
		byte[] resultDimsUnknown;
		String[] outputs;
		OutputInfo[] outputInfos;
		String aggBinInstrction;
		
		String output;
		OutputInfo outputinfo;
		String temp="temp";
	
	/*	GenerateTestMatrixBlock.runJob(numMappers, n, n, G, 1, bn, bn, System.currentTimeMillis(), 0.001, true);
		GenerateTestMatrixBlock.runJob(numMappers, 1, n, u, 1, 1, bn, System.currentTimeMillis(), 1, true);
		GenerateTestMatrixBlock.runJob(numMappers, n, 1, e, 1, bn, 1, System.currentTimeMillis(), 1, true);
		GenerateTestMatrixBlock.runJob(numMappers, n, 1, p, 1, bn, 1, System.currentTimeMillis(), 1, true);*/
		
		long start=System.currentTimeMillis();
		String t1="t1";
	
		ABMR.runJob(u+","+p, "ba+*", t1, numReducers, replication, 1, n, n, 1, 1, bn, bn, 1);
		
		String t2="t2";
		ABMR.runJob(e+","+t1, "ba+*", t2, numReducers, replication, n, 1, 1, 1, bn, 1, 1, 1);
		
		
		inputs=new String[]{G, p};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n};
		clens=new long[]{n, 1};
		brlens=new int[]{bn, bn};
		bclens=new int[]{bn, 1};
		instructionsInMapper="";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+* 0 1 2";
		output = temp;
		outputinfo = binaryoutputinfo;
				
		MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, aggBinInstrction, numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize);
		
		//alpha*temp+(1-alpha)*t2
		inputs=new String[]{temp, t2};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n};
		clens=new long[]{1, 1};
		brlens=new int[]{bn, bn};
		bclens=new int[]{1, 1};
		instructionsInMapper="s* 0 "+alpha+" 0,s* 1 "+(1-alpha)+" 1";
		aggInstructionsInReducer="a+ 0 0";
		otherInstructionsInReducer="b+ 0 1 0";
		resultIndexes=new byte[]{0};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{p};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
				resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.println(System.currentTimeMillis()-start);
		
		return 0;
	}
	
	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new PageRank(), args);
		System.exit(errCode);
	}
	
}
