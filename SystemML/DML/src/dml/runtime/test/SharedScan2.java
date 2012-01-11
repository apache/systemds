package dml.runtime.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import dml.runtime.matrix.GMR;
import dml.runtime.matrix.MMCJMR;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class SharedScan2 extends Configured implements Tool{

	public static InputInfo binaryinputinfo=InputInfo.BinaryBlockInputInfo;
	public static OutputInfo binaryoutputinfo=OutputInfo.BinaryBlockOutputInfo;
	
	private String A="A";
	private String D="D";
	String E="E";
	String F="F";
	long n;
	long m; 
	long k; 
	int bn; 
	int bm; 
	int bk; 
	int maxCap;
	int numReducers; 
	String out="out";
	
	//A*(E/F)
	public void sharedScan()
	throws Exception
	{
		int replication=1;
		long start;
		String[] inputs;
		InputInfo[] inputInfos;
		long[] rlens;
		long[] clens;
		int[] brlens;
		int[] bclens;
		String instructionsInMapper;
		String aggInstructionsInReducer;
		String aggBinInstrction;
		String output;
		OutputInfo outputinfo;
		int partialAggCacheSize;
		MatrixCharacteristics[] stats;
		String otherInstructionsInReducer;
		byte[] resultIndexes;
		byte[] resultDimsUnknown;
		String[] outputs;
		OutputInfo[] outputInfos;		
		String temp="temp";
		
		start=System.currentTimeMillis();
		
	//	System.out.println("@@@@@@@@@ shared scan");	
	//	System.out.println("\n----------------------------");
	//	System.out.println("GMR for mmult");
		start=System.currentTimeMillis();
		
		//out=max(A, 0)*( (t(E)+F) /D)
		inputs=new String[]{A, E, F};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n, n};
		clens=new long[]{m, m, m};
		brlens=new int[]{bn, bn, bn};
		bclens=new int[]{bm, bm, bm};
		instructionsInMapper="";
		aggInstructionsInReducer="";
		otherInstructionsInReducer="b/ 1 2 1,b* 0 1 0";
		resultIndexes=new byte[]{0};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{out};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
	}
	
	//A*(E/F)
	public void separate()
	throws Exception
	{
		int replication=1;
		long start;
		String[] inputs;
		InputInfo[] inputInfos;
		long[] rlens;
		long[] clens;
		int[] brlens;
		int[] bclens;
		String instructionsInMapper;
		String aggInstructionsInReducer;
		String aggBinInstrction;
		String output;
		OutputInfo outputinfo;
		int partialAggCacheSize;
		MatrixCharacteristics[] stats;
		String otherInstructionsInReducer;
		byte[] resultIndexes;
		byte[] resultDimsUnknown;
		String[] outputs;
		OutputInfo[] outputInfos;		
		String temp1="temp1";
		String temp2="temp2";
		
		/*		start=System.currentTimeMillis();
	//	System.out.println("@@@@@@@@@ NOT shared scan");
	//	System.out.println("\n----------------------------");
	//	System.out.println("t(C)");
		start=System.currentTimeMillis();
	
		//t(E)
		inputs=new String[]{E};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{m};
		clens=new long[]{n};
		brlens=new int[]{bm};
		bclens=new int[]{bn};
		instructionsInMapper="r' 0 0";
		aggInstructionsInReducer="";
		otherInstructionsInReducer="";
		resultIndexes=new byte[]{0};
		outputs=new String[]{temp2};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				0, replication, resultIndexes, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
	//	System.out.println("\n----------------------------");
	//	System.out.println("B*t(C)");
		start=System.currentTimeMillis();
		
		//T(E)+F
		inputs=new String[]{temp2, F};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n};
		clens=new long[]{m, m};
		brlens=new int[]{bn, bn};
		bclens=new int[]{bm, bm};
		instructionsInMapper="";
		aggInstructionsInReducer="";
		otherInstructionsInReducer="b+ 0 1 0";
		resultIndexes=new byte[]{0};
		outputs=new String[]{temp1};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		//System.out.println("\n----------------------------");
		//System.out.println("B%*%t(C)/D)"); */
		start=System.currentTimeMillis();
		
		//(T(E)+F/D)
		inputs=new String[]{E, F};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n};
		clens=new long[]{m, m};
		brlens=new int[]{bn, bn};
		bclens=new int[]{bm, bm};
		instructionsInMapper="";
		aggInstructionsInReducer="";
		otherInstructionsInReducer="b/ 0 1 2";
		resultIndexes=new byte[]{2};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{temp2};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		//System.out.println("\n----------------------------");
		//System.out.println("max(A, 0)");
		start=System.currentTimeMillis();
		
		//max(A, 0)
/*		inputs=new String[]{A};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{n};
		clens=new long[]{m};
		brlens=new int[]{bn};
		bclens=new int[]{bm};
		instructionsInMapper="smax 0 0 0";
		aggInstructionsInReducer="";
		otherInstructionsInReducer="";
		resultIndexes=new byte[]{0};
		outputs=new String[]{temp1};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				0, replication, resultIndexes, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");*/
		//System.out.println("\n----------------------------");
		//System.out.println("max(A, 0)*{ B%*%t(C)/D}");
		start=System.currentTimeMillis();
		
		//A*{ (t(E)+F)/D}
		inputs=new String[]{A, temp2};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n};
		clens=new long[]{m, m};
		brlens=new int[]{bn, bn};
		bclens=new int[]{bm, bm};
		instructionsInMapper="";
		aggInstructionsInReducer="";
		otherInstructionsInReducer="b* 0 1 2";
		resultIndexes=new byte[]{2};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{out};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		System.out.print((System.currentTimeMillis()-start)+"\t");
	}
	
	@Override
	public int run(String[] args) throws Exception {
		
		if(args.length<9)
		{
			System.err.println("[n] [m] [k] [bn] [bm] [bk] [# mappers] [# reducers] [sparsity]");
			System.exit(-1);
		}
		
		n=Long.parseLong(args[0]);
		m=Long.parseLong(args[1]);
		k=Long.parseLong(args[2]);
		bn=Integer.parseInt(args[3]);
		bm=Integer.parseInt(args[4]);
		bk=Integer.parseInt(args[5]);
		int numMappers=Integer.parseInt(args[6]);
		numReducers=Integer.parseInt(args[7]);
		double sparsity=Double.parseDouble(args[8]);
		maxCap=900000000;//(int)(100000000*0.75);
		
		System.out.print(n+"\t");
		
		GenerateTestMatrixBlock.runJob(numMappers, n, m, A, 1, bn, bm, System.currentTimeMillis(), sparsity, true);
		
		GenerateTestMatrixBlock.runJob(numMappers, n, m, E, 1, bn, bm, System.currentTimeMillis(), sparsity, true);
		
		GenerateTestMatrixBlock.runJob(numMappers, n, m, F, 1, bn, bm, System.currentTimeMillis(), sparsity, true);
		
		//GenerateTestMatrixBlock.runJob(numMappers, n, m, D, 1, bn, bm, System.currentTimeMillis(), sparsity, true);
		
		long totalStart=System.currentTimeMillis();
		
		sharedScan();
		System.out.print((System.currentTimeMillis()-totalStart)+"\t");
		totalStart=System.currentTimeMillis();
		separate();
		System.out.println((System.currentTimeMillis()-totalStart));
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new SharedScan2(), args);
		System.exit(errCode);
	}
}
