package dml.runtime.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.GMR;
import dml.runtime.matrix.MMCJMR;
import dml.runtime.matrix.MMRJMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.mapred.obsolete.ABMR;

public class TestMMult extends Configured implements Tool{
	InputInfo inputinfo;
	OutputInfo outputinfo;
	
	public static final String sps=Instruction.OPERAND_DELIM;
	public static final String tp=Instruction.VALUETYPE_PREFIX;
	
	boolean blockRepresentation=true;
	
	@Override
	public int run(String[] args) throws Exception {
	

		if(blockRepresentation)
		{
			inputinfo=InputInfo.BinaryBlockInputInfo;
			outputinfo=OutputInfo.BinaryBlockOutputInfo;
		}else
		{
			inputinfo=InputInfo.TextCellInputInfo;
			outputinfo=OutputInfo.TextCellOutputInfo;
		}

		mmult3(args);
		return 0;
	}
	
	public void mmult2(String[] args)throws Exception
	{
		if(args.length<10)
		{
			System.err.println("TestMMult: [file A] [file B] [m] [k] [n] [bm] [bk] [bn] [out file] [#reducers]");
			System.exit(-1);
		}
		String A=args[0];
		String B=args[1];
		long m=Long.parseLong(args[2]);
		long k=Long.parseLong(args[3]);
		long n=Long.parseLong(args[4]);
		int bm=Integer.parseInt(args[5]);
		int bk=Integer.parseInt(args[6]);
		int bn=Integer.parseInt(args[7]);
		String out=args[8];
		int numReducers=Integer.parseInt(args[9]);
		
		GenerateTestMatrixBlock.runJob(2, m, k, A, 1, bm, bk, System.currentTimeMillis(), 1, true);
		GenerateTestMatrixBlock.runJob(2, k, n, B, 1, bk, bn, System.currentTimeMillis(), 1, true);
		
		String inputs=A+","+B;
		String aggBinaryOperation="ba+*";
		int replication=1;
		ABMR.runJob(inputs, aggBinaryOperation, out, 
				numReducers, replication, m, k, k, n, bm, bk, bk, bn);
	}
	
	public void mmult3(String[] args)throws Exception 
	{
		if(args.length<10)
		{
			System.err.println("TestMMult: [file A] [file B] [m] [k] [n] [bm] [bk] [bn] [out file] [#reducers]");
			System.exit(-1);
		}
		String A=args[0];
		String B=args[1];
		long m=Long.parseLong(args[2]);
		long k=Long.parseLong(args[3]);
		long n=Long.parseLong(args[4]);
		int bm=Integer.parseInt(args[5]);
		int bk=Integer.parseInt(args[6]);
		int bn=Integer.parseInt(args[7]);
		String out=args[8];
		//int partialAggCacheSize=Integer.parseInt(args[9]);
		int numReducers=Integer.parseInt(args[9]);
		
		GenerateTestMatrixBlock.runJob(2, m, k, A, 1, bm, bk, System.currentTimeMillis(), 1, true);
		GenerateTestMatrixBlock.runJob(2, k, n, B, 1, bk, bn, System.currentTimeMillis(), 1, true);
		
		String[] inputs=new String[]{A, B};
		InputInfo[] inputInfos=new InputInfo[]{inputinfo, inputinfo};
		long[] rlens=new long[]{m, k};
		long[] clens=new long[]{k, n};
		int[] brlens=new int[]{bm, bk};
		int[] bclens=new int[]{bk, bn};
		
		String instructionsInMapper="r'"+sps+"0"+tp+"DOUBLE"+sps+"0"+tp+"DOUBLE";
		String aggInstructionsInReducer="";
		/*
		 * the aggregateBinary instruction for MMCJ
		 * matrix#2=aggregateBinary(ba+, matrix#0, matrix#1);
		 */
		String aggBinInstrction="ba+*"+sps+"0"+tp+"DOUBLE"+sps+"1"+tp+"DOUBLE"+sps+"2"+tp+"DOUBLE"; 
		
		int replication=1;
		
		byte[] resultIndexes=new byte[]{2};
		byte[] resultDimsUnknown = new byte[]{2};
		String[] outputs=new String[]{out};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.TextCellOutputInfo};	
		
		String otherInstructionsInReducer="s+"+sps+"2"+tp+"DOUBLE"+sps+"1.0"+tp+"DOUBLE"+sps+"2"+tp+"DOUBLE";
		//first perform A mmcj B
		blockRepresentation=false;
		MMRJMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
	
	public void mmult(String[] args)throws Exception 
	{
		if(args.length<10)
		{
			System.err.println("TestMMult: [file A] [file B] [m] [k] [n] [bm] [bk] [bn] [out file] [#reducers1] [#reducers2]");
			System.exit(-1);
		}
		String A=args[0];
		String B=args[1];
		long m=Long.parseLong(args[2]);
		long k=Long.parseLong(args[3]);
		long n=Long.parseLong(args[4]);
		int bm=Integer.parseInt(args[5]);
		int bk=Integer.parseInt(args[6]);
		int bn=Integer.parseInt(args[7]);
		String out=args[8];
		//int partialAggCacheSize=Integer.parseInt(args[9]);
		int numReducers1=Integer.parseInt(args[9]);
		int numReducers2=numReducers1;//Integer.parseInt(args[10]);
		
		
		String[] inputs=new String[]{A, B};
		InputInfo[] inputInfos=new InputInfo[]{inputinfo, inputinfo};
		long[] rlens=new long[]{m, k};
		long[] clens=new long[]{k, n};
		int[] brlens=new int[]{bm, bk};
		int[] bclens=new int[]{bk, bn};
		
		String instructionsInMapper="r':::0:DOUBLE:::0:DOUBLE";
		String aggInstructionsInReducer="";
		/*
		 * the aggregateBinary instruction for MMCJ
		 * matrix#2=aggregateBinary(ba+, matrix#0, matrix#1);
		 */
		String aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE"; 
		
		String output="temp";
		int replication=1;
		
		//first perform A mmcj B
		MMCJMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, aggBinInstrction, numReducers1, 
				replication, (byte)0, output, outputinfo);
		
		inputs=new String[]{"temp"};
		inputInfos=new InputInfo[]{inputinfo};
		rlens=new long[]{m};
		clens=new long[]{n};
		brlens=new int[]{bm};
		bclens=new int[]{bn};
		instructionsInMapper="";
		/*
		 * matrix#0=aggregate(sum, matrix#0)
		 */
		aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
		String otherInstructionsInReducer="s+:::0:DOUBLE:::1.0:DOUBLE:::0:DOUBLE";
		byte[] resultIndexes=new byte[]{0};
		byte[] resultDimsUnknown = new byte[]{0};
		String[] outputs=new String[]{out};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.TextCellOutputInfo};	
		GMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers2, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
	
	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new TestMMult(), args);
		System.exit(errCode);
	}
}
