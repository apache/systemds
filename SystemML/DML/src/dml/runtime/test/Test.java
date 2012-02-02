package dml.runtime.test;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Counters.Group;
import org.apache.hadoop.util.Tool;

import dml.runtime.matrix.GMR;
import dml.runtime.matrix.MMCJMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.mapred.obsolete.ABMR;

public class Test extends Configured implements Tool{
	InputInfo textinputinfo;
	OutputInfo textoutputinfo;
	
//	InputInfo binaryinputinfo=InputInfo.BinaryBlockInputInfo;
//	OutputInfo binaryoutputinfo=OutputInfo.BinaryBlockOutputInfo;
	boolean blockRepresentation=false;
	int partialAggCacheSize=10;
	
	@Override
	public int run(String[] args) throws Exception {
	
		if(blockRepresentation)
		{
			textinputinfo=InputInfo.BinaryBlockInputInfo;
			textoutputinfo=OutputInfo.BinaryBlockOutputInfo;
		}else
		{
			textinputinfo=InputInfo.TextCellInputInfo;
			textoutputinfo=OutputInfo.TextCellOutputInfo;
		}
	//	GenerateTestMatrix.runJob(1, 2, 2, "A", 1, false, 1, 1);
	//	reblock(args);
	//	streamThrough(args);
	//	unary(args);
	//	binary(args);
		mmult(args);
	//	mmult2(args);
	//	basicTest(args);
	//	mmult3(args);
		return 0;
	}
	
	//(A+0.5)'
	void streamThrough(String[] args)throws Exception
	{
		String[] inputs=new String[]{"A"};
		long[] rlens=new long[]{2};
		long[] clens=new long[]{2};
		int[] brlens=new int[]{2};
		int[] bclens=new int[]{1};
		InputInfo[] inputInfos=new InputInfo[]{textinputinfo};
		/*
		 * matrix#0=scalar(+, matrix#0, 0.5)
		 * matrix#0=reorg(transpose, matrix#0)
		 */
		String instructionsInMapper="s+:::0:DOUBLE:::0.5:DOUBLE:::0:DOUBLE:::,r':::0:DOUBLE:::0:DOUBLE";
		String aggInstructionsInReducer="";
		String otherInstructionsInReducer="";
		int numReducers=0;
		int replication=1;
		byte[] resultIndexes=new byte[]{0};
		byte[] resultDimsUnknown=new byte[]{0};
		String[] outputs=new String[]{"streamThrough_output0"};
		OutputInfo[] outputInfos=new OutputInfo[]{textoutputinfo};
		GMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
	
	//colum_sum( (A+0.5)') - 0.1
	public void unary(String[] args)throws Exception 
	{
		String[] inputs=new String[]{"A"};
		InputInfo[] inputInfos=new InputInfo[]{textinputinfo};
		long[] rlens=new long[]{2};
		long[] clens=new long[]{2};
		int[] brlens=new int[]{2};
		int[] bclens=new int[]{1};
		/*
		 * matrix#0=scalar(+, matrix#0, 0.5)
		 * matrix#0=reorg(transpose, matrix#0)
		 * matrix#0=aggregateUnary(column_sum, matrix#0)
		 */
		String instructionsInMapper="s+:::0:DOUBLE:::0.5:DOUBLE:::0:DOUBLE,r':::0:DOUBLE:::0:DOUBLE:::,uac+:::0:DOUBLE:::0:DOUBLE";
		/*
		 * matrix#0=aggregate(sum, matrix#0)
		 */
		String aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
		/*
		 * matrix#0=scalar(-, matrix#0, 0.1)
		 */
		String otherInstructionsInReducer="s-:::0:DOUBLE:::0.1:DOUBLE:::0:DOUBLE";
		int numReducers=1;
		int replication=1;
		byte[] resultIndexes=new byte[]{0};
		byte[] resultDimsUnknown=new byte[]{0};
		String[] outputs=new String[]{"unary_output0"};
		OutputInfo[] outputInfos=new OutputInfo[]{textoutputinfo};
		GMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
	
	// ((A+0.5)+A')-0.1
	public void binary(String[] args)throws Exception 
	{
		String[] inputs=new String[]{"A", "A"};
		InputInfo[] inputInfos=new InputInfo[]{textinputinfo, textinputinfo};
		long[] rlens=new long[]{2, 2};
		long[] clens=new long[]{2, 2};
		int[] brlens=new int[]{1, 2};
		int[] bclens=new int[]{2, 1};
		/*
		 * matrix#0=scalar(+, matrix#0, 0.5)
		 * matrix#1=reorg(transpose, matrix#1)
		 */
		String instructionsInMapper="s+:::0:DOUBLE:::0.5:DOUBLE:::0:DOUBLE,r':::1:DOUBLE:::1:DOUBLE";
		String aggInstructionsInReducer="";
		/*
		 * matrix#2=binary(+, matrix#0, matrix#1) //matrix#2 is the temp matrix
		 * matrix#2=scalar(-, matrix#2, 0.1)
		 */
		String otherInstructionsInReducer="b+:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE,s-:::2:DOUBLE:::0.1:DOUBLE:::2:DOUBLE";
		int numReducers=1;
		int replication=1;
		byte[] resultIndexes=new byte[]{2};
		byte[] resultDimsUnknown=new byte[]{0};
		String[] outputs=new String[]{"binary_output2"};
		OutputInfo[] outputInfos=new OutputInfo[]{textoutputinfo};
		GMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
	
	// ( (A')%*%A+(2*A) )-0.1
	public void mmult(String[] args)throws Exception 
	{
		String[] inputs=new String[]{"A"};
		InputInfo[] inputInfos=new InputInfo[]{textinputinfo};
		long[] rlens=new long[]{2};
		long[] clens=new long[]{2};
		int[] brlens=new int[]{1};
		int[] bclens=new int[]{1};
		/*
		 * matrix#2=reorg(transpose, matrix#0)
		 */
		String instructionsInMapper="r':::0:DOUBLE:::1:DOUBLE";
		String aggInstructionsInReducer="";
		/*
		 * the aggregateBinary instruction for MMCJ
		 * matrix#2=aggregateBinary(ba+, matrix#1, matrix#0);
		 */
		String aggBinInstrction="ba+*:::1:DOUBLE:::0:DOUBLE:::2:DOUBLE"; 
		
		String output="temp2";
		int numReducers=1;
		int replication=1;
		
		//first perform A mmcj (A')
		MMCJMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, aggBinInstrction, numReducers, 
				replication, (byte)0, output, textoutputinfo, partialAggCacheSize);
		
		inputs=new String[]{"temp2", "A"};
		inputInfos=new InputInfo[]{textinputinfo, textinputinfo};
		rlens=new long[]{2, 2};
		clens=new long[]{2, 2};
		brlens=new int[]{1, 1};
		bclens=new int[]{1, 1};
		/*
		 * matrix#1=scalar(*, matrix#1, 2.0)
		 */
		instructionsInMapper="s*:::1:DOUBLE:::2.0:DOUBLE:::1:DOUBLE";
		/*
		 * matrix#0=aggregate(sum, matrix#0)
		 */
		aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
		/*
		 * matrix#2=binary(+, matrix#0, matrix#1) //matrix#2 is the temp matrix
		 * matrix#2=scalar(-, matrix#2, 0.1)
		 */
		String otherInstructionsInReducer="b+:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE,s-:::2:DOUBLE:::0.1:DOUBLE:::2:DOUBLE";
		byte[] resultIndexes=new byte[]{2};
		byte[] resultDimsUnknown=new byte[]{0};
		String[] outputs=new String[]{"mmult_output2"};
		OutputInfo[] outputInfos=new OutputInfo[]{textoutputinfo};	
		GMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
	
	// A %*% row_sum(A')
	public void mmult2(String[] args)throws Exception 
	{
		String[] inputs=new String[]{"A"};
		InputInfo[] inputInfos=new InputInfo[]{textinputinfo};
		long[] rlens=new long[]{2};
		long[] clens=new long[]{2};
		int[] brlens=new int[]{1};
		int[] bclens=new int[]{2};
		/*
		 * matrix#1=reorg(transpose, matrix#0)
		 * matrix#1=aggregateUnary(row_sum, matrix#1)
		 */
		String instructionsInMapper="r':::0:DOUBLE:::1:DOUBLE,uar+:::1:DOUBLE:::1:DOUBLE";
		/*
		 * matrix#1=aggregate(sum, matrix#1)
		 */
		String aggInstructionsInReducer="a+:::1:DOUBLE:::1:DOUBLE";
		/*
		 * the aggregateBinary instruction for MMCJ
		 * matrix#2=aggregateBinary(ba+, matrix#0, matrix#1);
		 */
		String aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE"; 
		
		String output="temp";
		int numReducers=1;
		int replication=1;
		
		//first perform A mmcj column_sum(A')
		MMCJMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, aggBinInstrction, numReducers, 
				replication, (byte)0, output, textoutputinfo, partialAggCacheSize);
		
		inputs=new String[]{"temp"};
		inputInfos=new InputInfo[]{textinputinfo};
		rlens=new long[]{2};
		clens=new long[]{1};
		brlens=new int[]{1};
		bclens=new int[]{1};
		instructionsInMapper="";
		/*
		 * matrix#0=aggregate(sum, matrix#0)
		 */
		aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
		String otherInstructionsInReducer="";
		byte[] resultIndexes=new byte[]{0};
		byte[] resultDimsUnknown=new byte[]{0};
		String[] outputs=new String[]{"mmult2_output0"};
		OutputInfo[] outputInfos=new OutputInfo[]{textoutputinfo};
		
		GMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
	
	public void basicTest(String[] args)throws Exception 
	{
		String[] inputs=new String[]{"A", "A"};
		InputInfo[] inputInfos=new InputInfo[]{textinputinfo, textinputinfo};
		long[] rlens=new long[]{2, 2};
		long[] clens=new long[]{2, 2};
		int[] brlens=new int[]{1, 1};
		int[] bclens=new int[]{1, 2};
		String instructionsInMapper="";
		String aggInstructionsInReducer="";
		String aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE"; 
		String output="temp";
		int numReducers=1;
		int replication=1;
	
		MMCJMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, aggBinInstrction, numReducers, 
				replication, (byte)0, output, textoutputinfo, partialAggCacheSize);
		
		inputs=new String[]{"temp"};
		inputInfos=new InputInfo[]{textinputinfo};
		rlens=new long[]{2};
		clens=new long[]{2};
		brlens=new int[]{1};
		bclens=new int[]{2};
		instructionsInMapper="";
		aggInstructionsInReducer="";
		String otherInstructionsInReducer="";
		byte[] resultIndexes=new byte[]{0};
		byte[] resultDimsUnknown=new byte[]{0};
		String[] outputs=new String[]{"basicTest_output0"};
		OutputInfo[] outputInfos=new OutputInfo[]{textoutputinfo};
		
		GMR.runJob(blockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
	
	public void mmult3(String[] args)throws Exception
	{
		String inputs="A.binary,B.binary";
		String aggBinaryOperation="ba+*";
		String output="mmult3_out";
		int numReducers=1;
		int replication=1;
		int nr1=2;
		int nc1=2;
		int nr2=2;
		int nc2=2;
		int bnr1=1;
		int bnc1=2;
		int bnr2=2;
		int bnc2=1;
		ABMR.runJob(inputs, aggBinaryOperation, output, 
				numReducers, replication, nr1, nc1, nr2, nc2, bnr1, bnc1, bnr2, bnc2);
	}
	
	public static void main(String[] args) throws Exception {
		//int errCode = ToolRunner.run(new Configuration(), new Test(), args);
		//System.exit(errCode);
		double[][] A=new double[][] { { 1, 1 } };
		for(int i=0; i<A.length; i++)
			for(int j=0; j<A[i].length; j++)
				System.out.println(i+","+j+": "+A[i][j]);
	}
}
