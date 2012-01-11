package dml.runtime.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import dml.runtime.matrix.CMCOVMR;
import dml.runtime.matrix.CombineMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class TestCM extends Configured implements Tool{

	@Override
	public int run(String[] args) throws Exception {
		
		test2();
		
		return 0;
	}
	
	void test2()  throws Exception 
	{
		String V="V";
		String U="U";
		String W="W";
		String UVW="UVW";
		String UV="UV";
		
		long r=10, c=1; 
		int br=1, bc=1;
		
		boolean inBlockRepresentation=false;
		String[] inputs=new String[]{V, W, U};
		InputInfo[] inputInfos=new InputInfo[]{InputInfo.TextCellInputInfo, InputInfo.TextCellInputInfo, InputInfo.TextCellInputInfo};
		long[] rlens=new long[]{r, r, r};
		long[] clens=new long[]{1, 1, 1};
		int[] brlens=new int[]{br, br, br};
		int[] bclens=new int[]{1, 1, 1};
		String combineInstructions="combinebinary:::false:BOOLEAN:::2:DOUBLE:::0:DOUBLE:::3:DOUBLE,combinetertiary:::2:DOUBLE:::0:DOUBLE:::1:DOUBLE:::4:DOUBLE";
		int numReducers=3;
		int replication=1;
		byte[] resultIndexes=new byte[]{3, 4};
		String[] outputs=new String[]{UV, UVW};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.WeightedPairOutputInfo, OutputInfo.WeightedPairOutputInfo};
		CombineMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				combineInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
		
		inputs=new String[]{V, UV, UVW};
		inputInfos=new InputInfo[]{InputInfo.TextCellInputInfo, InputInfo.WeightedPairInputInfo, InputInfo.WeightedPairInputInfo};
		String instructionsInMapper="s+:::0:DOUBLE:::0.0:DOUBLE:::3:DOUBLE";
		String cmInstructions="cm:::3:DOUBLE:::4:INT:::5:DOUBLE,cov:::1:DOUBLE:::6:DOUBLE,cov:::2:DOUBLE:::7:DOUBLE";
		resultIndexes=new byte[]{5, 6, 7};
		outputs=new String[]{"out5", "out6", "out7"};
		outputInfos=new OutputInfo[]{OutputInfo.TextCellOutputInfo, OutputInfo.TextCellOutputInfo, OutputInfo.TextCellOutputInfo};
		CMCOVMR.runJob(inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				cmInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
	}
	
	void test1()  throws Exception 
	{
		String V="V";
		String U="U";
		String W="W";
		String VW="VW";
		String UVW="UVW";
		
		long r=10, c=1; 
		int br=1, bc=1;
		
		GenerateTestMatrixBlock.runJob(2, r, c, U, 1, br, bc, System.currentTimeMillis(), 1, true);
		GenerateTestMatrixBlock.runJob(2, r, c, V, 1, br, bc, System.currentTimeMillis(), 1, true);
		GenerateTestMatrixBlock.runJob(2, r, c, W, 1, br, bc, System.currentTimeMillis(), 1, true);
		
		boolean inBlockRepresentation=false;
		String[] inputs=new String[]{V, W, U};
		InputInfo[] inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo};
		long[] rlens=new long[]{r, r, r};
		long[] clens=new long[]{1, 1, 1};
		int[] brlens=new int[]{br, br, br};
		int[] bclens=new int[]{1, 1, 1};
		String combineInstructions="combinebinary:::true:BOOLEAN:::0:DOUBLE:::1:DOUBLE:::3:DOUBLE,combinetertiary:::2:DOUBLE:::0:DOUBLE:::1:DOUBLE:::4:DOUBLE";
		int numReducers=3;
		int replication=1;
		byte[] resultIndexes=new byte[]{3, 4};
		String[] outputs=new String[]{VW, UVW};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.WeightedPairOutputInfo, OutputInfo.WeightedPairOutputInfo};
		CombineMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				combineInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
		
		inputs=new String[]{V, VW, UVW};
		inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo, InputInfo.WeightedPairInputInfo, InputInfo.WeightedPairInputInfo};
		String instructionsInMapper="s+:::0:DOUBLE:::0.0:DOUBLE:::3:DOUBLE,s+:::1:DOUBLE:::0.0:DOUBLE:::4:DOUBLE";
		String cmInstructions="cm:::3:DOUBLE:::4:INT:::5:DOUBLE,cm:::4:DOUBLE:::4:INT:::6:DOUBLE,cov:::2:DOUBLE:::7:DOUBLE";
		resultIndexes=new byte[]{5, 6, 7};
		outputs=new String[]{"out5", "out6", "out7"};
		outputInfos=new OutputInfo[]{OutputInfo.TextCellOutputInfo, OutputInfo.TextCellOutputInfo, OutputInfo.TextCellOutputInfo};
		CMCOVMR.runJob(inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				cmInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
	}

	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new TestCM(), args);
		System.exit(errCode);
	}
}
