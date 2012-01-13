package dml.runtime.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import dml.runtime.matrix.CombineMR;
import dml.runtime.matrix.GroupedAggMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class TestGroupedAgg extends Configured implements Tool{

	@Override
	public int run(String[] args) throws Exception {
		
		test();
		
		return 0;
	}
	
	void test()  throws Exception 
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
		String combineInstructions="combinetertiary\u00b02\u00b7DOUBLE\u00b00\u00b7DOUBLE\u00b01\u00b7DOUBLE\u00b04\u00b7DOUBLE";
		int numReducers=3;
		int replication=1;
		byte[] resultIndexes=new byte[]{4};
		String[] outputs=new String[]{UVW};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.WeightedPairOutputInfo};
		CombineMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				combineInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
		
		inputs=new String[]{UVW};
		rlens=new long[]{r};
		clens=new long[]{1};
		brlens=new int[]{br};
		bclens=new int[]{1};
		inputInfos=new InputInfo[]{InputInfo.WeightedPairInputInfo};
		String grpaggInstructions="groupedagg\u00b00\u00b7DOUBLE\u00b0sum\u00b7STRING\u00b01\u00b7DOUBLE";
		String reduceInstructions="s-\u00b01\u00b7DOUBLE\u00b01\u00b7DOUBLE\u00b01\u00b7DOUBLE";
		resultIndexes=new byte[]{1};
		outputs=new String[]{"out1"};
		outputInfos=new OutputInfo[]{OutputInfo.TextCellOutputInfo};
		GroupedAggMR.runJob(inputs, inputInfos, rlens, clens, brlens, bclens, 
				grpaggInstructions, reduceInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
	}
	
	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new TestGroupedAgg(), args);
		System.exit(errCode);
	}
}
