package dml.runtime.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import dml.runtime.matrix.ReblockMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class ReBlockTest extends Configured implements Tool{

	@Override
	public int run(String[] args) throws Exception {
		if(args.length<11)
		{
			System.err.println("ReBlockTest: <input path> <input format> <#row> <#col> <#row per block> " +
					"<#col per block> <taget #row per block> <taget #col per block> <output path> <#reducers> <replication factor>");
			System.err.println("input format: 1(text cell), 2(binary cell), 3(binary block)");
			System.exit(-1);
		}
		String[] inputs=new String[]{args[0]};
		int format=Integer.parseInt(args[1]);
		InputInfo[] inputInfos=new InputInfo[1];
		switch(format)
		{
		case 1: inputInfos[0]=InputInfo.TextCellInputInfo;
		break;
		case 2: inputInfos[0]=InputInfo.BinaryCellInputInfo;
		break;
		case 3: inputInfos[0]=InputInfo.BinaryBlockInputInfo;
		break;
		}
		
		long[] rlens=new long[]{Long.parseLong(args[2])};
		long[] clens=new long[]{Long.parseLong(args[3])};
		int[] brlens=new int[]{Integer.parseInt(args[4])};
		int[] bclens=new int[]{Integer.parseInt(args[5])};
		
		int targetBrlens=Integer.parseInt(args[6]);
		int targetBclens=Integer.parseInt(args[7]);
		
		String[] outputs=new String[]{args[8]};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.BinaryBlockOutputInfo};
		
		String instructionsInMapper="";
		String reblockInstructions="rblk:::0:DOUBLE:::0:DOUBLE:::"+targetBrlens+":::"+targetBclens;
		String otherInstructionsInReducer="";
		byte[] resultIndexes=new byte[]{0};
		byte[] resultDimsUnknown=new byte[]{0};
		int numReducers=Integer.parseInt(args[9]);
		int replication=Integer.parseInt(args[10]);
		
		
		ReblockMR.runJob(inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				reblockInstructions, otherInstructionsInReducer, numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		return 0;
	}
	
	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new ReBlockTest(), args);
		System.exit(errCode);
	}

}
