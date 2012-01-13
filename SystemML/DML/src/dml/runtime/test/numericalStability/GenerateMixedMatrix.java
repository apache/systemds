package dml.runtime.test.numericalStability;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.GMR;
import dml.runtime.matrix.RandMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class GenerateMixedMatrix {

	public static final String sps=Instruction.OPERAND_DELIM;
	public static final String tp=Instruction.VALUETYPE_PREFIX;
	public static void main(String[] args) throws Exception {
		
		if(args.length<9)
		{
			System.out.println("GenerateMixedMatrix <file> <numRow> <numRowsPerBlock> <min1> <max1> <min2> <max2> <sparcity> <numReducers>");
			System.exit(-1);
		}
		
		String outdir=args[0];
		long n=Long.parseLong(args[1]);
		int npb=Integer.parseInt(args[2]);
		double min1=Double.parseDouble(args[3]);
		double max1=Double.parseDouble(args[4]);
		double min2=Double.parseDouble(args[5]);
		double max2=Double.parseDouble(args[6]);
		double sparcity=Double.parseDouble(args[7]);
		int[] brlens=new int[]{npb, npb};
		int[] bclens=new int[]{1, 1};
		
		String[] randInstructions=new String[]{"Rand"+sps+0+sps+0+sps+"rows="+n+sps+"cols=1"
				+sps+"min="+min1+sps+"max="+max1+sps+"sparsity=1.0"+sps+"pdf=uniform",
				"Rand"+sps+1+sps+1+sps+"rows="+n+sps+"cols=1"
				+sps+"min="+min2+sps+"max="+max2+sps+"sparsity="+sparcity+sps+"pdf=uniform"};
		
		String instructionsInMapper="";
		String aggInstructionsInReducer="";
		String otherInstructionsInReducer="";
		int numReducers=0;
		int replication=1;
		byte[] resultIndexes=new byte[]{0, 1};
		byte[] resultDimsUnknown=new byte[]{1, 1};
		String[] outputs=new String[]{"temp1", "temp2"};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.BinaryBlockOutputInfo, OutputInfo.BinaryBlockOutputInfo};
		if(npb==1)
			outputInfos=new OutputInfo[]{OutputInfo.TextCellOutputInfo, OutputInfo.TextCellOutputInfo};
		RandMR.runJob(randInstructions, brlens, bclens, instructionsInMapper, aggInstructionsInReducer, 
				otherInstructionsInReducer, numReducers, replication, resultIndexes, resultDimsUnknown, 
				outputs, outputInfos);
		
		String[] inputs=new String[]{"temp1", "temp2"};
		
		InputInfo[] inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo};
		if(npb==1)
			inputInfos=new InputInfo[]{InputInfo.TextCellInputInfo, InputInfo.TextCellInputInfo};
		
		long[] rlens=new long[]{n, n};
		long[] clens=new long[]{1, 1};
		numReducers=Integer.parseInt(args[8]);
		resultIndexes=new byte[]{2};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{outdir};
		otherInstructionsInReducer="b+"+sps+0+tp+"DOUBLE"+sps+1+tp+"DOUBLE"+sps+2+tp+"DOUBLE";
		outputInfos=new OutputInfo[]{OutputInfo.BinaryBlockOutputInfo};
		if(npb==1)
			outputInfos=new OutputInfo[]{OutputInfo.TextCellOutputInfo};
		GMR.runJob(npb>1, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
				resultIndexes, resultDimsUnknown, outputs, outputInfos);
	}
}
