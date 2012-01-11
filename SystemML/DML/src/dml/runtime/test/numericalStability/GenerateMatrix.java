package dml.runtime.test.numericalStability;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.RandMR;
import dml.runtime.matrix.io.OutputInfo;

public class GenerateMatrix {
	public static final String sps=Instruction.OPERAND_DELIM;
	public static final String tp=Instruction.VALUETYPE_PREFIX;
	public static void main(String[] args) throws Exception {
		
		if(args.length<6)
		{
			System.out.println("GenerateMatrix <file> <numRow> <numRowsPerBlock> <min> <max> <round?>");
			System.exit(-1);
		}
		
		String outdir=args[0];
		long n=Long.parseLong(args[1]);
		int npb=Integer.parseInt(args[2]);
		double min=Double.parseDouble(args[3]);
		double max=Double.parseDouble(args[4]);
		boolean round=Boolean.parseBoolean(args[5]);
		int[] brlens=new int[]{npb};
		int[] bclens=new int[]{1};
		
		String[] randInstructions=new String[]{"Rand"+sps+0+sps+0+sps+"rows="+n+sps+"cols=1"
				+sps+"min="+min+sps+"max="+max+sps+"sparsity=1.0"+sps+"pdf=uniform"};
		
		
		String instructionsInMapper="";
		if(round)
			instructionsInMapper="round"+sps+0+tp+"DOUBLE"+sps+0+tp+"DOUBLE";
		String aggInstructionsInReducer="";
		String otherInstructionsInReducer="";
		int numReducers=0;
		int replication=1;
		byte[] resultIndexes=new byte[]{0};
		byte[] resultDimsUnknown=new byte[]{1};
		String[] outputs=new String[]{outdir};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.BinaryBlockOutputInfo};
		if(npb==1)
			outputInfos=new OutputInfo[]{OutputInfo.TextCellOutputInfo};
		RandMR.runJob(randInstructions, brlens, bclens, instructionsInMapper, aggInstructionsInReducer, 
				otherInstructionsInReducer, numReducers, replication, resultIndexes, resultDimsUnknown, 
				outputs, outputInfos);
	}
}
