package dml.runtime.test.numericalStability;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.CMCOVMR;
import dml.runtime.matrix.GMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class SystemMLUnivariate {
	
	public static final String sps=Instruction.OPERAND_DELIM;
	public static final String tp=Instruction.VALUETYPE_PREFIX;
	public static String cmInstructionString(byte in, int p, byte out)
	{
		return "cm"+sps+in+tp+"DOUBLE"+sps+p+tp+"INT"+sps+out+tp+"DOUBLE";
	}
	
	public static String unaryInstructionString(String op, byte in, double cst, byte out)
	{
		return op+sps+in+tp+"DOUBLE"+sps+cst+tp+"DOUBLE"+sps+out+tp+"DOUBLE";
	}
	
	
	public static void main(String[] args) throws Exception {
	
		
		if(args.length<5)
		{
			System.out.println("SystemMLUnivariate <input> <output> <numRows> <numReducers> <block size>");
			System.exit(-1);
		}
		
		boolean inBlockRepresentation=false;
		String[] inputs=new String[]{args[0]};
		long n=Long.parseLong(args[2]);
		String outdir=args[1];
		int npb=Integer.parseInt(args[4]);
		InputInfo[] inputInfos=new InputInfo[]{InputInfo.TextCellInputInfo};
		if(npb>1)
		{
			inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo};
			inBlockRepresentation=true;
		}
		long[] rlens=new long[]{n};
		long[] clens=new long[]{1};
		int[] brlens=new int[]{npb};
		int[] bclens=new int[]{1};
		String instructionsInMapper="";
		String cmNcomInstructions="mean"+sps+0+tp+"DOUBLE"+sps+1+tp+"DOUBLE";
		cmNcomInstructions+=","+cmInstructionString((byte)0, 2, (byte)2);
		cmNcomInstructions+=","+cmInstructionString((byte)0, 3, (byte)3);
		cmNcomInstructions+=","+cmInstructionString((byte)0, 4, (byte)4);
		
		//cm\u00b00\u00b7DOUBLE\u00b02\u00b7INT\u00b03\u00b7DOUBLE
		int numReducers=Integer.parseInt(args[3]);
		int replication=1;
		byte[] resultIndexes=new byte[]{1, 2, 3, 4};
		String[] outputs=new String[]{outdir+"/Mean", outdir+"/CM2", outdir+"/CM3", outdir+"/CM4"};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo, 
				OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo};
		CMCOVMR.runJob(inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				cmNcomInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
		
		double mu=SystemMLCovariance.getResult(outdir, "Mean");
		double cm2=SystemMLCovariance.getResult(outdir, "CM2");
		double cm3=SystemMLCovariance.getResult(outdir, "CM3");
		double cm4=SystemMLCovariance.getResult(outdir, "CM4");

		double var = n/(n-1.0)*cm2;
		double std_dev = Math.sqrt(var);
		double g1 = cm3/var/std_dev;
		double g2 = cm4/var/var - 3;
		
		instructionsInMapper="uak+"+sps+"0"+tp+"DOUBLE"+sps+"1"+tp+"DOUBLE";
		String aggInstructionsInReducer="ak+"+sps+"1"+tp+"DOUBLE"+sps+"2"+tp+"DOUBLE"+sps+"true"+sps+"2";
		String otherInstructionsInReducer=unaryInstructionString("s*", (byte)2, 0, (byte)3);
		otherInstructionsInReducer+=","+unaryInstructionString("s+", (byte)3, var, (byte)4);
		otherInstructionsInReducer+=","+unaryInstructionString("s+", (byte)3, std_dev, (byte)5);
		otherInstructionsInReducer+=","+unaryInstructionString("s+", (byte)3, g1, (byte)6);
		otherInstructionsInReducer+=","+unaryInstructionString("s+", (byte)3, g2, (byte)7);
		//otherInstructionsInReducer+=","+unaryInstructionString("s/", (byte)2, n, (byte)8);
		
		resultIndexes=new byte[]{2, 4, 5, 6, 7};//, 8};
		outputs=new String[]{outdir+"/Summation", outdir+"/Variance", outdir+"/Std", 
				outdir+"/Skewness", outdir+"/Kurtosis"};//, outdir+"/Mean"};
		outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo,
				OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo};//,OutputInfo.BinaryCellOutputInfo};
		inBlockRepresentation=true;
		GMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, resultIndexes, 
				new byte[]{1,1,1,1,1}/*, 1}*/, outputs, outputInfos);

	}

}
