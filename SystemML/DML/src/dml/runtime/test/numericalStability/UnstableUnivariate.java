package dml.runtime.test.numericalStability;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.GMR;
import dml.runtime.matrix.RandMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class UnstableUnivariate {
	
	public static final String sps=Instruction.OPERAND_DELIM;
	public static final String tp=Instruction.VALUETYPE_PREFIX;
	public static String powerInstructionString(byte in, int cst, byte out)
	{
		return "s^"+sps+in+tp+"DOUBLE"+sps+cst+tp+"INT"+sps+out+tp+"DOUBLE";
	}
	public static String sumInstructionString(String op, byte in, byte out)
	{
		return op+sps+in+tp+"DOUBLE"+sps+out+tp+"DOUBLE";
	}
	public static void main(String[] args) throws Exception {
		
		if(args.length<5)
		{
			System.out.println("SystemMLUnivariate <input> <output> <numRows> <numReducers> <blocksize>");
			System.exit(-1);
		}
		
		
		boolean inBlockRepresentation=false;
		String[] inputs=new String[]{args[0]};
		long n=Long.parseLong(args[2]);
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
		
		String outdir=args[1];
		String recordReaderInstruction="";
		String instructionsInMapper=powerInstructionString((byte)0, 2, (byte)1);
		instructionsInMapper+=","+powerInstructionString((byte)0, 3, (byte)2);
		instructionsInMapper+=","+powerInstructionString((byte)0, 4, (byte)3);
		instructionsInMapper+=","+sumInstructionString("ua+", (byte)0, (byte)4);
		instructionsInMapper+=","+sumInstructionString("ua+", (byte)1, (byte)5);
		instructionsInMapper+=","+sumInstructionString("ua+", (byte)2, (byte)6);
		instructionsInMapper+=","+sumInstructionString("ua+", (byte)3, (byte)7);
		String aggInstructionsInReducer=sumInstructionString("a+", (byte)4, (byte)8);
		aggInstructionsInReducer+=","+sumInstructionString("a+", (byte)5, (byte)9);
		aggInstructionsInReducer+=","+sumInstructionString("a+", (byte)6, (byte)10);
		aggInstructionsInReducer+=","+sumInstructionString("a+", (byte)7, (byte)11);
		String otherInstructionsInReducer="";
		int numReducers=Integer.parseInt(args[3]);
		int replication=1;
		byte[] resultIndexes=new byte[]{8,9,10,11};
		byte[] resultDimsUnknown=new byte[]{1,1,1,1};
		String[] outputs=new String[]{outdir+"/Summation", outdir+"/S2", outdir+"/S3", outdir+"/S4"};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo,
				OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo};
		GMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				recordReaderInstruction, instructionsInMapper, aggInstructionsInReducer,
				otherInstructionsInReducer, numReducers, replication, resultIndexes, 
				resultDimsUnknown, outputs, outputInfos);
		
		double s1=SystemMLCovariance.getResult(outdir, "Summation");
		double s2=SystemMLCovariance.getResult(outdir, "S2");
		double s3=SystemMLCovariance.getResult(outdir, "S3");
		double s4=SystemMLCovariance.getResult(outdir, "S4");
		
		/*
		 * # Skewness
			//g1 = (s3 - 3*mu*s2 + 3*mu^2*s1 -n*mu^3)/(n*std_dev^3)
			g1 = (s3 - 3*mu*s2 + 2*n*mu^3)/(n*std*var)
			
		 * # Kurtosis (using binomial formula)
			g2 = (s4 - 4*s3*mu + 6*s2*mu^2 - 3*n*mu^4)/(n*var^2) - 3
			g2 = (sum(V^4) - 4*s3*mu + 6*s2*mu^2 - 3*n*mu^4)/(n*std_dev^4) - 3
		 */
		double mu = s1/n;
		double var= (s2-s1*mu)/(n-1);
		double std = Math.sqrt(var);
		double g1 = (s3 - 3*mu*s2 + 2*s1*mu*mu)/n/std/var;
		double g2 = (s4 - 4*s3*mu + 6*s2*mu*mu - 3*s1*Math.pow(mu,3))/n/var/var - 3;
		//System.out.println(g1);
		//System.out.println(g2);
		
		String[] randInstructions=new String[]{"Rand"+sps+0+sps+0+sps+"rows=1"+sps+"cols=1"
				+sps+"min=1.0"+sps+"max=1.0"+sps+"sparsity=1.0"+sps+"pdf=uniform"};
		brlens=new int[]{1};
		bclens=new int[]{1};
		instructionsInMapper=SystemMLUnivariate.unaryInstructionString("s*", (byte)0, mu, (byte)1);
		instructionsInMapper+=","+SystemMLUnivariate.unaryInstructionString("s*", (byte)0, var, (byte)2);
		instructionsInMapper+=","+SystemMLUnivariate.unaryInstructionString("s*", (byte)0, std, (byte)3);
		instructionsInMapper+=","+SystemMLUnivariate.unaryInstructionString("s*", (byte)0, g1, (byte)4);
		instructionsInMapper+=","+SystemMLUnivariate.unaryInstructionString("s*", (byte)0, g2, (byte)5);
		resultIndexes=new byte[]{1, 2,3,4, 5};
		outputs=new String[]{outdir+"/Mean", outdir+"/Variance", outdir+"/Std", 
				outdir+"/Skewness", outdir+"/Kurtosis"};
		outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo};
		RandMR.runJob(randInstructions, brlens, bclens, instructionsInMapper, "", "", 0, replication, 
				resultIndexes, resultIndexes, outputs, outputInfos);
	}

}
