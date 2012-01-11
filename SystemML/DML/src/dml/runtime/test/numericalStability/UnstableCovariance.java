package dml.runtime.test.numericalStability;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.GMR;
import dml.runtime.matrix.RandMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class UnstableCovariance {

	public static final String sps=Instruction.OPERAND_DELIM;
	public static final String tp=Instruction.VALUETYPE_PREFIX;
	/**
	 * @param args
	 * @throws Exception 
	 * 
	 * 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		
		if(args.length<6)
		{
			System.out.println("SystemMLCovariance <input1> <input2> <num Rows> <# reducers> <output> <blocksize>");
			System.exit(-1);
		}
		
		boolean inBlockRepresentation=false;
		String V=args[0];
		String U=args[1];
		String outdir=args[4];
		String[] inputs=new String[]{V, U};
		int npb=Integer.parseInt(args[5]);
		InputInfo[] inputInfos=new InputInfo[]{InputInfo.TextCellInputInfo, InputInfo.TextCellInputInfo};
		OutputInfo tempout=OutputInfo.BinaryCellOutputInfo;
		InputInfo tempin=InputInfo.BinaryCellInputInfo;
		if(npb>1)
		{
			inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo,};
			inBlockRepresentation=true;
			tempout=OutputInfo.BinaryBlockOutputInfo;
			tempin=InputInfo.BinaryBlockInputInfo;
		}
		long r=Long.parseLong(args[2]);
		long[] rlens=new long[]{r, r};
		long[] clens=new long[]{1, 1};
		int[] brlens=new int[]{npb, npb};
		int[] bclens=new int[]{1, 1};
		int numReducers=Integer.parseInt(args[3]);
		int replication=1;
		
		String instructionsInMapper=UnstableUnivariate.powerInstructionString((byte)0, 2, (byte)2);
		instructionsInMapper+=","+UnstableUnivariate.powerInstructionString((byte)1, 2, (byte)3);
		instructionsInMapper+=","+UnstableUnivariate.sumInstructionString("ua+", (byte)0, (byte)4);//sx
		instructionsInMapper+=","+UnstableUnivariate.sumInstructionString("ua+", (byte)1, (byte)5);//sy
		instructionsInMapper+=","+UnstableUnivariate.sumInstructionString("ua+", (byte)2, (byte)6);//sx2
		instructionsInMapper+=","+UnstableUnivariate.sumInstructionString("ua+", (byte)3, (byte)7);//sy2
		String aggInstructionsInReducer=UnstableUnivariate.sumInstructionString("a+", (byte)4, (byte)8);
		aggInstructionsInReducer+=","+UnstableUnivariate.sumInstructionString("a+", (byte)5, (byte)9);
		aggInstructionsInReducer+=","+UnstableUnivariate.sumInstructionString("a+", (byte)6, (byte)10);
		aggInstructionsInReducer+=","+UnstableUnivariate.sumInstructionString("a+", (byte)7, (byte)11);
		String otherInstructionsInReducer="b*"+sps+0+tp+"DOUBLE"+sps+1+tp+"DOUBLE"+sps+12+tp+"DOUBLE";//xy
		
		byte[] resultIndexes=new byte[]{8, 9, 10, 11, 12};
		byte[] resultDimsUnknown=new byte[]{0,0,0,0,0};
		String[] outputs=new String[]{outdir+"/Sx", outdir+"/Sy", outdir+"/Sx2", 
				outdir+"/Sy2", "XY"};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo,
				OutputInfo.BinaryCellOutputInfo,OutputInfo.BinaryCellOutputInfo,tempout};
		
		GMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		
		inputs=new String[]{"XY"};
		inputInfos=new InputInfo[]{tempin};
		rlens=new long[]{r};
		clens=new long[]{1};
		brlens=new int[]{npb};
		bclens=new int[]{1};
		instructionsInMapper=UnstableUnivariate.sumInstructionString("ua+", (byte)0, (byte)1);//sxy
		aggInstructionsInReducer=UnstableUnivariate.sumInstructionString("a+", (byte)1, (byte)2);
		otherInstructionsInReducer="";
		resultIndexes=new byte[]{2};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{outdir+"/Sxy"};
		outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo};
		GMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		double sx=SystemMLCovariance.getResult(outdir, "Sx");
		double sx2=SystemMLCovariance.getResult(outdir, "Sx2");
		double sy=SystemMLCovariance.getResult(outdir, "Sy");
		double sy2=SystemMLCovariance.getResult(outdir, "Sy2");
		double sxy=SystemMLCovariance.getResult(outdir, "Sxy");
		
		//covariance=(sxy-(sumx.sumy/n)/(n-1)
		double cov=(sxy-sx*sy/r)/(r-1);
		//var=(s2/n - mu*mu)*(n/(n-1))=(s2-s1*mu)/(n-1)
		double varx=(sx2-sx*sx/r)/(r-1);
		double vary=(sy2-sy*sy/r)/(r-1);
		
		double pearsonR=Double.NaN;
		if(varx>0&&vary>0)
		{
			
			double stdx=Math.sqrt(varx);
			double stdy=Math.sqrt(varx);
			pearsonR=cov/stdx/stdy;
		}
		
	/*	System.out.println("std1: "+stdx);
		System.out.println("std2: "+stdy);
		System.out.println("cov: "+cov);
		System.out.println("pearsonR: "+pearsonR);*/
		
		String[] randInstructions=new String[]{"Rand"+sps+0+sps+0+sps+"rows=1"+sps+"cols=1"
				+sps+"min=1.0"+sps+"max=1.0"+sps+"sparsity=1.0"+sps+"pdf=uniform"};
		brlens=new int[]{1};
		bclens=new int[]{1};
		instructionsInMapper=SystemMLUnivariate.unaryInstructionString("s*", (byte)0, cov, (byte)1);
		instructionsInMapper+=","+SystemMLUnivariate.unaryInstructionString("s*", (byte)0, pearsonR, (byte)2);
		resultIndexes=new byte[]{1, 2};
		outputs=new String[]{outdir+"/Covariance", outdir+"/PearsonR"};
		outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo};
		RandMR.runJob(randInstructions, brlens, bclens, instructionsInMapper, "", "", 0, replication, 
				resultIndexes, resultIndexes, outputs, outputInfos);
	}

}
