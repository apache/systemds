package dml.runtime.test.numericalStability;

import java.io.IOException;
import dml.runtime.matrix.CMCOVMR;
import dml.runtime.matrix.CombineMR;
import dml.runtime.matrix.RandMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.util.MapReduceTool;

public class SystemMLCovariance {

	public static double getResult(String folder,	String name) throws IOException {
		double[][] array;
		array = MapReduceTool.readMatrixFromHDFS(folder+"/"+name, InputInfo.BinaryCellInputInfo, 1, 1, 1, 1);
		return array[0][0];
	}
	
	public static void main(String[] args) throws Exception {
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
		if(npb>1)
		{
			inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo,};
			inBlockRepresentation=true;
		}
		long r=Long.parseLong(args[2]);
		long[] rlens=new long[]{r, r};
		long[] clens=new long[]{1, 1};
		int[] brlens=new int[]{npb, npb};
		int[] bclens=new int[]{1, 1};
		String combineInstructions="combinebinary\u00b0false\u00b7BOOLEAN\u00b00\u00b7DOUBLE\u00b01\u00b7DOUBLE\u00b02\u00b7DOUBLE";
		int numReducers=Integer.parseInt(args[3]);
		int replication=1;
		byte[] resultIndexes=new byte[]{2};
		String UV="UV";
		String[] outputs=new String[]{UV};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.WeightedPairOutputInfo};
	//	inBlockRepresentation=false;
		CombineMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				combineInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
		
		inputs=new String[]{U, V, UV};
		inputInfos=new InputInfo[]{inputInfos[0], inputInfos[1], InputInfo.WeightedPairInputInfo};
		rlens=new long[]{r, r, r};
		clens=new long[]{1, 1, 1};
		brlens=new int[]{npb, npb, 1};
		bclens=new int[]{1, 1, 1};
		String instructionsInMapper="";
		String cmNcomInstructions="cm\u00b00\u00b7DOUBLE\u00b02\u00b7INT\u00b03\u00b7DOUBLE,cm\u00b01\u00b7DOUBLE\u00b02\u00b7INT\u00b04\u00b7DOUBLE,cov\u00b02\u00b7DOUBLE\u00b05\u00b7DOUBLE";
		//"mean\u00b00\u00b7DOUBLE\u00b01\u00b7DOUBLE";
		resultIndexes=new byte[]{3,4,5};
		outputs=new String[]{outdir+"/Var1", outdir+"/Var2", outdir+"/Covariance"};
		outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo, OutputInfo.BinaryCellOutputInfo};
		CMCOVMR.runJob(inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				cmNcomInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
		double sigmaX=Math.sqrt(r/(r-1.0)*getResult(outdir, "Var1"));		
		double sigmaY=Math.sqrt(r/(r-1.0)*getResult(outdir, "Var2"));
		double covXY=getResult(outdir, "Covariance");
		double pearsonR = covXY/sigmaX/sigmaY;//*r/(r-1.0);
/*		System.out.println("std1: "+sigmaX);
		System.out.println("std2: "+sigmaY);
		System.out.println("cov: "+covXY);
		System.out.println("pearsonR: "+pearsonR);*/
		
		String[] randInstructions=new String[]{"Rand\u00b00\u00b00\u00b0rows=1\u00b0cols=1\u00b0min=1.0\u00b0max=1.0\u00b0sparsity=1.0\u00b0pdf=uniform"};
		brlens=new int[]{1};
		bclens=new int[]{1};
		instructionsInMapper="s*\u00b00\u00b7DOUBLE\u00b0"+pearsonR+"\u00b7DOUBLE\u00b01\u00b7DOUBLE";
		resultIndexes=new byte[]{1};
		outputs=new String[]{outdir+"/PearsonR"};
		outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo};
		RandMR.runJob(randInstructions, brlens, bclens, instructionsInMapper, "", "", 0, replication, 
				resultIndexes, resultIndexes, outputs, outputInfos);
	}
}
