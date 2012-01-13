package dml.runtime.test.numericalStability;

import dml.runtime.matrix.CMCOVMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;

public class StableMean {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		if(args.length<5)
		{
			System.out.println("StableMean <input> <output> <numRows> <numReducers> <blocksize>");
			System.exit(-1);
		}
		
		String[] inputs=new String[]{args[0]};
		InputInfo[] inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo};
		long[] rlens=new long[]{Long.parseLong(args[2])};
		long[] clens=new long[]{1};
		int[] brlens=new int[]{Integer.parseInt(args[4])};
		int[] bclens=new int[]{1};
		String instructionsInMapper="";
		String cmNcomInstructions="mean\u00b00\u00b7DOUBLE\u00b01\u00b7DOUBLE";
		int numReducers=Integer.parseInt(args[3]);
		int replication=3;
		byte[] resultIndexes=new byte[]{1};
		String[] outputs=new String[]{args[1]+"/Mean"};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.BinaryCellOutputInfo};
		CMCOVMR.runJob(inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				cmNcomInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);

	}

}
