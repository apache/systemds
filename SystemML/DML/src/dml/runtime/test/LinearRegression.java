package dml.runtime.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import dml.runtime.matrix.GMR;
import dml.runtime.matrix.MMCJMR;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MetaData;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.util.MapReduceTool;

public class LinearRegression extends Configured implements Tool{

	public static InputInfo binaryinputinfo=InputInfo.BinaryBlockInputInfo;
	public static OutputInfo binaryoutputinfo=OutputInfo.BinaryBlockOutputInfo;
	String X;
	String smallx;
	String y;
	String r="r";
	String p="p";
	String norm_r2_file="norm_r2";
	long n;
	long d;
	int bn;
	int bd;
	double tol;
	private int maxIter;
	private double lambda;
	private String q="q";
	
	@Override
	public int run(String[] args) throws Exception {
		
		if(args.length<13)
		{
			System.err.println("[A] [y] [x] [n] [d] [bn] [bd] [lambda] [tol] [maxIter] [# mappers] [# reducers] [aggregator buffer]");
			System.exit(-1);
		}
		
		long totalStart=System.currentTimeMillis();
		
		X=args[0];
		y=args[1];
		smallx=args[2];
		n=Long.parseLong(args[3]);
		d=Long.parseLong(args[4]);
		bn=Integer.parseInt(args[5]);
		bd=Integer.parseInt(args[6]);
		lambda=Double.parseDouble(args[7]);
		tol=Double.parseDouble(args[8]);
		maxIter=Integer.parseInt(args[9]);
		int numMappers=Integer.parseInt(args[10]);
		int numReducers=Integer.parseInt(args[11]);
		int partialAggCacheSize=Integer.parseInt(args[12]);
		int replication=1;
		
		String[] inputs;
		InputInfo[] inputInfos;
		long[] rlens;
		long[] clens;
		int[] brlens;
		int[] bclens;
		String instructionsInMapper;
		String aggInstructionsInReducer;
		String otherInstructionsInReducer;
		byte[] resultIndexes;
		byte[] resultDimsUnknown;
		String[] outputs;
		OutputInfo[] outputInfos;
		String aggBinInstrction;
		
		String output;
		OutputInfo outputinfo;
		String temp="temp";
		
		
		//-----------------
	//	GenerateTestMatrixBlock.runJob(numMappers, n, d, X, 1, bn, bd, System.currentTimeMillis(), 1, true);
		//------------------
		
		//
	//	GenerateTestMatrixBlock.runJob(numMappers, n, 1, y, 1, bn, 1, System.currentTimeMillis(), 1, true);
		
		//
//		GenerateTestMatrixBlock.runJob(numMappers, d, 1, smallx, 1, bd, 1, System.currentTimeMillis(), 1, true);
		
		
		
		//p = t(X) %*% y
		//mmult
		inputs=new String[]{X, y};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n};
		clens=new long[]{d, 1};
		brlens=new int[]{bn, bn};
		bclens=new int[]{bd, 1};
		instructionsInMapper="r':::0:DOUBLE:::0:DOUBLE";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
		output = temp;
		outputinfo = binaryoutputinfo;
		
				
		MetaData[] md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, aggBinInstrction, numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize).getMetaData();
		
	//	for(MatrixCharacteristics stat: stats)
	//		System.out.println(stat);
		
		//r=-p
		inputs=new String[]{temp};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{d};
		clens=new long[]{1};
		brlens=new int[]{bd};
		bclens=new int[]{1};
		instructionsInMapper="";
		aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
		otherInstructionsInReducer="s-r:::0:DOUBLE:::0:DOUBLE:::1:DOUBLE";
		resultIndexes=new byte[]{0, 1};
		resultDimsUnknown = new byte[]{0,0};
		outputs=new String[]{p, r};
		outputInfos=new OutputInfo[]{binaryoutputinfo, binaryoutputinfo};
		
		md=GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
				resultIndexes, resultDimsUnknown, outputs, outputInfos).getMetaData();
		
	//	for(MatrixCharacteristics stat: stats)
	//		System.out.println(stat);
		
		int iter = 0;
		
		//norm+r2=sum(r^2)
		inputs=new String[]{r};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{d};
		clens=new long[]{1};
		brlens=new int[]{bd};
		bclens=new int[]{1};
		instructionsInMapper="s^:::0:DOUBLE:::2:DOUBLE:::0:DOUBLE,ua+:::0:DOUBLE:::0:DOUBLE";
		aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
		otherInstructionsInReducer="";
		resultIndexes=new byte[]{0};
		resultDimsUnknown = new byte[]{0};
		outputs=new String[]{norm_r2_file};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		md=GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
				aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
				resultIndexes, resultDimsUnknown, outputs, outputInfos).getMetaData();
	//	for(MatrixCharacteristics stat: stats)
	//		System.out.println(stat);
		
		double norm_r2=MapReduceTool.readSingleNumberFromHDFSBlock(norm_r2_file);
		
		boolean converge = true;//(norm_r2 < tol*tol);
		
		System.out.print(n+"\t");
		long start=System.currentTimeMillis();
		
		while(!converge);
		{
			//in each iteration
			// q = ((t(X) %*% (X %*% p)) + lambda*p)
			
			//mmult: X%*%p
			inputs=new String[]{X, p};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{n, d};
			clens=new long[]{d, 1};
			brlens=new int[]{bn, bd};
			bclens=new int[]{bd, 1};
			instructionsInMapper="";
			aggInstructionsInReducer="";
			aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
			output = temp;
			outputinfo = binaryoutputinfo;
			partialAggCacheSize=900000000;
					
			MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, aggBinInstrction, numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize);
			
			inputs=new String[]{temp};
			inputInfos=new InputInfo[]{binaryinputinfo};
			rlens=new long[]{n};
			clens=new long[]{1};
			brlens=new int[]{bn};
			bclens=new int[]{1};
			instructionsInMapper="";
			aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
			otherInstructionsInReducer="";
			resultIndexes=new byte[]{0};
			resultDimsUnknown = new byte[]{0};
			String temp2="temp2";
			outputs=new String[]{temp2};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			
			GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
					resultIndexes, resultDimsUnknown, outputs, outputInfos);
			
			
			//mmult  t(X) %*% temp2
			inputs=new String[]{X, temp2};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{n, n};
			clens=new long[]{d, 1};
			brlens=new int[]{bn, bn};
			bclens=new int[]{bd, 1};
			instructionsInMapper="r':::0:DOUBLE:::0:DOUBLE";
			aggInstructionsInReducer="";
			aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
			output = temp;
			outputinfo = binaryoutputinfo;
			partialAggCacheSize=900000000;
					
			MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, aggBinInstrction, numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize);
			
			//q = (temp + lambda*p)
			inputs=new String[]{temp, p};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{d, d};
			clens=new long[]{1, 1};
			brlens=new int[]{bd, bd};
			bclens=new int[]{1, 1};
			instructionsInMapper="s*:::1:DOUBLE:::"+lambda+":DOUBLE:::1:DOUBLE";
			aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
			otherInstructionsInReducer="b+:::0:DOUBLE:::1:DOUBLE:::0:DOUBLE";
			resultIndexes=new byte[]{0};
			resultDimsUnknown = new byte[]{0};
			outputs=new String[]{q};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			
			GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
					resultIndexes, resultDimsUnknown, outputs, outputInfos);
			
			//alpha=(norm_r2)/(t(p) %*% q)
			
			inputs=new String[]{p, q};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{d, d};
			clens=new long[]{1, 1};
			brlens=new int[]{bd, bd};
			bclens=new int[]{1, 1};
			instructionsInMapper="r':::0:DOUBLE:::0:DOUBLE";
			aggInstructionsInReducer="";
			aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
			output = temp;
			outputinfo = binaryoutputinfo;
			partialAggCacheSize=900000000;
					
			MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, aggBinInstrction, numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize);
			
			//alpha = (norm_r2)/(t(p) %*% q)
			inputs=new String[]{temp};
			inputInfos=new InputInfo[]{binaryinputinfo};
			rlens=new long[]{1};
			clens=new long[]{1};
			brlens=new int[]{1};
			bclens=new int[]{1};
			instructionsInMapper="";
			aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
			otherInstructionsInReducer="";
			resultIndexes=new byte[]{0};
			resultDimsUnknown = new byte[]{0};
			outputs=new String[]{temp2};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			
			GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
					resultIndexes, resultDimsUnknown, outputs, outputInfos);
			
			double alpha = norm_r2/MapReduceTool.readSingleNumberFromHDFSBlock(temp2);
			
			//x = x + alpha[1,1] * p
			inputs=new String[]{smallx, p};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{d, d};
			clens=new long[]{1, 1};
			brlens=new int[]{bd, bd};
			bclens=new int[]{1,1};
			instructionsInMapper="s*:::1:DOUBLE:::"+alpha+":DOUBLE:::1:DOUBLE";
			aggInstructionsInReducer="";
			otherInstructionsInReducer="b+:::0:DOUBLE:::1:DOUBLE:::0:DOUBLE";
			resultIndexes=new byte[]{0};
			resultDimsUnknown = new byte[]{0};
			String newx="newx";
			outputs=new String[]{newx};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			
			GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
					resultIndexes, resultDimsUnknown, outputs, outputInfos);
			
			smallx=newx;
			
			double old_norm_r2 = norm_r2;
	
			//r = r + (alpha[1,1] * q)
			String newr="newr";
			
			inputs=new String[]{r, q};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{d, d};
			clens=new long[]{1, 1};
			brlens=new int[]{bd, bd};
			bclens=new int[]{1, 1};
			instructionsInMapper="s*:::1:DOUBLE:::"+alpha+":DOUBLE:::1:DOUBLE";
			aggInstructionsInReducer="";
			otherInstructionsInReducer="b+:::0:DOUBLE:::1:DOUBLE:::0:DOUBLE";
			resultIndexes=new byte[]{0};
			resultDimsUnknown = new byte[]{0};
			outputs=new String[]{newr};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			
			GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
					resultIndexes, resultDimsUnknown, outputs, outputInfos);
			r=newr;
			
			//norm_r2 =  norm(r, 'f')^2 
			inputs=new String[]{r};
			inputInfos=new InputInfo[]{binaryinputinfo};
			rlens=new long[]{d};
			clens=new long[]{1};
			brlens=new int[]{bd};
			bclens=new int[]{1};
			instructionsInMapper="s^:::0:DOUBLE:::2:DOUBLE:::0:DOUBLE,ua+:::0:DOUBLE:::0:DOUBLE";
			aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
			otherInstructionsInReducer="";
			resultIndexes=new byte[]{0};
			resultDimsUnknown = new byte[]{0};
			outputs=new String[]{norm_r2_file};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			
			GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
					resultIndexes, resultDimsUnknown, outputs, outputInfos);
			
			norm_r2=MapReduceTool.readSingleNumberFromHDFSBlock(norm_r2_file);
			double beta = norm_r2/old_norm_r2;
	
			//p = 0-r + (beta * p)
			String newp="newp";
			inputs=new String[]{r, p};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{d, d};
			clens=new long[]{1, 1};
			brlens=new int[]{bd, bd};
			bclens=new int[]{1, 1};
			instructionsInMapper="s-r:::0:DOUBLE:::0:DOUBLE:::0:DOUBLE,s*:::1:DOUBLE:::"+beta+":DOUBLE:::1:DOUBLE";
			aggInstructionsInReducer="";
			otherInstructionsInReducer="b+:::0:DOUBLE:::1:DOUBLE:::0:DOUBLE";
			resultIndexes=new byte[]{0};
			resultDimsUnknown = new byte[]{0};
			outputs=new String[]{newp};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			
			GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, instructionsInMapper, 
					aggInstructionsInReducer, otherInstructionsInReducer, numReducers, replication, 
					resultIndexes, resultDimsUnknown, outputs, outputInfos);
			p=newp;
			iter = iter + 1;
	
			converge = (norm_r2 < tol*tol) | (iter>maxIter);
		}
		
		System.out.println(System.currentTimeMillis()-start);
		
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new LinearRegression(), args);
		System.exit(errCode);
	}
}
