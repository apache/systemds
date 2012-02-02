package dml.runtime.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import dml.runtime.matrix.GMR;
import dml.runtime.matrix.MMCJMR;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.MatrixDimensionsMetaData;
import dml.runtime.matrix.MetaData;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.mapred.obsolete.ABMR;
import dml.runtime.util.MapReduceTool;
/*
 * This is the plan used for the ICDE paper.
 */

public class GNMF2 extends Configured implements Tool{

	public static InputInfo binaryinputinfo=InputInfo.BinaryBlockInputInfo;
	public static OutputInfo binaryoutputinfo=OutputInfo.BinaryBlockOutputInfo;
	@Override
	public int run(String[] args) throws Exception {
		
		if(args.length<11)
		{
			System.err.println("[location of V] [n] [m] [k] [bn] [bm] [bk] [location W] [location H] [# mappers] [# reducers]");
			System.exit(-1);
		}
		
		long totalStart=System.currentTimeMillis();
		
		String V=args[0];
		long n=Long.parseLong(args[1]);
		long m=Long.parseLong(args[2]);
		long k=Long.parseLong(args[3]);
		int bn=Integer.parseInt(args[4]);
		int bm=Integer.parseInt(args[5]);
		int bk=Integer.parseInt(args[6]);
		String W=args[7];
		String H=args[8];
		
		int numMappers=Integer.parseInt(args[9]);
		int numReducers=Integer.parseInt(args[10]);
		
		String finalW="finalW";
		String finalH="finalH";
		
		//int maxCap=Integer.parseInt(args[11]);//(int)(100000000*0.75);
		System.out.print(n+"\t");
		
		int replication=1;
		long start;
		String[] inputs;
		InputInfo[] inputInfos;
		long[] rlens;
		long[] clens;
		int[] brlens;
		int[] bclens;
		String instructionsInMapper;
		String aggInstructionsInReducer;
		String aggBinInstrction;
		String output;
		OutputInfo outputinfo;
		//int partialAggCacheSize;
		//MatrixCharacteristics[] stats;
		String otherInstructionsInReducer;
		byte[] resultIndexes;
		byte[] resultDimsUnknown;
		String[] outputs;
		OutputInfo[] outputInfos;		
		//MatrixCharacteristics[] T1_stats;
		//MatrixCharacteristics[] T2_stats;
		//MatrixCharacteristics[] T3_stats;
		MetaData[] md,T1_md,T2_md,T3_md;
		String T1="T1", T2="T2", T3="T3", temp="temp";
			
	//	System.out.println("\n----------------------------");
	//	System.out.println("generatring matrixes");
		start=System.currentTimeMillis();
		boolean use2MR=true;
		
		//-----------------
		//	GenerateTestMatrixBlock.runJob(numMappers, n, m, V, 1, bn, bm, System.currentTimeMillis(), 0.01, true);
		//------------------
		
		//W=randomMatrix(n,k)
		GenerateTestMatrixBlock.runJob(numMappers, n, k, W, 1, bn, bk, System.currentTimeMillis(), 1, true);
		
		//H=randomeMatrix(k,m)
		GenerateTestMatrixBlock.runJob(numMappers, k, m, H, 1, bk, bm, System.currentTimeMillis(), 1, true);
		
		System.out.print((System.currentTimeMillis()-start)+"\t"); 
		//System.out.println("\n----------------------------");
	//	System.out.println("T1=W'V");
		
		
		//[1] H=H*[ W'V / (W'W H)]
		
		//System.out.println("T2=W'W");
		start=System.currentTimeMillis();
		
		//T2=W'W    k X N X k
		inputs=new String[]{W};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{n};
		clens=new long[]{k};
		brlens=new int[]{bn};
		bclens=new int[]{bk};
		instructionsInMapper="r':::0:DOUBLE:::1:DOUBLE";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+*:::1:DOUBLE:::0:DOUBLE:::2:DOUBLE";
		output=temp;
		outputinfo=binaryoutputinfo;
	//	partialAggCacheSize=maxCap;//(int)(k*k/bk/bk);
		md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
				numReducers, replication, (byte)0, output, outputinfo).getMetaData();
		System.out.print((System.currentTimeMillis()-start)+"\t");
	//	System.out.println("\n----------------------------");
	//	System.out.println("GMR for mmult");
		start=System.currentTimeMillis();
		
		MatrixCharacteristics stats = ((MatrixDimensionsMetaData)md[0]).getMatrixCharacteristics();
		inputs=new String[]{temp};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{stats.numRows};
		clens=new long[]{stats.numColumns};
		brlens=new int[]{stats.numRowsPerBlock};
		bclens=new int[]{stats.numColumnsPerBlock};
		instructionsInMapper="";
		aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
		otherInstructionsInReducer="";
		resultIndexes=new byte[]{0};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{T2};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		T2_md=GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos).getMetaData();
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		MapReduceTool.deleteFileIfExistOnHDFS(new Path(temp), new JobConf(new Configuration()));
	//	System.out.println("\n----------------------------");
	//	System.out.println("T3=T2 H");
		start=System.currentTimeMillis();
		
		//T3=T2 H 
		ABMR.runJob(T2+","+H, "ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE", T3, numReducers, replication, k, k, k, m, bk, bk, bk, bm);
		System.out.print((System.currentTimeMillis()-start)+"\t");
		
		start=System.currentTimeMillis();
		//T1=W'V (k X n X m)
		
			inputs=new String[]{W, V};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{n, n};
			clens=new long[]{k, m};
			brlens=new int[]{bn, bn};
			bclens=new int[]{bk, bm};
			instructionsInMapper="r':::0:DOUBLE:::0:DOUBLE";
			aggInstructionsInReducer="";
			aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
			output=temp;
			outputinfo=binaryoutputinfo;
		//	partialAggCacheSize=maxCap;//Math.min((int)(k*m/bk/bm), (int)maxCap/bk/bm);
			md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
					instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
					numReducers, replication, (byte)0, output, outputinfo).getMetaData();
			System.out.print((System.currentTimeMillis()-start)+"\t");
		//	System.out.println("\n----------------------------");
		//	System.out.println("GMR for mmult");
		
		start=System.currentTimeMillis();
		
		//finalH=H*(T1/T3)
		inputs=new String[]{H, temp, T3};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo, binaryinputinfo};
		rlens=new long[]{k, k, k};
		clens=new long[]{m, m, m};
		brlens=new int[]{bk, bk, bk};
		bclens=new int[]{bm, bm, bm};
		instructionsInMapper="";
		aggInstructionsInReducer="a+:::1:DOUBLE:::1:DOUBLE";
		otherInstructionsInReducer="b/:::1:DOUBLE:::2:DOUBLE:::1:DOUBLE,b*:::0:DOUBLE:::1:DOUBLE:::0:DOUBLE";
		resultIndexes=new byte[]{0};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{finalH};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		MapReduceTool.deleteFileIfExistOnHDFS(new Path(temp), new JobConf(new Configuration()));
	//	System.out.println("\n----------------------------");
	//	System.out.println("T1=VH'");
		start=System.currentTimeMillis();
		
		//[2] W=W*[ VH' / ( W(HH')) ]
		start=System.currentTimeMillis();
		
		//T2=HH'
		H=finalH;
		inputs=new String[]{H};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{k};
		clens=new long[]{m};
		brlens=new int[]{bk};
		bclens=new int[]{bm};
		instructionsInMapper="r':::0:DOUBLE:::1:DOUBLE";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
		output=temp;
		outputinfo=binaryoutputinfo;
		//partialAggCacheSize=maxCap;//(int)(k*k/bk/bk);
		md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
				numReducers, replication, (byte)0, output, outputinfo).getMetaData();
		System.out.print((System.currentTimeMillis()-start)+"\t");
	//	System.out.println("\n----------------------------");
	//	System.out.println("GMR for mmult");
		start=System.currentTimeMillis();
		
		stats = ((MatrixDimensionsMetaData)md[0]).getMatrixCharacteristics();
		
		inputs=new String[]{temp};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{stats.numRows};
		clens=new long[]{stats.numColumns};
		brlens=new int[]{stats.numRowsPerBlock};
		bclens=new int[]{stats.numColumnsPerBlock};
		instructionsInMapper="";
		aggInstructionsInReducer="a+:::0:DOUBLE:::0:DOUBLE";
		otherInstructionsInReducer="";
		resultIndexes=new byte[]{0};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{T2};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		T2_md=GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos).getMetaData();
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		MapReduceTool.deleteFileIfExistOnHDFS(new Path(temp), new JobConf(new Configuration()));
	//	System.out.println("\n----------------------------");
	//	System.out.println("T3=W T2");
		start=System.currentTimeMillis();
		
		//T3=W T2 and finalW=W*(T1/T3)
		ABMR.runJob(W+","+T2, "ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE", T3, numReducers, replication, n, k, k, k, bn, bk, bk, bk);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		//System.out.println("\n----------------------------");
		//System.out.println("finalW=W*(T1/T3)");
		start=System.currentTimeMillis();
		
		//T1=VH' n x m x k
		inputs=new String[]{V, H};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, k};
		clens=new long[]{m, m};
		brlens=new int[]{bn, bk};
		bclens=new int[]{bm, bm};
		instructionsInMapper="r':::1:DOUBLE:::1:DOUBLE";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+*:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
		output=temp;
		outputinfo=binaryoutputinfo;
	//	partialAggCacheSize=maxCap;//Math.min((int)(n*k/bn/bk), maxCap/bn/bk);
		md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
				numReducers, replication, (byte)0, output, outputinfo).getMetaData();
		System.out.print((System.currentTimeMillis()-start)+"\t");
		//System.out.println("\n----------------------------");
		//System.out.println("GMR for mmult");
		start=System.currentTimeMillis();
		
		//finalW=W*(T1/T3)
		inputs=new String[]{W, temp, T3};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n, n};
		clens=new long[]{k, k, k};
		brlens=new int[]{bn, bn, bn};
		bclens=new int[]{bk, bk, bk};
		instructionsInMapper="";
		aggInstructionsInReducer="a+:::1:DOUBLE:::1:DOUBLE";
		otherInstructionsInReducer="b/:::1:DOUBLE:::2:DOUBLE:::1:DOUBLE,b*:::0:DOUBLE:::1:DOUBLE:::0:DOUBLE";
		resultIndexes=new byte[]{0};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{finalW};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		MapReduceTool.deleteFileIfExistOnHDFS(new Path(temp), new JobConf(new Configuration()));
		
		System.out.println((System.currentTimeMillis()-totalStart));
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new GNMF2(), args);
		System.exit(errCode);
	}
}
