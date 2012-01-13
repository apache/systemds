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
import dml.runtime.matrix.mapred.tentative.ABMR;
import dml.runtime.util.MapReduceTool;

public class GNMF extends Configured implements Tool{

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
		String finalW=args[7];
		String finalH=args[8];
		int numMappers=Integer.parseInt(args[9]);
		int numReducers=Integer.parseInt(args[10]);
		
		int maxCap=900000000;//(int)(100000000*0.75);
		System.out.print(n+"\t");
		
		String W="tempW";
		String H="tempH";
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
		int partialAggCacheSize;
		//MatrixCharacteristics[] stats;
		MetaData[] md;
		String otherInstructionsInReducer;
		byte[] resultIndexes;
		byte[] resultDimsUnknown;
		String[] outputs;
		OutputInfo[] outputInfos;		
		//MatrixCharacteristics[] T1_stats;
		//MatrixCharacteristics[] T2_stats;
		//MatrixCharacteristics[] T3_stats;
		MetaData[] T1_md, T2_md, T3_md;
		String T1="T1", T2="T2", T3="T3", temp="temp";
			
	//	System.out.println("\n----------------------------");
	//	System.out.println("generatring matrixes");
		start=System.currentTimeMillis();
		boolean use2MR=true;
		
		//-----------------
		//	GenerateTestMatrixBlock.runJob(numMappers, n, m, V, 1, bn, bm, System.currentTimeMillis(), 0.01);
		//------------------
		
		//W=randomMatrix(n,k)
	/*	GenerateTestMatrixBlock.runJob(numMappers, n, k, W, 1, bn, bk, System.currentTimeMillis(), 1, true);
		
		//H=randomeMatrix(k,m)
		GenerateTestMatrixBlock.runJob(numMappers, k, m, H, 1, bk, bm, System.currentTimeMillis(), 1, true);
		
		System.out.print((System.currentTimeMillis()-start)+"\t"); */
		//System.out.println("\n----------------------------");
	//	System.out.println("T1=W'V");
		start=System.currentTimeMillis();
		
		//[1] H=H*[ W'V / (W'W H)]
		
		//T1=W'V (k X n X m)
		if(use2MR)
		{
			inputs=new String[]{W, V};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{n, n};
			clens=new long[]{k, m};
			brlens=new int[]{bn, bn};
			bclens=new int[]{bk, bm};
			instructionsInMapper="r' 0 0";
			aggInstructionsInReducer="";
			aggBinInstrction="ba+* 0 1 2";
			output=temp;
			outputinfo=binaryoutputinfo;
			partialAggCacheSize=maxCap;//Math.min((int)(k*m/bk/bm), (int)maxCap/bk/bm);
			md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
					instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
					numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize).getMetaData();
			
			System.out.print((System.currentTimeMillis()-start)+"\t");
		//	System.out.println("\n----------------------------");
		//	System.out.println("GMR for mmult");
			start=System.currentTimeMillis();
			
			inputs=new String[]{temp};
			inputInfos=new InputInfo[]{binaryinputinfo};
			rlens=new long[]{k};
			clens=new long[]{m};
			brlens=new int[]{bk};
			bclens=new int[]{bm};
			instructionsInMapper="";
			aggInstructionsInReducer="a+ 0 0";
			otherInstructionsInReducer="";
			resultIndexes=new byte[]{0};
			resultDimsUnknown=new byte[]{0};
			outputs=new String[]{T1};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			
			T1_md=GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
					instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
					numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos).getMetaData();
			
			System.out.print((System.currentTimeMillis()-start)+"\t");
			
			MapReduceTool.deleteFileIfExistOnHDFS(new Path(temp), new JobConf(new Configuration()));
		}else
		{
			//T1=W'V (k X n X m)
			inputs=new String[]{W};
			inputInfos=new InputInfo[]{binaryinputinfo};
			rlens=new long[]{n};
			clens=new long[]{k};
			brlens=new int[]{bn};
			bclens=new int[]{bk};
			instructionsInMapper="r' 0 0";
			aggInstructionsInReducer="";
			otherInstructionsInReducer="";
			resultIndexes=new byte[]{0};
			resultDimsUnknown=new byte[]{0};
			outputs=new String[]{temp};
			outputInfos=new OutputInfo[]{binaryoutputinfo};
			GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
					instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
					0, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
			ABMR.runJob(temp+","+V, "ba+*", T1, numReducers, replication, k, n, n, m, bk, bn, bn, bm);
			System.out.print((System.currentTimeMillis()-start)+"\t");
		}
		
		//System.out.println("\n----------------------------");
		//System.out.println("T2=W'W");
		start=System.currentTimeMillis();
		
		//T2=W'W    k X N X k
		inputs=new String[]{W};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{n};
		clens=new long[]{k};
		brlens=new int[]{bn};
		bclens=new int[]{bk};
		instructionsInMapper="r' 0 1";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+* 1 0 2";
		output=temp;
		outputinfo=binaryoutputinfo;
		partialAggCacheSize=maxCap;//(int)(k*k/bk/bk);
		md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
				numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize).getMetaData();
		
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
		aggInstructionsInReducer="a+ 0 0";
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
		
		//T3=T2 H and finalH=H*(T1/T3)
		use2MR=false;
		if(use2MR)
		{
			inputs=new String[]{T2, H};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{k, k};
			clens=new long[]{k, m};
			brlens=new int[]{bk, bk};
			bclens=new int[]{bk, bm};
			instructionsInMapper="";
			aggInstructionsInReducer="";
			aggBinInstrction="ba+* 0 1 2";
			output=T3;
			outputinfo=binaryoutputinfo;
			partialAggCacheSize=maxCap;//Math.min((int)(k*m/bk/bm), (int)maxCap/bk/bm);
			T3_md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
					instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
					numReducers, replication, (byte) 0, output, outputinfo, partialAggCacheSize).getMetaData();
		}else
		{
			ABMR.runJob(T2+","+H, "ba+*", T3, numReducers, replication, k, k, k, m, bk, bk, bk, bm);
		}
		
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
	//	System.out.println("\n----------------------------");
	//	System.out.println("finalH=H*(T1/T3)");
		start=System.currentTimeMillis();
		
		//finalH=H*(T1/T3)
		inputs=new String[]{H, T1, T3};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo, binaryinputinfo};
		rlens=new long[]{k, k, k};
		clens=new long[]{m, m, m};
		brlens=new int[]{bk, bk, bk};
		bclens=new int[]{bm, bm, bm};
		instructionsInMapper="";
		if(use2MR)
			aggInstructionsInReducer="a+ 2 2";
		else
			aggInstructionsInReducer="";
		otherInstructionsInReducer="b/ 1 2 3,b* 0 3 4";
		resultIndexes=new byte[]{4};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{finalH};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
	//	System.out.println("\n----------------------------");
	//	System.out.println("T1=VH'");
		start=System.currentTimeMillis();
		
		//[2] W=W*[ VH' / ( W(HH')) ]
		H=finalH;
		//T1=VH' n x m x k
		inputs=new String[]{V, H};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, k};
		clens=new long[]{m, m};
		brlens=new int[]{bn, bk};
		bclens=new int[]{bm, bm};
		instructionsInMapper="r' 1 1";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+* 0 1 2";
		output=temp;
		outputinfo=binaryoutputinfo;
		partialAggCacheSize=maxCap;//Math.min((int)(n*k/bn/bk), maxCap/bn/bk);
		md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
				numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize).getMetaData();
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		//System.out.println("\n----------------------------");
		//System.out.println("GMR for mmult");
		start=System.currentTimeMillis();
		
		inputs=new String[]{temp};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{n};
		clens=new long[]{k};
		brlens=new int[]{bk};
		bclens=new int[]{bk};
		instructionsInMapper="";
		aggInstructionsInReducer="a+ 0 0";
		otherInstructionsInReducer="";
		resultIndexes=new byte[]{0};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{T1};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		T1_md=GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos).getMetaData();
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		MapReduceTool.deleteFileIfExistOnHDFS(new Path(temp), new JobConf(new Configuration()));
		//System.out.println("\n----------------------------");
		//System.out.println("T2=HH'");
		start=System.currentTimeMillis();
		
		//T2=HH'
		
		inputs=new String[]{H};
		inputInfos=new InputInfo[]{binaryinputinfo};
		rlens=new long[]{k};
		clens=new long[]{m};
		brlens=new int[]{bk};
		bclens=new int[]{bm};
		instructionsInMapper="r' 0 1";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+* 0 1 2";
		output=temp;
		outputinfo=binaryoutputinfo;
		partialAggCacheSize=maxCap;//(int)(k*k/bk/bk);
		md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
				numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize).getMetaData();
		
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
		aggInstructionsInReducer="a+ 0 0";
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
		use2MR=false;
		if(use2MR)
		{
			inputs=new String[]{W, T2};
			inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
			rlens=new long[]{n, k};
			clens=new long[]{k, k};
			brlens=new int[]{bn, bk};
			bclens=new int[]{bk, bk};
			instructionsInMapper="";
			aggInstructionsInReducer="";
			aggBinInstrction="ba+* 0 1 2";
			output=T3;
			outputinfo=binaryoutputinfo;
			partialAggCacheSize=maxCap;//Math.min((int)(n*k/bn/bk), maxCap/bn/bk);
			T3_md=MMCJMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
					instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, 
					numReducers, replication, (byte)0, output, outputinfo, partialAggCacheSize).getMetaData();
		}else
		{
			ABMR.runJob(W+","+T2, "ba+*", T3, numReducers, replication, n, k, k, k, bn, bk, bk, bk);
		}
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		//System.out.println("\n----------------------------");
		//System.out.println("finalW=W*(T1/T3)");
		start=System.currentTimeMillis();
		
		//finalW=W*(T1/T3)
		inputs=new String[]{W, T1, T3};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, n, n};
		clens=new long[]{k, k, k};
		brlens=new int[]{bn, bn, bn};
		bclens=new int[]{bk, bk, bk};
		instructionsInMapper="";
		aggInstructionsInReducer="a+ 2 2";
		otherInstructionsInReducer="b/ 1 2 3,b* 0 3 4";
		resultIndexes=new byte[]{4};
		resultDimsUnknown=new byte[]{0};
		outputs=new String[]{finalW};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		System.out.println((System.currentTimeMillis()-totalStart));
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new GNMF(), args);
		System.exit(errCode);
	}
}
