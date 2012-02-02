package dml.runtime.test.tentative;

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
import dml.runtime.test.GenerateTestMatrixBlock;
import dml.runtime.util.MapReduceTool;

public class TestOrder  extends Configured implements Tool{

	public static InputInfo binaryinputinfo=InputInfo.BinaryBlockInputInfo;
	public static OutputInfo binaryoutputinfo=OutputInfo.BinaryBlockOutputInfo;
	
	private String A="A";
	private String B="B";
	String C="C";
	long n;
	long m; 
	long k; 
	int bn; 
	int bm; 
	int bk; 
	int maxCap;
	int numReducers; 
	String out="out";
	
	//A(BC)
	void rightOrder() throws Exception
	{
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
		MetaData[] md;
		String otherInstructionsInReducer;
		byte[] resultIndexes;
		byte[] resultDimsUnknown;
		String[] outputs;
		OutputInfo[] outputInfos;		
		String temp="temp";
		String temp2="temp2";
		
		start=System.currentTimeMillis();
		
		//T2=HH'
		
		inputs=new String[]{B, C};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{k,m};
		clens=new long[]{m, k};
		brlens=new int[]{bk, bm};
		bclens=new int[]{bm, bk};
		instructionsInMapper="";
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
		outputs=new String[]{temp2};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
		MapReduceTool.deleteFileIfExistOnHDFS(new Path(temp), new JobConf(new Configuration()));
	//	System.out.println("\n----------------------------");
	//	System.out.println("T3=W T2");
		start=System.currentTimeMillis();
		
		//T3=W T2 and finalW=W*(T1/T3)
		ABMR.runJob(A+","+temp2, "ba+*", out, numReducers, replication, n, k, k, k, bn, bk, bk, bk);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
	}
	
	//(AB)C
	void wrongOrder() throws Exception
	{
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
		MetaData[] md;
		String otherInstructionsInReducer;
		byte[] resultIndexes;
		byte[] resultDimsUnknown;
		String[] outputs;
		OutputInfo[] outputInfos;		
		String temp="temp";
		String temp2="temp2";
		
		start=System.currentTimeMillis();
		
		ABMR.runJob(A+","+B, "ba+*", temp, numReducers, replication, n, k, k, m, bn, bk, bk, bm);
		System.out.print((System.currentTimeMillis()-start)+"\t");
		
		start=System.currentTimeMillis();
		
		inputs=new String[]{temp, C};
		inputInfos=new InputInfo[]{binaryinputinfo, binaryinputinfo};
		rlens=new long[]{n, m};
		clens=new long[]{m, k};
		brlens=new int[]{bn, bm};
		bclens=new int[]{bm, bk};
		instructionsInMapper="";
		aggInstructionsInReducer="";
		aggBinInstrction="ba+* 0 1 2";
		output=temp2;
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
		
		inputs=new String[]{temp2};
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
		outputs=new String[]{temp2};
		outputInfos=new OutputInfo[]{binaryoutputinfo};
		
		GMR.runJob(true, inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer, 
				numReducers, replication, resultIndexes, resultDimsUnknown, outputs, outputInfos);
		
		System.out.print((System.currentTimeMillis()-start)+"\t");
	}
	
	public int run(String[] args) throws Exception {
		
		if(args.length<9)
		{
			System.err.println("[n] [m] [k] [bn] [bm] [bk] [# mappers] [# reducers] [sparsity]");
			System.exit(-1);
		}
		
		n=Long.parseLong(args[0]);
		m=Long.parseLong(args[1]);
		k=Long.parseLong(args[2]);
		bn=Integer.parseInt(args[3]);
		bm=Integer.parseInt(args[4]);
		bk=Integer.parseInt(args[5]);
		int numMappers=Integer.parseInt(args[6]);
		numReducers=Integer.parseInt(args[7]);
		double sparsity=Double.parseDouble(args[8]);
		maxCap=100;//900000000;//(int)(100000000*0.75);
		
		System.out.print(n+"\t");
		
		GenerateTestMatrixBlock.runJob(numMappers, n, k, A, 1, bn, bk, System.currentTimeMillis(), sparsity, true);
		
		GenerateTestMatrixBlock.runJob(numMappers, k, m, B, 1, bk, bm, System.currentTimeMillis(), sparsity, true);
		
		GenerateTestMatrixBlock.runJob(numMappers, m, k, C, 1, bm, bk, System.currentTimeMillis(), sparsity, true);
		
		
		long totalStart=System.currentTimeMillis();
		rightOrder();
		System.out.print((System.currentTimeMillis()-totalStart)+"\t");
		totalStart=System.currentTimeMillis();
		wrongOrder();
		System.out.println((System.currentTimeMillis()-totalStart));
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int errCode = ToolRunner.run(new Configuration(), new TestOrder(), args);
		System.exit(errCode);
	}
	
}
