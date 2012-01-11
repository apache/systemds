package gnmf;

import java.io.IOException;
import java.net.URISyntaxException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;


public class MatrixGNMF
{
	public static void main(String[] args) throws IOException, URISyntaxException
	{
		if(args.length < 10)
		{
			System.out.println("missing parameters");
			System.out.println("expected parameters: [directory of v] [directory of w] [directory of h] " +
					"[k] [num mappers] [num reducers] [replication] [working directory] " +
					"[final directory of w] [final directory of h]");
			System.exit(1);
		}
		
		String vDir			= args[0];
		String wDir			= args[1];
		String hDir			= args[2];
		int k				= Integer.parseInt(args[3]);
		int numMappers		= Integer.parseInt(args[4]);
		int numReducers		= Integer.parseInt(args[5]);
		int replication		= Integer.parseInt(args[6]);
		String outputDir	= args[7];
		String wFinalDir	= args[8];
		String hFinalDir	= args[9];
		
		JobConf mainJob = new JobConf(MatrixGNMF.class);
		String vDirectory;
		String wDirectory;
		String hDirectory;
		
		FileSystem.get(mainJob).delete(new Path(outputDir));
		
		
		vDirectory = vDir;
		hDirectory = hDir;
		wDirectory = wDir;
		
		String workingDirectory;
		String resultDirectoryX;
		String resultDirectoryY;
		
		long start = System.currentTimeMillis();
		System.gc();
		System.out.println("starting calculation");
		System.out.print("calculating X = WT * V... ");
		workingDirectory = UpdateWHStep1.runJob(numMappers, numReducers, replication,
				UpdateWHStep1.UPDATE_TYPE_H, vDirectory, wDirectory, outputDir, k);
		resultDirectoryX = UpdateWHStep2.runJob(numMappers, numReducers, replication,
				workingDirectory, outputDir);
		FileSystem.get(mainJob).delete(new Path(workingDirectory));
		
		System.out.println("done");
		System.out.print("calculating Y = WT * W * H... ");
		workingDirectory = UpdateWHStep3.runJob(numMappers, numReducers, replication,
				wDirectory, outputDir);
		resultDirectoryY = UpdateWHStep4.runJob(numMappers, replication, workingDirectory,
				UpdateWHStep4.UPDATE_TYPE_H, hDirectory, outputDir);
		FileSystem.get(mainJob).delete(new Path(workingDirectory));
		System.out.println("done");
		
		System.out.print("calculating H = H .* X ./ Y... ");
		workingDirectory = UpdateWHStep5.runJob(numMappers, numReducers, replication,
				hDirectory, resultDirectoryX, resultDirectoryY, hFinalDir, k);
		System.out.println("done");
		
		FileSystem.get(mainJob).delete(new Path(resultDirectoryX));
		FileSystem.get(mainJob).delete(new Path(resultDirectoryY));
		
		System.out.print("storing back H... ");
		FileSystem.get(mainJob).delete(new Path(hDirectory));
		hDirectory = workingDirectory;
		System.out.println("done");
		
		System.out.print("calculating X = V * HT... ");
		workingDirectory = UpdateWHStep1.runJob(numMappers, numReducers, replication,
				UpdateWHStep1.UPDATE_TYPE_W, vDirectory, hDirectory, outputDir, k);
		resultDirectoryX = UpdateWHStep2.runJob(numMappers, numReducers, replication,
				workingDirectory, outputDir);
		FileSystem.get(mainJob).delete(new Path(workingDirectory));
		System.out.println("done");
		
		System.out.print("calculating Y = W * H * HT... ");
		workingDirectory = UpdateWHStep3.runJob(numMappers, numReducers, replication,
				hDirectory, outputDir);
		resultDirectoryY = UpdateWHStep4.runJob(numMappers, replication, workingDirectory,
				UpdateWHStep4.UPDATE_TYPE_W, wDirectory, outputDir);
		
		FileSystem.get(mainJob).delete(new Path(workingDirectory));
		System.out.println("done");
		System.out.print("calculating W = W .* X ./ Y... ");
		workingDirectory = UpdateWHStep5.runJob(numMappers, numReducers, replication,
				wDirectory, resultDirectoryX, resultDirectoryY, wFinalDir, k);
		System.out.println("done");
		
		FileSystem.get(mainJob).delete(new Path(resultDirectoryX));
		FileSystem.get(mainJob).delete(new Path(resultDirectoryY));
		
		System.out.print("storing back W... ");
		FileSystem.get(mainJob).delete(new Path(wDirectory));
		wDirectory = workingDirectory;
		System.out.println("done");
		
		long requiredTime = System.currentTimeMillis() - start;
		long requiredTimeMilliseconds = requiredTime % 1000;
		requiredTime -= requiredTimeMilliseconds;
		requiredTime /= 1000;
		long requiredTimeSeconds = requiredTime % 60;
		requiredTime -= requiredTimeSeconds;
		requiredTime /= 60;
		long requiredTimeMinutes = requiredTime % 60;
		requiredTime -= requiredTimeMinutes;
		requiredTime /= 60;
		long requiredTimeHours = requiredTime;
		System.out.println("required time: " + requiredTimeHours + "h " +
				requiredTimeMinutes + "m " + requiredTimeSeconds + "s");
		
//		System.out.println("W in: " + wDirectory);
//		System.out.println("H in: " + hDirectory);
//		System.out.print("now storing W and H in final directories... ");
//		FileSystem.get(mainJob).rename(new Path(wDirectory), new Path(wFinalDir));
//		FileSystem.get(mainJob).rename(new Path(hDirectory), new Path(hFinalDir));
//		System.out.println("done");
	}
}
