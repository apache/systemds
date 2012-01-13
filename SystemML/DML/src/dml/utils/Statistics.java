package dml.utils;

/**
 * This class captures all statistics.
 * 
 * @author Felix Hamborg
 * 
 */
public class Statistics {
	private static long lStartTime = 0;
	private static long lEndTime = 0;

	/** number of executed MR jobs */
	private static int iNoOfExecutedMRJobs = 0;

	/** number of compiled MR jobs */
	private static int iNoOfCompiledMRJobs = 0;

	public static void setNoOfExecutedMRJobs(int iNoOfExecutedMRJobs) {
		Statistics.iNoOfExecutedMRJobs = iNoOfExecutedMRJobs;
	}

	public static int getNoOfExecutedMRJobs() {
		return iNoOfExecutedMRJobs;
	}

	public static void setNoOfCompiledMRJobs(int iNoOfCompiledMRJobs) {
		Statistics.iNoOfCompiledMRJobs = iNoOfCompiledMRJobs;
	}

	public static int getNoOfCompiledMRJobs() {
		return iNoOfCompiledMRJobs;
	}

	/**
	 * Starts the timer, should be invoked immediately before invoking
	 * Program.execute()
	 */
	public static void startRunTimer() {
		lStartTime = System.nanoTime();
	}

	/**
	 * Stops the timer, should be invoked immediately after invoking
	 * Program.execute()
	 */
	public static void stopRunTimer() {
		lEndTime = System.nanoTime();
	}

	/**
	 * Returns the total time of run in nanoseconds.
	 * 
	 * @return
	 */
	public static long getRunTime() {
		return lEndTime - lStartTime;
	}

	/**
	 * Prints statistics.
	 * 
	 * @return
	 */
	public static String display() {
		StringBuilder sb = new StringBuilder();
		sb.append("SystemML Statistics:\n");
		double totalT = getRunTime()*1e-9; // nanoSec --> sec
		sb.append("Total time:\t\t" + totalT + " sec.\n");
		sb.append("Number of compiled MR Jobs:\t" + getNoOfCompiledMRJobs() + ".\n");
		sb.append("Number of executed MR Jobs:\t" + getNoOfExecutedMRJobs() + ".\n");

		return sb.toString();
	}
}
