package dml.runtime.test.numericalStability;

import java.io.IOException;

import dml.api.DMLScript;
import dml.parser.ParseException;
import dml.utils.DMLException;

public class RunSystemML {

	/**
	 * @param args
	 * @throws DMLException 
	 * @throws ParseException 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException, ParseException, DMLException {
		if(args.length<7)
		{
			System.out.println("RunSystemML <script folder> <univariate?> <stable?> <nRows> <output> <input1> <numerical>? <categorical>? <blocksize>?");
			System.exit(-1);
		}
		String scriptfolder=args[0];//"src/dml/runtime/test/numericalStability/scripts/";
		if(!scriptfolder.endsWith("/"))
			scriptfolder=scriptfolder+"/";
		boolean univariate=Boolean.parseBoolean(args[1]);
		boolean stable=Boolean.parseBoolean(args[2]);
		String nRows=args[3];
		String output=args[4];
		String input1=args[5];
		
		if(univariate)
		{
			String script=scriptfolder+"unstable_univariate.dml";
			if(stable)
				script=scriptfolder+"stable_univariate.dml";
			//#-f src/dml/runtime/test/numericalStability/scripts/stable_univariate.dml 
			//'ReferenceData/NumAcc2.dat.mtx' 1001 'mu' 'std' 'var' 'g2' 'g1' 's'
			String[] strs=new String[]{"-f", script, "'"+input1+"'", nRows, 
					"'"+output+"/Mean'", "'"+output+"/Std'", "'"+output+"/Variance'", "'"+output+"/Kurtosis'", 
					"'"+output+"/Skewness'", "'"+output+"/Summation'" };
			
			/*for(String str: strs)
			{
				System.out.println(str);
			}*/
			DMLScript.main(strs);
		}else
		{
			String input2=args[6];
			String input3=args[7];
			String blockSize=args[8];
			String script=scriptfolder+"unstable_bivariate.dml";
			if(stable)
				script=scriptfolder+"stable_bivariate.dml";
			
			//-f src/dml/runtime/test/numericalStability/scripts/stable_bivariate.dml 
			//'ReferenceData/NumAcc4.dat.mtx' 'ReferenceData/NumAcc3.dat.mtx' 
			//'ReferenceData/NumAcc2.dat.mtx' 1001 'pearson' 'eta' 'anovaF'
			//Grouped: Eta, AnovaF
			//Covariance: Covariance, PearsonR
			String[] strs=new String[]{"-f", script, "'"+input1+"'", "'"+input2+"'", "'"+input3+"'", nRows, 
					"'"+output+"/PearsonR'", "'"+output+"/Eta'", "'"+output+"/AnovaF'", "'"+output+"/Covariance'", blockSize};
			DMLScript.main(strs);
		}
	}

}
