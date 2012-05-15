package dml.test.components.parser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import dml.parser.DMLProgram;
import dml.parser.DMLQLParser;
import dml.parser.DMLTranslator;
import dml.utils.LanguageException;

/**
 * Different test cases for ParFOR loop dependency analysis:
 * 
 * * scalar tests - expected results
 *    1: no, 2: dep, 3: no, 4: no, 5: dep, 6: no, 7: no, 8: dep, 9: dep, 10: no   
 * * matrix 1D tests - expected results
 *    11: no, 12: no, 13: no, 14:dep, 15: no, 16: dep, 17: dep, 18: no, 19: no (DEP, hard), 20: no, 
 *    21: dep, 22: no, 23: no, 24: no, 25: no, 26:no, 29: ERR
 * * nested control structures
 *    27:dep                                                                    
 * * nested parallelism
 *    28: no, 
 * * range indexing
 *    30: no, 31: no, 32: dep
 * * set indexing
 *    33: dep, 34: dep, 35: no
 * 
 * 
 * @author mboehm
 *
 */
public class ParForDependencyAnalysisTest 
{
	public static String DIR = "./test/scripts/functions/parfor/";
	
	@Test
	public void testDependencyAnalysis1() { runTest("parfor1.dml", false); }
	
	@Test
	public void testDependencyAnalysis2() { runTest("parfor2.dml", true); }
	
	@Test
	public void testDependencyAnalysis3() { runTest("parfor3.dml", false); }
	
	@Test
	public void testDependencyAnalysis4() { runTest("parfor4.dml", false); }
	
	@Test
	public void testDependencyAnalysis5() { runTest("parfor5.dml", true); }
	
	@Test
	public void testDependencyAnalysis6() { runTest("parfor6.dml", false); }
	
	@Test
	public void testDependencyAnalysis7() { runTest("parfor7.dml", false); }
	
	@Test
	public void testDependencyAnalysis8() { runTest("parfor8.dml", true); }
	
	@Test
	public void testDependencyAnalysis9() { runTest("parfor9.dml", true); }
	
	@Test
	public void testDependencyAnalysis10() { runTest("parfor10.dml", false); }
	
	@Test
	public void testDependencyAnalysis11() { runTest("parfor11.dml", false); }
	
	@Test
	public void testDependencyAnalysis12() { runTest("parfor12.dml", false); }
	
	@Test
	public void testDependencyAnalysis13() { runTest("parfor13.dml", false); }
	
	@Test
	public void testDependencyAnalysis14() { runTest("parfor14.dml", true); }
	
	@Test
	public void testDependencyAnalysis15() { runTest("parfor15.dml", false); }
	
	@Test
	public void testDependencyAnalysis16() { runTest("parfor16.dml", true); }
	
	@Test
	public void testDependencyAnalysis17() { runTest("parfor17.dml", true); }
	
	@Test
	public void testDependencyAnalysis18() { runTest("parfor18.dml", false); }
	
	@Test
	public void testDependencyAnalysis19() { runTest("parfor19.dml", true); } //no (false) but not detectable by our applied tests 
	
	@Test
	public void testDependencyAnalysis20() { runTest("parfor20.dml", false); }
	
	@Test
	public void testDependencyAnalysis21() { runTest("parfor21.dml", true); }
	
	@Test
	public void testDependencyAnalysis22() { runTest("parfor22.dml", false); }
	
	@Test
	public void testDependencyAnalysis23() { runTest("parfor23.dml", false); }
	
	@Test
	public void testDependencyAnalysis24() { runTest("parfor24.dml", false); }
	
	@Test
	public void testDependencyAnalysis25() { runTest("parfor25.dml", false); }
	
	@Test
	public void testDependencyAnalysis26() { runTest("parfor26.dml", false); }
	
	@Test
	public void testDependencyAnalysis27() { runTest("parfor27.dml", true); }
	
	@Test
	public void testDependencyAnalysis28() { runTest("parfor28.dml", false); }
	
	@Test
	public void testDependencyAnalysis29() { runTest("parfor29.dml", true); } //ERR, but also dependency
	
	@Test
	public void testDependencyAnalysis30() { runTest("parfor30.dml", false); }
	
	@Test
	public void testDependencyAnalysis31() { runTest("parfor31.dml", false); } 
	
	@Test
	public void testDependencyAnalysis32() { runTest("parfor32.dml", true); }

	@Test
	public void testDependencyAnalysis33() { runTest("parfor33.dml", true); }
	
	@Test
	public void testDependencyAnalysis34() { runTest("parfor34.dml", true); } 
	
	@Test
	public void testDependencyAnalysis35() { runTest("parfor35.dml", false); }
	
	@Test
	public void testDependencyAnalysis36() { runTest("parfor36.dml", true); } 
	
	@Test
	public void testDependencyAnalysis37() { runTest("parfor37.dml", false); }
	
	
	private void runTest( String scriptFilename, boolean expectedException )
	{
		boolean raisedException = false;
		try
		{
			String dmlScriptString="";
			HashMap<String, String> argVals = new HashMap<String,String>();
			
			//read script
			BufferedReader in = new BufferedReader(new FileReader(DIR+scriptFilename));
			String s1 = null;
			while ((s1 = in.readLine()) != null)
				dmlScriptString += s1 + "\n";
			in.close();	
			
			//parsing and dependency analysis
			DMLQLParser parser = new DMLQLParser(dmlScriptString,argVals);
			DMLProgram prog = parser.parse();
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.validateParseTree(prog);	
		}
		catch(LanguageException ex)
		{
			raisedException = true;
			if(raisedException!=expectedException)
				ex.printStackTrace();
		}
		catch(Exception ex2)
		{
			ex2.printStackTrace();
			Assert.fail( "Unexpected exception occured during test run." );
		}
		
		//check correctness
		Assert.assertEquals(expectedException, raisedException);
	}
	
}
