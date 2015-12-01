/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.parser.python;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.BailErrorStrategy;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.DefaultErrorStrategy;
import org.antlr.v4.runtime.atn.PredictionMode;
import org.antlr.v4.runtime.misc.ParseCancellationException;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.AParserWrapper;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.ImportStatement;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParForStatement;
import org.apache.sysml.parser.ParForStatementBlock;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.parser.antlr4.DMLParserWrapper;
import org.apache.sysml.parser.python.PydmlParser.FunctionStatementContext;
import org.apache.sysml.parser.python.PydmlParser.PmlprogramContext;
import org.apache.sysml.parser.python.PydmlParser.StatementContext;
import org.apache.sysml.parser.python.PydmlSyntacticErrorListener.CustomDmlErrorListener;

/**
 * Logic of this wrapper is similar to DMLParserWrapper.
 * 
 * Note: ExpressionInfo and StatementInfo are simply wrapper objects and are reused in both DML and PyDML parsers.
 *
 */
public class PyDMLParserWrapper extends AParserWrapper
{
	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());

	/**
	 * Custom wrapper to convert statement into statement blocks. Called by doParse and in PydmlSyntacticValidator for for, parfor, while, ...
	 * @param current a statement
	 * @return corresponding statement block
	 */
	public static StatementBlock getStatementBlock(org.apache.sysml.parser.Statement current) {
		StatementBlock blk = null;
		if(current instanceof ParForStatement) {
			blk = new ParForStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof ForStatement) {
			blk = new ForStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof IfStatement) {
			blk = new IfStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof WhileStatement) {
			blk = new WhileStatementBlock();
			blk.addStatement(current);
		}
		else {
			// This includes ImportStatement
			blk = new StatementBlock();
			blk.addStatement(current);
		}
		return blk;
	}

	/**
	 * Parses the passed file with command line parameters. You can either pass both (local file) or just dmlScript (hdfs) or just file name (import command)
	 * @param fileName either full path or null --> only used for better error handling
	 * @param dmlScript required
	 * @param argVals
	 * @return
	 * @throws ParseException
	 */
	@Override
	public DMLProgram parse(String fileName, String dmlScript, HashMap<String,String> argVals) throws ParseException {
		DMLProgram prog = null;
		
		if(dmlScript == null || dmlScript.trim().isEmpty()) {
			throw new ParseException("Incorrect usage of parse. Please pass dmlScript not just filename");
		}
		
		// Set the pipeline required for ANTLR parsing
		PyDMLParserWrapper parser = new PyDMLParserWrapper();
		prog = parser.doParse(fileName, dmlScript, argVals);
		
		if(prog == null) {
			throw new ParseException("One or more errors found during parsing. (could not construct AST for file: " + fileName + "). Cannot proceed ahead.");
		}
		return prog;
	}

	/**
	 * This function is supposed to be called directly only from PydmlSyntacticValidator when it encounters 'import'
	 * @param fileName
	 * @return null if atleast one error
	 */
	public DMLProgram doParse(String fileName, String dmlScript, HashMap<String,String> argVals) throws ParseException {
		DMLProgram dmlPgm = null;
		
		ANTLRInputStream in;
		try {
			if(dmlScript == null) {
				dmlScript = DMLParserWrapper.readDMLScript(fileName);
			}
			
			InputStream stream = new ByteArrayInputStream(dmlScript.getBytes());
			in = new org.antlr.v4.runtime.ANTLRInputStream(stream);
//			else {
//				if(!(new File(fileName)).exists()) {
//					throw new ParseException("ERROR: Cannot open file:" + fileName);
//				}
//				in = new ANTLRInputStream(new FileInputStream(fileName));
//			}
		} catch (FileNotFoundException e) {
			throw new ParseException("ERROR: Cannot find file:" + fileName);
		} catch (IOException e) {
			throw new ParseException("ERROR: Cannot open file:" + fileName);
		} catch (LanguageException e) {
			throw new ParseException("ERROR: " + e.getMessage());
		}

		PmlprogramContext ast = null;
		CustomDmlErrorListener errorListener = new CustomDmlErrorListener();
		
		try {
			PydmlLexer lexer = new PydmlLexer(in);
			CommonTokenStream tokens = new CommonTokenStream(lexer);
			PydmlParser antlr4Parser = new PydmlParser(tokens);
			
			boolean tryOptimizedParsing = false; // For now no optimization, since it is not able to parse integer value. 
	
			if(tryOptimizedParsing) {
				// Try faster and simpler SLL
				antlr4Parser.getInterpreter().setPredictionMode(PredictionMode.SLL);
				antlr4Parser.removeErrorListeners();
				antlr4Parser.setErrorHandler(new BailErrorStrategy());
				try{
					ast = antlr4Parser.pmlprogram();
					// If successful, no need to try out full LL(*) ... SLL was enough
				}
				catch(ParseCancellationException ex) {
					// Error occurred, so now try full LL(*) for better error messages
					tokens.reset();
					antlr4Parser.reset();
					if(fileName != null) {
						errorListener.pushCurrentFileName(fileName);
					}
					else {
						errorListener.pushCurrentFileName("MAIN_SCRIPT");
					}
					// Set our custom error listener
					antlr4Parser.addErrorListener(errorListener);
					antlr4Parser.setErrorHandler(new DefaultErrorStrategy());
					antlr4Parser.getInterpreter().setPredictionMode(PredictionMode.LL);
					ast = antlr4Parser.pmlprogram();
				}
			}
			else {
				// Set our custom error listener
				antlr4Parser.removeErrorListeners();
				antlr4Parser.addErrorListener(errorListener);
				errorListener.pushCurrentFileName(fileName);
	
				// Now do the parsing
				ast = antlr4Parser.pmlprogram();
			}
		}
		catch(Exception e) {
			throw new ParseException("ERROR: Cannot parse the program:" + fileName);
		}
		

		try {
			// Now convert the parse tree into DMLProgram
			// Do syntactic validation while converting 
			ParseTree tree = ast;
			// And also do syntactic validation
			ParseTreeWalker walker = new ParseTreeWalker();
			PydmlSyntacticValidatorHelper helper = new PydmlSyntacticValidatorHelper(errorListener);
			PydmlSyntacticValidator validator = new PydmlSyntacticValidator(helper, fileName, argVals);
			walker.walk(validator, tree);
			errorListener.popFileName();
			if(errorListener.isAtleastOneError()) {
				return null;
			}
			dmlPgm = createDMLProgram(ast);
		}
		catch(Exception e) {
			throw new ParseException("ERROR: Cannot translate the parse tree into DMLProgram:" + e.getMessage());
		}
		
		return dmlPgm;
	}

	private DMLProgram createDMLProgram(PmlprogramContext ast) {

		DMLProgram dmlPgm = new DMLProgram();

		// First add all the functions
		for(FunctionStatementContext fn : ast.functionBlocks) {
			FunctionStatementBlock functionStmtBlk = new FunctionStatementBlock();
			functionStmtBlk.addStatement(fn.info.stmt);
			try {
				// TODO: currently the logic of nested namespace is not clear.
				String namespace = DMLProgram.DEFAULT_NAMESPACE;
				dmlPgm.addFunctionStatementBlock(namespace, fn.info.functionName, functionStmtBlk);
			} catch (LanguageException e) {
				LOG.error("line: " + fn.start.getLine() + ":" + fn.start.getCharPositionInLine() + " cannot process the function " + fn.info.functionName);
				return null;
			}
		}

		// Then add all the statements
		for(StatementContext stmtCtx : ast.blocks) {
			Statement current = stmtCtx.info.stmt;
			if(current == null) {
				LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot process the statement");
				return null;
			}
			
			// Ignore Newline logic 
			if(current.isEmptyNewLineStatement()) {
				continue;
			}

			if(current instanceof ImportStatement) {
				// Handle import statements separately
				if(stmtCtx.info.namespaces != null) {
					// Add the DMLProgram entries into current program
					for(Map.Entry<String, DMLProgram> entry : stmtCtx.info.namespaces.entrySet()) {
						dmlPgm.getNamespaces().put(entry.getKey(), entry.getValue());
//						// Don't add DMLProgram into the current program, just add function statements
//						// dmlPgm.getNamespaces().put(entry.getKey(), entry.getValue());
//						// Add function statements to current dml program
//						DMLProgram importedPgm = entry.getValue();
//
//						try {
//							for(FunctionStatementBlock importedFnBlk : importedPgm.getFunctionStatementBlocks()) {
//								if(importedFnBlk.getStatements() != null && importedFnBlk.getStatements().size() == 1) {
//									String functionName = ((FunctionStatement)importedFnBlk.getStatement(0)).getName();
//									dmlPgm.addFunctionStatementBlock(entry.getKey(), functionName, importedFnBlk);
//								}
//								else {
//									LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " incorrect number of functions in the imported function block .... strange");
//									return null;
//								}
//							}
//							if(importedPgm.getStatementBlocks() != null && importedPgm.getStatementBlocks().size() > 0) {
//								LOG.warn("Only the functions can be imported from the namespace " + entry.getKey());
//							}
//						} catch (LanguageException e) {
//							LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot import functions from the file in the import statement");
//							return null;
//						}
					}
				}
				else {
					LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot process the import statement");
					return null;
				}
			}

			// Now wrap statement into individual statement block
			// merge statement will take care of merging these blocks
			dmlPgm.addStatementBlock(getStatementBlock(current));
		}

		dmlPgm.mergeStatementBlocks();
		return dmlPgm;
	}
}
