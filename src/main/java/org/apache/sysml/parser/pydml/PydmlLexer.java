// Generated from org/apache/sysml/parser/pydml/Pydml.g4 by ANTLR 4.3
package org.apache.sysml.parser.pydml;

    // package org.apache.sysml.python;
    //import org.apache.sysml.parser.dml.StatementInfo;
    //import org.apache.sysml.parser.dml.ExpressionInfo;

import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class PydmlLexer extends Lexer {
	static { RuntimeMetaData.checkVersion("4.3", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__37=1, T__36=2, T__35=3, T__34=4, T__33=5, T__32=6, T__31=7, T__30=8, 
		T__29=9, T__28=10, T__27=11, T__26=12, T__25=13, T__24=14, T__23=15, T__22=16, 
		T__21=17, T__20=18, T__19=19, T__18=20, T__17=21, T__16=22, T__15=23, 
		T__14=24, T__13=25, T__12=26, T__11=27, T__10=28, T__9=29, T__8=30, T__7=31, 
		T__6=32, T__5=33, T__4=34, T__3=35, T__2=36, T__1=37, T__0=38, ID=39, 
		INT=40, DOUBLE=41, DIGIT=42, ALPHABET=43, COMMANDLINE_NAMED_ID=44, COMMANDLINE_POSITION_ID=45, 
		STRING=46, OPEN_BRACK=47, CLOSE_BRACK=48, OPEN_PAREN=49, CLOSE_PAREN=50, 
		NEWLINE=51, SKIP=52;
	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	public static final String[] tokenNames = {
		"'\\u0000'", "'\\u0001'", "'\\u0002'", "'\\u0003'", "'\\u0004'", "'\\u0005'", 
		"'\\u0006'", "'\\u0007'", "'\b'", "'\t'", "'\n'", "'\\u000B'", "'\f'", 
		"'\r'", "'\\u000E'", "'\\u000F'", "'\\u0010'", "'\\u0011'", "'\\u0012'", 
		"'\\u0013'", "'\\u0014'", "'\\u0015'", "'\\u0016'", "'\\u0017'", "'\\u0018'", 
		"'\\u0019'", "'\\u001A'", "'\\u001B'", "'\\u001C'", "'\\u001D'", "'\\u001E'", 
		"'\\u001F'", "' '", "'!'", "'\"'", "'#'", "'$'", "'%'", "'&'", "'''", 
		"'('", "')'", "'*'", "'+'", "','", "'-'", "'.'", "'/'", "'0'", "'1'", 
		"'2'", "'3'", "'4'"
	};
	public static final String[] ruleNames = {
		"T__37", "T__36", "T__35", "T__34", "T__33", "T__32", "T__31", "T__30", 
		"T__29", "T__28", "T__27", "T__26", "T__25", "T__24", "T__23", "T__22", 
		"T__21", "T__20", "T__19", "T__18", "T__17", "T__16", "T__15", "T__14", 
		"T__13", "T__12", "T__11", "T__10", "T__9", "T__8", "T__7", "T__6", "T__5", 
		"T__4", "T__3", "T__2", "T__1", "T__0", "ID", "INT", "DOUBLE", "DIGIT", 
		"ALPHABET", "EXP", "COMMANDLINE_NAMED_ID", "COMMANDLINE_POSITION_ID", 
		"STRING", "ESC", "OPEN_BRACK", "CLOSE_BRACK", "OPEN_PAREN", "CLOSE_PAREN", 
		"SPACES", "COMMENT", "LINE_JOINING", "NEWLINE", "SKIP"
	};


	    private boolean debugIndentRules = false;

	    // Indentation level stack
	    private java.util.Stack<Integer> indents = new java.util.Stack<Integer>();

	    // Extra tokens queue (see the NEWLINE rule).
	    private java.util.Queue<Token> tokens = new java.util.LinkedList<Token>();

	    // Number of opened braces, brackets and parenthesis.
	    private int opened = 0;

	    // This is only used to set the line number for dedent
	    private Token lastToken = null;


	    @Override
	    public void emit(Token t) {
	        if(debugIndentRules)
	            System.out.println("Emitted token:" + t);

	        super.setToken(t);
	        tokens.offer(t);
	    }


	    @Override
	    public Token nextToken() {
	        if (_input.LA(1) == EOF && !this.indents.isEmpty()) {
	            if(debugIndentRules)
	                System.out.println("EOF reached and expecting some DEDENTS, so emitting them");

	            tokens.poll();
	            this.emit(commonToken(PydmlParser.NEWLINE, "\n"));

	            // Now emit as much DEDENT tokens as needed.
	            while (!indents.isEmpty()) {
	                if(debugIndentRules)
	                    System.out.println("Emitting (inserted) DEDENTS");

	                this.emit(createDedent());
	                indents.pop();
	            }
	            // Put the EOF back on the token stream.
	            this.emit(commonToken(PydmlParser.EOF, "<EOF>"));
	        }
	        Token next = super.nextToken();
	        if (next.getChannel() == Token.DEFAULT_CHANNEL) {
	            // Keep track of the last token on the default channel.
	            this.lastToken = next;
	        }
	        Token retVal = tokens.isEmpty() ? next : tokens.poll();

	        if(debugIndentRules)
	            System.out.println("Returning nextToken: [" + retVal + "]<<" + tokens.isEmpty());

	        return retVal;
	    }

	    private Token createDedent() {
	        CommonToken dedent = commonToken(PydmlParser.DEDENT, "");
	        dedent.setLine(this.lastToken.getLine());
	        return dedent;
	    }

	    private CommonToken commonToken(int type, String text) {
	        // Nike: Main change: This logic was screwed up and was emitting additional 3 characters, so commenting it for now.
	        // int start = this.getCharIndex();
	        // int stop = start + text.length();
	        // return new CommonToken(this._tokenFactorySourcePair, type, DEFAULT_TOKEN_CHANNEL, start, stop);
	        return new CommonToken(type, text); // Main change
	    }

	    // Calculates the indentation level from the spaces:
	    // "Tabs are replaced (from left to right) by one to eight spaces
	    // such that the total number of characters up to and including
	    // the replacement is a multiple of eight [...]"
	    // https://docs.python.org/3.1/reference/lexical_analysis.html#indentation
	    static int getIndentationCount(String spaces) {
	        int count = 0;
	        for (char ch : spaces.toCharArray()) {
	            switch (ch) {
	                case '\t':
	                    count += 8 - (count % 8);
	                    break;
	                default:
	                    // A normal space char.
	                    count++;
	            }
	        }
	        return count;
	    }


	public PydmlLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "Pydml.g4"; }

	@Override
	public String[] getTokenNames() { return tokenNames; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	@Override
	public void action(RuleContext _localctx, int ruleIndex, int actionIndex) {
		switch (ruleIndex) {
		case 48: OPEN_BRACK_action((RuleContext)_localctx, actionIndex); break;

		case 49: CLOSE_BRACK_action((RuleContext)_localctx, actionIndex); break;

		case 50: OPEN_PAREN_action((RuleContext)_localctx, actionIndex); break;

		case 51: CLOSE_PAREN_action((RuleContext)_localctx, actionIndex); break;

		case 55: NEWLINE_action((RuleContext)_localctx, actionIndex); break;
		}
	}
	private void OPEN_PAREN_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 2: opened++; break;
		}
	}
	private void NEWLINE_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 4: 
		    String newLine = getText().replaceAll("[^\r\n]+", "");
		    String spaces = getText().replaceAll("[\r\n]+", "");
		    int next = _input.LA(1);
		    if (opened > 0 || next == '\r' || next == '\n' || next == '#') {
		        // If we're inside a list or on a blank line, ignore all indents,
		        // dedents and line breaks.
		        skip();
		        if(debugIndentRules) {
		            if(next == '\r' || next == '\n') {
		                    System.out.println("4.1 Skipping (blank lines)");
		            }
		            else if(next == '#') {
		                System.out.println("4.2 Skipping (comment)");
		            }
		            else {
		                System.out.println("4.2 Skipping something else");
		            }
		        }
		    }
		    else {
		        emit(commonToken(NEWLINE, newLine));

		        int indent = getIndentationCount(spaces);
		        int previous = indents.isEmpty() ? 0 : indents.peek();
		        if (indent == previous) {
		            if(debugIndentRules)
		                System.out.println("3. Skipping identation as of same size:" + next);

		            // skip indents of the same size as the present indent-size
		            skip();
		        }
		        else if (indent > previous) {
		            if(debugIndentRules)
		                System.out.println("1. Indent:" + next);

		            indents.push(indent);
		            emit(commonToken(PydmlParser.INDENT, spaces));
		        }
		        else {
		            // Possibly emit more than 1 DEDENT token.
		            while(!indents.isEmpty() && indents.peek() > indent) {
		                if(debugIndentRules)
		                    System.out.println("2. Dedent:" + next);

		                this.emit(createDedent());
		                indents.pop();
		            }
		        }
		    }
		 break;
		}
	}
	private void CLOSE_BRACK_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 1: opened--; break;
		}
	}
	private void CLOSE_PAREN_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 3: opened--; break;
		}
	}
	private void OPEN_BRACK_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 0: opened++; break;
		}
	}

	public static final String _serializedATN =
		"\3\u0430\ud6d1\u8206\uad2d\u4417\uaef1\u8d80\uaadd\2\66\u01c4\b\1\4\2"+
		"\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4"+
		"\13\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22"+
		"\t\22\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31"+
		"\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t"+
		" \4!\t!\4\"\t\"\4#\t#\4$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t"+
		"+\4,\t,\4-\t-\4.\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63\t\63\4\64"+
		"\t\64\4\65\t\65\4\66\t\66\4\67\t\67\48\t8\49\t9\4:\t:\3\2\3\2\3\3\3\3"+
		"\3\3\3\3\3\4\3\4\3\4\3\5\3\5\3\5\3\6\3\6\3\7\3\7\3\7\3\7\3\7\3\7\3\b\3"+
		"\b\3\b\3\t\3\t\3\n\3\n\3\n\3\n\3\13\3\13\3\13\3\f\3\f\3\f\3\f\3\f\3\f"+
		"\3\f\3\r\3\r\3\r\3\16\3\16\3\16\3\16\3\16\3\16\3\17\3\17\3\20\3\20\3\20"+
		"\3\20\3\20\3\20\3\21\3\21\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22"+
		"\3\22\3\22\3\22\3\23\3\23\3\23\3\23\3\23\3\23\3\23\3\23\3\23\3\23\3\23"+
		"\3\23\3\24\3\24\3\25\3\25\3\25\3\26\3\26\3\26\3\26\3\26\3\26\3\26\3\27"+
		"\3\27\3\30\3\30\3\30\3\31\3\31\3\32\3\32\3\32\3\33\3\33\3\34\3\34\3\34"+
		"\3\35\3\35\3\35\3\35\3\35\3\36\3\36\3\37\3\37\3\37\3 \3 \3!\3!\3!\3!\3"+
		"!\3!\3\"\3\"\3#\3#\3#\3$\3$\3$\3$\3$\3%\3%\3%\3%\3&\3&\3\'\3\'\3(\3(\3"+
		"(\3(\7(\u010b\n(\f(\16(\u010e\13(\3(\3(\5(\u0112\n(\3(\3(\3(\3(\7(\u0118"+
		"\n(\f(\16(\u011b\13(\3(\3(\3(\3(\3(\3(\3(\3(\3(\3(\3(\3(\5(\u0129\n(\3"+
		")\6)\u012c\n)\r)\16)\u012d\3)\5)\u0131\n)\3*\6*\u0134\n*\r*\16*\u0135"+
		"\3*\3*\7*\u013a\n*\f*\16*\u013d\13*\3*\5*\u0140\n*\3*\5*\u0143\n*\3*\6"+
		"*\u0146\n*\r*\16*\u0147\3*\5*\u014b\n*\3*\5*\u014e\n*\3*\3*\6*\u0152\n"+
		"*\r*\16*\u0153\3*\5*\u0157\n*\3*\5*\u015a\n*\5*\u015c\n*\3+\3+\3,\3,\3"+
		"-\3-\5-\u0164\n-\3-\3-\3.\3.\3.\3.\3.\7.\u016d\n.\f.\16.\u0170\13.\3/"+
		"\3/\6/\u0174\n/\r/\16/\u0175\3\60\3\60\3\60\7\60\u017b\n\60\f\60\16\60"+
		"\u017e\13\60\3\60\3\60\3\60\3\60\7\60\u0184\n\60\f\60\16\60\u0187\13\60"+
		"\3\60\5\60\u018a\n\60\3\61\3\61\3\61\3\62\3\62\3\62\3\63\3\63\3\63\3\64"+
		"\3\64\3\64\3\65\3\65\3\65\3\66\6\66\u019c\n\66\r\66\16\66\u019d\3\67\3"+
		"\67\7\67\u01a2\n\67\f\67\16\67\u01a5\13\67\38\38\58\u01a9\n8\38\58\u01ac"+
		"\n8\38\38\58\u01b0\n8\39\59\u01b3\n9\39\39\59\u01b7\n9\39\59\u01ba\n9"+
		"\39\39\3:\3:\3:\5:\u01c1\n:\3:\3:\4\u017c\u0185\2;\3\3\5\4\7\5\t\6\13"+
		"\7\r\b\17\t\21\n\23\13\25\f\27\r\31\16\33\17\35\20\37\21!\22#\23%\24\'"+
		"\25)\26+\27-\30/\31\61\32\63\33\65\34\67\359\36;\37= ?!A\"C#E$G%I&K\'"+
		"M(O)Q*S+U,W-Y\2[.]/_\60a\2c\61e\62g\63i\64k\2m\2o\2q\65s\66\3\2\13\4\2"+
		"NNnn\4\2C\\c|\4\2GGgg\4\2--//\4\2$$^^\4\2))^^\13\2$$))^^cdhhppttvvxx\4"+
		"\2\13\13\"\"\4\2\f\f\17\17\u01e8\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2"+
		"\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3\2\2\2\2\23\3\2"+
		"\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2\2\2\2\35\3\2\2\2"+
		"\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3\2\2\2\2)\3\2\2\2"+
		"\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3\2\2\2\2\65\3\2\2"+
		"\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3\2\2\2\2A\3\2\2\2"+
		"\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3\2\2\2\2K\3\2\2\2\2M\3\2\2\2\2O"+
		"\3\2\2\2\2Q\3\2\2\2\2S\3\2\2\2\2U\3\2\2\2\2W\3\2\2\2\2[\3\2\2\2\2]\3\2"+
		"\2\2\2_\3\2\2\2\2c\3\2\2\2\2e\3\2\2\2\2g\3\2\2\2\2i\3\2\2\2\2q\3\2\2\2"+
		"\2s\3\2\2\2\3u\3\2\2\2\5w\3\2\2\2\7{\3\2\2\2\t~\3\2\2\2\13\u0081\3\2\2"+
		"\2\r\u0083\3\2\2\2\17\u0089\3\2\2\2\21\u008c\3\2\2\2\23\u008e\3\2\2\2"+
		"\25\u0092\3\2\2\2\27\u0095\3\2\2\2\31\u009c\3\2\2\2\33\u009f\3\2\2\2\35"+
		"\u00a5\3\2\2\2\37\u00a7\3\2\2\2!\u00ad\3\2\2\2#\u00af\3\2\2\2%\u00bb\3"+
		"\2\2\2\'\u00c7\3\2\2\2)\u00c9\3\2\2\2+\u00cc\3\2\2\2-\u00d3\3\2\2\2/\u00d5"+
		"\3\2\2\2\61\u00d8\3\2\2\2\63\u00da\3\2\2\2\65\u00dd\3\2\2\2\67\u00df\3"+
		"\2\2\29\u00e2\3\2\2\2;\u00e7\3\2\2\2=\u00e9\3\2\2\2?\u00ec\3\2\2\2A\u00ee"+
		"\3\2\2\2C\u00f4\3\2\2\2E\u00f6\3\2\2\2G\u00f9\3\2\2\2I\u00fe\3\2\2\2K"+
		"\u0102\3\2\2\2M\u0104\3\2\2\2O\u0128\3\2\2\2Q\u012b\3\2\2\2S\u015b\3\2"+
		"\2\2U\u015d\3\2\2\2W\u015f\3\2\2\2Y\u0161\3\2\2\2[\u0167\3\2\2\2]\u0171"+
		"\3\2\2\2_\u0189\3\2\2\2a\u018b\3\2\2\2c\u018e\3\2\2\2e\u0191\3\2\2\2g"+
		"\u0194\3\2\2\2i\u0197\3\2\2\2k\u019b\3\2\2\2m\u019f\3\2\2\2o\u01a6\3\2"+
		"\2\2q\u01b6\3\2\2\2s\u01c0\3\2\2\2uv\7\61\2\2v\4\3\2\2\2wx\7f\2\2xy\7"+
		"g\2\2yz\7h\2\2z\6\3\2\2\2{|\7c\2\2|}\7u\2\2}\b\3\2\2\2~\177\7#\2\2\177"+
		"\u0080\7?\2\2\u0080\n\3\2\2\2\u0081\u0082\7=\2\2\u0082\f\3\2\2\2\u0083"+
		"\u0084\7y\2\2\u0084\u0085\7j\2\2\u0085\u0086\7k\2\2\u0086\u0087\7n\2\2"+
		"\u0087\u0088\7g\2\2\u0088\16\3\2\2\2\u0089\u008a\7,\2\2\u008a\u008b\7"+
		",\2\2\u008b\20\3\2\2\2\u008c\u008d\7?\2\2\u008d\22\3\2\2\2\u008e\u008f"+
		"\7h\2\2\u008f\u0090\7q\2\2\u0090\u0091\7t\2\2\u0091\24\3\2\2\2\u0092\u0093"+
		"\7k\2\2\u0093\u0094\7h\2\2\u0094\26\3\2\2\2\u0095\u0096\7u\2\2\u0096\u0097"+
		"\7q\2\2\u0097\u0098\7w\2\2\u0098\u0099\7t\2\2\u0099\u009a\7e\2\2\u009a"+
		"\u009b\7g\2\2\u009b\30\3\2\2\2\u009c\u009d\7>\2\2\u009d\u009e\7?\2\2\u009e"+
		"\32\3\2\2\2\u009f\u00a0\7u\2\2\u00a0\u00a1\7g\2\2\u00a1\u00a2\7v\2\2\u00a2"+
		"\u00a3\7y\2\2\u00a3\u00a4\7f\2\2\u00a4\34\3\2\2\2\u00a5\u00a6\7(\2\2\u00a6"+
		"\36\3\2\2\2\u00a7\u00a8\7H\2\2\u00a8\u00a9\7c\2\2\u00a9\u00aa\7n\2\2\u00aa"+
		"\u00ab\7u\2\2\u00ab\u00ac\7g\2\2\u00ac \3\2\2\2\u00ad\u00ae\7,\2\2\u00ae"+
		"\"\3\2\2\2\u00af\u00b0\7k\2\2\u00b0\u00b1\7o\2\2\u00b1\u00b2\7r\2\2\u00b2"+
		"\u00b3\7n\2\2\u00b3\u00b4\7g\2\2\u00b4\u00b5\7o\2\2\u00b5\u00b6\7g\2\2"+
		"\u00b6\u00b7\7p\2\2\u00b7\u00b8\7v\2\2\u00b8\u00b9\7g\2\2\u00b9\u00ba"+
		"\7f\2\2\u00ba$\3\2\2\2\u00bb\u00bc\7f\2\2\u00bc\u00bd\7g\2\2\u00bd\u00be"+
		"\7h\2\2\u00be\u00bf\7G\2\2\u00bf\u00c0\7z\2\2\u00c0\u00c1\7v\2\2\u00c1"+
		"\u00c2\7g\2\2\u00c2\u00c3\7t\2\2\u00c3\u00c4\7p\2\2\u00c4\u00c5\7c\2\2"+
		"\u00c5\u00c6\7n\2\2\u00c6&\3\2\2\2\u00c7\u00c8\7.\2\2\u00c8(\3\2\2\2\u00c9"+
		"\u00ca\7/\2\2\u00ca\u00cb\7@\2\2\u00cb*\3\2\2\2\u00cc\u00cd\7r\2\2\u00cd"+
		"\u00ce\7c\2\2\u00ce\u00cf\7t\2\2\u00cf\u00d0\7h\2\2\u00d0\u00d1\7q\2\2"+
		"\u00d1\u00d2\7t\2\2\u00d2,\3\2\2\2\u00d3\u00d4\7<\2\2\u00d4.\3\2\2\2\u00d5"+
		"\u00d6\7@\2\2\u00d6\u00d7\7?\2\2\u00d7\60\3\2\2\2\u00d8\u00d9\7~\2\2\u00d9"+
		"\62\3\2\2\2\u00da\u00db\7?\2\2\u00db\u00dc\7?\2\2\u00dc\64\3\2\2\2\u00dd"+
		"\u00de\7>\2\2\u00de\66\3\2\2\2\u00df\u00e0\7\61\2\2\u00e0\u00e1\7\61\2"+
		"\2\u00e18\3\2\2\2\u00e2\u00e3\7V\2\2\u00e3\u00e4\7t\2\2\u00e4\u00e5\7"+
		"w\2\2\u00e5\u00e6\7g\2\2\u00e6:\3\2\2\2\u00e7\u00e8\7@\2\2\u00e8<\3\2"+
		"\2\2\u00e9\u00ea\7q\2\2\u00ea\u00eb\7t\2\2\u00eb>\3\2\2\2\u00ec\u00ed"+
		"\7#\2\2\u00ed@\3\2\2\2\u00ee\u00ef\7k\2\2\u00ef\u00f0\7h\2\2\u00f0\u00f1"+
		"\7f\2\2\u00f1\u00f2\7g\2\2\u00f2\u00f3\7h\2\2\u00f3B\3\2\2\2\u00f4\u00f5"+
		"\7\'\2\2\u00f5D\3\2\2\2\u00f6\u00f7\7k\2\2\u00f7\u00f8\7p\2\2\u00f8F\3"+
		"\2\2\2\u00f9\u00fa\7g\2\2\u00fa\u00fb\7n\2\2\u00fb\u00fc\7u\2\2\u00fc"+
		"\u00fd\7g\2\2\u00fdH\3\2\2\2\u00fe\u00ff\7c\2\2\u00ff\u0100\7p\2\2\u0100"+
		"\u0101\7f\2\2\u0101J\3\2\2\2\u0102\u0103\7-\2\2\u0103L\3\2\2\2\u0104\u0105"+
		"\7/\2\2\u0105N\3\2\2\2\u0106\u010c\5W,\2\u0107\u010b\5W,\2\u0108\u010b"+
		"\5U+\2\u0109\u010b\7a\2\2\u010a\u0107\3\2\2\2\u010a\u0108\3\2\2\2\u010a"+
		"\u0109\3\2\2\2\u010b\u010e\3\2\2\2\u010c\u010a\3\2\2\2\u010c\u010d\3\2"+
		"\2\2\u010d\u010f\3\2\2\2\u010e\u010c\3\2\2\2\u010f\u0110\7\60\2\2\u0110"+
		"\u0112\3\2\2\2\u0111\u0106\3\2\2\2\u0111\u0112\3\2\2\2\u0112\u0113\3\2"+
		"\2\2\u0113\u0119\5W,\2\u0114\u0118\5W,\2\u0115\u0118\5U+\2\u0116\u0118"+
		"\7a\2\2\u0117\u0114\3\2\2\2\u0117\u0115\3\2\2\2\u0117\u0116\3\2\2\2\u0118"+
		"\u011b\3\2\2\2\u0119\u0117\3\2\2\2\u0119\u011a\3\2\2\2\u011a\u0129\3\2"+
		"\2\2\u011b\u0119\3\2\2\2\u011c\u011d\7k\2\2\u011d\u011e\7p\2\2\u011e\u011f"+
		"\7f\2\2\u011f\u0120\7g\2\2\u0120\u0121\7z\2\2\u0121\u0122\7\60\2\2\u0122"+
		"\u0123\7t\2\2\u0123\u0124\7g\2\2\u0124\u0125\7v\2\2\u0125\u0126\7w\2\2"+
		"\u0126\u0127\7t\2\2\u0127\u0129\7p\2\2\u0128\u0111\3\2\2\2\u0128\u011c"+
		"\3\2\2\2\u0129P\3\2\2\2\u012a\u012c\5U+\2\u012b\u012a\3\2\2\2\u012c\u012d"+
		"\3\2\2\2\u012d\u012b\3\2\2\2\u012d\u012e\3\2\2\2\u012e\u0130\3\2\2\2\u012f"+
		"\u0131\t\2\2\2\u0130\u012f\3\2\2\2\u0130\u0131\3\2\2\2\u0131R\3\2\2\2"+
		"\u0132\u0134\5U+\2\u0133\u0132\3\2\2\2\u0134\u0135\3\2\2\2\u0135\u0133"+
		"\3\2\2\2\u0135\u0136\3\2\2\2\u0136\u0137\3\2\2\2\u0137\u013b\7\60\2\2"+
		"\u0138\u013a\5U+\2\u0139\u0138\3\2\2\2\u013a\u013d\3\2\2\2\u013b\u0139"+
		"\3\2\2\2\u013b\u013c\3\2\2\2\u013c\u013f\3\2\2\2\u013d\u013b\3\2\2\2\u013e"+
		"\u0140\5Y-\2\u013f\u013e\3\2\2\2\u013f\u0140\3\2\2\2\u0140\u0142\3\2\2"+
		"\2\u0141\u0143\t\2\2\2\u0142\u0141\3\2\2\2\u0142\u0143\3\2\2\2\u0143\u015c"+
		"\3\2\2\2\u0144\u0146\5U+\2\u0145\u0144\3\2\2\2\u0146\u0147\3\2\2\2\u0147"+
		"\u0145\3\2\2\2\u0147\u0148\3\2\2\2\u0148\u014a\3\2\2\2\u0149\u014b\5Y"+
		"-\2\u014a\u0149\3\2\2\2\u014a\u014b\3\2\2\2\u014b\u014d\3\2\2\2\u014c"+
		"\u014e\t\2\2\2\u014d\u014c\3\2\2\2\u014d\u014e\3\2\2\2\u014e\u015c\3\2"+
		"\2\2\u014f\u0151\7\60\2\2\u0150\u0152\5U+\2\u0151\u0150\3\2\2\2\u0152"+
		"\u0153\3\2\2\2\u0153\u0151\3\2\2\2\u0153\u0154\3\2\2\2\u0154\u0156\3\2"+
		"\2\2\u0155\u0157\5Y-\2\u0156\u0155\3\2\2\2\u0156\u0157\3\2\2\2\u0157\u0159"+
		"\3\2\2\2\u0158\u015a\t\2\2\2\u0159\u0158\3\2\2\2\u0159\u015a\3\2\2\2\u015a"+
		"\u015c\3\2\2\2\u015b\u0133\3\2\2\2\u015b\u0145\3\2\2\2\u015b\u014f\3\2"+
		"\2\2\u015cT\3\2\2\2\u015d\u015e\4\62;\2\u015eV\3\2\2\2\u015f\u0160\t\3"+
		"\2\2\u0160X\3\2\2\2\u0161\u0163\t\4\2\2\u0162\u0164\t\5\2\2\u0163\u0162"+
		"\3\2\2\2\u0163\u0164\3\2\2\2\u0164\u0165\3\2\2\2\u0165\u0166\5Q)\2\u0166"+
		"Z\3\2\2\2\u0167\u0168\7&\2\2\u0168\u016e\5W,\2\u0169\u016d\5W,\2\u016a"+
		"\u016d\5U+\2\u016b\u016d\7a\2\2\u016c\u0169\3\2\2\2\u016c\u016a\3\2\2"+
		"\2\u016c\u016b\3\2\2\2\u016d\u0170\3\2\2\2\u016e\u016c\3\2\2\2\u016e\u016f"+
		"\3\2\2\2\u016f\\\3\2\2\2\u0170\u016e\3\2\2\2\u0171\u0173\7&\2\2\u0172"+
		"\u0174\5U+\2\u0173\u0172\3\2\2\2\u0174\u0175\3\2\2\2\u0175\u0173\3\2\2"+
		"\2\u0175\u0176\3\2\2\2\u0176^\3\2\2\2\u0177\u017c\7$\2\2\u0178\u017b\5"+
		"a\61\2\u0179\u017b\n\6\2\2\u017a\u0178\3\2\2\2\u017a\u0179\3\2\2\2\u017b"+
		"\u017e\3\2\2\2\u017c\u017d\3\2\2\2\u017c\u017a\3\2\2\2\u017d\u017f\3\2"+
		"\2\2\u017e\u017c\3\2\2\2\u017f\u018a\7$\2\2\u0180\u0185\7)\2\2\u0181\u0184"+
		"\5a\61\2\u0182\u0184\n\7\2\2\u0183\u0181\3\2\2\2\u0183\u0182\3\2\2\2\u0184"+
		"\u0187\3\2\2\2\u0185\u0186\3\2\2\2\u0185\u0183\3\2\2\2\u0186\u0188\3\2"+
		"\2\2\u0187\u0185\3\2\2\2\u0188\u018a\7)\2\2\u0189\u0177\3\2\2\2\u0189"+
		"\u0180\3\2\2\2\u018a`\3\2\2\2\u018b\u018c\7^\2\2\u018c\u018d\t\b\2\2\u018d"+
		"b\3\2\2\2\u018e\u018f\7]\2\2\u018f\u0190\b\62\2\2\u0190d\3\2\2\2\u0191"+
		"\u0192\7_\2\2\u0192\u0193\b\63\3\2\u0193f\3\2\2\2\u0194\u0195\7*\2\2\u0195"+
		"\u0196\b\64\4\2\u0196h\3\2\2\2\u0197\u0198\7+\2\2\u0198\u0199\b\65\5\2"+
		"\u0199j\3\2\2\2\u019a\u019c\t\t\2\2\u019b\u019a\3\2\2\2\u019c\u019d\3"+
		"\2\2\2\u019d\u019b\3\2\2\2\u019d\u019e\3\2\2\2\u019el\3\2\2\2\u019f\u01a3"+
		"\7%\2\2\u01a0\u01a2\n\n\2\2\u01a1\u01a0\3\2\2\2\u01a2\u01a5\3\2\2\2\u01a3"+
		"\u01a1\3\2\2\2\u01a3\u01a4\3\2\2\2\u01a4n\3\2\2\2\u01a5\u01a3\3\2\2\2"+
		"\u01a6\u01a8\7^\2\2\u01a7\u01a9\5k\66\2\u01a8\u01a7\3\2\2\2\u01a8\u01a9"+
		"\3\2\2\2\u01a9\u01af\3\2\2\2\u01aa\u01ac\7\17\2\2\u01ab\u01aa\3\2\2\2"+
		"\u01ab\u01ac\3\2\2\2\u01ac\u01ad\3\2\2\2\u01ad\u01b0\7\f\2\2\u01ae\u01b0"+
		"\7\17\2\2\u01af\u01ab\3\2\2\2\u01af\u01ae\3\2\2\2\u01b0p\3\2\2\2\u01b1"+
		"\u01b3\7\17\2\2\u01b2\u01b1\3\2\2\2\u01b2\u01b3\3\2\2\2\u01b3\u01b4\3"+
		"\2\2\2\u01b4\u01b7\7\f\2\2\u01b5\u01b7\7\17\2\2\u01b6\u01b2\3\2\2\2\u01b6"+
		"\u01b5\3\2\2\2\u01b7\u01b9\3\2\2\2\u01b8\u01ba\5k\66\2\u01b9\u01b8\3\2"+
		"\2\2\u01b9\u01ba\3\2\2\2\u01ba\u01bb\3\2\2\2\u01bb\u01bc\b9\6\2\u01bc"+
		"r\3\2\2\2\u01bd\u01c1\5k\66\2\u01be\u01c1\5m\67\2\u01bf\u01c1\5o8\2\u01c0"+
		"\u01bd\3\2\2\2\u01c0\u01be\3\2\2\2\u01c0\u01bf\3\2\2\2\u01c1\u01c2\3\2"+
		"\2\2\u01c2\u01c3\b:\7\2\u01c3t\3\2\2\2(\2\u010a\u010c\u0111\u0117\u0119"+
		"\u0128\u012d\u0130\u0135\u013b\u013f\u0142\u0147\u014a\u014d\u0153\u0156"+
		"\u0159\u015b\u0163\u016c\u016e\u0175\u017a\u017c\u0183\u0185\u0189\u019d"+
		"\u01a3\u01a8\u01ab\u01af\u01b2\u01b6\u01b9\u01c0\b\3\62\2\3\63\3\3\64"+
		"\4\3\65\5\39\6\b\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}