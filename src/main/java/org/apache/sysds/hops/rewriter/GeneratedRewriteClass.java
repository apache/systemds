package org.apache.sysds.hops.rewriter;

import java.util.ArrayList;
import java.util.function.Function;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;

public class GeneratedRewriteClass implements Function {

	@Override
	public Object apply( Object hi ) {
		if ( hi == null )
			return null;

		hi = _applyRewrite0((Hop) hi);		// /(a,1.0) => a
		hi = _applyRewrite1((Hop) hi);		// *(1.0,a) => a
		hi = _applyRewrite2((Hop) hi);		// *(a,1.0) => a
		hi = _applyRewrite3((Hop) hi);		// +(0.0,a) => a
		hi = _applyRewrite4((Hop) hi);		// +(a,0.0) => a
		hi = _applyRewrite5((Hop) hi);		// +(0.0,A) => A
		hi = _applyRewrite6((Hop) hi);		// +(A,0.0) => A
		hi = _applyRewrite7((Hop) hi);		// /(0.0,a) => 0.0
		hi = _applyRewrite8((Hop) hi);		// *(0.0,a) => 0.0
		hi = _applyRewrite9((Hop) hi);		// *(a,0.0) => 0.0
		hi = _applyRewrite13((Hop) hi);		// /(A,c) => *(A,/(1.0,c))
		hi = _applyRewrite14((Hop) hi);		// rowSums(*(a,B)) => *(a,rowSums(B))
		hi = _applyRewrite15((Hop) hi);		// rowSums(*(B,a)) => *(a,rowSums(B))
		hi = _applyRewrite16((Hop) hi);		// colSums(*(a,B)) => *(a,colSums(B))
		hi = _applyRewrite17((Hop) hi);		// colSums(*(B,a)) => *(a,colSums(B))
		hi = _applyRewrite18((Hop) hi);		// *(/(1.0,B),A) => /(A,B)
		hi = _applyRewrite19((Hop) hi);		// *(A,/(1.0,B)) => /(A,B)
		hi = _applyRewrite20((Hop) hi);		// *(/(1.0,B),a) => /(a,B)
		hi = _applyRewrite21((Hop) hi);		// *(a,/(1.0,B)) => /(a,B)
		hi = _applyRewrite22((Hop) hi);		// *(/(a,C),b) => /(*(a,b),C)
		hi = _applyRewrite23((Hop) hi);		// *(a,/(b,C)) => /(*(a,b),C)
		hi = _applyRewrite28((Hop) hi);		// -(0.0,-(B,A)) => -(A,B)
		hi = _applyRewrite29((Hop) hi);		// +(-(0.0,B),A) => -(A,B)
		hi = _applyRewrite30((Hop) hi);		// +(A,-(0.0,B)) => -(A,B)
		hi = _applyRewrite31((Hop) hi);		// -(0.0,-(B,a)) => -(a,B)
		hi = _applyRewrite32((Hop) hi);		// +(-(0.0,B),a) => -(a,B)
		hi = _applyRewrite33((Hop) hi);		// +(a,-(0.0,B)) => -(a,B)
		hi = _applyRewrite34((Hop) hi);		// -(0.0,-(b,A)) => -(A,b)
		hi = _applyRewrite35((Hop) hi);		// -(-(A,b),c) => -(A,+(b,c))
		hi = _applyRewrite36((Hop) hi);		// -(a,+(b,C)) => -(-(a,b),C)
		hi = _applyRewrite37((Hop) hi);		// -(a,+(C,b)) => -(-(a,b),C)
		hi = _applyRewrite38((Hop) hi);		// -(-(a,C),b) => -(-(a,b),C)
		hi = _applyRewrite39((Hop) hi);		// -(a,-(C,b)) => -(+(a,b),C)
		hi = _applyRewrite40((Hop) hi);		// +(-(a,C),b) => -(+(a,b),C)
		hi = _applyRewrite41((Hop) hi);		// +(a,-(b,C)) => -(+(a,b),C)
		hi = _applyRewrite42((Hop) hi);		// -(+(b,A),c) => +(A,-(b,c))
		hi = _applyRewrite43((Hop) hi);		// -(+(A,b),c) => +(A,-(b,c))
		hi = _applyRewrite44((Hop) hi);		// -(b,-(c,A)) => +(A,-(b,c))
		hi = _applyRewrite45((Hop) hi);		// +(-(A,c),b) => +(A,-(b,c))
		hi = _applyRewrite46((Hop) hi);		// +(b,-(A,c)) => +(A,-(b,c))
		hi = _applyRewrite47((Hop) hi);		// colSums(-(0.0,B)) => -(0.0,colSums(B))
		hi = _applyRewrite48((Hop) hi);		// rowSums(-(0.0,B)) => -(0.0,rowSums(B))
		hi = _applyRewrite49((Hop) hi);		// *(/(1.0,b),a) => /(a,b)
		hi = _applyRewrite50((Hop) hi);		// *(a,/(1.0,b)) => /(a,b)
		hi = _applyRewrite51((Hop) hi);		// -(0.0,-(b,a)) => -(a,b)
		hi = _applyRewrite52((Hop) hi);		// -(a,-(b,0.0)) => -(a,b)
		hi = _applyRewrite53((Hop) hi);		// +(-(0.0,b),a) => -(a,b)
		hi = _applyRewrite54((Hop) hi);		// +(a,-(0.0,b)) => -(a,b)
		hi = _applyRewrite55((Hop) hi);		// *(-(a,0.0),b) => *(a,b)
		hi = _applyRewrite56((Hop) hi);		// *(a,-(b,0.0)) => *(a,b)
		hi = _applyRewrite57((Hop) hi);		// /(-(a,0.0),b) => /(a,b)
		hi = _applyRewrite58((Hop) hi);		// -(A,-(b,0.0)) => -(A,b)
		hi = _applyRewrite59((Hop) hi);		// +(-(0.0,b),A) => -(A,b)
		hi = _applyRewrite60((Hop) hi);		// +(A,-(0.0,b)) => -(A,b)
		hi = _applyRewrite61((Hop) hi);		// *(-(b,0.0),A) => *(A,b)
		hi = _applyRewrite62((Hop) hi);		// *(A,-(b,0.0)) => *(A,b)
		hi = _applyRewrite63((Hop) hi);		// /(-(a,0.0),B) => /(a,B)
		hi = _applyRewrite65((Hop) hi);		// t(-(a,t(B))) => -(a,B)
		hi = _applyRewrite66((Hop) hi);		// t(-(t(A),b)) => -(A,b)
		hi = _applyRewrite67((Hop) hi);		// t(+(t(A),b)) => +(A,b)
		hi = _applyRewrite68((Hop) hi);		// t(+(b,t(A))) => +(A,b)
		hi = _applyRewrite69((Hop) hi);		// t(*(t(A),b)) => *(A,b)
		hi = _applyRewrite70((Hop) hi);		// t(*(b,t(A))) => *(A,b)
		hi = _applyRewrite71((Hop) hi);		// t(/(a,t(B))) => /(a,B)
		hi = _applyRewrite72((Hop) hi);		// +(*(C,A),*(B,A)) => *(A,+(B,C))
		hi = _applyRewrite73((Hop) hi);		// +(*(C,A),*(A,B)) => *(A,+(B,C))
		hi = _applyRewrite74((Hop) hi);		// +(*(A,C),*(B,A)) => *(A,+(B,C))
		hi = _applyRewrite75((Hop) hi);		// +(*(A,C),*(A,B)) => *(A,+(B,C))
		hi = _applyRewrite76((Hop) hi);		// *(t(*(a,C)),b) => *(*(a,b),t(C))
		hi = _applyRewrite77((Hop) hi);		// *(t(*(C,a)),b) => *(*(a,b),t(C))
		hi = _applyRewrite78((Hop) hi);		// *(a,t(*(b,C))) => *(*(a,b),t(C))
		hi = _applyRewrite79((Hop) hi);		// *(a,t(*(C,b))) => *(*(a,b),t(C))
		hi = _applyRewrite114((Hop) hi);		// *(t(/(a,C)),b) => /(*(a,b),t(C))
		hi = _applyRewrite115((Hop) hi);		// *(a,t(/(b,C))) => /(*(a,b),t(C))
		hi = _applyRewrite116((Hop) hi);		// colSums(/(*(a,B),C)) => *(a,colSums(/(B,C)))
		hi = _applyRewrite117((Hop) hi);		// colSums(/(*(B,a),C)) => *(a,colSums(/(B,C)))
		hi = _applyRewrite118((Hop) hi);		// colSums(*(/(a,C),B)) => *(a,colSums(/(B,C)))
		hi = _applyRewrite119((Hop) hi);		// colSums(*(B,/(a,C))) => *(a,colSums(/(B,C)))
		hi = _applyRewrite120((Hop) hi);		// rowSums(*(/(a,C),B)) => *(a,rowSums(/(B,C)))
		hi = _applyRewrite121((Hop) hi);		// rowSums(*(B,/(a,C))) => *(a,rowSums(/(B,C)))
		hi = _applyRewrite122((Hop) hi);		// rowSums(/(*(a,B),C)) => *(a,rowSums(/(B,C)))
		hi = _applyRewrite123((Hop) hi);		// rowSums(/(*(B,a),C)) => *(a,rowSums(/(B,C)))
		hi = _applyRewrite128((Hop) hi);		// *(/(*(b,A),D),c) => *(A,/(*(b,c),D))
		hi = _applyRewrite129((Hop) hi);		// *(/(*(A,b),D),c) => *(A,/(*(b,c),D))
		hi = _applyRewrite130((Hop) hi);		// *(b,/(*(c,A),D)) => *(A,/(*(b,c),D))
		hi = _applyRewrite131((Hop) hi);		// *(b,/(*(A,c),D)) => *(A,/(*(b,c),D))
		hi = _applyRewrite132((Hop) hi);		// *(/(/(a,C),D),b) => /(/(*(a,b),C),D)
		hi = _applyRewrite133((Hop) hi);		// *(/(a,C),/(b,D)) => /(/(*(a,b),C),D)
		hi = _applyRewrite134((Hop) hi);		// *(a,/(/(b,C),D)) => /(/(*(a,b),C),D)
		hi = _applyRewrite145((Hop) hi);		// t(-(t(A),B)) => -(A,t(B))
		hi = _applyRewrite146((Hop) hi);		// t(-(A,t(B))) => -(t(A),B)
		hi = _applyRewrite147((Hop) hi);		// -(t(A),t(B)) => t(-(A,B))
		hi = _applyRewrite148((Hop) hi);		// +(t(B),t(A)) => t(+(A,B))
		hi = _applyRewrite149((Hop) hi);		// t(+(t(A),B)) => +(A,t(B))
		hi = _applyRewrite150((Hop) hi);		// t(+(B,t(A))) => +(A,t(B))
		hi = _applyRewrite151((Hop) hi);		// -(t(-(a,C)),b) => -(-(a,b),t(C))
		hi = _applyRewrite152((Hop) hi);		// -(t(-(A,b)),c) => -(t(A),+(b,c))
		hi = _applyRewrite153((Hop) hi);		// -(-(-(a,D),C),b) => -(-(a,b),+(C,D))
		hi = _applyRewrite154((Hop) hi);		// -(-(a,C),+(b,D)) => -(-(a,b),+(C,D))
		hi = _applyRewrite155((Hop) hi);		// -(-(a,C),+(D,b)) => -(-(a,b),+(C,D))
		hi = _applyRewrite156((Hop) hi);		// -(-(-(A,c),B),d) => -(A,+(B,+(c,d)))
		hi = _applyRewrite157((Hop) hi);		// -(-(A,+(c,B)),d) => -(A,+(B,+(c,d)))
		hi = _applyRewrite158((Hop) hi);		// -(-(A,+(B,c)),d) => -(A,+(B,+(c,d)))
		hi = _applyRewrite159((Hop) hi);		// -(-(A,c),+(d,B)) => -(A,+(B,+(c,d)))
		hi = _applyRewrite160((Hop) hi);		// -(-(A,c),+(B,d)) => -(A,+(B,+(c,d)))
		hi = _applyRewrite161((Hop) hi);		// -(-(a,C),-(D,b)) => -(+(a,b),+(C,D))
		hi = _applyRewrite162((Hop) hi);		// -(a,-(D,-(b,C))) => -(+(a,b),+(C,D))
		hi = _applyRewrite163((Hop) hi);		// -(a,+(-(C,b),D)) => -(+(a,b),+(C,D))
		hi = _applyRewrite164((Hop) hi);		// -(a,+(D,-(C,b))) => -(+(a,b),+(C,D))
		hi = _applyRewrite165((Hop) hi);		// +(-(-(a,D),C),b) => -(+(a,b),+(C,D))
		hi = _applyRewrite166((Hop) hi);		// +(-(a,D),-(b,C)) => -(+(a,b),+(C,D))
		hi = _applyRewrite167((Hop) hi);		// +(a,-(-(b,C),D)) => -(+(a,b),+(C,D))
		hi = _applyRewrite168((Hop) hi);		// -(-(A,-(c,B)),d) => +(A,-(B,+(c,d)))
		hi = _applyRewrite169((Hop) hi);		// -(-(B,c),-(d,A)) => +(A,-(B,+(c,d)))
		hi = _applyRewrite170((Hop) hi);		// -(+(-(A,c),B),d) => +(A,-(B,+(c,d)))
		hi = _applyRewrite171((Hop) hi);		// -(+(B,-(A,c)),d) => +(A,-(B,+(c,d)))
		hi = _applyRewrite172((Hop) hi);		// +(-(B,c),-(A,d)) => +(A,-(B,+(c,d)))
		hi = _applyRewrite173((Hop) hi);		// +(t(+(a,C)),b) => +(+(a,b),t(C))
		hi = _applyRewrite174((Hop) hi);		// +(t(+(C,a)),b) => +(+(a,b),t(C))
		hi = _applyRewrite175((Hop) hi);		// +(a,t(+(b,C))) => +(+(a,b),t(C))
		hi = _applyRewrite176((Hop) hi);		// +(a,t(+(C,b))) => +(+(a,b),t(C))
		hi = _applyRewrite177((Hop) hi);		// -(b,-(-(D,c),A)) => +(A,-(+(b,c),D))
		hi = _applyRewrite178((Hop) hi);		// -(b,-(D,+(c,A))) => +(A,-(+(b,c),D))
		hi = _applyRewrite179((Hop) hi);		// -(b,-(D,+(A,c))) => +(A,-(+(b,c),D))
		hi = _applyRewrite180((Hop) hi);		// -(+(b,A),-(D,c)) => +(A,-(+(b,c),D))
		hi = _applyRewrite181((Hop) hi);		// -(+(A,b),-(D,c)) => +(A,-(+(b,c),D))
		hi = _applyRewrite182((Hop) hi);		// +(-(A,-(D,b)),c) => +(A,-(+(b,c),D))
		hi = _applyRewrite183((Hop) hi);		// +(b,-(A,-(D,c))) => +(A,-(+(b,c),D))
		hi = _applyRewrite184((Hop) hi);		// +(-(+(b,A),D),c) => +(A,-(+(b,c),D))
		hi = _applyRewrite185((Hop) hi);		// +(-(+(A,b),D),c) => +(A,-(+(b,c),D))
		hi = _applyRewrite186((Hop) hi);		// +(b,-(+(c,A),D)) => +(A,-(+(b,c),D))
		hi = _applyRewrite187((Hop) hi);		// +(b,-(+(A,c),D)) => +(A,-(+(b,c),D))
		hi = _applyRewrite188((Hop) hi);		// +(t(-(a,C)),b) => -(+(a,b),t(C))
		hi = _applyRewrite189((Hop) hi);		// +(a,t(-(b,C))) => -(+(a,b),t(C))
		hi = _applyRewrite190((Hop) hi);		// -(t(+(a,C)),b) => +(-(a,b),t(C))
		hi = _applyRewrite191((Hop) hi);		// -(t(+(C,a)),b) => +(-(a,b),t(C))
		hi = _applyRewrite192((Hop) hi);		// +(t(-(C,b)),a) => +(-(a,b),t(C))
		hi = _applyRewrite193((Hop) hi);		// +(a,t(-(C,b))) => +(-(a,b),t(C))
		hi = _applyRewrite194((Hop) hi);		// -(a,-(-(b,C),D)) => +(-(a,b),+(C,D))
		hi = _applyRewrite195((Hop) hi);		// -(+(a,D),-(b,C)) => +(-(a,b),+(C,D))
		hi = _applyRewrite196((Hop) hi);		// -(+(C,a),-(b,D)) => +(-(a,b),+(C,D))
		hi = _applyRewrite197((Hop) hi);		// +(-(C,-(b,D)),a) => +(-(a,b),+(C,D))
		hi = _applyRewrite198((Hop) hi);		// +(a,-(C,-(b,D))) => +(-(a,b),+(C,D))
		hi = _applyRewrite199((Hop) hi);		// -(-(A,-(D,b)),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite200((Hop) hi);		// -(-(b,D),-(c,A)) => +(A,-(-(b,c),D))
		hi = _applyRewrite201((Hop) hi);		// -(-(A,c),-(D,b)) => +(A,-(-(b,c),D))
		hi = _applyRewrite202((Hop) hi);		// -(b,-(D,-(A,c))) => +(A,-(-(b,c),D))
		hi = _applyRewrite203((Hop) hi);		// -(-(+(b,A),D),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite204((Hop) hi);		// -(-(+(A,b),D),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite205((Hop) hi);		// -(b,-(+(c,D),A)) => +(A,-(-(b,c),D))
		hi = _applyRewrite206((Hop) hi);		// -(b,-(+(D,c),A)) => +(A,-(-(b,c),D))
		hi = _applyRewrite207((Hop) hi);		// -(+(-(b,D),A),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite208((Hop) hi);		// -(+(A,-(b,D)),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite209((Hop) hi);		// -(+(b,A),+(c,D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite210((Hop) hi);		// -(+(b,A),+(D,c)) => +(A,-(-(b,c),D))
		hi = _applyRewrite211((Hop) hi);		// -(+(A,b),+(c,D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite212((Hop) hi);		// -(+(A,b),+(D,c)) => +(A,-(-(b,c),D))
		hi = _applyRewrite213((Hop) hi);		// -(b,+(-(c,A),D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite214((Hop) hi);		// -(b,+(D,-(c,A))) => +(A,-(-(b,c),D))
		hi = _applyRewrite215((Hop) hi);		// +(-(-(A,c),D),b) => +(A,-(-(b,c),D))
		hi = _applyRewrite216((Hop) hi);		// +(-(b,D),-(A,c)) => +(A,-(-(b,c),D))
		hi = _applyRewrite217((Hop) hi);		// +(-(A,c),-(b,D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite218((Hop) hi);		// +(b,-(-(A,c),D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite219((Hop) hi);		// +(-(A,+(c,D)),b) => +(A,-(-(b,c),D))
		hi = _applyRewrite220((Hop) hi);		// +(-(A,+(D,c)),b) => +(A,-(-(b,c),D))
		hi = _applyRewrite221((Hop) hi);		// +(b,-(A,+(c,D))) => +(A,-(-(b,c),D))
		hi = _applyRewrite222((Hop) hi);		// +(b,-(A,+(D,c))) => +(A,-(-(b,c),D))
		hi = _applyRewrite223((Hop) hi);		// colSums(-(t(A),b)) => t(rowSums(-(A,b)))
		hi = _applyRewrite224((Hop) hi);		// colSums(-(a,t(B))) => t(rowSums(-(a,B)))
		hi = _applyRewrite225((Hop) hi);		// rowSums(-(a,t(B))) => t(colSums(-(a,B)))
		hi = _applyRewrite226((Hop) hi);		// rowSums(-(t(A),b)) => t(colSums(-(A,b)))
		hi = _applyRewrite227((Hop) hi);		// rowSums(+(t(A),b)) => t(colSums(+(A,b)))
		hi = _applyRewrite228((Hop) hi);		// rowSums(+(b,t(A))) => t(colSums(+(A,b)))
		hi = _applyRewrite229((Hop) hi);		// colSums(+(t(A),b)) => t(rowSums(+(A,b)))
		hi = _applyRewrite230((Hop) hi);		// colSums(+(b,t(A))) => t(rowSums(+(A,b)))
		hi = _applyRewrite231((Hop) hi);		// *(t(A),t(B)) => t(*(A,B))
		hi = _applyRewrite232((Hop) hi);		// t(*(t(A),B)) => *(A,t(B))
		hi = _applyRewrite233((Hop) hi);		// t(*(B,t(A))) => *(A,t(B))
		hi = _applyRewrite235((Hop) hi);		// t(/(t(A),B)) => /(A,t(B))
		hi = _applyRewrite236((Hop) hi);		// t(/(A,t(B))) => /(t(A),B)
		hi = _applyRewrite237((Hop) hi);		// /(t(A),t(B)) => t(/(A,B))
		hi = _applyRewrite238((Hop) hi);		// colSums(/(a,t(B))) => t(rowSums(/(a,B)))
		hi = _applyRewrite239((Hop) hi);		// rowSums(/(a,t(B))) => t(colSums(/(a,B)))
		hi = _applyRewrite241((Hop) hi);		// %*%(*(a,C),*(b,D)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite242((Hop) hi);		// %*%(*(a,C),*(D,b)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite243((Hop) hi);		// %*%(*(C,a),*(b,D)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite244((Hop) hi);		// %*%(*(C,a),*(D,b)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite245((Hop) hi);		// %*%(/(a,C),*(b,D)) => %*%(/(*(a,b),C),D)
		hi = _applyRewrite246((Hop) hi);		// %*%(/(a,C),*(D,b)) => %*%(/(*(a,b),C),D)
		hi = _applyRewrite247((Hop) hi);		// %*%(*(b,A),/(c,D)) => %*%(A,/(*(b,c),D))
		hi = _applyRewrite248((Hop) hi);		// %*%(*(A,b),/(c,D)) => %*%(A,/(*(b,c),D))
		hi = _applyRewrite249((Hop) hi);		// t(%*%(t(B),A)) => %*%(t(A),B)
		hi = _applyRewrite250((Hop) hi);		// t(%*%(B,t(A))) => %*%(A,t(B))
		hi = _applyRewrite251((Hop) hi);		// %*%(t(B),t(A)) => t(%*%(A,B))
		return hi;
	}

	// Implementation of the rule /(a,1.0) => a
	private static Hop _applyRewrite0(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 1.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(a,1.0) => a");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule *(1.0,a) => a
	private static Hop _applyRewrite1(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(1.0,a) => a");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule *(a,1.0) => a
	private static Hop _applyRewrite2(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 1.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,1.0) => a");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule +(0.0,a) => a
	private static Hop _applyRewrite3(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(0.0,a) => a");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule +(a,0.0) => a
	private static Hop _applyRewrite4(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,0.0) => a");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule +(0.0,A) => A
	private static Hop _applyRewrite5(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(0.0,A) => A");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule +(A,0.0) => A
	private static Hop _applyRewrite6(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,0.0) => A");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule /(0.0,a) => 0.0
	private static Hop _applyRewrite7(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(0.0,a) => 0.0");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule *(0.0,a) => 0.0
	private static Hop _applyRewrite8(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(0.0,a) => 0.0");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule *(a,0.0) => 0.0
	private static Hop _applyRewrite9(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,0.0) => 0.0");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule /(A,c) => *(A,/(1.0,c))
	private static Hop _applyRewrite13(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,c) => *(A,/(1.0,c))");
		LiteralOp l1 = new LiteralOp( 1.0 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);

		return v3;
	}

	// Implementation of the rule rowSums(*(a,B)) => *(a,rowSums(B))
	private static Hop _applyRewrite14(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(*(a,B)) => *(a,rowSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule rowSums(*(B,a)) => *(a,rowSums(B))
	private static Hop _applyRewrite15(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(*(B,a)) => *(a,rowSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule colSums(*(a,B)) => *(a,colSums(B))
	private static Hop _applyRewrite16(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(*(a,B)) => *(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule colSums(*(B,a)) => *(a,colSums(B))
	private static Hop _applyRewrite17(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(*(B,a)) => *(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule *(/(1.0,B),A) => /(A,B)
	private static Hop _applyRewrite18(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(1.0,B),A) => /(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule *(A,/(1.0,B)) => /(A,B)
	private static Hop _applyRewrite19(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR || (l_hi_1_0.getValueType() != Types.ValueType.FP64 && l_hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,/(1.0,B)) => /(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v1;
	}

	// Implementation of the rule *(/(1.0,B),a) => /(a,B)
	private static Hop _applyRewrite20(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(1.0,B),a) => /(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule *(a,/(1.0,B)) => /(a,B)
	private static Hop _applyRewrite21(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR || (l_hi_1_0.getValueType() != Types.ValueType.FP64 && l_hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(1.0,B)) => /(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v1;
	}

	// Implementation of the rule *(/(a,C),b) => /(*(a,b),C)
	private static Hop _applyRewrite22(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(a,C),b) => /(*(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule *(a,/(b,C)) => /(*(a,b),C)
	private static Hop _applyRewrite23(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(b,C)) => /(*(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(0.0,-(B,A)) => -(A,B)
	private static Hop _applyRewrite28(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(0.0,-(B,A)) => -(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v1;
	}

	// Implementation of the rule +(-(0.0,B),A) => -(A,B)
	private static Hop _applyRewrite29(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(0.0,B),A) => -(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule +(A,-(0.0,B)) => -(A,B)
	private static Hop _applyRewrite30(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR || (l_hi_1_0.getValueType() != Types.ValueType.FP64 && l_hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,-(0.0,B)) => -(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v1;
	}

	// Implementation of the rule -(0.0,-(B,a)) => -(a,B)
	private static Hop _applyRewrite31(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(0.0,-(B,a)) => -(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v1;
	}

	// Implementation of the rule +(-(0.0,B),a) => -(a,B)
	private static Hop _applyRewrite32(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(0.0,B),a) => -(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule +(a,-(0.0,B)) => -(a,B)
	private static Hop _applyRewrite33(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR || (l_hi_1_0.getValueType() != Types.ValueType.FP64 && l_hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(0.0,B)) => -(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v1;
	}

	// Implementation of the rule -(0.0,-(b,A)) => -(A,b)
	private static Hop _applyRewrite34(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(0.0,-(b,A)) => -(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v1;
	}

	// Implementation of the rule -(-(A,b),c) => -(A,+(b,c))
	private static Hop _applyRewrite35(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,b),c) => -(A,+(b,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule -(a,+(b,C)) => -(-(a,b),C)
	private static Hop _applyRewrite36(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(b,C)) => -(-(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(a,+(C,b)) => -(-(a,b),C)
	private static Hop _applyRewrite37(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(C,b)) => -(-(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(-(a,C),b) => -(-(a,b),C)
	private static Hop _applyRewrite38(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,C),b) => -(-(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule -(a,-(C,b)) => -(+(a,b),C)
	private static Hop _applyRewrite39(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(C,b)) => -(+(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(a,C),b) => -(+(a,b),C)
	private static Hop _applyRewrite40(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(a,C),b) => -(+(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule +(a,-(b,C)) => -(+(a,b),C)
	private static Hop _applyRewrite41(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(b,C)) => -(+(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(+(b,A),c) => +(A,-(b,c))
	private static Hop _applyRewrite42(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(b,A),c) => +(A,-(b,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule -(+(A,b),c) => +(A,-(b,c))
	private static Hop _applyRewrite43(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,b),c) => +(A,-(b,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule -(b,-(c,A)) => +(A,-(b,c))
	private static Hop _applyRewrite44(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(c,A)) => +(A,-(b,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(A,c),b) => +(A,-(b,c))
	private static Hop _applyRewrite45(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,c),b) => +(A,-(b,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule +(b,-(A,c)) => +(A,-(b,c))
	private static Hop _applyRewrite46(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(A,c)) => +(A,-(b,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule colSums(-(0.0,B)) => -(0.0,colSums(B))
	private static Hop _applyRewrite47(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(-(0.0,B)) => -(0.0,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule rowSums(-(0.0,B)) => -(0.0,rowSums(B))
	private static Hop _applyRewrite48(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(-(0.0,B)) => -(0.0,rowSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule *(/(1.0,b),a) => /(a,b)
	private static Hop _applyRewrite49(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.FP64 && c_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(1.0,b),a) => /(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule *(a,/(1.0,b)) => /(a,b)
	private static Hop _applyRewrite50(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR || (l_hi_1_0.getValueType() != Types.ValueType.FP64 && l_hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(1.0,b)) => /(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v1;
	}

	// Implementation of the rule -(0.0,-(b,a)) => -(a,b)
	private static Hop _applyRewrite51(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(0.0,-(b,a)) => -(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v1;
	}

	// Implementation of the rule -(a,-(b,0.0)) => -(a,b)
	private static Hop _applyRewrite52(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_1_1.getValueType() != Types.ValueType.FP64 && l_hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(b,0.0)) => -(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v1;
	}

	// Implementation of the rule +(-(0.0,b),a) => -(a,b)
	private static Hop _applyRewrite53(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.FP64 && c_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(0.0,b),a) => -(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule +(a,-(0.0,b)) => -(a,b)
	private static Hop _applyRewrite54(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR || (l_hi_1_0.getValueType() != Types.ValueType.FP64 && l_hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(0.0,b)) => -(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v1;
	}

	// Implementation of the rule *(-(a,0.0),b) => *(a,b)
	private static Hop _applyRewrite55(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.FP64 && c_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.FP64 && l_hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(-(a,0.0),b) => *(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule *(a,-(b,0.0)) => *(a,b)
	private static Hop _applyRewrite56(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_1_1.getValueType() != Types.ValueType.FP64 && l_hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,-(b,0.0)) => *(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v1;
	}

	// Implementation of the rule /(-(a,0.0),b) => /(a,b)
	private static Hop _applyRewrite57(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.FP64 && c_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.FP64 && l_hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(-(a,0.0),b) => /(a,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule -(A,-(b,0.0)) => -(A,b)
	private static Hop _applyRewrite58(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_1_1.getValueType() != Types.ValueType.FP64 && l_hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(A,-(b,0.0)) => -(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v1;
	}

	// Implementation of the rule +(-(0.0,b),A) => -(A,b)
	private static Hop _applyRewrite59(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.FP64 && c_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.FP64 && l_hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(0.0,b),A) => -(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule +(A,-(0.0,b)) => -(A,b)
	private static Hop _applyRewrite60(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR || (l_hi_1_0.getValueType() != Types.ValueType.FP64 && l_hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,-(0.0,b)) => -(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v1;
	}

	// Implementation of the rule *(-(b,0.0),A) => *(A,b)
	private static Hop _applyRewrite61(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.FP64 && c_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.FP64 && l_hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(-(b,0.0),A) => *(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule *(A,-(b,0.0)) => *(A,b)
	private static Hop _applyRewrite62(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_1_1.getValueType() != Types.ValueType.FP64 && l_hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,-(b,0.0)) => *(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v1;
	}

	// Implementation of the rule /(-(a,0.0),B) => /(a,B)
	private static Hop _applyRewrite63(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.FP64 && c_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.FP64 && l_hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(-(a,0.0),B) => /(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule t(-(a,t(B))) => -(a,B)
	private static Hop _applyRewrite65(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(-(a,t(B))) => -(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule t(-(t(A),b)) => -(A,b)
	private static Hop _applyRewrite66(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(-(t(A),b)) => -(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule t(+(t(A),b)) => +(A,b)
	private static Hop _applyRewrite67(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(t(A),b)) => +(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule t(+(b,t(A))) => +(A,b)
	private static Hop _applyRewrite68(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(b,t(A))) => +(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule t(*(t(A),b)) => *(A,b)
	private static Hop _applyRewrite69(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(*(t(A),b)) => *(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule t(*(b,t(A))) => *(A,b)
	private static Hop _applyRewrite70(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(*(b,t(A))) => *(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule t(/(a,t(B))) => /(a,B)
	private static Hop _applyRewrite71(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(a,t(B))) => /(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule +(*(C,A),*(B,A)) => *(A,+(B,C))
	private static Hop _applyRewrite72(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_0_1 != hi_1_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(C,A),*(B,A)) => *(A,+(B,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(*(C,A),*(A,B)) => *(A,+(B,C))
	private static Hop _applyRewrite73(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(C,A),*(A,B)) => *(A,+(B,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(*(A,C),*(B,A)) => *(A,+(B,C))
	private static Hop _applyRewrite74(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_0_0 != hi_1_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(A,C),*(B,A)) => *(A,+(B,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(*(A,C),*(A,B)) => *(A,+(B,C))
	private static Hop _applyRewrite75(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(A,C),*(A,B)) => *(A,+(B,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule *(t(*(a,C)),b) => *(*(a,b),t(C))
	private static Hop _applyRewrite76(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(*(a,C)),b) => *(*(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(t(*(C,a)),b) => *(*(a,b),t(C))
	private static Hop _applyRewrite77(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(*(C,a)),b) => *(*(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(a,t(*(b,C))) => *(*(a,b),t(C))
	private static Hop _applyRewrite78(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,t(*(b,C))) => *(*(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(a,t(*(C,b))) => *(*(a,b),t(C))
	private static Hop _applyRewrite79(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,t(*(C,b))) => *(*(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(t(/(a,C)),b) => /(*(a,b),t(C))
	private static Hop _applyRewrite114(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(/(a,C)),b) => /(*(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(a,t(/(b,C))) => /(*(a,b),t(C))
	private static Hop _applyRewrite115(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,t(/(b,C))) => /(*(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule colSums(/(*(a,B),C)) => *(a,colSums(/(B,C)))
	private static Hop _applyRewrite116(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(/(*(a,B),C)) => *(a,colSums(/(B,C)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(/(*(B,a),C)) => *(a,colSums(/(B,C)))
	private static Hop _applyRewrite117(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(/(*(B,a),C)) => *(a,colSums(/(B,C)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(*(/(a,C),B)) => *(a,colSums(/(B,C)))
	private static Hop _applyRewrite118(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(*(/(a,C),B)) => *(a,colSums(/(B,C)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(*(B,/(a,C))) => *(a,colSums(/(B,C)))
	private static Hop _applyRewrite119(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.DIV || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(*(B,/(a,C))) => *(a,colSums(/(B,C)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule rowSums(*(/(a,C),B)) => *(a,rowSums(/(B,C)))
	private static Hop _applyRewrite120(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(*(/(a,C),B)) => *(a,rowSums(/(B,C)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule rowSums(*(B,/(a,C))) => *(a,rowSums(/(B,C)))
	private static Hop _applyRewrite121(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.DIV || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(*(B,/(a,C))) => *(a,rowSums(/(B,C)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule rowSums(/(*(a,B),C)) => *(a,rowSums(/(B,C)))
	private static Hop _applyRewrite122(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(/(*(a,B),C)) => *(a,rowSums(/(B,C)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule rowSums(/(*(B,a),C)) => *(a,rowSums(/(B,C)))
	private static Hop _applyRewrite123(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(/(*(B,a),C)) => *(a,rowSums(/(B,C)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(/(*(b,A),D),c) => *(A,/(*(b,c),D))
	private static Hop _applyRewrite128(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(*(b,A),D),c) => *(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(/(*(A,b),D),c) => *(A,/(*(b,c),D))
	private static Hop _applyRewrite129(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(*(A,b),D),c) => *(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(b,/(*(c,A),D)) => *(A,/(*(b,c),D))
	private static Hop _applyRewrite130(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(b,/(*(c,A),D)) => *(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(b,/(*(A,c),D)) => *(A,/(*(b,c),D))
	private static Hop _applyRewrite131(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(b,/(*(A,c),D)) => *(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(/(/(a,C),D),b) => /(/(*(a,b),C),D)
	private static Hop _applyRewrite132(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(/(a,C),D),b) => /(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(/(a,C),/(b,D)) => /(/(*(a,b),C),D)
	private static Hop _applyRewrite133(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(a,C),/(b,D)) => /(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule *(a,/(/(b,C),D)) => /(/(*(a,b),C),D)
	private static Hop _applyRewrite134(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(/(b,C),D)) => /(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule t(-(t(A),B)) => -(A,t(B))
	private static Hop _applyRewrite145(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(-(t(A),B)) => -(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(-(A,t(B))) => -(t(A),B)
	private static Hop _applyRewrite146(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(-(A,t(B))) => -(t(A),B)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule -(t(A),t(B)) => t(-(A,B))
	private static Hop _applyRewrite147(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(A),t(B)) => t(-(A,B))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(t(B),t(A)) => t(+(A,B))
	private static Hop _applyRewrite148(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(B),t(A)) => t(+(A,B))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_0, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule t(+(t(A),B)) => +(A,t(B))
	private static Hop _applyRewrite149(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(t(A),B)) => +(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(+(B,t(A))) => +(A,t(B))
	private static Hop _applyRewrite150(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(B,t(A))) => +(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule -(t(-(a,C)),b) => -(-(a,b),t(C))
	private static Hop _applyRewrite151(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(-(a,C)),b) => -(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(t(-(A,b)),c) => -(t(A),+(b,c))
	private static Hop _applyRewrite152(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(-(A,b)),c) => -(t(A),+(b,c))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(-(-(a,D),C),b) => -(-(a,b),+(C,D))
	private static Hop _applyRewrite153(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(-(a,D),C),b) => -(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(-(a,C),+(b,D)) => -(-(a,b),+(C,D))
	private static Hop _applyRewrite154(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,C),+(b,D)) => -(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(a,C),+(D,b)) => -(-(a,b),+(C,D))
	private static Hop _applyRewrite155(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,C),+(D,b)) => -(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(-(A,c),B),d) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite156(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(-(A,c),B),d) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(-(A,+(c,B)),d) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite157(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,+(c,B)),d) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(A,+(B,c)),d) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite158(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,+(B,c)),d) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(A,c),+(d,B)) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite159(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,c),+(d,B)) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(A,c),+(B,d)) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite160(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,c),+(B,d)) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(a,C),-(D,b)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite161(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,C),-(D,b)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(a,-(D,-(b,C))) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite162(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(D,-(b,C))) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(a,+(-(C,b),D)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite163(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(-(C,b),D)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(a,+(D,-(C,b))) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite164(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(D,-(C,b))) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(-(-(a,D),C),b) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite165(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(-(a,D),C),b) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(-(a,D),-(b,C)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite166(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(a,D),-(b,C)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(a,-(-(b,C),D)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite167(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(-(b,C),D)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(-(A,-(c,B)),d) => +(A,-(B,+(c,d)))
	private static Hop _applyRewrite168(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,-(c,B)),d) => +(A,-(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(B,c),-(d,A)) => +(A,-(B,+(c,d)))
	private static Hop _applyRewrite169(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(B,c),-(d,A)) => +(A,-(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(-(A,c),B),d) => +(A,-(B,+(c,d)))
	private static Hop _applyRewrite170(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(-(A,c),B),d) => +(A,-(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(+(B,-(A,c)),d) => +(A,-(B,+(c,d)))
	private static Hop _applyRewrite171(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(B,-(A,c)),d) => +(A,-(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(-(B,c),-(A,d)) => +(A,-(B,+(c,d)))
	private static Hop _applyRewrite172(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(B,c),-(A,d)) => +(A,-(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(t(+(a,C)),b) => +(+(a,b),t(C))
	private static Hop _applyRewrite173(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(+(a,C)),b) => +(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(t(+(C,a)),b) => +(+(a,b),t(C))
	private static Hop _applyRewrite174(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(+(C,a)),b) => +(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(a,t(+(b,C))) => +(+(a,b),t(C))
	private static Hop _applyRewrite175(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,t(+(b,C))) => +(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(a,t(+(C,b))) => +(+(a,b),t(C))
	private static Hop _applyRewrite176(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,t(+(C,b))) => +(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(b,-(-(D,c),A)) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite177(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(-(D,c),A)) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(b,-(D,+(c,A))) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite178(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(D,+(c,A))) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(b,-(D,+(A,c))) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite179(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(D,+(A,c))) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(+(b,A),-(D,c)) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite180(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(b,A),-(D,c)) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(A,b),-(D,c)) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite181(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,b),-(D,c)) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(-(A,-(D,b)),c) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite182(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,-(D,b)),c) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(b,-(A,-(D,c))) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite183(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(A,-(D,c))) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(-(+(b,A),D),c) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite184(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(+(b,A),D),c) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(-(+(A,b),D),c) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite185(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(+(A,b),D),c) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(b,-(+(c,A),D)) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite186(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(+(c,A),D)) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(b,-(+(A,c),D)) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite187(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(+(A,c),D)) => +(A,-(+(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(t(-(a,C)),b) => -(+(a,b),t(C))
	private static Hop _applyRewrite188(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(-(a,C)),b) => -(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(a,t(-(b,C))) => -(+(a,b),t(C))
	private static Hop _applyRewrite189(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,t(-(b,C))) => -(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(t(+(a,C)),b) => +(-(a,b),t(C))
	private static Hop _applyRewrite190(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(+(a,C)),b) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(t(+(C,a)),b) => +(-(a,b),t(C))
	private static Hop _applyRewrite191(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(+(C,a)),b) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(t(-(C,b)),a) => +(-(a,b),t(C))
	private static Hop _applyRewrite192(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(-(C,b)),a) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(a,t(-(C,b))) => +(-(a,b),t(C))
	private static Hop _applyRewrite193(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,t(-(C,b))) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(a,-(-(b,C),D)) => +(-(a,b),+(C,D))
	private static Hop _applyRewrite194(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(-(b,C),D)) => +(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(+(a,D),-(b,C)) => +(-(a,b),+(C,D))
	private static Hop _applyRewrite195(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(a,D),-(b,C)) => +(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(C,a),-(b,D)) => +(-(a,b),+(C,D))
	private static Hop _applyRewrite196(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(C,a),-(b,D)) => +(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(-(C,-(b,D)),a) => +(-(a,b),+(C,D))
	private static Hop _applyRewrite197(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(C,-(b,D)),a) => +(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(a,-(C,-(b,D))) => +(-(a,b),+(C,D))
	private static Hop _applyRewrite198(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(C,-(b,D))) => +(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, hi_1_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(-(A,-(D,b)),c) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite199(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,-(D,b)),c) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(b,D),-(c,A)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite200(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(b,D),-(c,A)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(A,c),-(D,b)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite201(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,c),-(D,b)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(b,-(D,-(A,c))) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite202(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(D,-(A,c))) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(-(+(b,A),D),c) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite203(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(+(b,A),D),c) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(-(+(A,b),D),c) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite204(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(+(A,b),D),c) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(b,-(+(c,D),A)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite205(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(+(c,D),A)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(b,-(+(D,c),A)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite206(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(+(D,c),A)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(+(-(b,D),A),c) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite207(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(-(b,D),A),c) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(+(A,-(b,D)),c) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite208(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,-(b,D)),c) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(+(b,A),+(c,D)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite209(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(b,A),+(c,D)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(b,A),+(D,c)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite210(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(b,A),+(D,c)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(A,b),+(c,D)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite211(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,b),+(c,D)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(A,b),+(D,c)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite212(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,b),+(D,c)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(b,+(-(c,A),D)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite213(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,+(-(c,A),D)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(b,+(D,-(c,A))) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite214(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,+(D,-(c,A))) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(-(-(A,c),D),b) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite215(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(-(A,c),D),b) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(-(b,D),-(A,c)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite216(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(b,D),-(A,c)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(-(A,c),-(b,D)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite217(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,c),-(b,D)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(b,-(-(A,c),D)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite218(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(-(A,c),D)) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(-(A,+(c,D)),b) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite219(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,+(c,D)),b) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(-(A,+(D,c)),b) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite220(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,+(D,c)),b) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(b,-(A,+(c,D))) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite221(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(A,+(c,D))) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(b,-(A,+(D,c))) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite222(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(A,+(D,c))) => +(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule colSums(-(t(A),b)) => t(rowSums(-(A,b)))
	private static Hop _applyRewrite223(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(-(t(A),b)) => t(rowSums(-(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MINUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(-(a,t(B))) => t(rowSums(-(a,B)))
	private static Hop _applyRewrite224(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(-(a,t(B))) => t(rowSums(-(a,B)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule rowSums(-(a,t(B))) => t(colSums(-(a,B)))
	private static Hop _applyRewrite225(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(-(a,t(B))) => t(colSums(-(a,B)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule rowSums(-(t(A),b)) => t(colSums(-(A,b)))
	private static Hop _applyRewrite226(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(-(t(A),b)) => t(colSums(-(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MINUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule rowSums(+(t(A),b)) => t(colSums(+(A,b)))
	private static Hop _applyRewrite227(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(+(t(A),b)) => t(colSums(+(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule rowSums(+(b,t(A))) => t(colSums(+(A,b)))
	private static Hop _applyRewrite228(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(+(b,t(A))) => t(colSums(+(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.PLUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule colSums(+(t(A),b)) => t(rowSums(+(A,b)))
	private static Hop _applyRewrite229(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(+(t(A),b)) => t(rowSums(+(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(+(b,t(A))) => t(rowSums(+(A,b)))
	private static Hop _applyRewrite230(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(+(b,t(A))) => t(rowSums(+(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.PLUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule *(t(A),t(B)) => t(*(A,B))
	private static Hop _applyRewrite231(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(A),t(B)) => t(*(A,B))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule t(*(t(A),B)) => *(A,t(B))
	private static Hop _applyRewrite232(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(*(t(A),B)) => *(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(*(B,t(A))) => *(A,t(B))
	private static Hop _applyRewrite233(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(*(B,t(A))) => *(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule t(/(t(A),B)) => /(A,t(B))
	private static Hop _applyRewrite235(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(t(A),B)) => /(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(/(A,t(B))) => /(t(A),B)
	private static Hop _applyRewrite236(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(A,t(B))) => /(t(A),B)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule /(t(A),t(B)) => t(/(A,B))
	private static Hop _applyRewrite237(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(t(A),t(B)) => t(/(A,B))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule colSums(/(a,t(B))) => t(rowSums(/(a,B)))
	private static Hop _applyRewrite238(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(/(a,t(B))) => t(rowSums(/(a,B)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule rowSums(/(a,t(B))) => t(colSums(/(a,B)))
	private static Hop _applyRewrite239(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(/(a,t(B))) => t(colSums(/(a,B)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule %*%(*(a,C),*(b,D)) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite241(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(a,C),*(b,D)) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_1, hi_1_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(a,C),*(D,b)) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite242(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(a,C),*(D,b)) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_1, hi_1_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(C,a),*(b,D)) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite243(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(C,a),*(b,D)) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_0, hi_1_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(C,a),*(D,b)) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite244(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(C,a),*(D,b)) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_0, hi_1_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(/(a,C),*(b,D)) => %*%(/(*(a,b),C),D)
	private static Hop _applyRewrite245(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(/(a,C),*(b,D)) => %*%(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(v2, hi_1_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(/(a,C),*(D,b)) => %*%(/(*(a,b),C),D)
	private static Hop _applyRewrite246(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(/(a,C),*(D,b)) => %*%(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(v2, hi_1_0);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(b,A),/(c,D)) => %*%(A,/(*(b,c),D))
	private static Hop _applyRewrite247(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(b,A),/(c,D)) => %*%(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_1, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(A,b),/(c,D)) => %*%(A,/(*(b,c),D))
	private static Hop _applyRewrite248(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(A,b),/(c,D)) => %*%(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule t(%*%(t(B),A)) => %*%(t(A),B)
	private static Hop _applyRewrite249(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(%*%(t(B),A)) => %*%(t(A),B)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_0_0);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(%*%(B,t(A))) => %*%(A,t(B))
	private static Hop _applyRewrite250(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(%*%(B,t(A))) => %*%(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_1_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule %*%(t(B),t(A)) => t(%*%(A,B))
	private static Hop _applyRewrite251(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(t(B),t(A)) => t(%*%(A,B))");
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_1_0, hi_0_0);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}
}
