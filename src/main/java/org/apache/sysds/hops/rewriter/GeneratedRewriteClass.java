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

		hi = _applyRewrite0((Hop) hi);		// *(1.0,a) => a
		hi = _applyRewrite1((Hop) hi);		// *(a,1.0) => a
		hi = _applyRewrite2((Hop) hi);		// /(a,1.0) => a
		hi = _applyRewrite3((Hop) hi);		// +(0.0,a) => a
		hi = _applyRewrite4((Hop) hi);		// +(a,0.0) => a
		hi = _applyRewrite5((Hop) hi);		// +(0.0,A) => A
		hi = _applyRewrite6((Hop) hi);		// +(A,0.0) => A
		hi = _applyRewrite7((Hop) hi);		// *(0.0,a) => 0.0
		hi = _applyRewrite8((Hop) hi);		// *(a,0.0) => 0.0
		hi = _applyRewrite9((Hop) hi);		// /(0.0,a) => 0.0
		//hi = _applyRewrite13((Hop) hi);		// /(A,c) => *(A,/(1.0,c))
		hi = _applyRewrite21((Hop) hi);		// colSums(*(a,B)) => *(a,colSums(B))
		hi = _applyRewrite22((Hop) hi);		// colSums(*(B,a)) => *(a,colSums(B))
		hi = _applyRewrite23((Hop) hi);		// rowSums(*(a,B)) => *(a,rowSums(B))
		hi = _applyRewrite24((Hop) hi);		// rowSums(*(B,a)) => *(a,rowSums(B))
		hi = _applyRewrite32((Hop) hi);		// *(/(1.0,B),a) => /(a,B)
		hi = _applyRewrite33((Hop) hi);		// *(a,/(1.0,B)) => /(a,B)
		hi = _applyRewrite34((Hop) hi);		// *(/(1.0,B),A) => /(A,B)
		hi = _applyRewrite35((Hop) hi);		// *(A,/(1.0,B)) => /(A,B)
		hi = _applyRewrite36((Hop) hi);		// *(/(a,C),b) => /(*(a,b),C)
		hi = _applyRewrite37((Hop) hi);		// *(a,/(b,C)) => /(*(a,b),C)
		hi = _applyRewrite42((Hop) hi);		// -(0.0,-(B,a)) => -(a,B)
		hi = _applyRewrite43((Hop) hi);		// +(-(0.0,B),a) => -(a,B)
		hi = _applyRewrite44((Hop) hi);		// +(a,-(0.0,B)) => -(a,B)
		hi = _applyRewrite45((Hop) hi);		// -(0.0,-(b,A)) => -(A,b)

		hi = _applyRewrite46((Hop) hi);		// -(0.0,-(B,A)) => -(A,B)
		hi = _applyRewrite47((Hop) hi);		// +(-(0.0,B),A) => -(A,B)
		hi = _applyRewrite48((Hop) hi);		// +(A,-(0.0,B)) => -(A,B)
		hi = _applyRewrite49((Hop) hi);		// -(-(A,b),c) => -(A,+(b,c))
		hi = _applyRewrite50((Hop) hi);		// -(a,+(b,C)) => -(-(a,b),C)
		hi = _applyRewrite51((Hop) hi);		// -(a,+(C,b)) => -(-(a,b),C)
		hi = _applyRewrite52((Hop) hi);		// -(-(a,C),b) => -(-(a,b),C)
		hi = _applyRewrite53((Hop) hi);		// -(a,-(C,b)) => -(+(a,b),C)
		hi = _applyRewrite54((Hop) hi);		// +(-(a,C),b) => -(+(a,b),C)
		hi = _applyRewrite55((Hop) hi);		// +(a,-(b,C)) => -(+(a,b),C)
		hi = _applyRewrite56((Hop) hi);		// -(+(b,A),c) => +(A,-(b,c))
		hi = _applyRewrite57((Hop) hi);		// -(+(A,b),c) => +(A,-(b,c))
		hi = _applyRewrite58((Hop) hi);		// -(b,-(c,A)) => +(A,-(b,c))
		hi = _applyRewrite59((Hop) hi);		// +(-(A,c),b) => +(A,-(b,c))
		hi = _applyRewrite60((Hop) hi);		// +(b,-(A,c)) => +(A,-(b,c))
		hi = _applyRewrite61((Hop) hi);		// colSums(-(0.0,B)) => -(0.0,colSums(B))
		hi = _applyRewrite62((Hop) hi);		// rowSums(-(0.0,B)) => -(0.0,rowSums(B))
		hi = _applyRewrite76((Hop) hi);		// rev(colSums(A)) => colSums(A)
		hi = _applyRewrite77((Hop) hi);		// *(/(1.0,b),a) => /(a,b)
		hi = _applyRewrite78((Hop) hi);		// *(a,/(1.0,b)) => /(a,b)
		hi = _applyRewrite79((Hop) hi);		// -(0.0,-(b,a)) => -(a,b)
		hi = _applyRewrite80((Hop) hi);		// -(a,-(b,0.0)) => -(a,b)
		hi = _applyRewrite81((Hop) hi);		// +(-(0.0,b),a) => -(a,b)
		hi = _applyRewrite82((Hop) hi);		// +(a,-(0.0,b)) => -(a,b)
		hi = _applyRewrite83((Hop) hi);		// *(-(a,0.0),b) => *(a,b)
		hi = _applyRewrite84((Hop) hi);		// *(a,-(b,0.0)) => *(a,b)
		hi = _applyRewrite85((Hop) hi);		// /(-(a,0.0),b) => /(a,b)
		hi = _applyRewrite88((Hop) hi);		// -(A,-(b,0.0)) => -(A,b)
		hi = _applyRewrite89((Hop) hi);		// +(-(0.0,b),A) => -(A,b)
		hi = _applyRewrite90((Hop) hi);		// +(A,-(0.0,b)) => -(A,b)
		hi = _applyRewrite91((Hop) hi);		// *(-(b,0.0),A) => *(A,b)
		hi = _applyRewrite92((Hop) hi);		// *(A,-(b,0.0)) => *(A,b)
		hi = _applyRewrite93((Hop) hi);		// /(-(a,0.0),B) => /(a,B)
		//hi = _applyRewrite94((Hop) hi);		// +(%*%(B,C),%*%(A,C)) => %*%(+(A,B),C)
		//hi = _applyRewrite95((Hop) hi);		// +(%*%(A,C),%*%(A,B)) => %*%(A,+(B,C))
		hi = _applyRewrite98((Hop) hi);		// rev(-(a,rev(B))) => -(a,B)
		hi = _applyRewrite99((Hop) hi);		// t(-(a,t(B))) => -(a,B)
		hi = _applyRewrite100((Hop) hi);		// rev(-(rev(A),b)) => -(A,b)
		hi = _applyRewrite101((Hop) hi);		// t(-(t(A),b)) => -(A,b)
		hi = _applyRewrite102((Hop) hi);		// rev(!=(rev(A),b)) => !=(A,b)
		hi = _applyRewrite103((Hop) hi);		// rev(!=(b,rev(A))) => !=(A,b)
		hi = _applyRewrite104((Hop) hi);		// t(!=(t(A),b)) => !=(A,b)
		hi = _applyRewrite105((Hop) hi);		// t(!=(b,t(A))) => !=(A,b)
		hi = _applyRewrite106((Hop) hi);		// rev(+(rev(A),b)) => +(A,b)
		hi = _applyRewrite107((Hop) hi);		// rev(+(b,rev(A))) => +(A,b)
		hi = _applyRewrite108((Hop) hi);		// t(+(t(A),b)) => +(A,b)
		hi = _applyRewrite109((Hop) hi);		// t(+(b,t(A))) => +(A,b)
		hi = _applyRewrite110((Hop) hi);		// rev(*(rev(A),b)) => *(A,b)
		hi = _applyRewrite111((Hop) hi);		// rev(*(b,rev(A))) => *(A,b)
		hi = _applyRewrite112((Hop) hi);		// t(*(t(A),b)) => *(A,b)
		hi = _applyRewrite113((Hop) hi);		// t(*(b,t(A))) => *(A,b)
		hi = _applyRewrite114((Hop) hi);		// rowSums(rev(*(a,B))) => *(a,rowSums(rev(B)))
		hi = _applyRewrite115((Hop) hi);		// rowSums(rev(*(B,a))) => *(a,rowSums(rev(B)))
		hi = _applyRewrite116((Hop) hi);		// colSums(rev(*(a,B))) => *(a,colSums(rev(B)))
		hi = _applyRewrite117((Hop) hi);		// colSums(rev(*(B,a))) => *(a,colSums(rev(B)))
		hi = _applyRewrite118((Hop) hi);		// rev(/(a,rev(B))) => /(a,B)
		hi = _applyRewrite119((Hop) hi);		// t(/(a,t(B))) => /(a,B)
		hi = _applyRewrite124((Hop) hi);		// +(*(C,A),*(B,A)) => *(A,+(B,C))
		hi = _applyRewrite125((Hop) hi);		// +(*(B,A),*(A,C)) => *(A,+(B,C))
		hi = _applyRewrite126((Hop) hi);		// +(*(A,C),*(B,A)) => *(A,+(B,C))
		hi = _applyRewrite127((Hop) hi);		// +(*(A,C),*(A,B)) => *(A,+(B,C))
		hi = _applyRewrite128((Hop) hi);		// *(t(*(a,C)),b) => *(*(a,b),t(C))
		hi = _applyRewrite129((Hop) hi);		// *(t(*(C,a)),b) => *(*(a,b),t(C))
		hi = _applyRewrite130((Hop) hi);		// *(a,t(*(b,C))) => *(*(a,b),t(C))
		hi = _applyRewrite131((Hop) hi);		// *(a,t(*(C,b))) => *(*(a,b),t(C))
		hi = _applyRewrite132((Hop) hi);		// *(rev(*(a,C)),b) => *(*(a,b),rev(C))
		hi = _applyRewrite133((Hop) hi);		// *(rev(*(C,a)),b) => *(*(a,b),rev(C))
		hi = _applyRewrite134((Hop) hi);		// *(a,rev(*(b,C))) => *(*(a,b),rev(C))
		hi = _applyRewrite135((Hop) hi);		// *(a,rev(*(C,b))) => *(*(a,b),rev(C))
		hi = _applyRewrite152((Hop) hi);		// *(t(/(a,C)),b) => /(*(a,b),t(C))
		hi = _applyRewrite153((Hop) hi);		// *(a,t(/(b,C))) => /(*(a,b),t(C))
		hi = _applyRewrite154((Hop) hi);		// *(rev(/(a,C)),b) => /(*(a,b),rev(C))
		hi = _applyRewrite155((Hop) hi);		// *(a,rev(/(b,C))) => /(*(a,b),rev(C))
		hi = _applyRewrite156((Hop) hi);		// %*%(colSums(B),*(a,C)) => *(a,%*%(colSums(B),C))
		hi = _applyRewrite157((Hop) hi);		// %*%(colSums(B),*(C,a)) => *(a,%*%(colSums(B),C))
		hi = _applyRewrite158((Hop) hi);		// %*%(*(a,B),rowSums(C)) => *(a,%*%(B,rowSums(C)))
		hi = _applyRewrite159((Hop) hi);		// %*%(*(B,a),rowSums(C)) => *(a,%*%(B,rowSums(C)))
		hi = _applyRewrite160((Hop) hi);		// colSums(/(*(a,B),C)) => *(a,colSums(/(B,C)))
		hi = _applyRewrite161((Hop) hi);		// colSums(/(*(B,a),C)) => *(a,colSums(/(B,C)))
		hi = _applyRewrite162((Hop) hi);		// colSums(*(/(a,C),B)) => *(a,colSums(/(B,C)))
		hi = _applyRewrite163((Hop) hi);		// colSums(*(B,/(a,C))) => *(a,colSums(/(B,C)))
		hi = _applyRewrite164((Hop) hi);		// rowSums(*(/(a,C),B)) => *(a,rowSums(/(B,C)))
		hi = _applyRewrite165((Hop) hi);		// rowSums(*(B,/(a,C))) => *(a,rowSums(/(B,C)))
		hi = _applyRewrite166((Hop) hi);		// rowSums(/(*(a,B),C)) => *(a,rowSums(/(B,C)))
		hi = _applyRewrite167((Hop) hi);		// rowSums(/(*(B,a),C)) => *(a,rowSums(/(B,C)))
		hi = _applyRewrite170((Hop) hi);		// *(/(*(a,C),D),b) => *(*(a,b),/(C,D))
		hi = _applyRewrite171((Hop) hi);		// *(/(*(C,a),D),b) => *(*(a,b),/(C,D))
		hi = _applyRewrite172((Hop) hi);		// *(a,/(*(b,C),D)) => *(*(a,b),/(C,D))
		hi = _applyRewrite173((Hop) hi);		// *(a,/(*(C,b),D)) => *(*(a,b),/(C,D))
		hi = _applyRewrite174((Hop) hi);		// *(/(/(a,C),D),b) => /(/(*(a,b),C),D)
		hi = _applyRewrite175((Hop) hi);		// *(/(a,C),/(b,D)) => /(/(*(a,b),C),D)
		hi = _applyRewrite176((Hop) hi);		// *(a,/(/(b,C),D)) => /(/(*(a,b),C),D)
		hi = _applyRewrite185((Hop) hi);		// !=(t(A),t(B)) => t(!=(A,B))
		hi = _applyRewrite186((Hop) hi);		// !=(rev(A),rev(A)) => rev(!=(A,A))
		hi = _applyRewrite187((Hop) hi);		// rev(-(rev(A),B)) => -(A,rev(B))
		hi = _applyRewrite188((Hop) hi);		// rev(-(A,rev(B))) => -(rev(A),B)
		hi = _applyRewrite189((Hop) hi);		// t(-(t(A),B)) => -(A,t(B))
		hi = _applyRewrite190((Hop) hi);		// t(-(A,t(B))) => -(t(A),B)
		hi = _applyRewrite191((Hop) hi);		// -(t(A),t(B)) => t(-(A,B))
		hi = _applyRewrite192((Hop) hi);		// +(t(B),t(A)) => t(+(A,B))
		hi = _applyRewrite193((Hop) hi);		// !=(rev(-(b,A)),A) => !=(A,-(b,A))
		hi = _applyRewrite194((Hop) hi);		// !=(A,rev(-(b,A))) => !=(A,-(b,A))
		hi = _applyRewrite195((Hop) hi);		// !=(-(b,rev(A)),A) => !=(A,-(b,A))
		hi = _applyRewrite196((Hop) hi);		// !=(-(b,A),rev(A)) => !=(A,-(b,A))
		hi = _applyRewrite197((Hop) hi);		// !=(A,-(b,rev(A))) => !=(A,-(b,A))
		hi = _applyRewrite198((Hop) hi);		// !=(rev(-(A,c)),A) => !=(A,-(A,c))
		hi = _applyRewrite199((Hop) hi);		// !=(A,rev(-(A,c))) => !=(A,-(A,c))
		hi = _applyRewrite200((Hop) hi);		// !=(-(rev(A),c),A) => !=(A,-(A,c))
		hi = _applyRewrite201((Hop) hi);		// !=(A,-(rev(A),c)) => !=(A,-(A,c))
		hi = _applyRewrite202((Hop) hi);		// !=(-(B,rev(A)),A) => !=(A,-(B,A))
		hi = _applyRewrite203((Hop) hi);		// !=(-(B,A),rev(A)) => !=(A,-(B,A))
		hi = _applyRewrite204((Hop) hi);		// !=(A,-(B,rev(A))) => !=(A,-(B,A))
		hi = _applyRewrite205((Hop) hi);		// !=(-(rev(A),C),A) => !=(A,-(A,C))
		hi = _applyRewrite206((Hop) hi);		// !=(-(A,C),rev(A)) => !=(A,-(A,C))
		hi = _applyRewrite207((Hop) hi);		// !=(A,-(rev(A),C)) => !=(A,-(A,C))
		hi = _applyRewrite208((Hop) hi);		// rev(!=(rev(A),B)) => !=(A,rev(B))
		hi = _applyRewrite209((Hop) hi);		// rev(!=(B,rev(A))) => !=(A,rev(B))
		hi = _applyRewrite210((Hop) hi);		// t(!=(t(A),B)) => !=(A,t(B))
		hi = _applyRewrite211((Hop) hi);		// t(!=(B,t(A))) => !=(A,t(B))
		hi = _applyRewrite212((Hop) hi);		// !=(rev(+(c,A)),A) => !=(A,+(A,c))
		hi = _applyRewrite213((Hop) hi);		// !=(rev(+(A,c)),A) => !=(A,+(A,c))
		hi = _applyRewrite214((Hop) hi);		// !=(A,rev(+(c,A))) => !=(A,+(A,c))
		hi = _applyRewrite215((Hop) hi);		// !=(A,rev(+(A,c))) => !=(A,+(A,c))
		hi = _applyRewrite216((Hop) hi);		// !=(+(rev(A),c),A) => !=(A,+(A,c))
		hi = _applyRewrite217((Hop) hi);		// !=(+(c,rev(A)),A) => !=(A,+(A,c))
		hi = _applyRewrite218((Hop) hi);		// !=(+(c,A),rev(A)) => !=(A,+(A,c))
		hi = _applyRewrite219((Hop) hi);		// !=(A,+(rev(A),c)) => !=(A,+(A,c))
		hi = _applyRewrite220((Hop) hi);		// !=(A,+(c,rev(A))) => !=(A,+(A,c))
		hi = _applyRewrite221((Hop) hi);		// !=(+(rev(A),C),A) => !=(A,+(A,C))
		hi = _applyRewrite222((Hop) hi);		// !=(+(C,rev(A)),A) => !=(A,+(A,C))
		hi = _applyRewrite223((Hop) hi);		// !=(+(C,A),rev(A)) => !=(A,+(A,C))
		hi = _applyRewrite224((Hop) hi);		// !=(+(A,C),rev(A)) => !=(A,+(A,C))
		hi = _applyRewrite225((Hop) hi);		// !=(A,+(rev(A),C)) => !=(A,+(A,C))
		hi = _applyRewrite226((Hop) hi);		// !=(A,+(C,rev(A))) => !=(A,+(A,C))
		hi = _applyRewrite227((Hop) hi);		// !=(!=(rev(A),c),A) => !=(A,!=(A,c))
		hi = _applyRewrite228((Hop) hi);		// !=(!=(c,rev(A)),A) => !=(A,!=(A,c))
		hi = _applyRewrite229((Hop) hi);		// !=(!=(c,A),rev(A)) => !=(A,!=(A,c))
		hi = _applyRewrite230((Hop) hi);		// !=(A,!=(rev(A),c)) => !=(A,!=(A,c))
		hi = _applyRewrite231((Hop) hi);		// !=(A,!=(c,rev(A))) => !=(A,!=(A,c))
		hi = _applyRewrite232((Hop) hi);		// !=(rev(!=(c,A)),A) => !=(A,!=(A,c))
		hi = _applyRewrite233((Hop) hi);		// !=(rev(!=(A,c)),A) => !=(A,!=(A,c))
		hi = _applyRewrite234((Hop) hi);		// !=(A,rev(!=(c,A))) => !=(A,!=(A,c))
		hi = _applyRewrite235((Hop) hi);		// !=(A,rev(!=(A,c))) => !=(A,!=(A,c))
		hi = _applyRewrite236((Hop) hi);		// !=(!=(rev(A),C),A) => !=(A,!=(A,C))
		hi = _applyRewrite237((Hop) hi);		// !=(!=(C,rev(A)),A) => !=(A,!=(A,C))
		hi = _applyRewrite238((Hop) hi);		// !=(!=(C,A),rev(A)) => !=(A,!=(A,C))
		hi = _applyRewrite239((Hop) hi);		// !=(!=(A,C),rev(A)) => !=(A,!=(A,C))
		hi = _applyRewrite240((Hop) hi);		// !=(A,!=(rev(A),C)) => !=(A,!=(A,C))
		hi = _applyRewrite241((Hop) hi);		// !=(A,!=(C,rev(A))) => !=(A,!=(A,C))
		hi = _applyRewrite242((Hop) hi);		// rev(+(rev(A),B)) => +(A,rev(B))
		hi = _applyRewrite243((Hop) hi);		// rev(+(B,rev(A))) => +(A,rev(B))
		hi = _applyRewrite244((Hop) hi);		// t(+(t(A),B)) => +(A,t(B))
		hi = _applyRewrite245((Hop) hi);		// t(+(B,t(A))) => +(A,t(B))
		hi = _applyRewrite246((Hop) hi);		// +(!=(rev(A),c),A) => +(A,!=(A,c))
		hi = _applyRewrite247((Hop) hi);		// +(!=(c,rev(A)),A) => +(A,!=(A,c))
		hi = _applyRewrite248((Hop) hi);		// +(A,!=(rev(A),c)) => +(A,!=(A,c))
		hi = _applyRewrite249((Hop) hi);		// +(A,!=(c,rev(A))) => +(A,!=(A,c))
		hi = _applyRewrite250((Hop) hi);		// +(rev(!=(c,A)),A) => +(A,!=(A,c))
		hi = _applyRewrite251((Hop) hi);		// +(rev(!=(A,c)),A) => +(A,!=(A,c))
		hi = _applyRewrite252((Hop) hi);		// +(A,rev(!=(c,A))) => +(A,!=(A,c))
		hi = _applyRewrite253((Hop) hi);		// +(A,rev(!=(A,c))) => +(A,!=(A,c))
		hi = _applyRewrite254((Hop) hi);		// +(!=(rev(A),C),A) => +(A,!=(A,C))
		hi = _applyRewrite255((Hop) hi);		// +(!=(C,rev(A)),A) => +(A,!=(A,C))
		hi = _applyRewrite256((Hop) hi);		// +(A,!=(rev(A),C)) => +(A,!=(A,C))
		hi = _applyRewrite257((Hop) hi);		// +(A,!=(C,rev(A))) => +(A,!=(A,C))
		hi = _applyRewrite258((Hop) hi);		// -(rev(!=(A,b)),A) => -(!=(A,b),A)
		hi = _applyRewrite259((Hop) hi);		// -(A,!=(rev(A),c)) => -(A,!=(A,c))
		hi = _applyRewrite260((Hop) hi);		// -(A,!=(c,rev(A))) => -(A,!=(A,c))
		hi = _applyRewrite261((Hop) hi);		// -(A,rev(!=(c,A))) => -(A,!=(A,c))
		hi = _applyRewrite262((Hop) hi);		// -(A,rev(!=(A,c))) => -(A,!=(A,c))
		hi = _applyRewrite263((Hop) hi);		// -(A,!=(rev(A),C)) => -(A,!=(A,C))
		hi = _applyRewrite264((Hop) hi);		// -(A,!=(C,rev(A))) => -(A,!=(A,C))
		hi = _applyRewrite265((Hop) hi);		// -(t(-(A,b)),c) => -(t(A),+(b,c))
		hi = _applyRewrite266((Hop) hi);		// -(t(-(a,C)),b) => -(-(a,b),t(C))
		hi = _applyRewrite267((Hop) hi);		// -(a,t(+(b,C))) => -(-(a,b),t(C))
		hi = _applyRewrite268((Hop) hi);		// -(a,t(+(C,b))) => -(-(a,b),t(C))
		hi = _applyRewrite269((Hop) hi);		// -(rev(-(A,b)),c) => -(rev(A),+(b,c))
		hi = _applyRewrite270((Hop) hi);		// -(rev(-(a,C)),b) => -(-(a,b),rev(C))
		hi = _applyRewrite271((Hop) hi);		// -(a,rev(+(b,C))) => -(-(a,b),rev(C))
		hi = _applyRewrite272((Hop) hi);		// -(a,rev(+(C,b))) => -(-(a,b),rev(C))
		hi = _applyRewrite273((Hop) hi);		// -(-(-(a,D),C),b) => -(-(a,b),+(C,D))
		hi = _applyRewrite274((Hop) hi);		// -(-(a,C),+(b,D)) => -(-(a,b),+(C,D))
		hi = _applyRewrite275((Hop) hi);		// -(-(a,D),+(C,b)) => -(-(a,b),+(C,D))
		hi = _applyRewrite276((Hop) hi);		// -(-(-(A,c),B),d) => -(A,+(B,+(c,d)))
		hi = _applyRewrite277((Hop) hi);		// -(-(A,+(c,B)),d) => -(A,+(B,+(c,d)))
		hi = _applyRewrite278((Hop) hi);		// -(-(A,+(B,c)),d) => -(A,+(B,+(c,d)))
		hi = _applyRewrite279((Hop) hi);		// -(-(A,c),+(d,B)) => -(A,+(B,+(c,d)))
		hi = _applyRewrite280((Hop) hi);		// -(-(A,c),+(B,d)) => -(A,+(B,+(c,d)))
		hi = _applyRewrite281((Hop) hi);		// -(a,rev(-(C,b))) => -(+(a,b),rev(C))
		hi = _applyRewrite282((Hop) hi);		// +(rev(-(a,C)),b) => -(+(a,b),rev(C))
		hi = _applyRewrite283((Hop) hi);		// +(a,rev(-(b,C))) => -(+(a,b),rev(C))
		hi = _applyRewrite284((Hop) hi);		// -(a,rev(-(b,C))) => +(-(a,b),rev(C))
		hi = _applyRewrite285((Hop) hi);		// -(rev(+(a,C)),b) => +(-(a,b),rev(C))
		hi = _applyRewrite286((Hop) hi);		// -(rev(+(C,a)),b) => +(-(a,b),rev(C))
		hi = _applyRewrite287((Hop) hi);		// +(rev(-(C,b)),a) => +(-(a,b),rev(C))
		hi = _applyRewrite288((Hop) hi);		// +(a,rev(-(C,b))) => +(-(a,b),rev(C))
		hi = _applyRewrite289((Hop) hi);		// +(rev(+(a,C)),b) => +(+(a,b),rev(C))
		hi = _applyRewrite290((Hop) hi);		// +(rev(+(C,a)),b) => +(+(a,b),rev(C))
		hi = _applyRewrite291((Hop) hi);		// +(a,rev(+(b,C))) => +(+(a,b),rev(C))
		hi = _applyRewrite292((Hop) hi);		// +(a,rev(+(C,b))) => +(+(a,b),rev(C))
		hi = _applyRewrite293((Hop) hi);		// -(-(a,C),-(D,b)) => -(+(a,b),+(C,D))
		hi = _applyRewrite294((Hop) hi);		// -(a,-(C,-(b,D))) => -(+(a,b),+(C,D))
		hi = _applyRewrite295((Hop) hi);		// -(a,+(-(D,b),C)) => -(+(a,b),+(C,D))
		hi = _applyRewrite296((Hop) hi);		// -(a,+(D,-(C,b))) => -(+(a,b),+(C,D))
		hi = _applyRewrite297((Hop) hi);		// +(-(-(a,C),D),b) => -(+(a,b),+(C,D))
		hi = _applyRewrite298((Hop) hi);		// +(-(a,D),-(b,C)) => -(+(a,b),+(C,D))
		hi = _applyRewrite299((Hop) hi);		// +(a,-(-(b,D),C)) => -(+(a,b),+(C,D))
		hi = _applyRewrite300((Hop) hi);		// -(-(A,-(c,B)),d) => +(A,-(B,+(c,d)))
		hi = _applyRewrite301((Hop) hi);		// -(-(B,c),-(d,A)) => +(A,-(B,+(c,d)))
		hi = _applyRewrite302((Hop) hi);		// -(+(-(B,c),A),d) => +(A,-(B,+(c,d)))
		hi = _applyRewrite303((Hop) hi);		// -(+(A,-(B,c)),d) => +(A,-(B,+(c,d)))
		hi = _applyRewrite304((Hop) hi);		// +(-(B,c),-(A,d)) => +(A,-(B,+(c,d)))
		hi = _applyRewrite305((Hop) hi);		// -(b,-(-(D,c),A)) => +(A,-(+(b,c),D))
		hi = _applyRewrite306((Hop) hi);		// -(b,-(D,+(c,A))) => +(A,-(+(b,c),D))
		hi = _applyRewrite307((Hop) hi);		// -(b,-(D,+(A,c))) => +(A,-(+(b,c),D))
		hi = _applyRewrite308((Hop) hi);		// -(+(b,A),-(D,c)) => +(A,-(+(b,c),D))
		hi = _applyRewrite309((Hop) hi);		// -(+(A,b),-(D,c)) => +(A,-(+(b,c),D))
		hi = _applyRewrite310((Hop) hi);		// +(-(A,-(D,b)),c) => +(A,-(+(b,c),D))
		hi = _applyRewrite311((Hop) hi);		// +(b,-(A,-(D,c))) => +(A,-(+(b,c),D))
		hi = _applyRewrite312((Hop) hi);		// +(-(+(b,A),D),c) => +(A,-(+(b,c),D))
		hi = _applyRewrite313((Hop) hi);		// +(-(+(A,b),D),c) => +(A,-(+(b,c),D))
		hi = _applyRewrite314((Hop) hi);		// +(b,-(+(c,A),D)) => +(A,-(+(b,c),D))
		hi = _applyRewrite315((Hop) hi);		// +(b,-(+(A,c),D)) => +(A,-(+(b,c),D))
		hi = _applyRewrite316((Hop) hi);		// -(c,-(-(d,B),A)) => +(A,+(B,-(c,d)))
		hi = _applyRewrite317((Hop) hi);		// -(+(c,B),-(d,A)) => +(A,+(B,-(c,d)))
		hi = _applyRewrite318((Hop) hi);		// -(+(B,c),-(d,A)) => +(A,+(B,-(c,d)))
		hi = _applyRewrite319((Hop) hi);		// +(-(A,-(d,B)),c) => +(A,+(B,-(c,d)))
		hi = _applyRewrite320((Hop) hi);		// +(c,-(A,-(d,B))) => +(A,+(B,-(c,d)))
		hi = _applyRewrite321((Hop) hi);		// -(-(A,-(D,b)),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite322((Hop) hi);		// -(-(b,D),-(c,A)) => +(A,-(-(b,c),D))
		hi = _applyRewrite323((Hop) hi);		// -(-(A,c),-(D,b)) => +(A,-(-(b,c),D))
		hi = _applyRewrite324((Hop) hi);		// -(b,-(D,-(A,c))) => +(A,-(-(b,c),D))
		hi = _applyRewrite325((Hop) hi);		// -(-(+(b,A),D),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite326((Hop) hi);		// -(-(+(A,b),D),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite327((Hop) hi);		// -(b,-(+(c,D),A)) => +(A,-(-(b,c),D))
		hi = _applyRewrite328((Hop) hi);		// -(b,-(+(D,c),A)) => +(A,-(-(b,c),D))
		hi = _applyRewrite329((Hop) hi);		// -(+(-(b,D),A),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite330((Hop) hi);		// -(+(b,A),+(c,D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite331((Hop) hi);		// -(+(b,A),+(D,c)) => +(A,-(-(b,c),D))
		hi = _applyRewrite332((Hop) hi);		// -(+(A,b),+(c,D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite333((Hop) hi);		// -(+(A,b),+(D,c)) => +(A,-(-(b,c),D))
		hi = _applyRewrite334((Hop) hi);		// -(+(A,-(b,D)),c) => +(A,-(-(b,c),D))
		hi = _applyRewrite335((Hop) hi);		// -(b,+(-(c,A),D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite336((Hop) hi);		// -(b,+(D,-(c,A))) => +(A,-(-(b,c),D))
		hi = _applyRewrite337((Hop) hi);		// +(-(-(A,c),D),b) => +(A,-(-(b,c),D))
		hi = _applyRewrite338((Hop) hi);		// +(-(b,D),-(A,c)) => +(A,-(-(b,c),D))
		hi = _applyRewrite339((Hop) hi);		// +(-(A,c),-(b,D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite340((Hop) hi);		// +(b,-(-(A,c),D)) => +(A,-(-(b,c),D))
		hi = _applyRewrite341((Hop) hi);		// +(-(A,+(c,D)),b) => +(A,-(-(b,c),D))
		hi = _applyRewrite342((Hop) hi);		// +(-(A,+(D,c)),b) => +(A,-(-(b,c),D))
		hi = _applyRewrite343((Hop) hi);		// +(b,-(A,+(c,D))) => +(A,-(-(b,c),D))
		hi = _applyRewrite344((Hop) hi);		// +(b,-(A,+(D,c))) => +(A,-(-(b,c),D))
		hi = _applyRewrite345((Hop) hi);		// -(a,t(-(C,b))) => -(+(a,b),t(C))
		hi = _applyRewrite346((Hop) hi);		// +(t(-(a,C)),b) => -(+(a,b),t(C))
		hi = _applyRewrite347((Hop) hi);		// +(a,t(-(b,C))) => -(+(a,b),t(C))
		hi = _applyRewrite348((Hop) hi);		// -(t(+(a,C)),b) => +(-(a,b),t(C))
		hi = _applyRewrite349((Hop) hi);		// -(t(+(C,a)),b) => +(-(a,b),t(C))
		hi = _applyRewrite350((Hop) hi);		// -(a,t(-(b,C))) => +(-(a,b),t(C))
		hi = _applyRewrite351((Hop) hi);		// +(t(-(C,b)),a) => +(-(a,b),t(C))
		hi = _applyRewrite352((Hop) hi);		// +(a,t(-(C,b))) => +(-(a,b),t(C))
		hi = _applyRewrite353((Hop) hi);		// +(t(+(a,C)),b) => +(+(a,b),t(C))
		hi = _applyRewrite354((Hop) hi);		// +(t(+(C,a)),b) => +(+(a,b),t(C))
		hi = _applyRewrite355((Hop) hi);		// +(a,t(+(b,C))) => +(+(a,b),t(C))
		hi = _applyRewrite356((Hop) hi);		// +(a,t(+(C,b))) => +(+(a,b),t(C))
		hi = _applyRewrite357((Hop) hi);		// colSums(-(t(A),b)) => t(rowSums(-(A,b)))
		hi = _applyRewrite358((Hop) hi);		// colSums(-(a,t(B))) => t(rowSums(-(a,B)))
		hi = _applyRewrite359((Hop) hi);		// rowSums(-(t(A),b)) => t(colSums(-(A,b)))
		hi = _applyRewrite360((Hop) hi);		// rowSums(-(a,t(B))) => t(colSums(-(a,B)))
		hi = _applyRewrite361((Hop) hi);		// colSums(!=(t(A),b)) => t(rowSums(!=(A,b)))
		hi = _applyRewrite362((Hop) hi);		// colSums(!=(b,t(A))) => t(rowSums(!=(A,b)))
		hi = _applyRewrite363((Hop) hi);		// rowSums(!=(t(A),b)) => t(colSums(!=(A,b)))
		hi = _applyRewrite364((Hop) hi);		// rowSums(!=(b,t(A))) => t(colSums(!=(A,b)))
		hi = _applyRewrite365((Hop) hi);		// colSums(+(t(A),b)) => t(rowSums(+(A,b)))
		hi = _applyRewrite366((Hop) hi);		// colSums(+(b,t(A))) => t(rowSums(+(A,b)))
		hi = _applyRewrite367((Hop) hi);		// rowSums(+(t(A),b)) => t(colSums(+(A,b)))
		hi = _applyRewrite368((Hop) hi);		// rowSums(+(b,t(A))) => t(colSums(+(A,b)))
		hi = _applyRewrite372((Hop) hi);		// *(t(A),t(B)) => t(*(A,B))
		hi = _applyRewrite373((Hop) hi);		// !=(*(rev(A),c),A) => !=(A,*(A,c))
		hi = _applyRewrite374((Hop) hi);		// !=(*(c,rev(A)),A) => !=(A,*(A,c))
		hi = _applyRewrite375((Hop) hi);		// !=(*(c,A),rev(A)) => !=(A,*(A,c))
		hi = _applyRewrite376((Hop) hi);		// !=(A,*(rev(A),c)) => !=(A,*(A,c))
		hi = _applyRewrite377((Hop) hi);		// !=(A,*(c,rev(A))) => !=(A,*(A,c))
		hi = _applyRewrite378((Hop) hi);		// !=(rev(*(c,A)),A) => !=(A,*(A,c))
		hi = _applyRewrite379((Hop) hi);		// !=(rev(*(A,c)),A) => !=(A,*(A,c))
		hi = _applyRewrite380((Hop) hi);		// !=(A,rev(*(c,A))) => !=(A,*(A,c))
		hi = _applyRewrite381((Hop) hi);		// !=(A,rev(*(A,c))) => !=(A,*(A,c))
		hi = _applyRewrite382((Hop) hi);		// !=(*(rev(A),C),A) => !=(A,*(A,C))
		hi = _applyRewrite383((Hop) hi);		// !=(*(C,rev(A)),A) => !=(A,*(A,C))
		hi = _applyRewrite384((Hop) hi);		// !=(*(C,A),rev(A)) => !=(A,*(A,C))
		hi = _applyRewrite385((Hop) hi);		// !=(*(A,C),rev(A)) => !=(A,*(A,C))
		hi = _applyRewrite386((Hop) hi);		// !=(A,*(rev(A),C)) => !=(A,*(A,C))
		hi = _applyRewrite387((Hop) hi);		// !=(A,*(C,rev(A))) => !=(A,*(A,C))
		hi = _applyRewrite388((Hop) hi);		// rev(*(rev(A),B)) => *(A,rev(B))
		hi = _applyRewrite389((Hop) hi);		// rev(*(B,rev(A))) => *(A,rev(B))
		hi = _applyRewrite390((Hop) hi);		// t(*(t(A),B)) => *(A,t(B))
		hi = _applyRewrite391((Hop) hi);		// t(*(B,t(A))) => *(A,t(B))
		hi = _applyRewrite392((Hop) hi);		// *(!=(rev(A),c),A) => *(A,!=(A,c))
		hi = _applyRewrite393((Hop) hi);		// *(!=(c,rev(A)),A) => *(A,!=(A,c))
		hi = _applyRewrite394((Hop) hi);		// *(A,!=(rev(A),c)) => *(A,!=(A,c))
		hi = _applyRewrite395((Hop) hi);		// *(A,!=(c,rev(A))) => *(A,!=(A,c))
		hi = _applyRewrite396((Hop) hi);		// *(rev(!=(c,A)),A) => *(A,!=(A,c))
		hi = _applyRewrite397((Hop) hi);		// *(rev(!=(A,c)),A) => *(A,!=(A,c))
		hi = _applyRewrite398((Hop) hi);		// *(A,rev(!=(c,A))) => *(A,!=(A,c))
		hi = _applyRewrite399((Hop) hi);		// *(A,rev(!=(A,c))) => *(A,!=(A,c))
		hi = _applyRewrite400((Hop) hi);		// *(!=(rev(A),C),A) => *(A,!=(A,C))
		hi = _applyRewrite401((Hop) hi);		// *(!=(C,rev(A)),A) => *(A,!=(A,C))
		hi = _applyRewrite402((Hop) hi);		// *(A,!=(rev(A),C)) => *(A,!=(A,C))
		hi = _applyRewrite403((Hop) hi);		// *(A,!=(C,rev(A))) => *(A,!=(A,C))
		hi = _applyRewrite405((Hop) hi);		// rev(/(rev(A),B)) => /(A,rev(B))
		hi = _applyRewrite406((Hop) hi);		// rev(/(A,rev(B))) => /(rev(A),B)
		hi = _applyRewrite407((Hop) hi);		// t(/(t(A),B)) => /(A,t(B))
		hi = _applyRewrite408((Hop) hi);		// t(/(A,t(B))) => /(t(A),B)
		hi = _applyRewrite409((Hop) hi);		// /(t(A),t(B)) => t(/(A,B))
		hi = _applyRewrite410((Hop) hi);		// !=(/(b,rev(A)),A) => !=(A,/(b,A))
		hi = _applyRewrite411((Hop) hi);		// !=(/(b,A),rev(A)) => !=(A,/(b,A))
		hi = _applyRewrite412((Hop) hi);		// !=(A,/(b,rev(A))) => !=(A,/(b,A))
		hi = _applyRewrite413((Hop) hi);		// !=(rev(/(b,A)),A) => !=(A,/(b,A))
		hi = _applyRewrite414((Hop) hi);		// !=(A,rev(/(b,A))) => !=(A,/(b,A))
		hi = _applyRewrite415((Hop) hi);		// !=(/(B,rev(A)),A) => !=(A,/(B,A))
		hi = _applyRewrite416((Hop) hi);		// !=(/(B,A),rev(A)) => !=(A,/(B,A))
		hi = _applyRewrite417((Hop) hi);		// !=(A,/(B,rev(A))) => !=(A,/(B,A))
		hi = _applyRewrite418((Hop) hi);		// !=(/(rev(A),C),A) => !=(A,/(A,C))
		hi = _applyRewrite419((Hop) hi);		// !=(/(A,C),rev(A)) => !=(A,/(A,C))
		hi = _applyRewrite420((Hop) hi);		// !=(A,/(rev(A),C)) => !=(A,/(A,C))
		hi = _applyRewrite421((Hop) hi);		// /(rev(!=(A,b)),A) => /(!=(A,b),A)
		hi = _applyRewrite422((Hop) hi);		// /(A,rev(!=(c,A))) => /(A,!=(A,c))
		hi = _applyRewrite423((Hop) hi);		// /(A,rev(!=(A,c))) => /(A,!=(A,c))
		hi = _applyRewrite424((Hop) hi);		// /(A,!=(rev(A),c)) => /(A,!=(A,c))
		hi = _applyRewrite425((Hop) hi);		// /(A,!=(c,rev(A))) => /(A,!=(A,c))
		hi = _applyRewrite426((Hop) hi);		// /(A,!=(rev(A),C)) => /(A,!=(A,C))
		hi = _applyRewrite427((Hop) hi);		// /(A,!=(C,rev(A))) => /(A,!=(A,C))
		hi = _applyRewrite428((Hop) hi);		// colSums(/(a,t(B))) => t(rowSums(/(a,B)))
		hi = _applyRewrite429((Hop) hi);		// rowSums(/(a,t(B))) => t(colSums(/(a,B)))
		hi = _applyRewrite438((Hop) hi);		// !=(A,rev(rowSums(A))) => !=(A,rowSums(A))
		hi = _applyRewrite439((Hop) hi);		// !=(A,rowSums(rev(A))) => !=(A,rowSums(A))
		hi = _applyRewrite440((Hop) hi);		// !=(A,colSums(rev(A))) => !=(A,colSums(A))
		hi = _applyRewrite441((Hop) hi);		// +(A,colSums(rev(A))) => +(A,colSums(A))
		hi = _applyRewrite442((Hop) hi);		// *(A,colSums(rev(A))) => *(A,colSums(A))
		hi = _applyRewrite443((Hop) hi);		// %*%(*(a,C),*(b,D)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite444((Hop) hi);		// %*%(*(a,C),*(D,b)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite445((Hop) hi);		// %*%(*(C,a),*(b,D)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite446((Hop) hi);		// %*%(*(C,a),*(D,b)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite447((Hop) hi);		// *(%*%(*(a,C),D),b) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite448((Hop) hi);		// *(%*%(*(C,a),D),b) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite449((Hop) hi);		// *(%*%(C,*(D,a)),b) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite450((Hop) hi);		// *(a,%*%(*(b,C),D)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite451((Hop) hi);		// *(a,%*%(*(C,b),D)) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite452((Hop) hi);		// *(a,%*%(C,*(b,D))) => *(*(a,b),%*%(C,D))
		hi = _applyRewrite453((Hop) hi);		// %*%(/(a,C),*(b,D)) => %*%(/(*(a,b),C),D)
		hi = _applyRewrite454((Hop) hi);		// %*%(/(a,C),*(D,b)) => %*%(/(*(a,b),C),D)
		hi = _applyRewrite455((Hop) hi);		// *(%*%(/(a,C),D),b) => %*%(/(*(a,b),C),D)
		hi = _applyRewrite456((Hop) hi);		// *(a,%*%(/(b,C),D)) => %*%(/(*(a,b),C),D)
		hi = _applyRewrite457((Hop) hi);		// %*%(*(b,A),/(c,D)) => %*%(A,/(*(b,c),D))
		hi = _applyRewrite458((Hop) hi);		// %*%(*(A,b),/(c,D)) => %*%(A,/(*(b,c),D))
		hi = _applyRewrite459((Hop) hi);		// *(%*%(A,/(b,D)),c) => %*%(A,/(*(b,c),D))
		hi = _applyRewrite460((Hop) hi);		// *(b,%*%(A,/(c,D))) => %*%(A,/(*(b,c),D))
		hi = _applyRewrite461((Hop) hi);		// t(%*%(t(B),A)) => %*%(t(A),B)
		hi = _applyRewrite462((Hop) hi);		// t(%*%(B,t(A))) => %*%(A,t(B))
		hi = _applyRewrite463((Hop) hi);		// %*%(t(B),t(A)) => t(%*%(A,B))
		hi = _applyRewrite464((Hop) hi);		// !=(%*%(B,rev(A)),A) => !=(A,%*%(B,A))
		hi = _applyRewrite465((Hop) hi);		// !=(A,%*%(B,rev(A))) => !=(A,%*%(B,A))
		hi = _applyRewrite466((Hop) hi);		// !=(rev(%*%(A,C)),A) => !=(A,%*%(A,C))
		hi = _applyRewrite467((Hop) hi);		// !=(A,rev(%*%(A,C))) => !=(A,%*%(A,C))
		hi = _applyRewrite468((Hop) hi);		// !=(%*%(rev(A),C),A) => !=(A,%*%(A,C))
		hi = _applyRewrite469((Hop) hi);		// !=(A,%*%(rev(A),C)) => !=(A,%*%(A,C))
		hi = _applyRewrite470((Hop) hi);		// rev(%*%(!=(b,A),A)) => %*%(!=(A,b),A)
		hi = _applyRewrite471((Hop) hi);		// rev(%*%(!=(A,b),A)) => %*%(!=(A,b),A)
		hi = _applyRewrite472((Hop) hi);		// %*%(!=(rev(A),b),A) => %*%(!=(A,b),A)
		hi = _applyRewrite473((Hop) hi);		// %*%(!=(b,rev(A)),A) => %*%(!=(A,b),A)
		hi = _applyRewrite474((Hop) hi);		// %*%(rev(!=(b,A)),A) => %*%(!=(A,b),A)
		hi = _applyRewrite475((Hop) hi);		// %*%(rev(!=(A,b)),A) => %*%(!=(A,b),A)
		hi = _applyRewrite476((Hop) hi);		// %*%(!=(rev(A),B),A) => %*%(!=(A,B),A)
		hi = _applyRewrite477((Hop) hi);		// %*%(!=(B,rev(A)),A) => %*%(!=(A,B),A)
		hi = _applyRewrite478((Hop) hi);		// %*%(A,!=(rev(A),c)) => %*%(A,!=(A,c))
		hi = _applyRewrite479((Hop) hi);		// %*%(A,!=(c,rev(A))) => %*%(A,!=(A,c))
		hi = _applyRewrite480((Hop) hi);		// %*%(A,rev(!=(c,A))) => %*%(A,!=(A,c))
		hi = _applyRewrite481((Hop) hi);		// %*%(A,rev(!=(A,c))) => %*%(A,!=(A,c))
		hi = _applyRewrite482((Hop) hi);		// %*%(A,!=(rev(A),C)) => %*%(A,!=(A,C))
		hi = _applyRewrite483((Hop) hi);		// %*%(A,!=(C,rev(A))) => %*%(A,!=(A,C))
		hi = _applyRewrite484((Hop) hi);		// rev(-(colSums(A),b)) => -(colSums(A),b)
		hi = _applyRewrite485((Hop) hi);		// rev(-(a,colSums(B))) => -(a,colSums(B))
		hi = _applyRewrite486((Hop) hi);		// rev(!=(colSums(B),a)) => !=(a,colSums(B))
		hi = _applyRewrite487((Hop) hi);		// rev(!=(a,colSums(B))) => !=(a,colSums(B))
		hi = _applyRewrite488((Hop) hi);		// rev(t(rowSums(A))) => t(rowSums(A))
		hi = _applyRewrite489((Hop) hi);		// rev(+(colSums(B),a)) => +(a,colSums(B))
		hi = _applyRewrite490((Hop) hi);		// rev(+(a,colSums(B))) => +(a,colSums(B))
		hi = _applyRewrite491((Hop) hi);		// rev(*(colSums(B),a)) => *(a,colSums(B))
		hi = _applyRewrite492((Hop) hi);		// rev(*(a,colSums(B))) => *(a,colSums(B))
		hi = _applyRewrite493((Hop) hi);		// rev(/(a,colSums(B))) => /(a,colSums(B))
		hi = _applyRewrite494((Hop) hi);		// *(colSums(/(a,C)),b) => colSums(/(*(a,b),C))
		hi = _applyRewrite495((Hop) hi);		// *(a,colSums(/(b,C))) => colSums(/(*(a,b),C))
		hi = _applyRewrite496((Hop) hi);		// *(rowSums(/(a,C)),b) => rowSums(/(*(a,b),C))
		hi = _applyRewrite497((Hop) hi);		// *(a,rowSums(/(b,C))) => rowSums(/(*(a,b),C))
		hi = _applyRewrite498((Hop) hi);		// rev(%*%(colSums(A),B)) => %*%(colSums(A),B)
		return hi;
	}

	// Implementation of the rule *(1.0,a) => a
	private static Hop _applyRewrite0(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite1(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule /(a,1.0) => a
	private static Hop _applyRewrite2(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule +(0.0,a) => a
	private static Hop _applyRewrite3(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
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

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
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

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(0.0,a) => 0.0
	private static Hop _applyRewrite7(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite8(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule /(0.0,a) => 0.0
	private static Hop _applyRewrite9(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule /(A,c) => *(A,/(1.0,c))
	private static Hop _applyRewrite13(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule colSums(*(a,B)) => *(a,colSums(B))
	private static Hop _applyRewrite21(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite22(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule rowSums(*(a,B)) => *(a,rowSums(B))
	private static Hop _applyRewrite23(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite24(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(/(1.0,B),a) => /(a,B)
	private static Hop _applyRewrite32(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite33(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(/(1.0,B),A) => /(A,B)
	private static Hop _applyRewrite34(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite35(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(/(a,C),b) => /(*(a,b),C)
	private static Hop _applyRewrite36(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite37(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(0.0,-(B,a)) => -(a,B)
	private static Hop _applyRewrite42(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite43(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite44(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite45(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(0.0,-(B,A)) => -(A,B)
	private static Hop _applyRewrite46(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite47(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite48(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(-(A,b),c) => -(A,+(b,c))
	private static Hop _applyRewrite49(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite50(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite51(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite52(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite53(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite54(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite55(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite56(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite57(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite58(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite59(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite60(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite61(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite62(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule rev(colSums(A)) => colSums(A)
	private static Hop _applyRewrite76(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0 = (AggUnaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.AggOp.SUM || !c_hi_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(colSums(A)) => colSums(A)");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0, Types.AggOp.SUM, Types.Direction.Col);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v1;
	}

	// Implementation of the rule *(/(1.0,b),a) => /(a,b)
	private static Hop _applyRewrite77(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite78(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite79(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite80(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite81(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite82(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite83(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR|| !l_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite84(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite85(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR|| !l_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite88(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite89(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite90(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite91(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR|| !l_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite92(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite93(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR|| !l_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule +(%*%(B,C),%*%(A,C)) => %*%(+(A,B),C)
	private static Hop _applyRewrite94(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_0_1 != hi_1_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(%*%(B,C),%*%(A,C)) => %*%(+(A,B),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_0, Types.OpOp2.PLUS);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(%*%(A,C),%*%(A,B)) => %*%(A,+(B,C))
	private static Hop _applyRewrite95(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(%*%(A,C),%*%(A,B)) => %*%(A,+(B,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0_1, Types.OpOp2.PLUS);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule rev(-(a,rev(B))) => -(a,B)
	private static Hop _applyRewrite98(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(-(a,rev(B))) => -(a,B)");
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

	// Implementation of the rule t(-(a,t(B))) => -(a,B)
	private static Hop _applyRewrite99(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule rev(-(rev(A),b)) => -(A,b)
	private static Hop _applyRewrite100(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(-(rev(A),b)) => -(A,b)");
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

	// Implementation of the rule t(-(t(A),b)) => -(A,b)
	private static Hop _applyRewrite101(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule rev(!=(rev(A),b)) => !=(A,b)
	private static Hop _applyRewrite102(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(!=(rev(A),b)) => !=(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule rev(!=(b,rev(A))) => !=(A,b)
	private static Hop _applyRewrite103(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(!=(b,rev(A))) => !=(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule t(!=(t(A),b)) => !=(A,b)
	private static Hop _applyRewrite104(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(!=(t(A),b)) => !=(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule t(!=(b,t(A))) => !=(A,b)
	private static Hop _applyRewrite105(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(!=(b,t(A))) => !=(A,b)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule rev(+(rev(A),b)) => +(A,b)
	private static Hop _applyRewrite106(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(+(rev(A),b)) => +(A,b)");
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

	// Implementation of the rule rev(+(b,rev(A))) => +(A,b)
	private static Hop _applyRewrite107(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(+(b,rev(A))) => +(A,b)");
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

	// Implementation of the rule t(+(t(A),b)) => +(A,b)
	private static Hop _applyRewrite108(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite109(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule rev(*(rev(A),b)) => *(A,b)
	private static Hop _applyRewrite110(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(*(rev(A),b)) => *(A,b)");
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

	// Implementation of the rule rev(*(b,rev(A))) => *(A,b)
	private static Hop _applyRewrite111(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(*(b,rev(A))) => *(A,b)");
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

	// Implementation of the rule t(*(t(A),b)) => *(A,b)
	private static Hop _applyRewrite112(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite113(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule rowSums(rev(*(a,B))) => *(a,rowSums(rev(B)))
	private static Hop _applyRewrite114(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(rev(*(a,B))) => *(a,rowSums(rev(B)))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule rowSums(rev(*(B,a))) => *(a,rowSums(rev(B)))
	private static Hop _applyRewrite115(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(rev(*(B,a))) => *(a,rowSums(rev(B)))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule colSums(rev(*(a,B))) => *(a,colSums(rev(B)))
	private static Hop _applyRewrite116(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(rev(*(a,B))) => *(a,colSums(rev(B)))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule colSums(rev(*(B,a))) => *(a,colSums(rev(B)))
	private static Hop _applyRewrite117(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(rev(*(B,a))) => *(a,colSums(rev(B)))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule rev(/(a,rev(B))) => /(a,B)
	private static Hop _applyRewrite118(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(/(a,rev(B))) => /(a,B)");
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

	// Implementation of the rule t(/(a,t(B))) => /(a,B)
	private static Hop _applyRewrite119(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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
	private static Hop _applyRewrite124(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule +(*(B,A),*(A,C)) => *(A,+(B,C))
	private static Hop _applyRewrite125(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(B,A),*(A,C)) => *(A,+(B,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.PLUS);
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
	private static Hop _applyRewrite126(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
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
	private static Hop _applyRewrite127(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite128(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite129(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite130(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite131(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(rev(*(a,C)),b) => *(*(a,b),rev(C))
	private static Hop _applyRewrite132(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(rev(*(a,C)),b) => *(*(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule *(rev(*(C,a)),b) => *(*(a,b),rev(C))
	private static Hop _applyRewrite133(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(rev(*(C,a)),b) => *(*(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule *(a,rev(*(b,C))) => *(*(a,b),rev(C))
	private static Hop _applyRewrite134(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,rev(*(b,C))) => *(*(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule *(a,rev(*(C,b))) => *(*(a,b),rev(C))
	private static Hop _applyRewrite135(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,rev(*(C,b))) => *(*(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_0, Types.ReOrgOp.REV);
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
	private static Hop _applyRewrite152(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite153(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(rev(/(a,C)),b) => /(*(a,b),rev(C))
	private static Hop _applyRewrite154(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(rev(/(a,C)),b) => /(*(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule *(a,rev(/(b,C))) => /(*(a,b),rev(C))
	private static Hop _applyRewrite155(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,rev(/(b,C))) => /(*(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule %*%(colSums(B),*(a,C)) => *(a,%*%(colSums(B),C))
	private static Hop _applyRewrite156(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0 = (AggUnaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.AggOp.SUM || !c_hi_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(colSums(B),*(a,C)) => *(a,%*%(colSums(B),C))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0, Types.AggOp.SUM, Types.Direction.Col);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_1_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(colSums(B),*(C,a)) => *(a,%*%(colSums(B),C))
	private static Hop _applyRewrite157(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0 = (AggUnaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.AggOp.SUM || !c_hi_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(colSums(B),*(C,a)) => *(a,%*%(colSums(B),C))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0, Types.AggOp.SUM, Types.Direction.Col);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_1_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(a,B),rowSums(C)) => *(a,%*%(B,rowSums(C)))
	private static Hop _applyRewrite158(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1 = (AggUnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.AggOp.SUM || !c_hi_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(a,B),rowSums(C)) => *(a,%*%(B,rowSums(C)))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_1_0, Types.AggOp.SUM, Types.Direction.Row);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_1, v1);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(B,a),rowSums(C)) => *(a,%*%(B,rowSums(C)))
	private static Hop _applyRewrite159(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1 = (AggUnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.AggOp.SUM || !c_hi_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(B,a),rowSums(C)) => *(a,%*%(B,rowSums(C)))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_1_0, Types.AggOp.SUM, Types.Direction.Row);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v1);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule colSums(/(*(a,B),C)) => *(a,colSums(/(B,C)))
	private static Hop _applyRewrite160(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite161(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite162(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite163(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.DIV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite164(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite165(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.DIV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite166(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite167(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(/(*(a,C),D),b) => *(*(a,b),/(C,D))
	private static Hop _applyRewrite170(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(*(a,C),D),b) => *(*(a,b),/(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_1, Types.OpOp2.DIV);
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

	// Implementation of the rule *(/(*(C,a),D),b) => *(*(a,b),/(C,D))
	private static Hop _applyRewrite171(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(*(C,a),D),b) => *(*(a,b),/(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.DIV);
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

	// Implementation of the rule *(a,/(*(b,C),D)) => *(*(a,b),/(C,D))
	private static Hop _applyRewrite172(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(*(b,C),D)) => *(*(a,b),/(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_1, hi_1_1, Types.OpOp2.DIV);
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

	// Implementation of the rule *(a,/(*(C,b),D)) => *(*(a,b),/(C,D))
	private static Hop _applyRewrite173(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(*(C,b),D)) => *(*(a,b),/(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_0, hi_1_1, Types.OpOp2.DIV);
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

	// Implementation of the rule *(/(/(a,C),D),b) => /(/(*(a,b),C),D)
	private static Hop _applyRewrite174(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite175(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite176(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule !=(t(A),t(B)) => t(!=(A,B))
	private static Hop _applyRewrite185(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(t(A),t(B)) => t(!=(A,B))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule !=(rev(A),rev(A)) => rev(!=(A,A))
	private static Hop _applyRewrite186(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(A),rev(A)) => rev(!=(A,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_0, Types.OpOp2.NOTEQUAL);
		ReorgOp v2 = HopRewriteUtils.createReorg(v1, Types.ReOrgOp.REV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule rev(-(rev(A),B)) => -(A,rev(B))
	private static Hop _applyRewrite187(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(-(rev(A),B)) => -(A,rev(B))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule rev(-(A,rev(B))) => -(rev(A),B)
	private static Hop _applyRewrite188(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(-(A,rev(B))) => -(rev(A),B)");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule t(-(t(A),B)) => -(A,t(B))
	private static Hop _applyRewrite189(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite190(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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
	private static Hop _applyRewrite191(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
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
	private static Hop _applyRewrite192(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule !=(rev(-(b,A)),A) => !=(A,-(b,A))
	private static Hop _applyRewrite193(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(-(b,A)),A) => !=(A,-(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(-(b,A))) => !=(A,-(b,A))
	private static Hop _applyRewrite194(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(-(b,A))) => !=(A,-(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(-(b,rev(A)),A) => !=(A,-(b,A))
	private static Hop _applyRewrite195(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(-(b,rev(A)),A) => !=(A,-(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(-(b,A),rev(A)) => !=(A,-(b,A))
	private static Hop _applyRewrite196(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(-(b,A),rev(A)) => !=(A,-(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,-(b,rev(A))) => !=(A,-(b,A))
	private static Hop _applyRewrite197(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,-(b,rev(A))) => !=(A,-(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(rev(-(A,c)),A) => !=(A,-(A,c))
	private static Hop _applyRewrite198(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(-(A,c)),A) => !=(A,-(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(-(A,c))) => !=(A,-(A,c))
	private static Hop _applyRewrite199(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(-(A,c))) => !=(A,-(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(-(rev(A),c),A) => !=(A,-(A,c))
	private static Hop _applyRewrite200(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(-(rev(A),c),A) => !=(A,-(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,-(rev(A),c)) => !=(A,-(A,c))
	private static Hop _applyRewrite201(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,-(rev(A),c)) => !=(A,-(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(-(B,rev(A)),A) => !=(A,-(B,A))
	private static Hop _applyRewrite202(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(-(B,rev(A)),A) => !=(A,-(B,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(-(B,A),rev(A)) => !=(A,-(B,A))
	private static Hop _applyRewrite203(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(-(B,A),rev(A)) => !=(A,-(B,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,-(B,rev(A))) => !=(A,-(B,A))
	private static Hop _applyRewrite204(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,-(B,rev(A))) => !=(A,-(B,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(-(rev(A),C),A) => !=(A,-(A,C))
	private static Hop _applyRewrite205(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(-(rev(A),C),A) => !=(A,-(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(-(A,C),rev(A)) => !=(A,-(A,C))
	private static Hop _applyRewrite206(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(-(A,C),rev(A)) => !=(A,-(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,-(rev(A),C)) => !=(A,-(A,C))
	private static Hop _applyRewrite207(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,-(rev(A),C)) => !=(A,-(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule rev(!=(rev(A),B)) => !=(A,rev(B))
	private static Hop _applyRewrite208(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(!=(rev(A),B)) => !=(A,rev(B))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_1, Types.ReOrgOp.REV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule rev(!=(B,rev(A))) => !=(A,rev(B))
	private static Hop _applyRewrite209(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(!=(B,rev(A))) => !=(A,rev(B))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0, Types.ReOrgOp.REV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule t(!=(t(A),B)) => !=(A,t(B))
	private static Hop _applyRewrite210(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(!=(t(A),B)) => !=(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(!=(B,t(A))) => !=(A,t(B))
	private static Hop _applyRewrite211(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(!=(B,t(A))) => !=(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(rev(+(c,A)),A) => !=(A,+(A,c))
	private static Hop _applyRewrite212(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(+(c,A)),A) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(rev(+(A,c)),A) => !=(A,+(A,c))
	private static Hop _applyRewrite213(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(+(A,c)),A) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(+(c,A))) => !=(A,+(A,c))
	private static Hop _applyRewrite214(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(+(c,A))) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(+(A,c))) => !=(A,+(A,c))
	private static Hop _applyRewrite215(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(+(A,c))) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(+(rev(A),c),A) => !=(A,+(A,c))
	private static Hop _applyRewrite216(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(+(rev(A),c),A) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(+(c,rev(A)),A) => !=(A,+(A,c))
	private static Hop _applyRewrite217(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(+(c,rev(A)),A) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(+(c,A),rev(A)) => !=(A,+(A,c))
	private static Hop _applyRewrite218(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(+(c,A),rev(A)) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,+(rev(A),c)) => !=(A,+(A,c))
	private static Hop _applyRewrite219(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,+(rev(A),c)) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,+(c,rev(A))) => !=(A,+(A,c))
	private static Hop _applyRewrite220(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,+(c,rev(A))) => !=(A,+(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(+(rev(A),C),A) => !=(A,+(A,C))
	private static Hop _applyRewrite221(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(+(rev(A),C),A) => !=(A,+(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(+(C,rev(A)),A) => !=(A,+(A,C))
	private static Hop _applyRewrite222(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(+(C,rev(A)),A) => !=(A,+(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(+(C,A),rev(A)) => !=(A,+(A,C))
	private static Hop _applyRewrite223(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(+(C,A),rev(A)) => !=(A,+(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(+(A,C),rev(A)) => !=(A,+(A,C))
	private static Hop _applyRewrite224(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(+(A,C),rev(A)) => !=(A,+(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,+(rev(A),C)) => !=(A,+(A,C))
	private static Hop _applyRewrite225(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,+(rev(A),C)) => !=(A,+(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,+(C,rev(A))) => !=(A,+(A,C))
	private static Hop _applyRewrite226(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,+(C,rev(A))) => !=(A,+(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(!=(rev(A),c),A) => !=(A,!=(A,c))
	private static Hop _applyRewrite227(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(!=(rev(A),c),A) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(!=(c,rev(A)),A) => !=(A,!=(A,c))
	private static Hop _applyRewrite228(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(!=(c,rev(A)),A) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(!=(c,A),rev(A)) => !=(A,!=(A,c))
	private static Hop _applyRewrite229(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(!=(c,A),rev(A)) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,!=(rev(A),c)) => !=(A,!=(A,c))
	private static Hop _applyRewrite230(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,!=(rev(A),c)) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,!=(c,rev(A))) => !=(A,!=(A,c))
	private static Hop _applyRewrite231(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,!=(c,rev(A))) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(rev(!=(c,A)),A) => !=(A,!=(A,c))
	private static Hop _applyRewrite232(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(!=(c,A)),A) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(rev(!=(A,c)),A) => !=(A,!=(A,c))
	private static Hop _applyRewrite233(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(!=(A,c)),A) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(!=(c,A))) => !=(A,!=(A,c))
	private static Hop _applyRewrite234(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(!=(c,A))) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(!=(A,c))) => !=(A,!=(A,c))
	private static Hop _applyRewrite235(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(!=(A,c))) => !=(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(!=(rev(A),C),A) => !=(A,!=(A,C))
	private static Hop _applyRewrite236(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(!=(rev(A),C),A) => !=(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(!=(C,rev(A)),A) => !=(A,!=(A,C))
	private static Hop _applyRewrite237(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(!=(C,rev(A)),A) => !=(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(!=(C,A),rev(A)) => !=(A,!=(A,C))
	private static Hop _applyRewrite238(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(!=(C,A),rev(A)) => !=(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(!=(A,C),rev(A)) => !=(A,!=(A,C))
	private static Hop _applyRewrite239(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(!=(A,C),rev(A)) => !=(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,!=(rev(A),C)) => !=(A,!=(A,C))
	private static Hop _applyRewrite240(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,!=(rev(A),C)) => !=(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,!=(C,rev(A))) => !=(A,!=(A,C))
	private static Hop _applyRewrite241(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,!=(C,rev(A))) => !=(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule rev(+(rev(A),B)) => +(A,rev(B))
	private static Hop _applyRewrite242(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(+(rev(A),B)) => +(A,rev(B))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule rev(+(B,rev(A))) => +(A,rev(B))
	private static Hop _applyRewrite243(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(+(B,rev(A))) => +(A,rev(B))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule t(+(t(A),B)) => +(A,t(B))
	private static Hop _applyRewrite244(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite245(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule +(!=(rev(A),c),A) => +(A,!=(A,c))
	private static Hop _applyRewrite246(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(!=(rev(A),c),A) => +(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule +(!=(c,rev(A)),A) => +(A,!=(A,c))
	private static Hop _applyRewrite247(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(!=(c,rev(A)),A) => +(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule +(A,!=(rev(A),c)) => +(A,!=(A,c))
	private static Hop _applyRewrite248(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,!=(rev(A),c)) => +(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule +(A,!=(c,rev(A))) => +(A,!=(A,c))
	private static Hop _applyRewrite249(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,!=(c,rev(A))) => +(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule +(rev(!=(c,A)),A) => +(A,!=(A,c))
	private static Hop _applyRewrite250(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(rev(!=(c,A)),A) => +(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule +(rev(!=(A,c)),A) => +(A,!=(A,c))
	private static Hop _applyRewrite251(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(rev(!=(A,c)),A) => +(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule +(A,rev(!=(c,A))) => +(A,!=(A,c))
	private static Hop _applyRewrite252(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,rev(!=(c,A))) => +(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule +(A,rev(!=(A,c))) => +(A,!=(A,c))
	private static Hop _applyRewrite253(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,rev(!=(A,c))) => +(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule +(!=(rev(A),C),A) => +(A,!=(A,C))
	private static Hop _applyRewrite254(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(!=(rev(A),C),A) => +(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule +(!=(C,rev(A)),A) => +(A,!=(A,C))
	private static Hop _applyRewrite255(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(!=(C,rev(A)),A) => +(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule +(A,!=(rev(A),C)) => +(A,!=(A,C))
	private static Hop _applyRewrite256(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,!=(rev(A),C)) => +(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule +(A,!=(C,rev(A))) => +(A,!=(A,C))
	private static Hop _applyRewrite257(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,!=(C,rev(A))) => +(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule -(rev(!=(A,b)),A) => -(!=(A,b),A)
	private static Hop _applyRewrite258(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(rev(!=(A,b)),A) => -(!=(A,b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule -(A,!=(rev(A),c)) => -(A,!=(A,c))
	private static Hop _applyRewrite259(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(A,!=(rev(A),c)) => -(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule -(A,!=(c,rev(A))) => -(A,!=(A,c))
	private static Hop _applyRewrite260(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(A,!=(c,rev(A))) => -(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule -(A,rev(!=(c,A))) => -(A,!=(A,c))
	private static Hop _applyRewrite261(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(A,rev(!=(c,A))) => -(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule -(A,rev(!=(A,c))) => -(A,!=(A,c))
	private static Hop _applyRewrite262(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(A,rev(!=(A,c))) => -(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule -(A,!=(rev(A),C)) => -(A,!=(A,C))
	private static Hop _applyRewrite263(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(A,!=(rev(A),C)) => -(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule -(A,!=(C,rev(A))) => -(A,!=(A,C))
	private static Hop _applyRewrite264(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(A,!=(C,rev(A))) => -(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule -(t(-(A,b)),c) => -(t(A),+(b,c))
	private static Hop _applyRewrite265(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(t(-(a,C)),b) => -(-(a,b),t(C))
	private static Hop _applyRewrite266(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(a,t(+(b,C))) => -(-(a,b),t(C))
	private static Hop _applyRewrite267(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,t(+(b,C))) => -(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
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

	// Implementation of the rule -(a,t(+(C,b))) => -(-(a,b),t(C))
	private static Hop _applyRewrite268(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,t(+(C,b))) => -(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_0);
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

	// Implementation of the rule -(rev(-(A,b)),c) => -(rev(A),+(b,c))
	private static Hop _applyRewrite269(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(rev(-(A,b)),c) => -(rev(A),+(b,c))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule -(rev(-(a,C)),b) => -(-(a,b),rev(C))
	private static Hop _applyRewrite270(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(rev(-(a,C)),b) => -(-(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule -(a,rev(+(b,C))) => -(-(a,b),rev(C))
	private static Hop _applyRewrite271(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,rev(+(b,C))) => -(-(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule -(a,rev(+(C,b))) => -(-(a,b),rev(C))
	private static Hop _applyRewrite272(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,rev(+(C,b))) => -(-(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule -(-(-(a,D),C),b) => -(-(a,b),+(C,D))
	private static Hop _applyRewrite273(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite274(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(-(a,D),+(C,b)) => -(-(a,b),+(C,D))
	private static Hop _applyRewrite275(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,D),+(C,b)) => -(-(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, hi_0_1, Types.OpOp2.PLUS);
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
	private static Hop _applyRewrite276(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite277(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite278(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite279(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite280(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(a,rev(-(C,b))) => -(+(a,b),rev(C))
	private static Hop _applyRewrite281(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,rev(-(C,b))) => -(+(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule +(rev(-(a,C)),b) => -(+(a,b),rev(C))
	private static Hop _applyRewrite282(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(rev(-(a,C)),b) => -(+(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule +(a,rev(-(b,C))) => -(+(a,b),rev(C))
	private static Hop _applyRewrite283(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,rev(-(b,C))) => -(+(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule -(a,rev(-(b,C))) => +(-(a,b),rev(C))
	private static Hop _applyRewrite284(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,rev(-(b,C))) => +(-(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule -(rev(+(a,C)),b) => +(-(a,b),rev(C))
	private static Hop _applyRewrite285(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(rev(+(a,C)),b) => +(-(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule -(rev(+(C,a)),b) => +(-(a,b),rev(C))
	private static Hop _applyRewrite286(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(rev(+(C,a)),b) => +(-(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule +(rev(-(C,b)),a) => +(-(a,b),rev(C))
	private static Hop _applyRewrite287(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(rev(-(C,b)),a) => +(-(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule +(a,rev(-(C,b))) => +(-(a,b),rev(C))
	private static Hop _applyRewrite288(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,rev(-(C,b))) => +(-(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule +(rev(+(a,C)),b) => +(+(a,b),rev(C))
	private static Hop _applyRewrite289(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(rev(+(a,C)),b) => +(+(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule +(rev(+(C,a)),b) => +(+(a,b),rev(C))
	private static Hop _applyRewrite290(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(rev(+(C,a)),b) => +(+(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_0_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule +(a,rev(+(b,C))) => +(+(a,b),rev(C))
	private static Hop _applyRewrite291(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,rev(+(b,C))) => +(+(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule +(a,rev(+(C,b))) => +(+(a,b),rev(C))
	private static Hop _applyRewrite292(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,rev(+(C,b))) => +(+(a,b),rev(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createReorg(hi_1_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule -(-(a,C),-(D,b)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite293(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(a,-(C,-(b,D))) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite294(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(C,-(b,D))) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, hi_1_1_1, Types.OpOp2.PLUS);
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

	// Implementation of the rule -(a,+(-(D,b),C)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite295(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(-(D,b),C)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0_0, Types.OpOp2.PLUS);
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
	private static Hop _applyRewrite296(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule +(-(-(a,C),D),b) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite297(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(-(a,C),D),b) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_1, Types.OpOp2.PLUS);
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
	private static Hop _applyRewrite298(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule +(a,-(-(b,D),C)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite299(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(-(b,D),C)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0_1, Types.OpOp2.PLUS);
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
	private static Hop _applyRewrite300(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite301(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(+(-(B,c),A),d) => +(A,-(B,+(c,d)))
	private static Hop _applyRewrite302(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(-(B,c),A),d) => +(A,-(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MINUS);
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

	// Implementation of the rule -(+(A,-(B,c)),d) => +(A,-(B,+(c,d)))
	private static Hop _applyRewrite303(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,-(B,c)),d) => +(A,-(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.MINUS);
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

	// Implementation of the rule +(-(B,c),-(A,d)) => +(A,-(B,+(c,d)))
	private static Hop _applyRewrite304(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(b,-(-(D,c),A)) => +(A,-(+(b,c),D))
	private static Hop _applyRewrite305(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite306(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite307(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite308(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite309(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite310(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite311(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite312(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite313(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite314(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite315(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(c,-(-(d,B),A)) => +(A,+(B,-(c,d)))
	private static Hop _applyRewrite316(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(-(d,B),A)) => +(A,+(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_1, v1, Types.OpOp2.PLUS);
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

	// Implementation of the rule -(+(c,B),-(d,A)) => +(A,+(B,-(c,d)))
	private static Hop _applyRewrite317(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(c,B),-(d,A)) => +(A,+(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.PLUS);
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

	// Implementation of the rule -(+(B,c),-(d,A)) => +(A,+(B,-(c,d)))
	private static Hop _applyRewrite318(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(B,c),-(d,A)) => +(A,+(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);
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

	// Implementation of the rule +(-(A,-(d,B)),c) => +(A,+(B,-(c,d)))
	private static Hop _applyRewrite319(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,-(d,B)),c) => +(A,+(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_1, v1, Types.OpOp2.PLUS);
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

	// Implementation of the rule +(c,-(A,-(d,B))) => +(A,+(B,-(c,d)))
	private static Hop _applyRewrite320(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(A,-(d,B))) => +(A,+(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_1, v1, Types.OpOp2.PLUS);
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

	// Implementation of the rule -(-(A,-(D,b)),c) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite321(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite322(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite323(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite324(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite325(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite326(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite327(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite328(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite329(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(+(b,A),+(c,D)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite330(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite331(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite332(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite333(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(+(A,-(b,D)),c) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite334(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(b,+(-(c,A),D)) => +(A,-(-(b,c),D))
	private static Hop _applyRewrite335(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite336(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite337(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite338(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite339(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite340(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite341(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite342(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite343(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite344(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(a,t(-(C,b))) => -(+(a,b),t(C))
	private static Hop _applyRewrite345(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,t(-(C,b))) => -(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_0);
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

	// Implementation of the rule +(t(-(a,C)),b) => -(+(a,b),t(C))
	private static Hop _applyRewrite346(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite347(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite348(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite349(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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

	// Implementation of the rule -(a,t(-(b,C))) => +(-(a,b),t(C))
	private static Hop _applyRewrite350(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,t(-(b,C))) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
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

	// Implementation of the rule +(t(-(C,b)),a) => +(-(a,b),t(C))
	private static Hop _applyRewrite351(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite352(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule +(t(+(a,C)),b) => +(+(a,b),t(C))
	private static Hop _applyRewrite353(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite354(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite355(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite356(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule colSums(-(t(A),b)) => t(rowSums(-(A,b)))
	private static Hop _applyRewrite357(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite358(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule rowSums(-(t(A),b)) => t(colSums(-(A,b)))
	private static Hop _applyRewrite359(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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

	// Implementation of the rule rowSums(-(a,t(B))) => t(colSums(-(a,B)))
	private static Hop _applyRewrite360(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule colSums(!=(t(A),b)) => t(rowSums(!=(A,b)))
	private static Hop _applyRewrite361(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(!=(t(A),b)) => t(rowSums(!=(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule colSums(!=(b,t(A))) => t(rowSums(!=(A,b)))
	private static Hop _applyRewrite362(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(!=(b,t(A))) => t(rowSums(!=(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule rowSums(!=(t(A),b)) => t(colSums(!=(A,b)))
	private static Hop _applyRewrite363(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(!=(t(A),b)) => t(colSums(!=(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule rowSums(!=(b,t(A))) => t(colSums(!=(A,b)))
	private static Hop _applyRewrite364(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(!=(b,t(A))) => t(colSums(!=(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
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
	private static Hop _applyRewrite365(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite366(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule rowSums(+(t(A),b)) => t(colSums(+(A,b)))
	private static Hop _applyRewrite367(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite368(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule *(t(A),t(B)) => t(*(A,B))
	private static Hop _applyRewrite372(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule !=(*(rev(A),c),A) => !=(A,*(A,c))
	private static Hop _applyRewrite373(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(*(rev(A),c),A) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(*(c,rev(A)),A) => !=(A,*(A,c))
	private static Hop _applyRewrite374(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(*(c,rev(A)),A) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(*(c,A),rev(A)) => !=(A,*(A,c))
	private static Hop _applyRewrite375(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(*(c,A),rev(A)) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,*(rev(A),c)) => !=(A,*(A,c))
	private static Hop _applyRewrite376(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,*(rev(A),c)) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,*(c,rev(A))) => !=(A,*(A,c))
	private static Hop _applyRewrite377(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,*(c,rev(A))) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(rev(*(c,A)),A) => !=(A,*(A,c))
	private static Hop _applyRewrite378(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(*(c,A)),A) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(rev(*(A,c)),A) => !=(A,*(A,c))
	private static Hop _applyRewrite379(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(*(A,c)),A) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(*(c,A))) => !=(A,*(A,c))
	private static Hop _applyRewrite380(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(*(c,A))) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(*(A,c))) => !=(A,*(A,c))
	private static Hop _applyRewrite381(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(*(A,c))) => !=(A,*(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(*(rev(A),C),A) => !=(A,*(A,C))
	private static Hop _applyRewrite382(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(*(rev(A),C),A) => !=(A,*(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(*(C,rev(A)),A) => !=(A,*(A,C))
	private static Hop _applyRewrite383(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(*(C,rev(A)),A) => !=(A,*(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(*(C,A),rev(A)) => !=(A,*(A,C))
	private static Hop _applyRewrite384(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(*(C,A),rev(A)) => !=(A,*(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(*(A,C),rev(A)) => !=(A,*(A,C))
	private static Hop _applyRewrite385(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(*(A,C),rev(A)) => !=(A,*(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,*(rev(A),C)) => !=(A,*(A,C))
	private static Hop _applyRewrite386(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,*(rev(A),C)) => !=(A,*(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,*(C,rev(A))) => !=(A,*(A,C))
	private static Hop _applyRewrite387(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,*(C,rev(A))) => !=(A,*(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule rev(*(rev(A),B)) => *(A,rev(B))
	private static Hop _applyRewrite388(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(*(rev(A),B)) => *(A,rev(B))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule rev(*(B,rev(A))) => *(A,rev(B))
	private static Hop _applyRewrite389(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(*(B,rev(A))) => *(A,rev(B))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule t(*(t(A),B)) => *(A,t(B))
	private static Hop _applyRewrite390(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite391(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule *(!=(rev(A),c),A) => *(A,!=(A,c))
	private static Hop _applyRewrite392(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(!=(rev(A),c),A) => *(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule *(!=(c,rev(A)),A) => *(A,!=(A,c))
	private static Hop _applyRewrite393(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(!=(c,rev(A)),A) => *(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule *(A,!=(rev(A),c)) => *(A,!=(A,c))
	private static Hop _applyRewrite394(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,!=(rev(A),c)) => *(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule *(A,!=(c,rev(A))) => *(A,!=(A,c))
	private static Hop _applyRewrite395(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,!=(c,rev(A))) => *(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule *(rev(!=(c,A)),A) => *(A,!=(A,c))
	private static Hop _applyRewrite396(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(rev(!=(c,A)),A) => *(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule *(rev(!=(A,c)),A) => *(A,!=(A,c))
	private static Hop _applyRewrite397(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(rev(!=(A,c)),A) => *(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule *(A,rev(!=(c,A))) => *(A,!=(A,c))
	private static Hop _applyRewrite398(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,rev(!=(c,A))) => *(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule *(A,rev(!=(A,c))) => *(A,!=(A,c))
	private static Hop _applyRewrite399(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,rev(!=(A,c))) => *(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule *(!=(rev(A),C),A) => *(A,!=(A,C))
	private static Hop _applyRewrite400(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(!=(rev(A),C),A) => *(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule *(!=(C,rev(A)),A) => *(A,!=(A,C))
	private static Hop _applyRewrite401(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(!=(C,rev(A)),A) => *(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule *(A,!=(rev(A),C)) => *(A,!=(A,C))
	private static Hop _applyRewrite402(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,!=(rev(A),C)) => *(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule *(A,!=(C,rev(A))) => *(A,!=(A,C))
	private static Hop _applyRewrite403(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,!=(C,rev(A))) => *(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule rev(/(rev(A),B)) => /(A,rev(B))
	private static Hop _applyRewrite405(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(/(rev(A),B)) => /(A,rev(B))");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_1, Types.ReOrgOp.REV);
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

	// Implementation of the rule rev(/(A,rev(B))) => /(rev(A),B)
	private static Hop _applyRewrite406(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(/(A,rev(B))) => /(rev(A),B)");
		ReorgOp v1 = HopRewriteUtils.createReorg(hi_0_0, Types.ReOrgOp.REV);
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

	// Implementation of the rule t(/(t(A),B)) => /(A,t(B))
	private static Hop _applyRewrite407(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite408(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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
	private static Hop _applyRewrite409(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule !=(/(b,rev(A)),A) => !=(A,/(b,A))
	private static Hop _applyRewrite410(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(/(b,rev(A)),A) => !=(A,/(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(/(b,A),rev(A)) => !=(A,/(b,A))
	private static Hop _applyRewrite411(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(/(b,A),rev(A)) => !=(A,/(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,/(b,rev(A))) => !=(A,/(b,A))
	private static Hop _applyRewrite412(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,/(b,rev(A))) => !=(A,/(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(rev(/(b,A)),A) => !=(A,/(b,A))
	private static Hop _applyRewrite413(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(/(b,A)),A) => !=(A,/(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(/(b,A))) => !=(A,/(b,A))
	private static Hop _applyRewrite414(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(/(b,A))) => !=(A,/(b,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0_0, hi_0, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(/(B,rev(A)),A) => !=(A,/(B,A))
	private static Hop _applyRewrite415(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(/(B,rev(A)),A) => !=(A,/(B,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(/(B,A),rev(A)) => !=(A,/(B,A))
	private static Hop _applyRewrite416(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_1 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(/(B,A),rev(A)) => !=(A,/(B,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,/(B,rev(A))) => !=(A,/(B,A))
	private static Hop _applyRewrite417(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,/(B,rev(A))) => !=(A,/(B,A))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(/(rev(A),C),A) => !=(A,/(A,C))
	private static Hop _applyRewrite418(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(/(rev(A),C),A) => !=(A,/(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(/(A,C),rev(A)) => !=(A,/(A,C))
	private static Hop _applyRewrite419(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_0_0 != hi_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(/(A,C),rev(A)) => !=(A,/(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule !=(A,/(rev(A),C)) => !=(A,/(A,C))
	private static Hop _applyRewrite420(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,/(rev(A),C)) => !=(A,/(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule /(rev(!=(A,b)),A) => /(!=(A,b),A)
	private static Hop _applyRewrite421(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(rev(!=(A,b)),A) => /(!=(A,b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule /(A,rev(!=(c,A))) => /(A,!=(A,c))
	private static Hop _applyRewrite422(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,rev(!=(c,A))) => /(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule /(A,rev(!=(A,c))) => /(A,!=(A,c))
	private static Hop _applyRewrite423(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,rev(!=(A,c))) => /(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule /(A,!=(rev(A),c)) => /(A,!=(A,c))
	private static Hop _applyRewrite424(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,!=(rev(A),c)) => /(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule /(A,!=(c,rev(A))) => /(A,!=(A,c))
	private static Hop _applyRewrite425(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,!=(c,rev(A))) => /(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule /(A,!=(rev(A),C)) => /(A,!=(A,C))
	private static Hop _applyRewrite426(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,!=(rev(A),C)) => /(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule /(A,!=(C,rev(A))) => /(A,!=(A,C))
	private static Hop _applyRewrite427(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,!=(C,rev(A))) => /(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule colSums(/(a,t(B))) => t(rowSums(/(a,B)))
	private static Hop _applyRewrite428(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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
	private static Hop _applyRewrite429(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || !c_hi.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule !=(A,rev(rowSums(A))) => !=(A,rowSums(A))
	private static Hop _applyRewrite438(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1_0 = (AggUnaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.AggOp.SUM || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1_0.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(rowSums(A))) => !=(A,rowSums(A))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,rowSums(rev(A))) => !=(A,rowSums(A))
	private static Hop _applyRewrite439(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1 = (AggUnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.AggOp.SUM || !c_hi_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rowSums(rev(A))) => !=(A,rowSums(A))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0, Types.AggOp.SUM, Types.Direction.Row);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(A,colSums(rev(A))) => !=(A,colSums(A))
	private static Hop _applyRewrite440(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1 = (AggUnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.AggOp.SUM || !c_hi_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,colSums(rev(A))) => !=(A,colSums(A))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule +(A,colSums(rev(A))) => +(A,colSums(A))
	private static Hop _applyRewrite441(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1 = (AggUnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.AggOp.SUM || !c_hi_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,colSums(rev(A))) => +(A,colSums(A))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule *(A,colSums(rev(A))) => *(A,colSums(A))
	private static Hop _applyRewrite442(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1 = (AggUnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.AggOp.SUM || !c_hi_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(A,colSums(rev(A))) => *(A,colSums(A))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule %*%(*(a,C),*(b,D)) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite443(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite444(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite445(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite446(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(%*%(*(a,C),D),b) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite447(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(%*%(*(a,C),D),b) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_0_1, hi_0_1);
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

	// Implementation of the rule *(%*%(*(C,a),D),b) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite448(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(%*%(*(C,a),D),b) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_0_0, hi_0_1);
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

	// Implementation of the rule *(%*%(C,*(D,a)),b) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite449(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MULT || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(%*%(C,*(D,a)),b) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_0, hi_0_1_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule *(a,%*%(*(b,C),D)) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite450(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,%*%(*(b,C),D)) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_1_0_1, hi_1_1);
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

	// Implementation of the rule *(a,%*%(*(C,b),D)) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite451(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,%*%(*(C,b),D)) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_1_0_0, hi_1_1);
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

	// Implementation of the rule *(a,%*%(C,*(b,D))) => *(*(a,b),%*%(C,D))
	private static Hop _applyRewrite452(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MULT || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,%*%(C,*(b,D))) => *(*(a,b),%*%(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MULT);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_1_0, hi_1_1_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule %*%(/(a,C),*(b,D)) => %*%(/(*(a,b),C),D)
	private static Hop _applyRewrite453(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite454(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(%*%(/(a,C),D),b) => %*%(/(*(a,b),C),D)
	private static Hop _applyRewrite455(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(%*%(/(a,C),D),b) => %*%(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(v2, hi_0_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(a,%*%(/(b,C),D)) => %*%(/(*(a,b),C),D)
	private static Hop _applyRewrite456(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,%*%(/(b,C),D)) => %*%(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(v2, hi_1_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule %*%(*(b,A),/(c,D)) => %*%(A,/(*(b,c),D))
	private static Hop _applyRewrite457(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite458(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
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

	// Implementation of the rule *(%*%(A,/(b,D)),c) => %*%(A,/(*(b,c),D))
	private static Hop _applyRewrite459(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.DIV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX || !hi_0_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(%*%(A,/(b,D)),c) => %*%(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule *(b,%*%(A,/(c,D))) => %*%(A,/(*(b,c),D))
	private static Hop _applyRewrite460(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.DIV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(b,%*%(A,/(c,D))) => %*%(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_1_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule t(%*%(t(B),A)) => %*%(t(A),B)
	private static Hop _applyRewrite461(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
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

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
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
	private static Hop _applyRewrite462(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
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
	private static Hop _applyRewrite463(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
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

	// Implementation of the rule !=(%*%(B,rev(A)),A) => !=(A,%*%(B,A))
	private static Hop _applyRewrite464(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(%*%(B,rev(A)),A) => !=(A,%*%(B,A))");
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_0_0, hi_0_1_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule !=(A,%*%(B,rev(A))) => !=(A,%*%(B,A))
	private static Hop _applyRewrite465(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,%*%(B,rev(A))) => !=(A,%*%(B,A))");
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_1_0, hi_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule !=(rev(%*%(A,C)),A) => !=(A,%*%(A,C))
	private static Hop _applyRewrite466(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0_0) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(rev(%*%(A,C)),A) => !=(A,%*%(A,C))");
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_0_0_0, hi_0_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,rev(%*%(A,C))) => !=(A,%*%(A,C))
	private static Hop _applyRewrite467(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1_0) )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,rev(%*%(A,C))) => !=(A,%*%(A,C))");
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_0, hi_1_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule !=(%*%(rev(A),C),A) => !=(A,%*%(A,C))
	private static Hop _applyRewrite468(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
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

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(%*%(rev(A),C),A) => !=(A,%*%(A,C))");
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_0_0_0, hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule !=(A,%*%(rev(A),C)) => !=(A,%*%(A,C))
	private static Hop _applyRewrite469(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.NOTEQUAL || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: !=(A,%*%(rev(A),C)) => !=(A,%*%(A,C))");
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_0, hi_1_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule rev(%*%(!=(b,A),A)) => %*%(!=(A,b),A)
	private static Hop _applyRewrite470(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_0_1 != hi_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(%*%(!=(b,A),A)) => %*%(!=(A,b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_0_0, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_0_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule rev(%*%(!=(A,b),A)) => %*%(!=(A,b),A)
	private static Hop _applyRewrite471(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_0_0 != hi_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(%*%(!=(A,b),A)) => %*%(!=(A,b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule %*%(!=(rev(A),b),A) => %*%(!=(A,b),A)
	private static Hop _applyRewrite472(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(!=(rev(A),b),A) => %*%(!=(A,b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule %*%(!=(b,rev(A)),A) => %*%(!=(A,b),A)
	private static Hop _applyRewrite473(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(!=(b,rev(A)),A) => %*%(!=(A,b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_1_0);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule %*%(rev(!=(b,A)),A) => %*%(!=(A,b),A)
	private static Hop _applyRewrite474(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(rev(!=(b,A)),A) => %*%(!=(A,b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_0_0, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_0_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule %*%(rev(!=(A,b)),A) => %*%(!=(A,b),A)
	private static Hop _applyRewrite475(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.REV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(rev(!=(A,b)),A) => %*%(!=(A,b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule %*%(!=(rev(A),B),A) => %*%(!=(A,B),A)
	private static Hop _applyRewrite476(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.REV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(!=(rev(A),B),A) => %*%(!=(A,B),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.NOTEQUAL);
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

	// Implementation of the rule %*%(!=(B,rev(A)),A) => %*%(!=(A,B),A)
	private static Hop _applyRewrite477(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.REV || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(!=(B,rev(A)),A) => %*%(!=(A,B),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_1_0);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule %*%(A,!=(rev(A),c)) => %*%(A,!=(A,c))
	private static Hop _applyRewrite478(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(A,!=(rev(A),c)) => %*%(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule %*%(A,!=(c,rev(A))) => %*%(A,!=(A,c))
	private static Hop _applyRewrite479(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(A,!=(c,rev(A))) => %*%(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule %*%(A,rev(!=(c,A))) => %*%(A,!=(A,c))
	private static Hop _applyRewrite480(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(A,rev(!=(c,A))) => %*%(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule %*%(A,rev(!=(A,c))) => %*%(A,!=(A,c))
	private static Hop _applyRewrite481(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.REV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(A,rev(!=(A,c))) => %*%(A,!=(A,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule %*%(A,!=(rev(A),C)) => %*%(A,!=(A,C))
	private static Hop _applyRewrite482(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_0 = (ReorgOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.ReOrgOp.REV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_0 != hi_1_0_0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(A,!=(rev(A),C)) => %*%(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v2;
	}

	// Implementation of the rule %*%(A,!=(C,rev(A))) => %*%(A,!=(A,C))
	private static Hop _applyRewrite483(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if (hi_1_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1_1 = (ReorgOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.ReOrgOp.REV || !c_hi_1_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_0 != hi_1_1_0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(A,!=(C,rev(A))) => %*%(A,!=(A,C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.NOTEQUAL);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v2;
	}

	// Implementation of the rule rev(-(colSums(A),b)) => -(colSums(A),b)
	private static Hop _applyRewrite484(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_0 = (AggUnaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.AggOp.SUM || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(-(colSums(A),b)) => -(colSums(A),b)");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule rev(-(a,colSums(B))) => -(a,colSums(B))
	private static Hop _applyRewrite485(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_1 = (AggUnaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.AggOp.SUM || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(-(a,colSums(B))) => -(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule rev(!=(colSums(B),a)) => !=(a,colSums(B))
	private static Hop _applyRewrite486(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_0 = (AggUnaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.AggOp.SUM || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(!=(colSums(B),a)) => !=(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule rev(!=(a,colSums(B))) => !=(a,colSums(B))
	private static Hop _applyRewrite487(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.NOTEQUAL || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_1 = (AggUnaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.AggOp.SUM || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(!=(a,colSums(B))) => !=(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.NOTEQUAL);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule rev(t(rowSums(A))) => t(rowSums(A))
	private static Hop _applyRewrite488(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_0 = (AggUnaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.AggOp.SUM || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_0.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(t(rowSums(A))) => t(rowSums(A))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0_0, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule rev(+(colSums(B),a)) => +(a,colSums(B))
	private static Hop _applyRewrite489(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_0 = (AggUnaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.AggOp.SUM || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(+(colSums(B),a)) => +(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule rev(+(a,colSums(B))) => +(a,colSums(B))
	private static Hop _applyRewrite490(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_1 = (AggUnaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.AggOp.SUM || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(+(a,colSums(B))) => +(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule rev(*(colSums(B),a)) => *(a,colSums(B))
	private static Hop _applyRewrite491(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_0 = (AggUnaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.AggOp.SUM || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(*(colSums(B),a)) => *(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule rev(*(a,colSums(B))) => *(a,colSums(B))
	private static Hop _applyRewrite492(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_1 = (AggUnaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.AggOp.SUM || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(*(a,colSums(B))) => *(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule rev(/(a,colSums(B))) => /(a,colSums(B))
	private static Hop _applyRewrite493(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if (hi_0_1.getParent().size() > 1)
			return hi;
		if ( !(hi_0_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_1 = (AggUnaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.AggOp.SUM || !c_hi_0_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX || !hi_0_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(/(a,colSums(B))) => /(a,colSums(B))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_1_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule *(colSums(/(a,C)),b) => colSums(/(*(a,b),C))
	private static Hop _applyRewrite494(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0 = (AggUnaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.AggOp.SUM || !c_hi_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(colSums(/(a,C)),b) => colSums(/(*(a,b),C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1, Types.OpOp2.DIV);
		AggUnaryOp v3 = HopRewriteUtils.createAggUnaryOp(v2, Types.AggOp.SUM, Types.Direction.Col);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(a,colSums(/(b,C))) => colSums(/(*(a,b),C))
	private static Hop _applyRewrite495(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1 = (AggUnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.AggOp.SUM || !c_hi_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,colSums(/(b,C))) => colSums(/(*(a,b),C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_1, Types.OpOp2.DIV);
		AggUnaryOp v3 = HopRewriteUtils.createAggUnaryOp(v2, Types.AggOp.SUM, Types.Direction.Col);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(rowSums(/(a,C)),b) => rowSums(/(*(a,b),C))
	private static Hop _applyRewrite496(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0 = (AggUnaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.AggOp.SUM || !c_hi_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(rowSums(/(a,C)),b) => rowSums(/(*(a,b),C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1, Types.OpOp2.DIV);
		AggUnaryOp v3 = HopRewriteUtils.createAggUnaryOp(v2, Types.AggOp.SUM, Types.Direction.Row);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(a,rowSums(/(b,C))) => rowSums(/(*(a,b),C))
	private static Hop _applyRewrite497(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_1 = (AggUnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.AggOp.SUM || !c_hi_1.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_1.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,rowSums(/(b,C))) => rowSums(/(*(a,b),C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_1, Types.OpOp2.DIV);
		AggUnaryOp v3 = HopRewriteUtils.createAggUnaryOp(v2, Types.AggOp.SUM, Types.Direction.Row);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule rev(%*%(colSums(A),B)) => %*%(colSums(A),B)
	private static Hop _applyRewrite498(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.REV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0_0 = (AggUnaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.AggOp.SUM || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( !(c_hi_0_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rev(%*%(colSums(A),B)) => %*%(colSums(A),B)");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0_0, Types.AggOp.SUM, Types.Direction.Col);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}
}