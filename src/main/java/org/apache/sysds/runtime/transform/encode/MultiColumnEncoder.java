package org.apache.sysds.runtime.transform.encode;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.IndexRange;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MultiColumnEncoder implements Encoder {

    private List<ColumnEncoder> _columnEncoders = null;
    private FrameBlock _meta = null;
    protected static final Log LOG = LogFactory.getLog(MultiColumnEncoder.class.getName());

    public <T extends ColumnEncoder> MultiColumnEncoder(List<T> columnEncoders){
        _columnEncoders = columnEncoders.stream().map(encoder -> encoder).collect(Collectors.toList());
    }

    public MultiColumnEncoder() {
        _columnEncoders = new ArrayList<>();
    }

    @Override
    public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
        try {
            build(in);
            _meta = new FrameBlock(in.getNumColumns(), Types.ValueType.STRING);
            for( ColumnEncoder columnEncoder : _columnEncoders)
                _meta = columnEncoder.getMetaData(_meta);
            for( ColumnEncoder columnEncoder : _columnEncoders)
                columnEncoder.initMetaData(_meta);

            //apply meta data
            for( ColumnEncoder columnEncoder : _columnEncoders)
                out = columnEncoder.apply(in, out);
        }
		catch(Exception ex) {
            LOG.error("Failed transform-encode frame with \n" + this);
            throw ex;
        }

		return out;
    }

    @Override
    public void build(FrameBlock in) {
        for( ColumnEncoder columnEncoder : _columnEncoders)
            columnEncoder.build(in);
    }

    @Override
    public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
        try {
            for( ColumnEncoder columnEncoder : _columnEncoders)
                out = columnEncoder.apply(in, out);
        }
        catch(Exception ex) {
            LOG.error("Failed to transform-apply frame with \n" + this);
            throw ex;
        }
        return out;
    }

    @Override
    public FrameBlock getMetaData(FrameBlock out) {
        if( _meta != null )
            return _meta;
        for( ColumnEncoder columnEncoder : _columnEncoders)
            columnEncoder.getMetaData(out);
        return out;
    }

    @Override
    public void initMetaData(FrameBlock meta) {
        for( ColumnEncoder columnEncoder : _columnEncoders)
            columnEncoder.initMetaData(meta);
    }

    @Override
    public void prepareBuildPartial() {
        for( Encoder encoder : _columnEncoders )
            encoder.prepareBuildPartial();
    }

    @Override
    public void buildPartial(FrameBlock in) {
        for( Encoder encoder : _columnEncoders )
            encoder.buildPartial(in);
    }

    @Override
    public MatrixBlock getColMapping(FrameBlock meta, MatrixBlock out) {
        List<ColumnEncoderDummycode> dc = getColumnEncoders(ColumnEncoderDummycode.class);
        if(!dc.isEmpty()){
            for (Encoder encoder: dc)
                out = encoder.getColMapping(meta,out);
        }else{
            for(int i=0; i<out.getNumRows(); i++) {
                out.quickSetValue(i, 0, i+1);
                out.quickSetValue(i, 1, i+1);
                out.quickSetValue(i, 2, i+1);
            }
        }
        return out;
    }

    @Override
    public void updateIndexRanges(long[] beginDims, long[] endDims) {
        _columnEncoders.forEach(encoder -> encoder.updateIndexRanges(beginDims, endDims));
    }


    public List<ColumnEncoder> getColumnEncoders(){
        return _columnEncoders;
    }

    public List<ColumnEncoder> getEncodersForID(int colID){
        return _columnEncoders.stream().filter(encoder -> encoder._colID == colID).collect(Collectors.toList());
    }


    public MultiColumnEncoder subRangeEncoder(IndexRange ixRange){
        List<ColumnEncoder> encoders = new ArrayList<>();
        for(long i = ixRange.colStart; i < ixRange.colEnd; i++){
            encoders.addAll(getEncodersForID((int) i));
        }
        return new MultiColumnEncoder(encoders);
    }


    public <T extends  ColumnEncoder> MultiColumnEncoder subRangeEncoder(IndexRange ixRange, Class<T> type){
        List<ColumnEncoder> encoders = new ArrayList<>();
        for(long i = ixRange.colStart; i < ixRange.colEnd; i++){
            encoders.add(getColumnEncoder((int) i, type));
        }
        return new MultiColumnEncoder(encoders);
    }

    public void mergeReplace(MultiColumnEncoder multiEncoder) {
        for(ColumnEncoder otherEncoder: multiEncoder._columnEncoders){
            ColumnEncoder encoder = getColumnEncoder(otherEncoder._colID, otherEncoder.getClass());
            if(encoder != null){
                _columnEncoders.remove(encoder);
            }
            _columnEncoders.add(otherEncoder);
        }
    }

    public void mergeAt(Encoder other, int row) {
        if(other instanceof MultiColumnEncoder){
            for(ColumnEncoder encoder: ((MultiColumnEncoder) other)._columnEncoders){
                addEncoder(encoder, row);
            }
        }else{
            addEncoder((ColumnEncoder) other, row);
        }
    }

    private void addEncoder(ColumnEncoder encoder, int row){
        //Check if same encoder exists
        ColumnEncoder presentEncoder = getColumnEncoder(encoder._colID, encoder.getClass());
        if(presentEncoder != null){
            presentEncoder.mergeAt(encoder, row);
        }else{
            //Check if CompositeEncoder for this colID exists
            ColumnEncoderComposite presentComposite = getColumnEncoder(encoder._colID, ColumnEncoderComposite.class);
            if(presentComposite != null){
                // if here encoder can never be a CompositeEncoder
                presentComposite.mergeAt(encoder, row);
            }else{
                if(encoder instanceof ColumnEncoderComposite){
                    _columnEncoders.add(encoder);
                }else{
                    List<ColumnEncoder> list = new ArrayList<>();
                    list.add(encoder);
                    _columnEncoders.add(new ColumnEncoderComposite(list));
                }
            }
        }
    }

    public int getNumExtraCols(){
        List<ColumnEncoderDummycode> dc = getColumnEncoders(ColumnEncoderDummycode.class);
        if(dc.isEmpty()){
           return 0;
        }
        return dc.stream().map(ColumnEncoderDummycode::getNumCols).mapToInt(i->i).max().getAsInt();
    }


    public <T extends ColumnEncoder> boolean containsEncoderForID(int colID, Class<T> type){
        return getColumnEncoders(type).stream().anyMatch(encoder -> encoder.getColID() == colID);
    }

    public <T extends ColumnEncoder> List<T> getColumnEncoders(Class<T> type){
        // TODO cache results for faster access
        List<T> ret = new ArrayList<>();
        for(ColumnEncoder encoder: _columnEncoders){
            if (encoder.getClass().equals(ColumnEncoderComposite.class)){
                encoder = ((ColumnEncoderComposite) encoder).getEncoder(type);
            }
            if (encoder.getClass().equals(type)){
                ret.add((T) encoder);
            }
        }
        return ret;
    }

    public <T extends ColumnEncoder> T getColumnEncoder(int colID, Class<T> type){
        for(T encoder: getColumnEncoders(type)){
            if(encoder._colID == colID){
                return encoder;
            }
        }
        return null;
    }

    public <T extends ColumnEncoder, E> List<E> getFromAll(Class<T> type, Function<? super T, ? extends  E> mapper){
        return getColumnEncoders(type).stream().map(mapper).collect(Collectors.toList());
    }

    public <T extends ColumnEncoder> int[] getFromAllIntArray(Class<T> type, Function<? super T, ? extends  Integer> mapper){
        return getFromAll(type, mapper).stream().mapToInt(i->i).toArray();
    }

    public <T extends ColumnEncoder> double[] getFromAllDoubleArray(Class<T> type, Function<? super T, ? extends  Double> mapper){
        return getFromAll(type, mapper).stream().mapToDouble(i->i).toArray();
    }

    public <T extends ColumnEncoder, E> void applyToAll(Class<T> type, Consumer<? super T> function){
        getColumnEncoders(type).forEach(function);
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(_columnEncoders.size());
        for(ColumnEncoder columnEncoder : _columnEncoders) {
            out.writeByte(EncoderFactory.getEncoderType(columnEncoder));
            columnEncoder.writeExternal(out);
        }
        out.writeBoolean(_meta != null);
        if(_meta != null)
            _meta.write(out);
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        int encodersSize = in.readInt();
        _columnEncoders = new ArrayList<>();
        for(int i = 0; i < encodersSize; i++) {
            ColumnEncoder columnEncoder = EncoderFactory.createInstance(in.readByte());
            columnEncoder.readExternal(in);
            _columnEncoders.add(columnEncoder);
        }
        if (in.readBoolean()) {
            FrameBlock meta = new FrameBlock();
            meta.readFields(in);
            _meta = meta;
        }
    }


}
