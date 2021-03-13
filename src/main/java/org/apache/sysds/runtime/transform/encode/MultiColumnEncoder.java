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
import java.util.HashSet;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MultiColumnEncoder implements Encoder {

    private List<ColumnEncoderComposite> _columnEncoders = null;
    private FrameBlock _meta = null;
    protected static final Log LOG = LogFactory.getLog(MultiColumnEncoder.class.getName());

    public MultiColumnEncoder(List<ColumnEncoderComposite> columnEncoders){
        _columnEncoders = columnEncoders;
    }

    public MultiColumnEncoder() {
        _columnEncoders = new ArrayList<>();
    }

    public MatrixBlock encode(FrameBlock in) {
        MatrixBlock out = null;
        try {
            build(in);
            _meta = new FrameBlock(in.getNumColumns(), Types.ValueType.STRING);
            for( ColumnEncoder columnEncoder : _columnEncoders)
                _meta = columnEncoder.getMetaData(_meta);
            for( ColumnEncoder columnEncoder : _columnEncoders)
                columnEncoder.initMetaData(_meta);
            resolveInterEncoderDependencies();
            //apply meta data
            out = apply(in);
        }
		catch(Exception ex) {
            LOG.error("Failed transform-encode frame with \n" + this);
            throw ex;
        }

		return out;
    }

    public void build(FrameBlock in) {
        for( ColumnEncoder columnEncoder : _columnEncoders)
            columnEncoder.build(in);
    }

    public MatrixBlock apply(FrameBlock in) {
        MatrixBlock out = null;
        try {
            resolveInterEncoderDependencies();
            for( ColumnEncoderComposite columnEncoder : _columnEncoders){
                if(out == null)
                    out = columnEncoder.apply(in);
                else
                    out = out.append(columnEncoder.apply(in), null);
            }
        }
        catch(Exception ex) {
            LOG.error("Failed to transform-apply frame with \n" + this);
            throw ex;
        }
        return out;
    }

    @Override
    public FrameBlock getMetaData(FrameBlock meta) {
        if( _meta != null )
            return _meta;
        for( ColumnEncoder columnEncoder : _columnEncoders)
            columnEncoder.getMetaData(meta);
        return meta;
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

    /**
     * Obtain the column mapping of encoded frames based on the passed
     * meta data frame.
     *
     * @param meta meta data frame block
     * @param out output matrix
     * @return matrix with column mapping (one row per attribute)
     */
    public MatrixBlock getColMapping(FrameBlock meta) {
        MatrixBlock out = new MatrixBlock(meta.getNumColumns(), 3, false);
        List<ColumnEncoderDummycode> dc = getColumnEncoders(ColumnEncoderDummycode.class);

        for(int i = 0, ni = 0; i < out.getNumRows(); i++){
            final int colID = i + 1; // 1-based
            int nColID = ni + 1;
            List<ColumnEncoderDummycode> encoder = dc.stream().filter(e -> e.getColID() == colID).collect(Collectors.toList());
            assert encoder.size() <= 1;
            if(encoder.size() == 1){
                ni += meta.getColumnMetadata(i).getNumDistinct();
            }else{
                ni++;
            }
            out.quickSetValue(i, 0, colID);
            out.quickSetValue(i, 1, nColID);
            out.quickSetValue(i, 2, ni);
        }
        return out;
    }

    @Override
    public void updateIndexRanges(long[] beginDims, long[] endDims) {
        _columnEncoders.forEach(encoder -> encoder.updateIndexRanges(beginDims, endDims));
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(_columnEncoders.size());
        for(ColumnEncoder columnEncoder : _columnEncoders) {
            out.writeInt(columnEncoder._colID);
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
            int colID = in.readInt();
            ColumnEncoderComposite columnEncoder = new ColumnEncoderComposite();
            columnEncoder.readExternal(in);
            columnEncoder.setColID(colID);
            _columnEncoders.add(columnEncoder);
        }
        if (in.readBoolean()) {
            FrameBlock meta = new FrameBlock();
            meta.readFields(in);
            _meta = meta;
        }
    }


    public <T extends ColumnEncoder> List<T> getColumnEncoders(Class<T> type){
        // TODO cache results for faster access
        List<T> ret = new ArrayList<>();
        for(ColumnEncoder encoder: _columnEncoders){
            if (encoder.getClass().equals(ColumnEncoderComposite.class) && type != ColumnEncoderComposite.class){
                encoder = ((ColumnEncoderComposite) encoder).getEncoder(type);
            }
            if (encoder != null && encoder.getClass().equals(type)){
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

    public <T extends ColumnEncoder> double[] getFromAllDoubleArray(Class<T> type, Function<? super T, ? extends  Double> mapper) {
        return getFromAll(type, mapper).stream().mapToDouble(i -> i).toArray();
    }


    public List<ColumnEncoderComposite> getColumnEncoders(){
        return _columnEncoders;
    }

    public List<ColumnEncoderComposite> getCompositeEncodersForID(int colID){
        return _columnEncoders.stream().filter(encoder -> encoder._colID == colID).collect(Collectors.toList());
    }

    public <T extends ColumnEncoder> List<Class<T>> getEncoderTypes(int colID){
        HashSet<Class<T>> set = new HashSet<>();
        for(ColumnEncoderComposite encoderComp: _columnEncoders){
            if(encoderComp._colID != colID && colID != -1)
                continue;
            for(ColumnEncoder encoder: encoderComp.getEncoders()){
                set.add((Class<T>) encoder.getClass());
            }
        }
        return new ArrayList<>(set);
    }

    public <T extends ColumnEncoder> List<Class<T>> getEncoderTypes(){
        return getEncoderTypes(-1);
    }

    public int getNumExtraCols(){
        List<ColumnEncoderDummycode> dc = getColumnEncoders(ColumnEncoderDummycode.class);
        if(dc.isEmpty()){
            return 0;
        }
        return dc.stream().map(ColumnEncoderDummycode::getDomainSize).mapToInt(i->i).sum() - dc.size();
    }

    public <T extends ColumnEncoder> boolean containsEncoderForID(int colID, Class<T> type){
        return getColumnEncoders(type).stream().anyMatch(encoder -> encoder.getColID() == colID);
    }

    public <T extends ColumnEncoder, E> void applyToAll(Class<T> type, Consumer<? super T> function){
        getColumnEncoders(type).forEach(function);
    }


    public MultiColumnEncoder subRangeEncoder(IndexRange ixRange){
        List<ColumnEncoderComposite> encoders = new ArrayList<>();
        for(long i = ixRange.colStart; i < ixRange.colEnd; i++){
            encoders.addAll(getCompositeEncodersForID((int) i));
        }
        return new MultiColumnEncoder(encoders);
    }


    public <T extends  ColumnEncoder> MultiColumnEncoder subRangeEncoder(IndexRange ixRange, Class<T> type){
        List<T> encoders = new ArrayList<>();
        for(long i = ixRange.colStart; i < ixRange.colEnd; i++){
            encoders.add(getColumnEncoder((int) i, type));
        }
        if(type.equals(ColumnEncoderComposite.class))
            return new MultiColumnEncoder((List<ColumnEncoderComposite>) encoders);
        else
            return new MultiColumnEncoder(encoders.stream().map(ColumnEncoderComposite::new).collect(Collectors.toList()));
    }

    public void mergeReplace(MultiColumnEncoder multiEncoder) {
        for(ColumnEncoderComposite otherEncoder: multiEncoder._columnEncoders){
            ColumnEncoderComposite encoder = getColumnEncoder(otherEncoder._colID, otherEncoder.getClass());
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
                    _columnEncoders.add((ColumnEncoderComposite) encoder);
                }else{
                    _columnEncoders.add(new ColumnEncoderComposite(encoder));
                }
            }
        }
    }


    public void resolveInterEncoderDependencies(){
        int domainSize = 0;
        for(ColumnEncoderDummycode dc: getColumnEncoders(ColumnEncoderDummycode.class)){
            domainSize += dc._domainSize - 1;
            for (ColumnEncoder encoder: _columnEncoders.stream().filter(encoder -> encoder._colID > dc._colID).collect(Collectors.toList())){
                encoder.shiftOutCol(dc._domainSize-1);
            }
        }
        for(ColumnEncoderDummycode dc: getColumnEncoders(ColumnEncoderDummycode.class)){
            dc._dummycodedLength = domainSize + dc._clen;
        }
    }


}
