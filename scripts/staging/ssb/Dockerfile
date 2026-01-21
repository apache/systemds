# Help: https://docs.docker.com/compose/gettingstarted/#step-1-set-up
# https://docs.docker.com/build/building/multi-stage/
# Star Schema Benchmark data set generator (ssb-dbgen): 
# https://github.com/eyalroz/ssb-dbgen.git

# We use multi-stage method.

# First create the datagen docker container
FROM alpine:latest AS datagen
ENV SCALE = 0.1
RUN apk update
RUN apk add git gcc cmake make musl-dev
RUN git clone https://github.com/eyalroz/ssb-dbgen.git --depth 1
# Build the generator
WORKDIR /ssb-dbgen
RUN echo "build and generate data with datagen."
RUN cmake -B ./build && cmake --build ./build
# Run the generator (with -s )
RUN build/dbgen -b dists.dss -v -s $SCALE
RUN mkdir -p ../data
RUN mv *.tbl ../data

# Second: use the systemds docker container
# And execute a selected query. 
FROM apache/systemds:latest
WORKDIR /input
WORKDIR /systemds
ENV QUERY=q1_1
WORKDIR /systemds
COPY queries queries
COPY --from=datagen data data 
CMD ["queries/$QUERY.dml","data"]

#TODO: Currently only accepting one query each. To expand to accept more queries.
#COPY helper.sh helper.sh
#RUN chmod u+x helper.sh
# COPY host_data_dir container_data_dir
#COPY data/very_small_s0_01_dataset data

#CMD if [ "$QUERY" = "all" ]; then \
#      ["./helper.sh", "A", "B"]; \
#    else \
#      ["queries/$QUERY.dml","data"]; \
#    fi 
