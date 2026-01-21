# Help: https://docs.docker.com/compose/gettingstarted/#step-1-set-up
# Star Schema Benchmark data set generator (ssb-dbgen): 
# https://github.com/eyalroz/ssb-dbgen.git

FROM alpine:latest
ENV SCALE = 0.1
WORKDIR /ssb-dbgen
ENV query_no = "0" #run the query number.
RUN apk update
RUN apk add git gcc cmake make musl-dev
RUN git clone https://github.com/eyalroz/ssb-dbgen.git --depth 1
# Build the generator
WORKDIR /ssb-dbgen/ssb-dbgen
RUN echo "build and generate data with datagen."
RUN cmake -B ./build && cmake --build ./build
## Run the generator (with -s )
RUN build/dbgen -b dists.dss -v -s $scale

# TODO: Some code for copying the data files into a volume to
#make it available for other container and the host.
#This is working:

FROM apache/systemds:latest
WORKDIR /input
WORKDIR /systemds
ENV QUERY=q1_1
WORKDIR /systemds
COPY queries queries
# COPY host_data_dir container_data_dir
COPY data/very_small_s0_01_dataset data 
CMD ["queries/$QUERY.dml","data"]

#TODO: Currently only accepting one query each.
#FROM apache/systemds:latest
#WORKDIR /input
#ENV QUERY=q1_1
#WORKDIR /systemds
#COPY queries queries
#COPY helper.sh helper.sh
#RUN chmod u+x helper.sh
# COPY host_data_dir container_data_dir
#COPY data/very_small_s0_01_dataset data

#CMD if [ "$QUERY" = "all" ]; then \
#      ["./helper.sh", "A", "B"]; \
#    else \
#      ["queries/$QUERY.dml","data"]; \
#    fi 
