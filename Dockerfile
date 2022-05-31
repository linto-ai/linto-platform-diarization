FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y g++ zlib1g-dev make automake libtool-bin git autoconf && \
    apt-get install -y subversion libatlas3-base bzip2 wget python2.7 python3 && \
    ln -s /usr/bin/python2.7 /usr/bin/python && \
    ln -s -f bash /bin/sh

RUN git clone https://github.com/kaldi-asr/kaldi /opt/kaldi
RUN cd /opt/kaldi/tools && make && find /opt/kaldi/tools -type f -name "*.o" -delete
RUN cd /opt/kaldi/src && ./configure --shared && make depend && make && find /opt/kaldi/src -type f -name "*.o" -delete
RUN rm -rf /opt/kaldi/.git

WORKDIR /opt/kaldi

# Install python dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/tango4j/Auto-Tuning-Spectral-Clustering.git
CMD [ "sh", "-c", "service ssh start; bash"]
