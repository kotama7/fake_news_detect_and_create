FROM python:3.10.11

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
    
RUN apt-get -y install git

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y openssh-server

# SSHの設定
RUN mkdir /var/run/sshd
RUN echo 'root:Docker!' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN pip install --upgrade pip==23.2.1
RUN pip install --upgrade setuptools

WORKDIR /root/workspaces

COPY . .

EXPOSE 22
EXPOSE 8000

RUN pip install -r requirements.txt

CMD ["/usr/sbin/sshd", "-D"]