FROM ros:noetic

RUN apt-get update && apt-get install -y vim mesa-utils ros-noetic-pinocchio python3-pip
RUN pip install scipy numpy

