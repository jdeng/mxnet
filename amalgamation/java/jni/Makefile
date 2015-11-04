CXX = ~/android/arm-linux-androideabi-4.9/bin/arm-linux-androideabi-g++
CXXFLAGS = -std=c++11 -fPIC -shared  -O3
all: ../../mxnet_predict-all.cc Predictor.cc
	$(CXX) $(CXXFLAGS) -I ../.. -o mxnet_predict.so predictor.cc
