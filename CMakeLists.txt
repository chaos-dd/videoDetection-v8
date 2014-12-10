
PROJECT (VD8)

SET(CMAKE_CXX_COMPILER "/usr/bin/clang++")

SET(CMAKE_CXX_FLAGS "-std=c++11 -O0 -ggdb ")
SET(SRC_LIST colorFeat.cpp video_det_param.cpp video_proc_utilities.cpp video_det_utilities.cpp video_det_engine.cpp video_det_param.cpp)

SET(SRC_LIST ${SRC_LIST} main.cpp)


ADD_EXECUTABLE(main ${SRC_LIST})

TARGET_LINK_LIBRARIES(main opencv_core opencv_highgui opencv_imgproc opencv_features2d flann_s boost_system boost_filesystem)


