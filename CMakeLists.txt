cmake_minimum_required(VERSION 3.5)

project(deep_picker)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(dnn OBJECT  dnn_picker.cpp clique.cpp ann_data.c)
add_library(commandline OBJECT commandline.cpp )
add_library(jsoncpp OBJECT jsoncpp.cpp )
add_library(libcerf STATIC erfcx.c err_fcts.c im_w_of_x.c w_of_z.c)



# 2D peak picking
add_executable(deep_picker main_peak_picking.cpp spectrum_io.cpp spectrum_pick.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline> )
target_link_libraries(deep_picker libcerf)

find_package(Ceres)
if(Ceres_FOUND)
    add_executable(voigt_fit main_peak_fitting.cpp spectrum_io.cpp spectrum_fit.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
    target_include_directories(voigt_fit PRIVATE  ${CERES_INCLUDE_DIRS})
    target_include_directories(voigt_fit PRIVATE  ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(voigt_fit ${CERES_LIBRARIES})
    target_link_libraries(voigt_fit libcerf)
endif()
