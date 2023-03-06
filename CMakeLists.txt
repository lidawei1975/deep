cmake_minimum_required(VERSION 2.8)

project(deep_picker)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})



add_library(dnn OBJECT  dnn_picker.cpp clique.cpp ann_data.c)
add_library(dnn_d OBJECT dnn_picker.cpp clique.cpp ann_data.c )
target_compile_definitions(dnn_d PUBLIC LDW_DEBUG)
add_library(commandline OBJECT commandline.cpp )
add_library(jsoncpp OBJECT jsoncpp.cpp )
add_library(libcerf STATIC erfcx.c err_fcts.c im_w_of_x.c w_of_z.c)


# voigt convolution auxiliary tool
add_executable(voigt_conv main_voigt_conv.cpp $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(voigt_conv libcerf)

# 2D peak picking
add_executable(deep_picker main_peak_picking.cpp spectrum_io.cpp spectrum_pick.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline> )
target_link_libraries(deep_picker libcerf)

add_executable(simple_picker main_simple_picking.cpp spectrum_io.cpp spectrum_simple_picking.cpp $<TARGET_OBJECTS:commandline> )



# 1d peak picking
add_executable(deep_picker_1d main_peak_picking_1d.cpp spectrum_pick_1d.cpp spectrum_io_1d.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(deep_picker_1d libcerf)


# 1d phase correction
add_executable(phase_1d main_phasing_1d_v2.cpp phasing_dnn.cpp spectrum_io_1d.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 1d fwhh estimation
add_executable(fwhh_1d main_fwhh_1d.cpp spectrum_fwhh_1d.cpp spectrum_io_1d.cpp fwhh_dnn.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)


# fitting part, need Ceres

find_package(Ceres)

if(Ceres_FOUND)
    
 
    
    add_executable(voigt_fit main_peak_fitting.cpp spectrum_io.cpp spectrum_fit.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
    target_include_directories(voigt_fit PRIVATE  ${CERES_INCLUDE_DIRS})
    target_include_directories(voigt_fit PRIVATE  ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(voigt_fit ${CERES_LIBRARIES})
    target_link_libraries(voigt_fit libcerf)


    add_executable(peak_fit_1d main_peak_fitting_1d.cpp spectrum_io_1d.cpp spectrum_fit_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
    target_include_directories(peak_fit_1d PRIVATE  ${CERES_INCLUDE_DIRS})
    target_include_directories(peak_fit_1d PRIVATE  ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(peak_fit_1d ${CERES_LIBRARIES})
    target_link_libraries(peak_fit_1d libcerf)


endif()

