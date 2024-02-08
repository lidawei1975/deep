cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

project(deep_picker VERSION 2.0)

configure_file(DeepConfig.h.in DeepConfig.h)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories("${PROJECT_BINARY_DIR}")

add_library(dnn OBJECT dnn_picker.cpp clique.cpp ann_data.c dnn_base.cpp cubic_spline.cpp)
add_library(commandline OBJECT commandline.cpp)
add_library(jsoncpp OBJECT jsoncpp.cpp)
add_library(libcerf STATIC erfcx.c err_fcts.c im_w_of_x.c w_of_z.c)

# voigt convolution auxiliary tool
add_executable(voigt_conv main_voigt_conv.cpp $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(voigt_conv libcerf)

# 1d peak picking
add_executable(deep_picker_1d main_peak_picking_1d.cpp spectrum_pick_1d.cpp spectrum_fwhh_1d.cpp fwhh_dnn.cpp spectrum_io_1d.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(deep_picker_1d libcerf)

# 1d phase correction
add_executable(phase_1d main_phasing_1d.cpp spectrum_fwhh_1d.cpp fwhh_dnn.cpp dnn_base.cpp spectrum_io_1d.cpp spectrum_phasing_1d.cpp spectrum_baseline_1d.cpp cubic_spline.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 1d fid processing
add_executable(fid_1d main_fid_1d.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 1d baseline correction
add_executable(baseline_1d main_baseline_1d.cpp spectrum_baseline_1d.cpp kiss_fft.c spectrum_io_1d.cpp cubic_spline.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_include_directories(baseline_1d PRIVATE ${EIGEN3_INCLUDE_DIR})

# 1d plotting
add_executable(plot_spectrum_1d main_plot_1d.cpp spectrum_io_1d.cpp commandline.cpp jsoncpp.cpp) 

# 1d fwhh estimation
add_executable(fwhh_1d main_fwhh_1d.cpp spectrum_fwhh_1d.cpp spectrum_io_1d.cpp fwhh_dnn.cpp dnn_base.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2d fwhh estimation
add_executable(fwhh_2d main_fwhh.cpp spectrum_fwhh.cpp spectrum_io.cpp fwhh_dnn.cpp dnn_base.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D peak picking
add_executable(deep_picker main_peak_picking.cpp spectrum_io.cpp spectrum_fwhh.cpp spectrum_pick.cpp fwhh_dnn.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline>)
target_link_libraries(deep_picker libcerf)

add_executable(simple_picker main_simple_picking.cpp spectrum_io.cpp spectrum_simple_picking.cpp $<TARGET_OBJECTS:commandline>)

# 1D database query. There are here because we need to run DP part of the code.
# db_match for colamr1d
add_executable(db_1d main_1d_database.cpp db_1d.cpp commandline.cpp jsoncpp.cpp)

# fitting part, need Ceres
find_package(Ceres)

if(Ceres_FOUND)
  add_executable(db_match_1d main_1d.cpp db_match_1d.cpp db_1d.cpp hungary.cpp commandline.cpp jsoncpp.cpp spectrum_io_1d.cpp  fwhh_dnn.cpp spectrum_fwhh_1d.cpp spectrum_pick_1d.cpp spectrum_fit_1d.cpp kiss_fft.c $<TARGET_OBJECTS:dnn>)
  target_include_directories(db_match_1d PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(db_match_1d PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(db_match_1d ${CERES_LIBRARIES})
  target_link_libraries(db_match_1d libcerf)

  add_executable(db_match_global_1d main_global_1d.cpp db_match_global_1d.cpp hungary.cpp commandline.cpp jsoncpp.cpp spectrum_io_1d.cpp)
  target_include_directories(db_match_global_1d PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(db_match_global_1d PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(db_match_global_1d ${CERES_LIBRARIES})
  target_link_libraries(db_match_global_1d libcerf)
endif()



if(Ceres_FOUND)
  add_executable(voigt_fit main_peak_fitting.cpp spectrum_io.cpp spectrum_fit.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(voigt_fit PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(voigt_fit PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(voigt_fit ${CERES_LIBRARIES})
  target_link_libraries(voigt_fit libcerf)

  add_executable(peak_fit_1d main_peak_fitting_1d.cpp spectrum_io_1d.cpp spectrum_fit_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(peak_fit_1d PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(peak_fit_1d PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(peak_fit_1d ${CERES_LIBRARIES})
  target_link_libraries(peak_fit_1d libcerf)
endif()
