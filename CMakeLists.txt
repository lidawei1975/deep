cmake_minimum_required(VERSION 2.8)





if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

project(deep_picker VERSION 1.2)

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

# Only build for webserver

# voigt convolution auxiliary tool
add_executable(voigt_conv main_voigt_conv.cpp $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(voigt_conv libcerf)

# simulate 2D spectrum and draw contour plot auxiliary tool
add_executable(simulate_2d main_simulate_2d.cpp spectrum_prediction.cpp contour.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(simulate_2d libcerf)

# 1d plotting
add_executable(plot_spectrum_1d main_plot_1d.cpp spectrum_io_1d.cpp commandline.cpp jsoncpp.cpp)

# 1D database browse tool
add_executable(db_1d main_1d_database.cpp db_1d.cpp commandline.cpp jsoncpp.cpp)

# 1D database list sanctuary check tool
add_executable(db_1d_check main_1d_check.cpp commandline.cpp jsoncpp.cpp)






# 1D phase correction
add_executable(phase_1d main_phasing_1d.cpp spectrum_fwhh_1d.cpp fwhh_dnn.cpp  phase_dnn.cpp dnn_base.cpp spectrum_io_1d.cpp spectrum_phasing_1d.cpp spectrum_baseline_1d.cpp cubic_spline.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D phase correction
add_executable(phase main_phasing.cpp spectrum_io.cpp spectrum_phasing.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 1D fid processing
add_executable(fid_1d main_fid_1d.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D fid processing
add_executable(fid main_fid.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 1D baseline correction
add_executable(baseline_1d main_baseline_1d.cpp spectrum_baseline_1d.cpp kiss_fft.c spectrum_io_1d.cpp cubic_spline.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_include_directories(baseline_1d PRIVATE ${EIGEN3_INCLUDE_DIR})


# 1D fwhh estimation
add_executable(fwhh_1d main_fwhh_1d.cpp spectrum_fwhh_1d.cpp spectrum_io_1d.cpp fwhh_dnn.cpp dnn_base.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D fwhh estimation
add_executable(fwhh_2d main_fwhh.cpp spectrum_fwhh.cpp spectrum_io.cpp fwhh_dnn.cpp dnn_base.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D peak matching of two peak files
add_executable(peak_match main_peak_match.cpp  $<TARGET_OBJECTS:commandline> )

# 1d peak picking
add_executable(deep_picker_1d main_peak_picking_1d.cpp spectrum_pick_1d.cpp spectrum_fwhh_1d.cpp fwhh_dnn.cpp spectrum_io_1d.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(deep_picker_1d libcerf)

# 2D peak picking
add_executable(deep_picker main_peak_picking.cpp spectrum_io.cpp spectrum_fwhh.cpp spectrum_pick.cpp fwhh_dnn.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline>)
target_link_libraries(deep_picker libcerf)

# 2D simple peak picking
add_executable(simple_picker main_simple_picking.cpp spectrum_io.cpp spectrum_simple_picking.cpp $<TARGET_OBJECTS:commandline>)


# build a alternative peak fitting program for 2D, using my own minimization algorithm instead of Ceres
add_executable(voigt_fit_2 main_peak_fitting.cpp spectrum_io.cpp spectrum_fit.cpp kiss_fft.c lmminimizer.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_include_directories(voigt_fit_2 PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(voigt_fit_2 libcerf)
target_compile_definitions(voigt_fit_2 PRIVATE LMMIN)


# fitting part, need Ceres
find_package(Ceres)

# peak fitting programs for both 1D and 2D
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

  # 1D DOSY fitting
  add_executable(dosy main_dosy_fit.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(dosy PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(dosy PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(dosy ${CERES_LIBRARIES})
endif()
