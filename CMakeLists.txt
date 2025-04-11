cmake_minimum_required(VERSION 3.5)





if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

project(deep_picker VERSION 1.2)

configure_file(DeepConfig.h.in DeepConfig.h)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories("${PROJECT_BINARY_DIR}")

add_library(dnn OBJECT dnn_picker.cpp clique.cpp ann_data.c dnn_base.cpp cubic_spline.cpp)
add_library(commandline OBJECT commandline.cpp)
add_library(jsoncpp OBJECT jsoncpp.cpp)
add_library(libcerf STATIC erfcx.c err_fcts.c im_w_of_x.c w_of_z.c)




# 1D phase correction
add_executable(phase_1d main_phasing_1d.cpp spectrum_fwhh_1d.cpp fwhh_dnn.cpp  phase_dnn.cpp dnn_base.cpp  spectrum_phasing_1d.cpp spectrum_baseline_1d.cpp cubic_spline.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D phase correction
add_executable(phase main_phasing.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c spectrum_phasing.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 1D fid processing
add_executable(fid_1d main_fid_1d.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D fid processing
add_executable(fid main_fid.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 1D baseline correction
add_executable(baseline_1d main_baseline_1d.cpp spectrum_baseline_1d.cpp kiss_fft.c fid_1d.cpp kiss_fft.c cubic_spline.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_include_directories(baseline_1d PRIVATE ${EIGEN3_INCLUDE_DIR})


# 1D fwhh estimation
add_executable(fwhh_1d main_fwhh_1d.cpp spectrum_fwhh_1d.cpp fid_1d.cpp kiss_fft.c fwhh_dnn.cpp dnn_base.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D fwhh estimation
add_executable(fwhh_2d main_fwhh.cpp spectrum_fwhh.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c fwhh_dnn.cpp dnn_base.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

# 2D peak matching of two peak files
add_executable(peak_match main_peak_match.cpp peak_manipulation.cpp $<TARGET_OBJECTS:commandline> )

# 1d peak picking
add_executable(deep_picker_1d main_peak_picking_1d.cpp spectrum_pick_1d.cpp spectrum_fwhh_1d.cpp fwhh_dnn.cpp  fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(deep_picker_1d libcerf)

# 2D peak picking
add_executable(deep_picker main_peak_picking.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c spectrum_fwhh.cpp spectrum_pick.cpp fwhh_dnn.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(deep_picker libcerf)

# 2D simple peak picking
add_executable(simple_picker main_simple_picking.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c spectrum_simple_picking.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)


# build a alternative peak fitting program for 2D, using my own minimization algorithm instead of Ceres
add_executable(voigt_fit_2 main_peak_fitting.cpp fid_2d.cpp fid_1d.cpp spectrum_fit.cpp cost_functions.cpp kiss_fft.c lmminimizer.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_include_directories(voigt_fit_2 PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(voigt_fit_2 libcerf)
target_compile_definitions(voigt_fit_2 PRIVATE LMMIN)

# build voigt_fit_2 with openmp support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_executable(voigt_fit_2_omp main_peak_fitting.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c spectrum_fit.cpp cost_functions.cpp kiss_fft.c lmminimizer.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(voigt_fit_2_omp PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(voigt_fit_2_omp libcerf)
  target_compile_definitions(voigt_fit_2_omp PRIVATE LMMIN)
  target_compile_options(voigt_fit_2_omp PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(voigt_fit_2_omp  OpenMP::OpenMP_CXX)
  target_compile_definitions(voigt_fit_2_omp PRIVATE USE_OPENMP)
endif()

# fitting part, need Ceres
find_package(Ceres)

# peak fitting programs for both 1D and 2D
if(Ceres_FOUND)
  add_executable(voigt_fit main_peak_fitting.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c spectrum_fit.cpp cost_functions.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(voigt_fit PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(voigt_fit PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(voigt_fit ${CERES_LIBRARIES})
  target_link_libraries(voigt_fit libcerf)

  add_executable(peak_fit_1d main_peak_fitting_1d.cpp fid_1d.cpp kiss_fft.c spectrum_fit_1d.cpp cost_functions.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(peak_fit_1d PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(peak_fit_1d PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(peak_fit_1d ${CERES_LIBRARIES})
  target_link_libraries(peak_fit_1d libcerf)

  # 1D DOSY fitting
  add_executable(dosy main_dosy_fit.cpp dosy.cpp peak_manipulation.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(dosy PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(dosy PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(dosy ${CERES_LIBRARIES})
endif()

#peak fitting program for 2D, using ceres and openmp
if(Ceres_FOUND AND OpenMP_CXX_FOUND)
  add_executable(voigt_fit_omp main_peak_fitting.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c spectrum_fit.cpp cost_functions.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(voigt_fit_omp PRIVATE ${CERES_INCLUDE_DIRS})
  target_include_directories(voigt_fit_omp PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(voigt_fit_omp ${CERES_LIBRARIES})
  target_link_libraries(voigt_fit_omp libcerf)
  target_compile_options(voigt_fit_omp PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(voigt_fit_omp OpenMP::OpenMP_CXX)
  target_compile_definitions(voigt_fit_omp PRIVATE USE_OPENMP)
endif()


# test programs
configure_file(test/test_dp.pl ${CMAKE_CURRENT_BINARY_DIR}/test_dp.pl COPYONLY)
configure_file(test/test_vf.pl ${CMAKE_CURRENT_BINARY_DIR}/test_vf.pl COPYONLY)
configure_file(test/test_vf2.pl ${CMAKE_CURRENT_BINARY_DIR}/test_vf2.pl COPYONLY)
configure_file(test/test_vf2_omp.pl ${CMAKE_CURRENT_BINARY_DIR}/test_vf2_omp.pl COPYONLY)
configure_file(test/plane001.ft2 ${CMAKE_CURRENT_BINARY_DIR}/plane001.ft2 COPYONLY)
configure_file(test/plane002.ft2 ${CMAKE_CURRENT_BINARY_DIR}/plane002.ft2 COPYONLY)
configure_file(test/plane003.ft2 ${CMAKE_CURRENT_BINARY_DIR}/plane003.ft2 COPYONLY)
# test programs and files for peak_match
configure_file(test/test_peak_match.pl ${CMAKE_CURRENT_BINARY_DIR}/test_peak_match.pl COPYONLY)
configure_file(test/ass.list ${CMAKE_CURRENT_BINARY_DIR}/ass.list COPYONLY)
configure_file(test/peaks_ass.tab ${CMAKE_CURRENT_BINARY_DIR}/peaks_ass.tab COPYONLY)
# test programs for fid and phase (2D)
configure_file(test/test_fid_phase.pl ${CMAKE_CURRENT_BINARY_DIR}/test_fid_phase.pl COPYONLY)
configure_file(test/ser ${CMAKE_CURRENT_BINARY_DIR}/ser COPYONLY)
configure_file(test/acqus ${CMAKE_CURRENT_BINARY_DIR}/acqus COPYONLY)
configure_file(test/acqu2s ${CMAKE_CURRENT_BINARY_DIR}/acqu2s COPYONLY)


enable_testing()

add_test(NAME test_deep_picker COMMAND /usr/bin/env perl test_dp.pl)

if(Ceres_FOUND)
  add_test(NAME test_voigt_fit COMMAND /usr/bin/env perl test_vf.pl)
  set_tests_properties(test_voigt_fit PROPERTIES DEPENDS test_deep_picker)

  add_test(NAME test_voigt_fit_pseudo3D COMMAND /usr/bin/env perl test_vf2.pl)
  set_tests_properties(test_voigt_fit_pseudo3D PROPERTIES DEPENDS test_voigt_fit)
endif()

if(Ceres_FOUND AND OpenMP_CXX_FOUND)
  add_test(NAME test_voigt_fit_pseudo3D_omp COMMAND /usr/bin/env perl test_vf2_omp.pl)
  set_tests_properties(test_voigt_fit_pseudo3D_omp PROPERTIES DEPENDS test_voigt_fit)
endif()

add_test(NAME test_peak_match COMMAND /usr/bin/env perl test_peak_match.pl)


add_executable(test_functors test_functors.cpp cost_functions.cpp kiss_fft.c)
target_compile_definitions(test_functors PRIVATE LMMIN)
target_link_libraries(test_functors libcerf)
add_test(NAME test_functors COMMAND test_functors)

add_test(NAME test_fid_phase COMMAND /usr/bin/env perl test_fid_phase.pl)