cmake_minimum_required(VERSION 3.5)


project(deep_picker VERSION 1.2)

configure_file(DeepConfig.h.in DeepConfig.h)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install path" FORCE)
endif()


if(EMSCRIPTEN)
  # Allow searching for packages in the host system for header-only libs like Eigen
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

find_package(Eigen3 REQUIRED)

if(EMSCRIPTEN)
  # Restore to ONLY for safety with other dependencies
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories("${PROJECT_BINARY_DIR}")

if(EMSCRIPTEN)
  # WebAssembly build logic
  # Define source files for WASM
  file(GLOB ALL_CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
  file(GLOB ALL_C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

  set(WASM_SOURCES)
  foreach(FILE ${ALL_CPP_FILES})
    get_filename_component(FNAME ${FILE} NAME)
    if(NOT FNAME MATCHES "^test|^main|^db_match" AND NOT FNAME STREQUAL "dosy.cpp")
      list(APPEND WASM_SOURCES ${FILE})
    endif()
  endforeach()
  list(APPEND WASM_SOURCES ${ALL_C_FILES})

  # Create the executable (which Emscripten turns into JS/WASM)
  add_executable(webdp1d_cpp ${WASM_SOURCES})
  
  # Include Eigen3
  target_include_directories(webdp1d_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})

# Compile options and definitions
  target_compile_definitions(webdp1d_cpp PRIVATE WEBASSEMBLY LMMIN)
  target_compile_options(webdp1d_cpp PRIVATE -msimd128 -flto -O3 -w -sMEMORY64)
  
  # Link options
  target_link_options(webdp1d_cpp PRIVATE
    "-sMEMORY64"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=\"webdp1d_cpp\""
    "-sWASM=1"
    "-lembind"
    "-msimd128"
    "-flto"
    "-O3"
    "-sEXPORTED_RUNTIME_METHODS=['HEAPF32','HEAP32','HEAPU32','HEAPU8']"
    "-sEXPORTED_FUNCTIONS=['_malloc','_free']"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sMAXIMUM_MEMORY=16GB"
  )
  
  # Set output name and suffix
  set_target_properties(webdp1d_cpp PROPERTIES OUTPUT_NAME "webdp1d_cpp")
  set_target_properties(webdp1d_cpp PROPERTIES SUFFIX ".js")

  # Custom target for 'make webass'
  add_custom_target(webass DEPENDS webdp1d_cpp)

else()
  # Native build logic
  add_compile_options("$<$<NOT:$<CONFIG:Debug>>:-march=x86-64-v3>")

  add_library(dnn OBJECT dnn_picker.cpp clique.cpp ann_data.c dnn_base.cpp cubic_spline.cpp)
  add_library(commandline OBJECT commandline.cpp)
  add_library(jsoncpp OBJECT jsoncpp.cpp)
  add_library(libcerf STATIC erfcx.c err_fcts.c im_w_of_x.c w_of_z.c)
  add_library(fid_base_lib STATIC fid_base.cpp)

  # Only build for webserver

  # voigt convolution auxiliary tool
  add_executable(voigt_conv main_voigt_conv.cpp $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(voigt_conv libcerf)

  # simulate 2D spectrum and draw contour plot auxiliary tool
  add_executable(simulate_2d main_simulate_2d.cpp spectrum_prediction.cpp contour.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(simulate_2d libcerf)

  # 1d plotting for webserver
  add_executable(plot_spectrum_1d main_plot_1d.cpp fid_1d.cpp commandline.cpp jsoncpp.cpp kiss_fft.c)
  target_link_libraries(plot_spectrum_1d fid_base_lib)

  # 2d plotting for webserver
  add_executable(plot_spectrum_2d main_plot_2d.cpp fid_2d.cpp  kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(plot_spectrum_2d fid_base_lib)



  # 1D phase correction
  add_executable(phase_1d main_phasing_1d.cpp spectrum_fwhh_1d.cpp fwhh_dnn.cpp  phase_dnn.cpp dnn_base.cpp  spectrum_phasing_1d.cpp spectrum_baseline_1d.cpp cubic_spline.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(phase_1d fid_base_lib)

  # 2D phase correction
  add_executable(phase main_phasing.cpp fid_2d.cpp kiss_fft.c spectrum_phasing.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(phase fid_base_lib)

  # 1D fid processing
  add_executable(fid_1d main_fid_1d.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(fid_1d fid_base_lib)

  # 2D fid processing
  add_executable(fid main_fid.cpp fid_2d.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(fid fid_base_lib)

  # 3D fid processing
  add_executable(fid_3d main_fid_3d.cpp fid_3d.cpp lmminimizer.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(fid_3d fid_base_lib)

  # 2D fid and phase correction in one step
  # add_executable(fid_phase main_fid_phasing.cpp fid_2d.cpp fid_1d.cpp kiss_fft.c fid_2d.cpp spectrum_phasing.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)

  # 1D baseline correction
  add_executable(baseline_1d main_baseline_1d.cpp spectrum_baseline_1d.cpp kiss_fft.c fid_1d.cpp kiss_fft.c cubic_spline.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(baseline_1d fid_base_lib)
  target_include_directories(baseline_1d PRIVATE ${EIGEN3_INCLUDE_DIR})

  # 1D, urea signal removal
  add_executable(urea main_urea.cpp  kiss_fft.c fid_1d.cpp kiss_fft.c cubic_spline.cpp lmminimizer.cpp cost_functions.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(urea PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(urea fid_base_lib)
  target_link_libraries(urea libcerf)
  target_compile_definitions(urea PRIVATE LMMIN)

  # 1D fwhh estimation
  add_executable(fwhh_1d main_fwhh_1d.cpp spectrum_fwhh_1d.cpp fid_1d.cpp kiss_fft.c fwhh_dnn.cpp dnn_base.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(fwhh_1d fid_base_lib)

  # 2D fwhh estimation
  add_executable(fwhh_2d main_fwhh.cpp spectrum_fwhh.cpp fid_2d.cpp kiss_fft.c  fwhh_dnn.cpp dnn_base.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(fwhh_2d fid_base_lib)

  # 2D peak matching of two peak files
  add_executable(peak_match main_peak_match.cpp peak_manipulation.cpp $<TARGET_OBJECTS:commandline> )

  # 1d peak picking
  add_executable(deep_picker_1d main_peak_picking_1d.cpp spectrum_pick_1d.cpp spectrum_fwhh_1d.cpp fwhh_dnn.cpp fid_1d.cpp kiss_fft.c $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(deep_picker_1d fid_base_lib)
  target_link_libraries(deep_picker_1d libcerf)

  # 2D peak picking
  add_executable(deep_picker main_peak_picking.cpp fid_2d.cpp kiss_fft.c spectrum_fwhh.cpp spectrum_pick.cpp fwhh_dnn.cpp $<TARGET_OBJECTS:dnn> $<TARGET_OBJECTS:commandline>  $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(deep_picker fid_base_lib)
  target_link_libraries(deep_picker libcerf)

  # 2D simple peak picking
  add_executable(simple_picker main_simple_picking.cpp fid_2d.cpp kiss_fft.c spectrum_simple_picking.cpp $<TARGET_OBJECTS:commandline>  $<TARGET_OBJECTS:jsoncpp>)
  target_link_libraries(simple_picker fid_base_lib)


  # peak file manupilation
  add_executable(peak_manipulation main_tab_peak.cpp peak_manipulation.cpp $<TARGET_OBJECTS:commandline>)

# 3D spectrum deconvolution / peak picking
add_executable(spectrum_deconv_3d main_peak_pickingfitting_3d.cpp spectrum_fit_3d.cpp spectrum_fit_3d_cost.cpp fid_3d.cpp  kiss_fft.c  lmminimizer.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
target_link_libraries(spectrum_deconv_3d fid_base_lib)


  # for web server only. need to copy run_fid.pl and run_colmarq_dp_vf.pl to the build directory
  configure_file(run_fid.pl ${CMAKE_CURRENT_BINARY_DIR}/run_fid.pl COPYONLY)
  configure_file(run_colmarq_dp_vf.pl ${CMAKE_CURRENT_BINARY_DIR}/run_colmarq_dp_vf.pl COPYONLY)


  # fitting part, need Ceres
  find_package(Ceres)


  # database query programs for the webserver. They simulated theoretical spectra of database item according to input experimental conditions.
  if(Ceres_FOUND )
    # 1D DOSY fitting
    add_executable(dosy main_dosy_fit.cpp dosy.cpp peak_manipulation $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
    target_include_directories(dosy PRIVATE ${CERES_INCLUDE_DIRS})
    target_include_directories(dosy PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(dosy ${CERES_LIBRARIES})

    # spectrum simulation from spin system intermediate file
    add_executable(simulate_1d main_spin_simulation.cpp spectrum_simulation.cpp hungary.cpp fid_1d.cpp  fwhh_dnn.cpp spectrum_fwhh_1d.cpp spectrum_pick_1d.cpp spectrum_fit_1d.cpp cost_functions.cpp kiss_fft.c  $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp> $<TARGET_OBJECTS:dnn>)
    target_link_libraries(simulate_1d fid_base_lib)
    target_link_libraries(simulate_1d libcerf)
    target_include_directories(simulate_1d PRIVATE ${CERES_INCLUDE_DIRS})
    target_include_directories(simulate_1d PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(simulate_1d ${CERES_LIBRARIES})
  endif()


  # build a alternative peak fitting program for 2D, using my own minimization algorithm instead of Ceres
  add_executable(voigt_fit_2 main_peak_fitting.cpp fid_2d.cpp kiss_fft.c spectrum_fit.cpp cost_functions.cpp kiss_fft.c lmminimizer.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(voigt_fit_2 PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(voigt_fit_2 fid_base_lib)
  target_link_libraries(voigt_fit_2 libcerf)
  target_compile_definitions(voigt_fit_2 PRIVATE LMMIN)

  # build a alternative peak fitting program for 1D, using my own minimization algorithm instead of Ceres
  add_executable(peak_fit_1d_2 main_peak_fitting_1d.cpp fid_1d.cpp spectrum_fit_1d.cpp cost_functions.cpp kiss_fft.c lmminimizer.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
  target_include_directories(peak_fit_1d_2 PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(peak_fit_1d_2 fid_base_lib)
  target_link_libraries(peak_fit_1d_2 libcerf)
  target_compile_definitions(peak_fit_1d_2 PRIVATE LMMIN)

  # build voigt_fit_2 with openmp support
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    add_executable(voigt_fit_2_omp main_peak_fitting.cpp fid_2d.cpp kiss_fft.c spectrum_fit.cpp cost_functions.cpp kiss_fft.c lmminimizer.cpp $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
    target_include_directories(voigt_fit_2_omp PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(voigt_fit_2_omp fid_base_lib)
    target_link_libraries(voigt_fit_2_omp libcerf)
    target_compile_definitions(voigt_fit_2_omp PRIVATE LMMIN)
    target_compile_options(voigt_fit_2_omp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(voigt_fit_2_omp  OpenMP::OpenMP_CXX)
    target_compile_definitions(voigt_fit_2_omp PRIVATE USE_OPENMP)
  endif()


  # peak fitting programs for both 1D and 2D
  if(Ceres_FOUND)
    add_executable(voigt_fit main_peak_fitting.cpp fid_2d.cpp kiss_fft.c spectrum_fit.cpp cost_functions.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
    target_include_directories(voigt_fit PRIVATE ${CERES_INCLUDE_DIRS})
    target_include_directories(voigt_fit PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(voigt_fit fid_base_lib)
    target_link_libraries(voigt_fit ${CERES_LIBRARIES})
    target_link_libraries(voigt_fit libcerf)

    add_executable(peak_fit_1d main_peak_fitting_1d.cpp fid_1d.cpp spectrum_fit_1d.cpp cost_functions.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
    target_include_directories(peak_fit_1d PRIVATE ${CERES_INCLUDE_DIRS})
    target_include_directories(peak_fit_1d PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(peak_fit_1d fid_base_lib)
    target_link_libraries(peak_fit_1d ${CERES_LIBRARIES})
    target_link_libraries(peak_fit_1d libcerf)
  endif()

  #peak fitting program for 2D, using ceres and openmp
  if(Ceres_FOUND AND OpenMP_CXX_FOUND)
    add_executable(voigt_fit_omp main_peak_fitting.cpp fid_2d.cpp kiss_fft.c spectrum_fit.cpp cost_functions.cpp kiss_fft.c $<TARGET_OBJECTS:commandline> $<TARGET_OBJECTS:jsoncpp>)
    target_include_directories(voigt_fit_omp PRIVATE ${CERES_INCLUDE_DIRS})
    target_include_directories(voigt_fit_omp PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(voigt_fit_omp fid_base_lib)
    target_link_libraries(voigt_fit_omp ${CERES_LIBRARIES})
    target_link_libraries(voigt_fit_omp libcerf)
    target_compile_options(voigt_fit_omp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(voigt_fit_omp OpenMP::OpenMP_CXX)
    target_compile_definitions(voigt_fit_omp PRIVATE USE_OPENMP)
  endif()

  # test programs and data files for DP and VF
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

  # test program and files for dp_1d and vf_1d
  configure_file(test/test_dp_1d.pl ${CMAKE_CURRENT_BINARY_DIR}/test_dp_1d.pl COPYONLY)
  configure_file(test/test_vf_1d.pl ${CMAKE_CURRENT_BINARY_DIR}/test_vf_1d.pl COPYONLY)
  configure_file(test/test_vf_2_1d.pl ${CMAKE_CURRENT_BINARY_DIR}/test_vf_2_1d.pl COPYONLY)

  configure_file(test/test.ft1 ${CMAKE_CURRENT_BINARY_DIR}/test.ft1 COPYONLY)

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

  add_test(NAME test_deep_picker_1d COMMAND /usr/bin/env perl test_dp_1d.pl)
  add_test(NAME test_voigt_fit_1d COMMAND /usr/bin/env perl test_vf_1d.pl)
  set_tests_properties(test_voigt_fit_1d PROPERTIES DEPENDS test_deep_picker_1d)
  add_test(NAME test_voigt_fit_2_1d COMMAND /usr/bin/env perl test_vf_2_1d.pl)
  set_tests_properties(test_voigt_fit_2_1d PROPERTIES DEPENDS test_deep_picker_1d)

  add_test(NAME test_peak_match COMMAND /usr/bin/env perl test_peak_match.pl)


  add_executable(test_functors test_functors.cpp cost_functions.cpp kiss_fft.c)
  target_compile_definitions(test_functors PRIVATE LMMIN)
  target_link_libraries(test_functors libcerf)
  add_test(NAME test_functors COMMAND test_functors)

add_test(NAME test_fid_phase COMMAND /usr/bin/env perl test_fid_phase.pl)


  install(TARGETS 
    fid_1d fid_3d phase_1d plot_spectrum_1d baseline_1d urea peak_manipulation deep_picker_1d peak_fit_1d dosy
    simulate_1d
    fid phase fwhh_2d plot_spectrum_2d  simple_picker deep_picker voigt_fit
    simulate_2d spectrum_deconv_3d
    RUNTIME DESTINATION release
  )

  install(FILES
    run_fid.pl run_colmarq_dp_vf.pl
    DESTINATION release
  )
endif()
