cmake_minimum_required(VERSION 3.10)
project(SpectroVis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(PkgConfig REQUIRED)
find_package(PNG REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
pkg_check_modules(SNDFILE REQUIRED sndfile)

add_executable(spectrovis spectrovis.cpp)

target_include_directories(spectrovis PRIVATE
    ${FFTW3_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
)

target_link_libraries(spectrovis
    ${FFTW3_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    PNG::PNG
    m
)