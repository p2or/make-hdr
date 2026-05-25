cmake_minimum_required(VERSION 3.13)
project(make_hdr LANGUAGES CXX)

set(PLUGIN_NAME "MakeHDR")
option(BUILD_OFX "Build the ${PLUGIN_NAME} OFX plugin" ON)
option(BUILD_CLI "Build standalone ${PLUGIN_NAME} command-line interface" OFF)

option(USE_ACCELERATE "Use macOS Accelerate framework instead of clapack" OFF)
option(NUKE_DEBUGGING "Configure IDE debugger to launch Nuke" ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if (WIN32)
	add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
elseif (APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -include stdio.h")
elseif (UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif()

if (APPLE AND USE_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    set(MATH_LIBRARIES ${ACCELERATE_FRAMEWORK})
else()
    include_directories(${CMAKE_SOURCE_DIR}/modules/clapack/INCLUDE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/modules/clapack)
    set(MATH_LIBRARIES f2c blas lapack)
    # Bundled clapack uses f2c C ABI: no Fortran hidden args, long BLAS/LAPACK integers.
    set(ARMA_CLAPACK_DEFS ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS ARMA_BLAS_LONG)
endif()

if (BUILD_OFX)
    if (WIN32)
        set(OFX_PLUGIN_PATH "C:/Program Files/Common Files/OFX/Plugins" CACHE PATH "OFX Plugin Install Path")
        set(INSTALL_BIN_PATH ${OFX_PLUGIN_PATH}/${PLUGIN_NAME}.ofx.bundle/Contents/Win64)
    elseif (APPLE)
        set(OFX_PLUGIN_PATH "/Library/OFX/Plugins" CACHE PATH "OFX Plugin Install Path")
        set(INSTALL_BIN_PATH ${OFX_PLUGIN_PATH}/${PLUGIN_NAME}.ofx.bundle/Contents/MacOS)
    elseif (UNIX)
        set(OFX_PLUGIN_PATH "/usr/OFX/Plugins" CACHE PATH "OFX Plugin Install Path")
        set(INSTALL_BIN_PATH ${OFX_PLUGIN_PATH}/${PLUGIN_NAME}.ofx.bundle/Contents/Linux-x86-64)
    endif()
    set(INSTALL_RES_PATH ${OFX_PLUGIN_PATH}/${PLUGIN_NAME}.ofx.bundle/Contents/Resources)

    add_subdirectory(${CMAKE_SOURCE_DIR}/modules/openfx/Support/Library)

    add_library(make_hdr MODULE 
        ${CMAKE_SOURCE_DIR}/source/resources.h
        ${CMAKE_SOURCE_DIR}/source/processor.h
        ${CMAKE_SOURCE_DIR}/source/solver.h
        ${CMAKE_SOURCE_DIR}/source/effect.h
        ${CMAKE_SOURCE_DIR}/source/effect.cpp
        )

    target_include_directories(make_hdr PRIVATE
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/modules/armadillo/include
        ${CMAKE_SOURCE_DIR}/modules/openfx/include
        ${CMAKE_SOURCE_DIR}/modules/openfx/Support/include
        ${CMAKE_SOURCE_DIR}/modules/openfx/Support/Plugins/include
        ${CMAKE_SOURCE_DIR}/modules/spdlog/include)

    if (APPLE AND NOT USE_ACCELERATE)
        target_compile_definitions(make_hdr PRIVATE ${ARMA_CLAPACK_DEFS})
    endif()

    target_link_libraries(make_hdr ${MATH_LIBRARIES} OfxSupport)

    # MSVC/Windows-specific compile options and definitions.
    if (WIN32 AND MSVC)
        target_compile_options(make_hdr PRIVATE /utf-8)
        target_compile_definitions(make_hdr PRIVATE NOMINMAX)

        # Workaround for constexpr mutex constructor in MSVC 1930+ (Visual Studio 2019+).
        if (MSVC_VERSION GREATER_EQUAL 1920)
            target_compile_definitions(make_hdr PUBLIC _CRT_SECURE_NO_WARNINGS _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
        endif()
    endif()

    set_target_properties(make_hdr
        PROPERTIES 
        LINKER_LANGUAGE CXX
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        PREFIX ""
        SUFFIX ".ofx"
        OUTPUT_NAME "${PLUGIN_NAME}")

    if (NUKE_DEBUGGING)
        find_package(Nuke)

        if (NUKE_FOUND)
            set_target_properties(make_hdr
                PROPERTIES
                VS_DEBUGGER_COMMAND ${NUKE_EXECUTABLE}
                VS_DEBUGGER_COMMAND_ARGUMENTS ${CMAKE_SOURCE_DIR}/test/room.nk
                XCODE_SCHEME_EXECUTABLE ${NUKE_EXECUTABLE}
                XCODE_SCHEME_ARGUMENTS ${CMAKE_SOURCE_DIR}/test/room.nk)
        endif()
    endif()

    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT make_hdr)

    # Install the plugin binary and resource files.
    install(TARGETS make_hdr DESTINATION ${INSTALL_BIN_PATH})
    install(FILES ${CMAKE_SOURCE_DIR}/icons/net.sf.openfx.make_hdr.png DESTINATION ${INSTALL_RES_PATH})
    install(FILES ${CMAKE_SOURCE_DIR}/icons/net.sf.openfx.make_hdr.svg DESTINATION ${INSTALL_RES_PATH})
endif()

if (BUILD_CLI)
    # 1. CMake config package (works on all platforms when OIIO is installed with cmake support)
    find_package(OpenImageIO CONFIG QUIET)

    # 2. pkg-config fallback (Linux distro packages, Homebrew, MacPorts, vcpkg with pkgconf)
    if (NOT OpenImageIO_FOUND)
        find_package(PkgConfig QUIET)
        if (PkgConfig_FOUND)
            pkg_check_modules(OIIO QUIET OpenImageIO)
            if (OIIO_FOUND)
                add_library(OpenImageIO::OpenImageIO INTERFACE IMPORTED)
                target_include_directories(OpenImageIO::OpenImageIO INTERFACE ${OIIO_INCLUDE_DIRS})
                target_link_libraries(OpenImageIO::OpenImageIO INTERFACE ${OIIO_LIBRARIES})
                target_link_directories(OpenImageIO::OpenImageIO INTERFACE ${OIIO_LIBRARY_DIRS})
                set(OpenImageIO_FOUND TRUE)
            endif()
        endif()
    endif()

    # 3. Manual search — common install locations on macOS, Linux and Windows (vcpkg)
    if (NOT OpenImageIO_FOUND)
        find_path(OIIO_INCLUDE_DIR OpenImageIO/imageio.h
            PATHS
                /opt/homebrew/opt/openimageio/include   # macOS Homebrew keg-only
                /opt/homebrew/include                   # macOS Homebrew
                /usr/local/include                      # macOS legacy / Linux local
                /usr/include                            # Linux system
                "$ENV{VCPKG_ROOT}/installed/x64-windows/include"    # Windows vcpkg
                "$ENV{VCPKG_ROOT}/installed/arm64-windows/include")
        find_library(OIIO_LIBRARY
            NAMES OpenImageIO OpenImageIO_D
            PATHS
                /opt/homebrew/opt/openimageio/lib
                /opt/homebrew/lib
                /usr/local/lib
                /usr/lib /usr/lib64
                "$ENV{VCPKG_ROOT}/installed/x64-windows/lib"
                "$ENV{VCPKG_ROOT}/installed/arm64-windows/lib")
        if (OIIO_INCLUDE_DIR AND OIIO_LIBRARY)
            add_library(OpenImageIO::OpenImageIO UNKNOWN IMPORTED)
            set_target_properties(OpenImageIO::OpenImageIO PROPERTIES
                IMPORTED_LOCATION "${OIIO_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${OIIO_INCLUDE_DIR}")
            set(OpenImageIO_FOUND TRUE)
        else()
            message(FATAL_ERROR
                "OpenImageIO not found. Install with:\n"
                "  macOS:   brew install openimageio\n"
                "  Ubuntu:  apt install libopenimageio-dev\n"
                "  Windows: vcpkg install openimageio\n"
                "Or set CMAKE_PREFIX_PATH to your OIIO install root.")
        endif()
    endif()

    add_executable(make_hdr_cli
        ${CMAKE_SOURCE_DIR}/source/resources.h
        ${CMAKE_SOURCE_DIR}/source/solver.h
        ${CMAKE_SOURCE_DIR}/source/merge.h
        ${CMAKE_SOURCE_DIR}/source/cli.cpp)

    target_compile_definitions(make_hdr_cli PRIVATE MAKE_HDR_CLI ARMA_WARN_LEVEL=0)

    target_include_directories(make_hdr_cli PRIVATE
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/modules/armadillo/include)

    if (APPLE AND USE_ACCELERATE)
        target_link_libraries(make_hdr_cli PRIVATE ${ACCELERATE_FRAMEWORK} OpenImageIO::OpenImageIO)
    else()
        if (APPLE)
            target_compile_definitions(make_hdr_cli PRIVATE ${ARMA_CLAPACK_DEFS})
        endif()
        target_link_libraries(make_hdr_cli PRIVATE f2c blas lapack OpenImageIO::OpenImageIO)
    endif()

    set_target_properties(make_hdr_cli PROPERTIES
        OUTPUT_NAME makehdr
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES)

    if (WIN32 AND MSVC)
        target_compile_options(make_hdr_cli PRIVATE /utf-8)
        target_compile_definitions(make_hdr_cli PRIVATE NOMINMAX)
    endif()

    install(TARGETS make_hdr_cli DESTINATION bin)

endif()