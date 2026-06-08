# 1. CMake config package: OIIO is installed with cmake support
find_package(OpenImageIO CONFIG QUIET)

# 2. pkg-config fallback: Linux distro packages, Homebrew, MacPorts, vcpkg with pkgconf
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

# 3. Manual search: common install locations on macOS, Linux and Windows (vcpkg)
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
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenImageIO
    REQUIRED_VARS OpenImageIO_FOUND
    FAIL_MESSAGE
        "OpenImageIO not found. Install with:\n"
        "  macOS:   brew install openimageio\n"
        "  Ubuntu:  apt install libopenimageio-dev\n"
        "  Windows: vcpkg install openimageio\n"
        "Or set CMAKE_PREFIX_PATH to your OIIO install root.")
