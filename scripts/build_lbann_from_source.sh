#!/bin/bash

# "spack" is just a shell function; it may not be exported to this
# scope. Just to be sure, reload the shell integration.
if [ -n "${SPACK_ROOT}" ]; then
    source ${SPACK_ROOT}/share/spack/setup-env.sh
fi

SPACK_VERSION=$(spack --version | sed 's/-.*//g')
MIN_SPACK_VERSION=0.13.3

source $(dirname ${BASH_SOURCE})/utilities.sh

compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
VALID_SPACK=$?

if [[ ${VALID_SPACK} -eq 2 ]]; then
    echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
    exit 1
fi

# Detect system parameters
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
ARCH=$(uname -m)
SYS=$(uname -s)

SCRIPT=${BASH_SOURCE}

if [[ ${SYS} = "Darwin" ]]; then
SCRIPTS_DIR=$(normpath $(dirname $(osx_realpath ${SCRIPT})))
else
SCRIPTS_DIR=$(realpath $(dirname ${SCRIPT}))
fi

LBANN_HOME=$(dirname ${SCRIPTS_DIR})
SPACK_ENV_DIR=${LBANN_HOME}/spack_environments

# Identify the center that we are running at
CENTER=
# String to identify the default compiler - DON'T use this for picking a compiler
COMPILER=
BUILD_SUFFIX=
if [[ ${SYS} = "Darwin" ]]; then
    CENTER="osx"
    COMPILER="clang"
    BUILD_SUFFIX=llnl.gov
else
    CORI=$([[ $(hostname) =~ (cori|cgpu) ]] && echo 1 || echo 0)
    if [[ ${CORI} -eq 1 ]]; then
        CENTER="nersc"
        # Make sure to purge and setup the modules properly prior to finding the Spack architecture
        source ${SPACK_ENV_DIR}/${CENTER}/setup_modules.sh
        BUILD_SUFFIX=nersc.gov
    else
        CENTER="llnl_lc"
        BUILD_SUFFIX=llnl.gov
    fi
    COMPILER="gnu"
fi

SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)

SCRIPT=$(basename ${BASH_SOURCE})
ENABLE_GPUS=ON
ENABLE_DISTCONV=OFF
ENABLE_DIHYDROGEN=OFF
EXEC_ENV=TRUE
BUILD_TYPE=Release
VERBOSE=0
DETERMINISTIC=OFF
LBANN_ENV=lbann-dev-${SPACK_ARCH_TARGET}

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "script ${BASH_SOURCE[0]} is being sourced ..."
    EXEC_ENV="FALSE"
fi

CORE_BUILD_PATH="${LBANN_HOME}/build/${COMPILER}.${BUILD_TYPE}.${CLUSTER}.${BUILD_SUFFIX}"

################################################################
# Help message
################################################################

function help_message {
    local SCRIPT=$(basename ${BASH_SOURCE})
    local N=$(tput sgr0)    # Normal text
    local C=$(tput setf 4)  # Colored text
    cat << EOF
Build LBANN on an LLNL LC system.
Can be called anywhere in the LBANN project tree.
Usage: ${SCRIPT} [options]
Options:
  ${C}--help${N}               Display this help message and exit.
  ${C}--debug${N}              Build with debug flag.
  ${C}--verbose${N}            Verbose output.
  ${C}-e | --env${N}           Build and install LBANN using the spack environment provided: default=lbann-dev
  ${C}-p | --prefix${N}        Build and install LBANN headers and dynamic library into subdirectorys at this path prefix.
  ${C}-i | --install-dir${N}   Install LBANN headers and dynamic library into the install directory: default=${CORE_BUILD_PATH}/install
  ${C}-b | --build-dir${N}     Specify alternative build directory: default=${CORE_BUILD_PATH}/build
  ${C}--disable-gpus${N}       Disable GPUS
  ${C}--instrument${N}         Use -finstrument-functions flag, for profiling stack traces
  ${C}-s | --superbuild${N}    Superbuild LBANN with hydrogen and aluminum
  ${C}-c | --distconv${N}      Enable the DistConv library
EOF
}

################################################################
# Parse command-line arguments
################################################################

while :; do
    case ${1} in
        -h|--help)
            # Help message
            help_message
            if [[ ${EXEC_ENV} == "FALSE" ]]; then
                return
            else
                exit 1
            fi
            ;;
        -e|--env)
            # Change default build directory
            if [ -n "${2}" ]; then
                LBANN_ENV=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -b|--build-dir)
            # Change default build directory
            if [ -n "${2}" ]; then
                if [[ ${2} = "." ]]; then
                    BUILD_DIR=${SPACK_ENV_DIR}/${2}
                elif [[ ${2} = /* ]]; then
                    BUILD_DIR=${2}
                else
                    BUILD_DIR=${SPACK_ENV_DIR}/${2}
                fi
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -i|--install-dir)
            # Specify install directory
            if [ -n "${2}" ]; then
                if [[ ${2} = "." ]]; then
                    INSTALL_DIR=${SPACK_ENV_DIR}/${2}
                elif [[ ${2} = /* ]]; then
                    INSTALL_DIR=${2}
                else
                    INSTALL_DIR=${SPACK_ENV_DIR}/${2}
                fi
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -p|--prefix)
            # Change default build directory
            if [ -n "${2}" ]; then
                if [[ ${2} = "." ]]; then
                    BUILD_DIR=${SPACK_ENV_DIR}/${2}/build
                    INSTALL_DIR=${SPACK_ENV_DIR}/${2}/install
                elif [[ ${2} = /* ]]; then
                    BUILD_DIR=${2}/build
                    INSTALL_DIR=${2}/install
                else
                    BUILD_DIR=${SPACK_ENV_DIR}/${2}/build
                    INSTALL_DIR=${SPACK_ENV_DIR}/${2}/install
                fi
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --disable-gpus)
            ENABLE_GPUS=OFF
            ;;
        -v|--verbose)
            # Verbose output
            VERBOSE=1
            ;;
        -d|--debug)
            # Debug mode
            BUILD_TYPE=Debug
            DETERMINISTIC=ON
            ;;
        --instrument)
            INSTRUMENT="-finstrument-functions -ldl"
            ;;
        -s|--superbuild)
            # Debug mode
            SUPERBUILD="superbuild_lbann_with_hydrogen_and_aluminum.sh"
            ;;
        -c|--distconv)
            ENABLE_DISTCONV=ON
            ENABLE_DIHYDROGEN=ON
            # CUDA is required for Distconv
            ENABLE_GPUS=ON
            # MPI-CUDA backend is required for Distconv
            ALUMINUM_WITH_MPI_CUDA=ON
            ;;
        -?*)
            # Unknown option
            echo "Unknown option (${1})" >&2
            exit 1
            ;;
        *)
            # Break loop if there are no more options
            break
    esac
    shift
done

BUILD_DIR="${BUILD_DIR:-${CORE_BUILD_PATH}/build}"
INSTALL_DIR="${INSTALL_DIR:-${CORE_BUILD_PATH}/install}"

export LBANN_HOME=${LBANN_HOME}
export LBANN_BUILD_DIR=${BUILD_DIR}
export LBANN_INSTALL_DIR=${INSTALL_DIR}

CMD="mkdir -p ${BUILD_DIR}"
echo ${CMD}
${CMD}
CMD="mkdir -p ${INSTALL_DIR}"
echo ${CMD}
${CMD}

SUPERBUILD="${SUPERBUILD:-cmake_lbann.sh}"
if [[ ${SYS} = "Darwin" ]]; then
    OSX_VER=$(sw_vers -productVersion)
    ENABLE_GPUS=OFF
fi

CMD="cd ${LBANN_BUILD_DIR}"
echo ${CMD}
${CMD}
echo ${PWD}

SPACK_ENV_CMD=
if [[ ${LBANN_ENV} ]]; then
    SPACK_ENV_CMD="spack env activate -p ${LBANN_ENV}"
    ${SPACK_ENV_CMD}
fi

if [[ ${SYS} = "Darwin" ]]; then
    export DYLD_LIBRARY_PATH=/System/Library/Frameworks/ImageIO.framework/Resources/:/usr/lib/:${DYLD_LIBRARY_PATH}
fi

C_FLAGS="${INSTRUMENT} -fno-omit-frame-pointer"
CXX_FLAGS="-DLBANN_SET_EL_RNG ${INSTRUMENT} -fno-omit-frame-pointer"

if [ "${ARCH}" == "x86_64" ]; then
    CXX_FLAGS="-march=native ${CXX_FLAGS}"
else
    CXX_FLAGS="-mcpu=native -mtune=native ${CXX_FLAGS}"
fi

source ${SPACK_ENV_DIR}/${SUPERBUILD}

ninja install

echo "To rebuild the environment:"
echo "    ${SPACK_ENV_CMD}"
echo "    cd ${LBANN_BUILD_DIR}"
echo "    ninja install"
