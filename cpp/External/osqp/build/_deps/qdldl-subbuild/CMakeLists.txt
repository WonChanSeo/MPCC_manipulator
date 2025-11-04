# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(qdldl-populate NONE)



include(ExternalProject)
ExternalProject_Add(qdldl-populate
                     "UPDATE_DISCONNECTED" "False"
                    SOURCE_DIR          "/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp/../qdldl"
                    BINARY_DIR          "/home/mms-wonchan/git/MPCC_manipulator/cpp/External/osqp/build/_deps/qdldl-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


