# - Try to find CAFFE
#
# The following variables are optionally searched for defaults
#  CAFFE_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  CAFFE_FOUND
#  CAFFE_INCLUDE_DIRS
#  CAFFE_LIBRARIES

include(FindPackageHandleStandardArgs)

set(CAFFE_ROOT_DIR "/opt/caffe")

set(CAFFE_INCLUDE_DIR "/opt/caffe/include" "/opt/caffe/build/src" "/opt/caffe/src")
set(CAFFE_LIBRARY "/opt/caffe/build/lib/libcaffe.so")

#set(CAFFE_ROOT_DIR "/Users/dementrock/libs/caffe")
#
#set(CAFFE_INCLUDE_DIR "/Users/dementrock/libs/caffe/include" "/Users/dementrock/libs/caffe/build/src" "/Users/dementrock/libs/caffe/src")
#set(CAFFE_LIBRARY "/Users/dementrock/libs/caffe/build/lib/libcaffe.so")


# find_library(CAFFE_LIBRARY caffe
#   PATHS ${CAFFE_ROOT_DIR}
#   PATH_SUFFIXES
#   build/lib)

find_package_handle_standard_args(CAFFE DEFAULT_MSG
  CAFFE_INCLUDE_DIR CAFFE_LIBRARY)

if(CAFFE_FOUND)
  set(CAFFE_INCLUDE_DIRS ${CAFFE_INCLUDE_DIR})
  set(CAFFE_LIBRARIES ${CAFFE_LIBRARY})
endif()
