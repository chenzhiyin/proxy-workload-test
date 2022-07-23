#ifndef WORKLOAD_PROXY_TEST_PARSER_UTILITY
#define WORKLOAD_PROXY_TEST_PARSER_UTILITY

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/load_library.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"

class ServingTestConfig {
private:
  /* data */
public:
  ServingTestConfig();

  std::string model_name;
  std::string signature_name;
  std::string version;
  int threads_num;
  int intra_op;
  int inter_op;
  int interval;

  std::vector<std::string> libraries;

  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions sess_options;
  tensorflow::RunOptions run_options;

  std::vector<std::string> output_tensor_names;
  std::vector<std::string> output_tensor_aliases;

  google::protobuf::Map<std::string, tensorflow::TensorProto> input;
  google::protobuf::Map<std::string, tensorflow::TensorInfo> info;
};

int config_parse(char *config_file, ServingTestConfig *config);

#endif // WORKLOAD_PROXY_TEST_PARSER_UTILITY