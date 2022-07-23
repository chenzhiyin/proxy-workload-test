// rapidjson/example/simpledom/simpledom.cpp`
#include "parser_utility.h"
#include <random>

ServingTestConfig::ServingTestConfig() {
  version = "1";
  signature_name = "serving_default";
  threads_num = 4;
  intra_op = -1;
  inter_op = -1;
  interval = 30;
}

int generate_data_to_tensor(tensorflow::TensorProto *tensor, int min, int max,
                            int size) {
  std::uniform_int_distribution<int> u1(min, max);
  std::uniform_real_distribution<double> u2((double)min, (double)max);
  std::default_random_engine e;
  e.seed(time(0));

  const tensorflow::DataType dtype = tensor->dtype();

  for (int64_t i = 0; i < size; ++i) {
    switch (dtype) {
    case tensorflow::DT_FLOAT:
      tensor->add_float_val(u2(e));
      break;

    case tensorflow::DT_DOUBLE:
      tensor->add_double_val(u2(e));
      break;

    case tensorflow::DT_INT32:
    case tensorflow::DT_INT16:
    case tensorflow::DT_INT8:
    case tensorflow::DT_UINT8:
      tensor->add_int_val(u1(e));
      break;

    case tensorflow::DT_INT64:
      tensor->add_int64_val(u1(e));
      break;

    case tensorflow::DT_UINT32:
      tensor->add_uint32_val(u1(e));
      break;

    case tensorflow::DT_UINT64:
      tensor->add_uint64_val(u1(e));
      break;

    default:
      return -1;
    }
  }

  return 0;
}

int generate_value_to_tensor(tensorflow::TensorProto *tensor,
                             const std::string &script) {
  tensorflow::TensorShapeProto *shape = tensor->mutable_tensor_shape();
  int size = 1;
  for (int64_t i = 0; i < shape->dim_size(); ++i) {
    size *= shape->dim(i).size();
  }

  int start = script.find("random(") + sizeof("random(") - 1;
  if (start < 0) {
    return -1;
  }

  int end = script.find(")", start);
  if (end < 0) {
    return -1;
  }

  std::string param_val = script.substr(start, end - start);
  int pos = param_val.find(",");
  if (pos < 0) {
    return -1;
  }

  int min = std::stoi(param_val.substr(0, pos));
  int max = std::stoi(param_val.substr(pos + 1));
  return generate_data_to_tensor(tensor, min, max, size);
}

int generate_shape_to_tensor(tensorflow::TensorShapeProto *shape,
                             const std::string &script) {
  int start = script.find("shape(") + sizeof("shape(") - 1;
  if (start < 0) {
    return -1;
  }

  int end = script.find(")", start);
  if (end < 0) {
    return -1;
  }

  std::string shape_val = script.substr(start, end - start);

  int num = 0;
  while (shape_val.size() > 0) {
    int pos = shape_val.find(",");
    if (pos > 0) {
      num = std::stoi(shape_val.substr(0, pos));
      shape_val = shape_val.substr(pos + 1);
    } else {
      num = std::stoi(shape_val);
      shape_val = "";
    }

    if (num <= 0) {
      return -1;
    }

    shape->add_dim()->set_size(num);
  }

  if (num <= 0) {
    return -1;
  }

  return 0;
}

int add_value_to_tensor(const rapidjson::Value &val, tensorflow::DataType dtype,
                        tensorflow::TensorProto *tensor) {
  switch (dtype) {
  case tensorflow::DT_FLOAT:
    if (!val.IsNumber())
      return -1;
    tensor->add_float_val(val.GetFloat());
    break;

  case tensorflow::DT_DOUBLE:
    if (!val.IsNumber())
      return -1;
    tensor->add_double_val(val.GetDouble());
    break;

  case tensorflow::DT_INT32:
  case tensorflow::DT_INT16:
  case tensorflow::DT_INT8:
  case tensorflow::DT_UINT8:
    if (!val.IsInt())
      return -1;
    tensor->add_int_val(val.GetInt());
    break;

  case tensorflow::DT_STRING:
    if (!val.IsString())
      return -1;
    tensor->add_string_val(val.GetString(), val.GetStringLength());
    break;

  case tensorflow::DT_INT64:
    if (!val.IsInt64())
      return -1;
    tensor->add_int64_val(val.GetInt64());
    break;

  case tensorflow::DT_BOOL:
    if (!val.IsBool())
      return -1;
    tensor->add_bool_val(val.GetBool());
    break;

  case tensorflow::DT_UINT32:
    if (!val.IsUint())
      return -1;
    tensor->add_uint32_val(val.GetUint());
    break;

  case tensorflow::DT_UINT64:
    if (!val.IsUint64())
      return -1;
    tensor->add_uint64_val(val.GetUint64());
    break;

  default:
    return -1;
  }
  return 0;
}

// Fills tensor values.
int fill_tensor_proto(const rapidjson::Value &val, int level,
                      tensorflow::DataType dtype,
                      tensorflow::TensorProto *tensor) {
  const auto rank = tensor->tensor_shape().dim_size();
  if (!val.IsArray()) {
    // DOM tree for a (dense) tensor will always have all values
    // at same (leaf) level equal to the rank of the tensor.
    if (level != rank) {
      fprintf(stderr, "JSON inputs rank error.\n ");
      return -1;
    }

    if (add_value_to_tensor(val, dtype, tensor)) {
      fprintf(stderr, "JSON inputs rank error.\n ");
      return -1;
    }

    return 0;
  }

  // If list is nested deeper than rank, stop processing.
  if (level >= rank) {
    fprintf(stderr, "JSON inputs rank error.\n ");
    return -1;
  }

  // Ensure list is of expected size for our level.
  if (val.Size() != tensor->tensor_shape().dim(level).size()) {
    fprintf(stderr, "Encountered list at unexpected size: ", val.Size(),
            " at level: ", level,
            " expected size: ", tensor->tensor_shape().dim(level).size());
    return -1;
  }

  // All OK, recurse into elements of the list.
  for (const auto &v : val.GetArray()) {
    if (fill_tensor_proto(v, level + 1, dtype, tensor)) {
      return -1;
    }
  }

  return 0;
}

void get_dense_tensor_shape(const rapidjson::Value &val,
                            tensorflow::TensorShapeProto *shape) {
  if (!val.IsArray())
    return;
  const auto size = val.Size();
  shape->add_dim()->set_size(size);
  if (size > 0) {
    get_dense_tensor_shape(val[0], shape);
  }
}

int tensor_data_parse(const rapidjson::Value &val,
                      tensorflow::TensorProto *tensor) {
  if (val.IsArray()) {
    get_dense_tensor_shape(val, tensor->mutable_tensor_shape());
    if (fill_tensor_proto(val, 0 /* level */, tensor->dtype(), tensor) != 0) {
      return -1;
    }
  } else if (val.IsString()) {
    std::string script = val.GetString();
    if (generate_shape_to_tensor(tensor->mutable_tensor_shape(), script) != 0) {
      return -1;
    }
    if (generate_value_to_tensor(tensor, script)) {
      return -1;
    }
  } else {
    return -1;
  }

  return 0;
}

int input_data_parse(const rapidjson::Value::MemberIterator &itr,
                     ServingTestConfig *config) {
  const rapidjson::Value &val = itr->value;
  if (!val.IsObject()) {
    auto *tensor = &(config->input)[config->info.begin()->first];
    tensor->set_dtype(config->info.begin()->second.dtype());
    if (tensor_data_parse(val, tensor) != 0) {
      return -1;
    }
  } else {
    for (const auto &kv : config->info) {
      const auto &name = kv.first;
      auto item = val.FindMember(name.c_str());
      if (item == val.MemberEnd()) {
        fprintf(stderr, "Missing named input: %s  in 'inputs' object.\n ",
                name);
        return -1;
      }

      const auto dtype = kv.second.dtype();
      auto *tensor = &(config->input)[name];
      tensor->set_dtype(dtype);
      tensor->mutable_tensor_shape()->Clear();
      if (tensor_data_parse(val, tensor) != 0) {
        return -1;
      }
    }
  }
  return 0;
}

int metadata_parse(ServingTestConfig *config) {
  char tmp[256];
  getcwd(tmp, 256);
  std::string current_dir(tmp);
  std::string export_path =
      current_dir + "/models/" + config->model_name + "/" + config->version;

  for (int i = 0; i < config->libraries.size(); ++i) {
    std::string library_path;
    if (config->libraries[i].at(0) == '/') {
      library_path = config->libraries[i];
    } else {
      library_path = current_dir + "/" + config->libraries[i];
    }

    void *handler = NULL;
    tensorflow::Status status = tensorflow::internal::LoadDynamicLibrary(
        library_path.c_str(), &handler);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      return -1;
    }
  }

  tensorflow::ConfigProto &conf = config->sess_options.config;
  if (config->inter_op != -1) {
    conf.set_inter_op_parallelism_threads(config->inter_op);
  }

  if (config->intra_op != -1) {
    conf.set_intra_op_parallelism_threads(config->intra_op);
  }
  conf.set_use_per_session_threads(true);

  if (!tensorflow::LoadSavedModel(config->sess_options, config->run_options,
                                  export_path, {"serve"}, &config->bundle)
           .ok()) {
    fprintf(stderr, "Fail to load model: %s.\n ", export_path);
    return -1;
  }

  auto iter = config->bundle.GetSignatures().find(config->signature_name);
  if (iter == config->bundle.GetSignatures().end()) {
    fprintf(stderr, "Missing named input: %s  in 'inputs' object.\n ",
            config->signature_name);
    return -1;
  }

  const tensorflow::SignatureDef &signature = iter->second;
  config->info = signature.inputs();

  for (auto &iter : signature.outputs()) {
    config->output_tensor_names.emplace_back(iter.second.name());
    config->output_tensor_aliases.emplace_back(iter.first);
  }

  return 0;
}

int config_parse(char *config_file, ServingTestConfig *config) {
  if (config_file == NULL) {
    config_file = (char *)"./config/default.json";
  }

  std::ifstream in(config_file);
  if (!in.is_open()) {
    fprintf(stderr, "fail to read json file: %s\n", config_file);
    return -1;
  }

  std::string json_content((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  in.close();

  rapidjson::Document doc;
  if (doc.Parse(json_content.c_str()).HasParseError() ||
      !doc.HasMember("model_name") || !doc["model_name"].IsString()) {
    fprintf(stderr, "fail to parse json file: %s\n", config_file);
    return -1;
  }

  config->model_name = doc["model_name"].GetString();
  if (doc.HasMember("version") && doc["version"].IsString()) {
    config->version = doc["version"].GetString();
  }

  if (doc.HasMember("signature_name") && doc["signature_name"].IsString()) {
    config->signature_name = doc["signature_name"].GetString();
  }

  if (doc.HasMember("args")) {
    const rapidjson::Value &args = doc["args"];
    if (args.HasMember("max_threads_num")) {
      config->threads_num = args["max_threads_num"].GetInt();
    }

    if (args.HasMember("intra_op")) {
      config->intra_op = args["intra_op"].GetInt();
    }

    if (args.HasMember("inter_op")) {
      config->inter_op = args["inter_op"].GetInt();
    }

    if (args.HasMember("interval")) {
      config->interval = args["interval"].GetInt();
    }
  }

  if (doc.HasMember("libraries") && doc["libraries"].IsArray()) {
    auto libraries = doc["libraries"].GetArray();
    for (int i = 0; i < libraries.Size(); ++i) {
      config->libraries.emplace_back(libraries[i].GetString());
    }
  }

  if (metadata_parse(config) != 0) {
    return -1;
  }

  auto itr = doc.FindMember("inputs");
  if (itr != doc.MemberEnd()) {
    input_data_parse(itr, config);
  }

  return 0;
}

int sesssion_run() { return 0; }

int test(int argc, char *argv[]) {
  ServingTestConfig config;
  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

  char *file_path = NULL;
  if (argc > 1) {
    file_path = argv[1];
  }

  int rc = config_parse(file_path, &config);
  if (rc != 0) {
    return -1;
  }

  for (auto &input : config.input) {
    tensorflow::Tensor tensor;
    const std::string &alias = input.first;
    auto iter = config.info.find(alias);
    if (iter == config.info.end()) {
      return -1;
    }

    if (!tensor.FromProto(input.second)) {
      return -1;
    }

    // for(int j=0; j<tensor.NumElements(); ++j){
    //     std::cout << tensor.flat<double>()(j) << " ";
    // }
    // std::cout << iter->second.name() << std::endl;

    inputs.emplace_back(std::make_pair(iter->second.name(), tensor));
  }

  tensorflow::Status status = config.bundle.session->Run(
      inputs, config.output_tensor_names, {}, &outputs);
  if (!status.ok()) {
    return -1;
  }

  for (int i = 0; i < outputs.size(); ++i) {
    for (int j = 0; j < outputs[i].NumElements(); ++j) {
      std::cout << (outputs[i]).flat<float>()(j) << " ";
    }
  }
  std::cout << std::endl;
  return 0;
}