#include "parser_utility.h"

using namespace tensorflow;

bool g_stop = false;

typedef struct {
  pid_t pid;
  long cost;
  float qps;
  int loop;
} perf_info_t;

pid_t gettid() { return syscall(SYS_gettid); }

long time_now() {
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return (tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

static void inline run_session(
    Session *session, std::vector<std::pair<std::string, Tensor>> *inputs,
    std::vector<std::string> *output_names,
    std::vector<tensorflow::Tensor> *outputs, int times) {
  for (int i = 0; i < 20; i++) {
    // Status status = session->Run(*inputs,
    // {"Lets_regard_it_as_final_output:0"}, {}, &outputs);
    Status status = session->Run(*inputs, *output_names, {}, outputs);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      return;
    }
  }
}

void model_worker(Session *session,
                  std::vector<std::pair<std::string, Tensor>> *inputs,
                  std::vector<std::string> *output_names, perf_info_t *info) {
  // initialize the number of worker threads
  // tensorflow::RunOptions run_options;
  // run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
  // tensorflow::RunMetadata run_metadata;
  // Load graph protobuf

  // initialize tensor inputs

  /*std::string concat_input_name = "concat_input";
  fill_input(bs, graph_def, concat_input_name, inputs);

  std::string op_input_name = "op_input";
  fill_input(bs, graph_def, op_input_name, inputs);*/

  // The session will initialize the outputs
  // std::cout << "TID(" << gettid() << ")is running" << std::endl;
  std::vector<tensorflow::Tensor> outputs;
  // Warm up
  run_session(session, inputs, output_names, &outputs, 20);
  // Run the session, evaluating our "output" operation from the graph
  long start = time_now();
  info->loop = 0;
  while (!g_stop) {
    Status status = session->Run(*inputs, *output_names, {}, &outputs);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      return;
    }
    info->loop += 1;
  }
  long end = time_now();

  // Ramping down
  run_session(session, inputs, output_names, &outputs, 20);

  info->pid = gettid();
  info->cost = end - start;
  info->qps = info->loop * 1000.0f / (end - start);

  return;
}

/**
 * @brief deep model for click through rate prediction
 * @details [long description]
 *
 * @param argv[1] inter_op
 * @param argv[2] intra_op
 * @param argv[3] input_graph
 * @param argv[4] model_num
 * @param argv[5] batch_size
 *
 * @return [description]:
 */
int main(int argc, char *argv[]) {
  std::vector<std::thread> threads;
  ServingTestConfig config;

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

  char *file_path = NULL;
  if (argc > 1) {
    file_path = argv[1];
    std::cout << argv[1] << std::endl;
  }

  if (config_parse(file_path, &config) != 0) {
    std::cerr << "init config failed" << std::endl;
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

    inputs.emplace_back(std::make_pair(iter->second.name(), tensor));
  }

  std::vector<perf_info_t> info(config.threads_num);
  for (int i = 0; i < config.threads_num; i++) {
    threads.push_back(std::thread(model_worker, config.bundle.GetSession(),
                                  &inputs, &config.output_tensor_names,
                                  &info[i]));
  }

  sleep(config.interval);
  g_stop = true;

  for (int i = 0; i < config.threads_num; i++) {
    threads[i].join();
  }

  float total = 0;
  for (int i = 0; i < config.threads_num; i++) {
    std::cout << "TID: " << info[i].pid << std::endl;
    std::cout << "Latency: " << (1.0 * info[i].cost / (info[i].loop)) << "ms";
    std::cout << "  Loops: " << info[i].loop;
    std::cout << "  Total Time: " << info[i].cost << "ms" << std::endl;
    std::cout << std::endl;
    total += info[i].qps;
  }
  std::cout << "Total QPS: " << total << std::endl;

  return 0;
}
