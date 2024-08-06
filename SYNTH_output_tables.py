import warnings

from os import makedirs, path

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option("display.max_rows", None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

algorithms = [
    ("SEP", "CUT"),
    ("SVM", "CUT"),
    ("SVM", "SIM")
]

dimension = 150
instance_numbers = list(range(30))

max_n_queries = 2000
n_query_checkpoints = []  # [8, 40, 200, 1000]
n_query_list = n_query_checkpoints + [max_n_queries]

output_directory = "SYNTH_output/tables"
if not path.exists(output_directory):
    makedirs(output_directory)

columns = ["algorithm", "n_solved_instances", "last_relative_error", "running_time", "n_queries"]

for n_queries in n_query_list:

    with open("{}/table_{}.txt".format(output_directory, n_queries), "w") as f:
        f.write("Maximum number of queries: {}, dimension: {}".format(max_n_queries, dimension))
    print("Maximum number of queries: {}, dimension: {}".format(max_n_queries, dimension))

    results_df = pd.DataFrame(columns=columns)

    for separator_name, strategy_name in algorithms:

        input_directory = "SYNTH_output/logs/{}+{}".format(separator_name, strategy_name)
        results = pd.read_json("{}/{}_results.json".format(input_directory, n_queries))

        algorithm_name = "{}+{}".format(separator_name, strategy_name)

        results_df = pd.concat([
            results_df, pd.DataFrame(dict(zip(columns, [[algorithm_name],
                                                        [results.solved.to_numpy().sum()],
                                                        [results.last_relative_error.mean()],
                                                        [results.elapsed_time.mean()],
                                                        [results.n_queries.mean()]])))])

    with open("{}/table_{}.txt".format(output_directory, n_queries), "a") as f:
        f.write(results_df.sort_values(by="algorithm").head(results_df.shape[0]).to_string(index=False))
    print(results_df.sort_values(by="algorithm").head(results_df.shape[0]).to_string(index=False))
