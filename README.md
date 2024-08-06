# actively_learning_combinatorial_optimization_using_a_membership_oracle
Code and data for: Actively learning combinatorial optimization using a membership oracle, submitted to NeurIPS 2024.

## Data

The 30 MIT-Math CSPP instances are in the directory "data/MIT_instances". Each file has name "<instance_number>.pickle", where <instance_number> is the progressive id of the instance, from 0 to 29.

The 30 SYNTH-MIT CSPP instances are in the directory "data/SYNTH_instances". Each file has name "150_<instance_number>.pickle", where <instance_number> is the progressive id of the instance, from 0 to 29.

The instances for the knapsack experiments (KNAP) are generated automatically at test time (i.e., when the script "KNAP_test.py" is executed) using the generator defined in "KNAP_generator.py". The generator is a Python re-implementation of a C knapsack generator as cited in the paper.

The directory "data/MIT_subjects" contains the file "subjects.csv" with the information about the 150 MIT courses on which the MIT-Math CSPP instances are based. The information is organized in 6 columns separated by the symbol "|". The columns are:
- "code": progressive code of the course;
- "name": name of the course;
- "units": credits in the form (a, b, c) as described in the paper;
- "prerequisites": prerequisites of the course as a list of lists containing course codes. The semantics is that at least one course from each sub-list has to be included in the study plan;
- "corequisites": corequisites of the course in the same form as the prerequisites;
- "alternatives": alternatives of the course as a list of course codes.

## Code

For each of the 3 instance categories (KNAP, MIT, SYNTH), the following Python files are present:
- "<instance_category>\_test.py": executes the test on the corresponging instances and saves the output file in the directory "<instance_category>\_output/logs";
- "<instance_category>\_strategy.py": contains the implementation of the CUT and SIM sampling strategies for the specific instance category;
- "<instance_category>\_optimizer.py": contains the code for execution of the optimization step for the specific instance category;
- "<instance_category>\_output\_graphs.py": reads the output of "<instance_category>\_test.py" from "<instance_category>\_output/logs" and outputs the objective error graphs in  "<instance_category>\_output/graphs";
- "<instance_category>\_output\_tables.py": reads the output of "<instance_category>\_test.py" from "<instance_category>\_output/logs" and outputs the table(s) of additional computational results in  "<instance_category>\_output/tables".

The remaining Python files are the following:
- "separator.py": implementation of the SEP and SVM separation methods;
- "oracle.py": contains the class that allows to make oracle calls;
- "functions.py": additional functions.