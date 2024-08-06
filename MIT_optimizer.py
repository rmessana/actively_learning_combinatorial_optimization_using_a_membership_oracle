import gurobipy as gp
import numpy as np
import pandas as pd


class MITOptimizer:

    def __init__(self, no_good_cuts: bool = True):

        self.no_good_cuts = no_good_cuts

        self.name = "opt"
        self.acronym = "OPT"

        df = pd.read_csv("data/MIT_subjects/subjects.csv", sep="|", index_col=False)
        self.units = np.array([sum(eval(u)) for u in df.units])
        self.prerequisites = [eval(p) for p in df.prerequisites]
        self.corequisites = [eval(c) for c in df.corequisites]
        self.alternatives = [eval(a) for a in df.alternatives]

    def step(self, status: dict):

        b = status["b"]
        w = status["w"]
        points = status["points"]

        dimension = self.units.size

        model = gp.Model("optimization_strategy_model")
        variables = model.addVars(range(dimension), vtype=gp.GRB.BINARY)
        model.addConstr(b + gp.quicksum(w[j] * variables[j] for j in range(dimension)) >= 0)
        for j, p in enumerate(self.prerequisites):
            for code_list in p:
                model.addConstr(variables[j] <= gp.quicksum(variables[code] for code in code_list))
        for j, c in enumerate(self.corequisites):
            for code_list in c:
                model.addConstr(variables[j] <= gp.quicksum(variables[code] for code in code_list))
        for j, a in enumerate(self.alternatives):
            model.addConstr(variables[j] + gp.quicksum(variables[code] for code in a) <= 1)
        if self.no_good_cuts:
            for point in points["feasible"]:
                model.addConstr(gp.quicksum(variables[j] * (point[j] < 0.5) for j in range(dimension)) >= 1 - 1e-8)
            for point in points["infeasible"]:
                model.addConstr(gp.quicksum(variables[j] * (point[j] > 0.5) for j in range(dimension)) <=
                                round(point.sum()) - 1 + 1e-8)
        model.setObjective(gp.quicksum(self.units[j] * variables[j] for j in range(dimension)))
        model.setParam("OutputFlag", 0)
        model.setParam("Threads", 4)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()

        if model.Status == 2:
            optimal_solution = np.array([round(variables[j].X) for j in range(dimension)])
            message = "(strategy: optimization)"
        else:
            optimal_solution = None
            message = "infeasible model (strategy: optimization)"

        return optimal_solution, message
