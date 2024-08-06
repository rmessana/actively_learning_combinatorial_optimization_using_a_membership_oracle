import gurobipy as gp
import numpy as np


class Optimizer:

    def __init__(self, no_good_cuts: bool = True):

        self.no_good_cuts = no_good_cuts

        self.name = "opt"
        self.acronym = "OPT"

    def step(self, status: dict):

        values = status["values"]
        b = status["b"]
        w = status["w"]
        points = status["points"]

        dimension = values.size

        model = gp.Model("optimization_strategy_model")
        variables = model.addVars(range(dimension), vtype=gp.GRB.BINARY)
        model.addConstr(b + gp.quicksum(w[j] * variables[j] for j in range(dimension)) >= 0)
        if self.no_good_cuts:
            for point in points["feasible"]:
                model.addConstr(gp.quicksum(variables[j] * (point[j] < 0.5) for j in range(dimension)) >= 1 - 1e-8)
            for point in points["infeasible"]:
                model.addConstr(gp.quicksum(variables[j] * (point[j] > 0.5) for j in range(dimension)) <=
                                round(point.sum()) - 1 + 1e-8)
        model.setObjective(gp.quicksum(values[j] * variables[j] for j in range(dimension)))
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
