import gurobipy as gp
import numpy as np
import pandas as pd


class Strategy:

    def __init__(self, dimension: int, no_good_cuts: bool = True, improving_point_constraint: bool = False):

        self.dimension = dimension
        self.no_good_cuts = no_good_cuts
        self.improving_point_constraint = improving_point_constraint

    def step(self, status: dict):
        pass


class MITCUTStrategy(Strategy):

    def __init__(self, dimension: int, no_good_cuts: bool = True, improving_point_constraint: bool = False):

        super().__init__(dimension, no_good_cuts, improving_point_constraint)

        self.name = "cut"
        self.acronym = "CUT"

        df = pd.read_csv("data/MIT_subjects/subjects.csv", sep="|", index_col=False)
        self.prerequisites = [eval(p) for p in df.prerequisites]
        self.corequisites = [eval(c) for c in df.corequisites]
        self.alternatives = [eval(a) for a in df.alternatives]

    def step(self, status: dict):

        b = status["b"]
        w = status["w"]

        if b == 0:
            optimal_solution = np.zeros(self.dimension)
            optimal_solution[0] = 1
            message = "(strategy: cut)"
            return optimal_solution, message

        points = status["points"]
        weights = - w / b

        model = gp.Model("cut_strategy_model")
        p_variables = model.addVars(range(self.dimension), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                    vtype=gp.GRB.CONTINUOUS)
        x_variables = model.addVars(range(self.dimension), vtype=gp.GRB.BINARY)

        model.addConstr(gp.quicksum(p_variables[j] * x_variables[j] for j in range(self.dimension)) == 1)
        for j, p in enumerate(self.prerequisites):
            for code_list in p:
                model.addConstr(x_variables[j] <= gp.quicksum(x_variables[code] for code in code_list))
        for j, c in enumerate(self.corequisites):
            for code_list in c:
                model.addConstr(x_variables[j] <= gp.quicksum(x_variables[code] for code in code_list))
        for j, a in enumerate(self.alternatives):
            model.addConstr(x_variables[j] + gp.quicksum(x_variables[code] for code in a) <= 1)
        if self.improving_point_constraint:
            values = status["values"]
            best_point = status["best_point"]
            best_objective = np.dot(values, best_point)
            model.addConstr(best_objective - gp.quicksum(values[j] * x_variables[j]
                                                         for j in range(self.dimension)) <= 0)
        if self.no_good_cuts:
            for point in points["feasible"]:
                model.addConstr(gp.quicksum(x_variables[j] * (point[j] < 0.5) for j in range(self.dimension)) >= 1)
            for point in points["infeasible"]:
                model.addConstr(gp.quicksum(x_variables[j] * (point[j] > 0.5) for j in range(self.dimension)) <=
                                round(point.sum()) - 1)

        model.setObjective(gp.quicksum((weights[j] - p_variables[j]) ** 2 for j in range(self.dimension)))
        model.ModelSense = gp.GRB.MINIMIZE

        model.setParam("OutputFlag", 0)
        model.setParam("Threads", 4)
        model.setParam("NodeLimit", 50000)
        model.optimize()

        if model.Status in {2, 8, 9}:
            try:
                optimal_solution = np.array([round(x_variables[j].X) for j in range(self.dimension)])
                message = "(strategy: cut)"
            except:
                optimal_solution = None
                message = "time limit reached (strategy: cut)"
        else:
            optimal_solution = None
            message = "no more points available (strategy: cut)"

        return optimal_solution, message


class MITSIMStrategy(Strategy):

    def __init__(self, dimension: int, no_good_cuts: bool = True, improving_point_constraint: bool = False):

        super().__init__(dimension, no_good_cuts, improving_point_constraint)

        self.name = "sim"
        self.acronym = "SIM"

        df = pd.read_csv("data/MIT_subjects/subjects.csv", sep="|", index_col=False)
        self.prerequisites = [eval(p) for p in df.prerequisites]
        self.corequisites = [eval(c) for c in df.corequisites]
        self.alternatives = [eval(a) for a in df.alternatives]

    def step(self, status: dict):

        b = status["b"]
        w = status["w"]
        points = status["points"]

        model = gp.Model("simple_margin_strategy_model")

        x_variables = model.addVars(range(self.dimension), vtype=gp.GRB.BINARY)
        u_variable = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
        abs_u_variable = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)

        model.addConstr(u_variable == b + gp.quicksum(w[j] * x_variables[j] for j in range(self.dimension)))
        model.addConstr(abs_u_variable == gp.abs_(u_variable))
        for j, p in enumerate(self.prerequisites):
            for code_list in p:
                model.addConstr(x_variables[j] <= gp.quicksum(x_variables[code] for code in code_list))
        for j, c in enumerate(self.corequisites):
            for code_list in c:
                model.addConstr(x_variables[j] <= gp.quicksum(x_variables[code] for code in code_list))
        for j, a in enumerate(self.alternatives):
            model.addConstr(x_variables[j] + gp.quicksum(x_variables[code] for code in a) <= 1)
        if self.improving_point_constraint:
            values = status["values"]
            best_point = status["best_point"]
            best_objective = np.dot(values, best_point)
            model.addConstr(best_objective - gp.quicksum(values[j] * x_variables[j]
                                                         for j in range(self.dimension)) <= 0)
        if self.no_good_cuts:
            for point in points["feasible"]:
                model.addConstr(gp.quicksum(x_variables[j] * (point[j] < 0.5) for j in range(self.dimension)) >= 1)
            for point in points["infeasible"]:
                model.addConstr(gp.quicksum(x_variables[j] * (point[j] > 0.5) for j in range(self.dimension)) <=
                                round(point.sum()) - 1)

        model.setObjective(abs_u_variable)
        model.ModelSense = gp.GRB.MINIMIZE

        model.setParam("OutputFlag", 0)
        model.setParam("Threads", 4)
        model.setParam("NodeLimit", 50000)
        model.optimize()

        if model.Status in {2, 8, 9}:
            try:
                optimal_solution = np.array([round(x_variables[j].X) for j in range(self.dimension)])
                if b + np.dot(w, optimal_solution) >= 0:
                    if b + np.dot(w, optimal_solution) >= 1:
                        location = "before positive margin"
                    else:
                        location = "positive margin"
                else:
                    if b + np.dot(w, optimal_solution) <= -1:
                        location = "beyond negative margin"
                    else:
                        location = "negative margin"
                message = "(strategy: sim) (location: {})".format(location)
            except:
                optimal_solution = None
                message = "time limit reached (strategy: sim)"
        else:
            optimal_solution = None
            message = "no more points available (strategy: sim)"

        return optimal_solution, message
