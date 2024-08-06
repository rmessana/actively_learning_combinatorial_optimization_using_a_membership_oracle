import gurobipy as gp
import numpy as np


class LinearSeparator:

    def __init__(self, data: dict):

        self.data = data

        self.dimension = data["feasible"].shape[1]

    def update(self, new_data: dict):
        pass

    def solve(self, verbose: int = 0):
        pass


class SEPLinearSeparator(LinearSeparator):

    def __init__(self, data: dict):

        super().__init__(data)

        self.R = np.sqrt(self.dimension)
        x = np.append(np.zeros(self.dimension), 1.0)
        self.A = np.array([x / self.dual_norm(x)])
        x = np.append(- np.ones(self.dimension), - 1.0)
        self.A = np.concatenate([self.A, [x / self.dual_norm(x)]], axis=0)
        for j in range(self.dimension):
            x = np.append(np.zeros(self.dimension), 0.0)
            x[j] = - 1.0
            self.A = np.concatenate([self.A, [x / self.dual_norm(x)]], axis=0)
            x = np.append(np.zeros(self.dimension), 1.0)
            x[j] = 1.0
            self.A = np.concatenate([self.A, [x / self.dual_norm(x)]], axis=0)

    def dual_norm(self, x):

        return (1 / np.sqrt(2)) * np.linalg.norm(np.append(x[:-1] * self.R, 1))

    def update(self, new_data: np.ndarray):

        for x in new_data["feasible"]:
            x = np.append(x, 1)
            self.A = np.concatenate([self.A, [x / self.dual_norm(x)]], axis=0)
        for x in new_data["infeasible"]:
            x = np.append(-x, -1)
            self.A = np.concatenate([self.A, [x / self.dual_norm(x)]], axis=0)

    def solve_nomod(self, new_data: dict, verbose: int = 0):
        
        A = self.A.copy()
        for x in new_data["feasible"]:
            x = np.append(x, 1)
            A = np.concatenate([A, [x / self.dual_norm(x)]], axis=0)
        for x in new_data["infeasible"]:
            x = np.append(-x, -1)
            A = np.concatenate([A, [x / self.dual_norm(x)]], axis=0)

        model = gp.Model("model")
        p_variables = model.addVars(range(self.dimension + 1), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                                    ub=gp.GRB.INFINITY)
        c_variables = model.addVars(range(A.shape[0]), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)
        model.addConstr(c_variables.sum() == 1)
        model.addConstrs((gp.quicksum(c_variables[i] * A[i, j] for i in range(A.shape[0])) == p_variables[j]
                          for j in range(self.dimension + 1)))
        model.setObjective(0.25 * (np.power(self.R, 2) * gp.quicksum(p_variables[j] * p_variables[j]
                                                                     for j in range(self.dimension)) +
                                   p_variables[self.dimension] * p_variables[self.dimension]))

        model.ModelSense = gp.GRB.MINIMIZE
        model.setParam('OutputFlag', verbose)
        model.setParam("Threads", 4)
        model.optimize()

        if model.Status == 2:
            p = np.array([p_variables[j].X for j in range(self.dimension + 1)])
            gradient = np.append(0.5 * np.power(self.R, 2) * p[:-1], 0.5 * p[-1])
            weights = - gradient[:-1] / gradient[-1]
            w = - weights
            solution = {"b": 1.0, "w": w}
        else:
            solution = None
            
        return solution

    def solve(self, verbose: int = 0):

        model = gp.Model("model")
        p_variables = model.addVars(range(self.dimension + 1), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                                    ub=gp.GRB.INFINITY)
        c_variables = model.addVars(range(self.A.shape[0]), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)
        model.addConstr(c_variables.sum() == 1)
        model.addConstrs((gp.quicksum(c_variables[i] * self.A[i, j] for i in range(self.A.shape[0])) == p_variables[j]
                          for j in range(self.dimension + 1)))
        model.setObjective(0.25 * (np.power(self.R, 2) * gp.quicksum(p_variables[j] * p_variables[j]
                                                                     for j in range(self.dimension)) +
                                   p_variables[self.dimension] * p_variables[self.dimension]))

        model.ModelSense = gp.GRB.MINIMIZE
        model.setParam('OutputFlag', verbose)
        model.setParam("Threads", 4)
        model.optimize()

        if model.Status == 2:
            p = np.array([p_variables[j].X for j in range(self.dimension + 1)])
            gradient = np.append(0.5 * np.power(self.R, 2) * p[:-1], 0.5 * p[-1])
            weights = - gradient[:-1] / gradient[-1]
            w = - weights
            solution = {"b": 1.0, "w": w}
        else:
            solution = None
            
        return solution


class SVMLinearSeparator(LinearSeparator):

    def __init__(self, data: dict):

        super().__init__(data)

        self.model = gp.Model("linear_separation_model")
        self.variables = {"b": self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                                 name="b"),
                          'w': self.model.addVars([j for j in range(self.dimension)], vtype=gp.GRB.CONTINUOUS,
                                                  lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="w")}
        self.model.addConstrs((self.variables["b"] + gp.quicksum(data["feasible"][i, j] * self.variables['w'][j]
                                                                 for j in range(self.dimension)) >= 1
                               for i in range(self.data["feasible"].shape[0])), "feasible_point_separation_constraints")
        self.model.addConstrs((self.variables["b"] + gp.quicksum(data["infeasible"][i, j] * self.variables['w'][j]
                                                                 for j in range(self.dimension)) <= - 1
                               for i in range(self.data["infeasible"].shape[0])),
                              "infeasible_point_separation_constraints")
        self.model.setObjective(gp.quicksum(self.variables['w'][j] * self.variables['w'][j]
                                            for j in range(self.dimension)))
        self.model.setParam("Threads", 4)
        self.model.ModelSense = gp.GRB.MINIMIZE

    def update(self, new_data: dict):

        self.model.addConstrs((self.variables["b"] + gp.quicksum(new_data["feasible"][i, j] * self.variables['w'][j]
                                                                 for j in range(self.dimension)) >= 1
                               for i in range(new_data["feasible"].shape[0])))
        self.model.addConstrs((self.variables["b"] + gp.quicksum(new_data["infeasible"][i, j] * self.variables['w'][j]
                                                                 for j in range(self.dimension)) <= - 1
                               for i in range(new_data["infeasible"].shape[0])))
        
    def solve_nomod(self, new_data: dict, verbose: int = 0):
        
        temp_model = self.model.copy()
        temp_variables = {"b": temp_model.getVarByName("b"),
                          "w": [temp_model.getVarByName("w[{}]".format(j)) for j in range(self.dimension)]}
        temp_model.addConstrs(
            (temp_variables["b"] + gp.quicksum(new_data["feasible"][i, j] * temp_variables["w"][j]
                                               for j in range(self.dimension)) >= 1 
             for i in range(new_data["feasible"].shape[0]))), 
        temp_model.addConstrs(
            (temp_variables["b"] + gp.quicksum(new_data["infeasible"][i, j] * temp_variables["w"][j]
                                               for j in range(self.dimension)) <= - 1
             for i in range(new_data["infeasible"].shape[0])))

        temp_model.setParam("OutputFlag", verbose)
        temp_model.setParam("Threads", 4)
        for method_code in {2: "barrier", 1: "dual simplex", 0: "primal simplex"}:
            print({2: "barrier", 1: "dual simplex", 0: "primal simplex"}[method_code])
            temp_model.setParam("Method", method_code)
            temp_model.optimize()
            if temp_model.Status == 2:
                break

        if temp_model.Status == 2:
            b = temp_variables["b"].X
            w = np.array([temp_variables["w"][j].X for j in range(self.dimension)])
            solution = {"b": b, "w": w}
        else:
            print("barrier + BarConvTol")
            self.model.setParam("Method", 2)
            self.model.setParam("BarConvTol", 0.1)
            temp_model.optimize()
            if temp_model.Status == 2:
                b = temp_variables["b"].X
                w = np.array([temp_variables["w"][j].X for j in range(self.dimension)])
                solution = {"b": b, "w": w}
            else:
                solution = None

        return solution

    def solve(self, verbose: int = 0):

        self.model.setParam("OutputFlag", verbose)
        self.model.setParam("Threads", 4)
        for method_code in {2: "barrier", 1: "dual simplex", 0: "primal simplex"}:
            print({2: "barrier", 1: "dual simplex", 0: "primal simplex"}[method_code])
            self.model.setParam("Method", method_code)
            self.model.optimize()
            if self.model.Status == 2:
                break

        if self.model.Status == 2:
            b = self.variables["b"].X
            w = np.array([self.variables["w"][j].X for j in range(self.dimension)])
            solution = {"b": b, "w": w}
        else:
            print("barrier + BarConvTol")
            self.model.setParam("Method", 2)
            self.model.setParam("BarConvTol", 0.1)
            self.model.optimize()
            if self.model.Status == 2:
                b = self.variables["b"].X
                w = np.array([self.variables["w"][j].X for j in range(self.dimension)])
                solution = {"b": b, "w": w}
            else:
                solution = None

        return solution
