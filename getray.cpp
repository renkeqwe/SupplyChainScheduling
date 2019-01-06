void get_ray(const node&tree_node, IloNumArray &alpha, IloNumArray &beta, IloNumArray &gamma)
{
	IloEnv env;
	IloModel model(env);

	IloNumVarArray alpha_var(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
	IloNumVarArray beta_var(env, m - 1, -IloInfinity, IloInfinity, ILOFLOAT);
	IloNumVarArray gamma_var(env, m, -IloInfinity, IloInfinity, ILOFLOAT);

	IloExpr sumA(env);
	for (IloInt s = 0; s < tree_node.omega[0].size(); ++s) {
		for (IloInt j = 0; j < n; ++j) {
			sumA += alpha_var[j] * tree_node.omega[0][s].a[j];
		}
		sumA += (tree_node.omega[0][s].r_s + tree_node.omega[0][s].P)*beta_var[0] + gamma_var[0];
		model.add(sumA <= 0);
		sumA.clear();
	}

	for (IloInt i = 1; i < m - 1; ++i) {
		for (IloInt s = 0; s < tree_node.omega[i].size(); ++s) {
			for (IloInt j = 0; j < n; ++j) {
				sumA += alpha_var[j] * tree_node.omega[i][s].a[j];
			}
			sumA += (tree_node.omega[i][s].r_s + tree_node.omega[i][s].P)*beta_var[i] - tree_node.omega[i][s].r_s*beta_var[i - 1] + gamma_var[i];
			model.add(sumA <= 0);
			sumA.clear();
		}
	}

	for (IloInt s = 0; s < tree_node.omega[m - 1].size(); ++s) {
		for (IloInt j = 0; j < n; ++j) {
			sumA += alpha_var[j] * tree_node.omega[m - 1][s].a[j];
		}
		sumA += (-tree_node.omega[m - 1][s].r_s*beta_var[m - 2] + gamma_var[m - 1]);
		model.add(sumA <= 0);
		sumA.clear();
	}

	IloObjective obj = IloMinimize(env, 1);
	model.add(obj);

	IloCplex cplex(env);
	cplex.extract(model);

	cplex.setParam(IloCplex::Param::Simplex::Display, 0);
	cplex.setParam(IloCplex::Param::ParamDisplay, 0);

	cplex.solve();
	if (cplex.getStatus() == IloAlgorithm::Optimal) {

		for (IloInt j = 0; j < n; ++j) alpha[j] = cplex.getValue(alpha_var[j]);
		for (IloInt i = 0; i < m - 1; ++i) beta[i] = cplex.getValue(beta_var[i]);
		for (IloInt i = 0; i < m; ++i) gamma[i] = cplex.getValue(gamma_var[i]);

	}
	else {
		cout << "The model cannot be solved in get_ray!" << endl;
	}

}
