from rlaopt_experiments.isolation import ProblemWorker


def test_problem_worker_enforces_hard_timeout():
    specification = {"n": 16, "p": 8, "alpha": 1.0, "factor_seed": 1, "response_seed": 2}
    worker = ProblemWorker(specification, "cpu", "synthetic_ridge")
    try:
        assert worker.wait_until_ready(30)["kind"] == "ready"
        outcome = worker.solve(
            {
                "solver": "torch_qr",
                "ridge": 1e-3,
                "native_tolerance": 0.0,
                "kkt_tolerance": 1e-6,
                "max_iters": 16,
                "timeout_seconds": 300,
                "rank": 4,
                "nystrom_seed": 3,
            },
            timeout_seconds=1e-6,
        )
        assert outcome["kind"] == "timeout"
        assert not worker.alive
    finally:
        worker.close()


def test_problem_worker_dispatches_selected_ridge_suite():
    specification = {"n": 16, "p": 8, "alpha": 1.0, "factor_seed": 1, "response_seed": 2}
    worker = ProblemWorker(specification, "cpu", "synthetic_ridge")
    try:
        ready = worker.wait_until_ready(30)
        assert ready["kind"] == "ready"
        assert ready["worker_metadata"]["problem_generator"]

        outcome = worker.solve(
            {
                "solver": "torch_qr",
                "ridge": 1e-3,
                "native_tolerance": 0.0,
                "kkt_tolerance": 1e-6,
                "max_iters": 16,
                "timeout_seconds": 30,
                "rank": 4,
                "nystrom_seed": 3,
            },
            timeout_seconds=30,
        )

        assert outcome["kind"] == "result"
        assert outcome["native_success"]
        assert outcome["runtime_eligible"]
        assert outcome["accuracy"]["external_success"]
        assert outcome["accuracy"]["relative_kkt"] < 1e-6
        assert outcome["diagnostics"]["full_condition_number"] > 1
    finally:
        worker.close()
