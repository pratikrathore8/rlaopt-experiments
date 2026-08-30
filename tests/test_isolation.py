from rlaopt_experiments.isolation import ProblemWorker


def test_problem_worker_enforces_hard_timeout():
    specification = {"n": 16, "p": 8, "alpha": 1.0, "factor_seed": 1, "response_seed": 2}
    worker = ProblemWorker(specification, "cpu")
    try:
        assert worker.wait_until_ready(30)["kind"] == "ready"
        outcome = worker.solve({
            "solver": "torch_qr", "ridge": 1e-3, "native_tolerance": 0.0,
            "kkt_tolerance": 1e-6, "max_iters": 16, "timeout_seconds": 300,
            "rank": 4, "nystrom_seed": 3,
        }, timeout_seconds=1e-6)
        assert outcome["kind"] == "timeout"
        assert not worker.alive
    finally:
        worker.close()
