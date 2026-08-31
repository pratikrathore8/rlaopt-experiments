from rlaopt_experiments.sharding import shard_cpu_records


def test_shards_are_complete_balanced_and_rotate_solvers():
    records = [
        {"n": n, "p": n, "alpha": 1.0, "seed": 0, "solver": solver}
        for n in (256, 512)
        for solver in ("nystrom", "cg", "lsqr", "qr")
    ]

    first, second = shard_cpu_records(records)

    assert len(first) == len(second) == 4
    assert {id(record) for record in first}.isdisjoint({id(record) for record in second})
    assert {id(record) for record in first + second} == {id(record) for record in records}
    assert {record["solver"] for record in first} == {"nystrom", "cg", "lsqr", "qr"}
    assert {record["solver"] for record in second} == {"nystrom", "cg", "lsqr", "qr"}
