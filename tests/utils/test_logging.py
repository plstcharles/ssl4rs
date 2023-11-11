import os
import pathlib
import pickle
import time

import numpy as np
import pytest

import ssl4rs.utils.logging


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path / "test_logs"


@pytest.fixture
def logger(temp_dir):
    assert not pathlib.Path(temp_dir).exists()
    loggr = ssl4rs.utils.logging.DebugLogger(output_root_path=temp_dir)
    assert pathlib.Path(temp_dir).exists()
    assert pathlib.Path(loggr.root_dir).exists()
    assert loggr.root_dir == str(temp_dir)
    log_dir = pathlib.Path(loggr.log_dir)
    assert log_dir.exists()
    pickle_paths = list(log_dir.glob("*.pkl"))
    assert len(pickle_paths) == 1
    with open(pickle_paths[0], "rb") as fd:
        metadata = pickle.load(fd)
        assert "runtime_hash" in metadata["kwargs"]
    return loggr


def test_log_hyperparams(logger):
    params = {"param1": 10, "param2": "test"}
    logger.log_hyperparams(params)
    log_files = sorted(list(pathlib.Path(logger.log_dir).glob("*.pkl")))
    with open(log_files[-1], "rb") as fd:
        data = pickle.load(fd)
        assert data["func_name"] == "log_hyperparams"
        assert data["kwargs"]["params"] == params


def test_log_metrics(logger):
    metrics = {"accuracy": 0.95, "loss": 0.05}
    step = 100
    logger.log_metrics(metrics, step=step)
    log_files = sorted(list(pathlib.Path(logger.log_dir).glob("*.pkl")))
    with open(log_files[-1], "rb") as fd:
        data = pickle.load(fd)
        assert data["func_name"] == "log_metrics"
        assert data["kwargs"]["metrics"] == metrics
        assert data["kwargs"]["step"] == step


def test_log_graph(logger):
    model = "dummy_model"
    input_array = [1, 2, 3]
    logger.log_graph(model, input_array=input_array)
    log_files = sorted(list(pathlib.Path(logger.log_dir).glob("*.pkl")))
    with open(log_files[-1], "rb") as fd:
        data = pickle.load(fd)
        assert data["func_name"] == "log_graph"
        assert data["kwargs"]["model"] == model
        assert data["kwargs"]["input_array"] == input_array


def test_log_with_name_conflict_avoidance(logger):
    competing_logger = ssl4rs.utils.logging.DebugLogger(output_root_path=logger.root_dir)
    logger.log_hyperparams({"param": 1})
    logger.log_hyperparams({"param": 2})
    competing_logger.log_hyperparams({"param": 3})
    competing_logger.log_hyperparams({"param": 4})
    log_files = sorted(list(pathlib.Path(logger.log_dir).glob("*.pkl")))
    competing_log_files = sorted(list(pathlib.Path(competing_logger.log_dir).glob("*.pkl")))
    assert len(competing_log_files) == 3  # init + 2 log calls
    assert len(log_files) == 3  # init + 2 log calls
    with open(log_files[-1], "rb") as fd:
        assert pickle.load(fd)["kwargs"]["params"]["param"] == 2
    with open(competing_log_files[-1], "rb") as fd:
        assert pickle.load(fd)["kwargs"]["params"]["param"] == 4


def test_logger_arbitrary_method_call(logger):
    logger.log_custom("potato")
    log_files = sorted(list(pathlib.Path(logger.log_dir).glob("*.pkl")))
    with open(log_files[-1], "rb") as fd:
        data = pickle.load(fd)
        assert data["func_name"] == "log_custom"
        assert data["args"][0] == "potato"


def test_logger_parse_metrics_logs(logger):
    logger.log_hyperparams({"param": 1})  # should never show up in parsed logs
    logger.log_metrics({"accuracy": 0.95, "loss": 0.05}, step=100)
    logger.log_metrics({"loss": 0.25, "potato": 1}, step=200)
    metrics = ssl4rs.utils.logging.DebugLogger.parse_metric_logs(logger.log_dir)
    assert metrics.shape == (2, 4)
    assert metrics[metrics["step"] == 100]["loss"].item() == 0.05
    assert metrics[metrics["step"] == 200]["loss"].item() == 0.25
    assert np.isnan(metrics[metrics["step"] == 200]["accuracy"].item())
