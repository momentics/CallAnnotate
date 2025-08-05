import pytest
import torch
import logging
from unittest.mock import Mock
from app.models_registry import ModelRegistry

def test_singleton_and_load_and_unload_and_clear_and_memory():
    reg1 = ModelRegistry()
    reg2 = ModelRegistry()
    assert reg1 is reg2

    # loader
    class M:
        def parameters(self): return []
    logger = logging.getLogger("test")
    model = reg1.get_model(logger, "k", lambda: M(), info=1)
    loaded = reg1.get_loaded_models()
    assert "k" in loaded
    mem = reg1.get_memory_usage()
    assert "total_models" in mem
    # clear
    reg1.clear_all(logger)
    assert reg1.get_loaded_models() == {}
