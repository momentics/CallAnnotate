# -*- coding: utf-8 -*-
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

from app.models_registry import ModelRegistry

def test_singleton():
    r1 = ModelRegistry()
    r2 = ModelRegistry()
    assert r1 is r2

def test_get_and_unload_model():
    registry = ModelRegistry()
    def loader():
        # simple object
        return [1, 2, 3]
    key = "test"
    m = registry.get_model(key, loader)
    assert m == [1,2,3]
    # second get returns same
    m2 = registry.get_model(key, lambda: [])
    assert m2 == [1,2,3]
    # unload
    assert registry.unload_model(key)
    # after unload, loader called again
    m3 = registry.get_model(key, loader)
    assert m3 == [1,2,3]

def test_memory_usage_and_clear_all():
    registry = ModelRegistry()
    registry.clear_all()
    info = registry.get_memory_usage()
    assert info["total_models"] == 0
