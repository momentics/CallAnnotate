============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-8.4.1, pluggy-1.5.0 -- C:\Users\momentics\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: C:\CallAnnotate
configfile: pytest.ini
plugins: anyio-3.7.1, asyncio-1.1.0, cov-6.2.1, mock-3.14.1
asyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 24 items

tests/test_app_endpoints.py::test_health_and_info PASSED                 [  4%]
tests/test_app_endpoints.py::test_create_job_too_large PASSED            [  8%]
tests/test_app_endpoints.py::test_create_job_unsupported_type PASSED     [ 12%]
tests/test_app_endpoints.py::test_delete_job_endpoint PASSED             [ 16%]
tests/test_config.py::test_recognition_validator FAILED                  [ 20%]
tests/test_config.py::test_load_basic_yaml PASSED                        [ 25%]
tests/test_config.py::test_load_settings_env_override PASSED             [ 29%]
tests/test_models_registry.py::test_singleton PASSED                     [ 33%]
tests/test_models_registry.py::test_get_and_unload_model PASSED          [ 37%]
tests/test_models_registry.py::test_memory_usage_and_clear_all PASSED    [ 41%]
tests/test_stages_base.py::test_process_success PASSED                   [ 45%]
tests/test_stages_base.py::test_process_error PASSED                     [ 50%]
tests/test_stages_base.py::test_timing_and_model_info PASSED             [ 54%]
tests/test_utils.py::test_validate_audio_file_ok PASSED                  [ 58%]
tests/test_utils.py::test_validate_audio_file_bad_ext PASSED             [ 62%]
tests/test_utils.py::test_extract_audio_metadata PASSED                  [ 66%]
tests/test_utils.py::test_ensure_directory_and_cleanup PASSED            [ 70%]
tests/test_utils.py::test_format_duration PASSED                         [ 75%]
tests/test_utils.py::test_get_supported_audio_formats PASSED             [ 79%]
tests/test_websocket_integration.py::TestWebSocketIntegration::test_websocket_connection PASSED [ 83%]
tests/test_websocket_integration.py::TestWebSocketIntegration::test_websocket_task_subscription PASSED [ 87%]
tests/test_websocket_integration.py::TestWebSocketIntegration::test_websocket_audio_upload PASSED [ 91%]
tests/test_websocket_integration.py::TestWebSocketIntegration::test_websocket_disconnect PASSED [ 95%]
tests/test_websocket_integration.py::test_websocket_manager_functionality PASSED [100%]

================================== FAILURES ===================================
_________________________ test_recognition_validator __________________________
tests\test_config.py:20: in test_recognition_validator
    assert Path(cfg.embeddings_path).exists()
           ^^^^^^^^^^^^^^^^^^^^^^^^^
C:\Users\momentics\AppData\Local\Programs\Python\Python312\Lib\site-packages\fastapi\param_functions.py:306: in Path
    return params.Path(
C:\Users\momentics\AppData\Local\Programs\Python\Python312\Lib\site-packages\fastapi\params.py:182: in __init__
    assert default is ..., "Path parameters cannot have a default value"
           ^^^^^^^^^^^^^^
E   AssertionError: Path parameters cannot have a default value
============================== warnings summary ===============================
tests/test_config.py::test_recognition_validator
  C:\Users\momentics\AppData\Local\Programs\Python\Python312\Lib\inspect.py:1007: UserWarning: Module 'speechbrain.pretrained' was deprecated, redirecting to 'speechbrain.inference'. Please update your script. This is a change from SpeechBrain 1.0. See: https://github.com/speechbrain/speechbrain/releases/tag/v1.0.0
    if ismodule(module) and hasattr(module, '__file__'):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/test_config.py::test_recognition_validator - AssertionError: Pat...
=================== 1 failed, 23 passed, 1 warning in 3.04s ===================
