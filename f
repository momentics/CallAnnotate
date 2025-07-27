============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-8.4.1, pluggy-1.5.0 -- C:\Users\momentics\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: C:\CallAnnotate
configfile: pytest.ini
plugins: anyio-4.9.0, asyncio-1.1.0, cov-6.2.1
asyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 9 items

tests/test_app_endpoints.py::test_health_and_info ERROR                  [ 11%]
tests/test_app_endpoints.py::test_create_job_too_large ERROR             [ 22%]
tests/test_app_endpoints.py::test_create_job_unsupported_type ERROR      [ 33%]
tests/test_app_endpoints.py::test_delete_job_endpoint ERROR              [ 44%]
tests/test_websocket_integration.py::test_websocket_connection_and_heartbeat PASSED [ 55%]
tests/test_websocket_integration.py::test_websocket_task_subscription PASSED [ 66%]
tests/test_websocket_integration.py::test_websocket_audio_upload PASSED  [ 77%]
tests/test_websocket_integration.py::test_websocket_connection_disconnect PASSED [ 88%]
tests/test_websocket_integration.py::test_websocket_invalid_message PASSED [100%]

=================================== ERRORS ====================================
___________________ ERROR at setup of test_health_and_info ____________________
file C:\CallAnnotate\tests\test_app_endpoints.py, line 11
  def test_health_and_info(client):
E       fixture 'client' not found
>       available fixtures: _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, event_loop_policy, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, mock_dependencies, monkeypatch, no_cover, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, setup_test_environment, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory
>       use 'pytest --fixtures [testpath]' for help on them.

C:\CallAnnotate\tests\test_app_endpoints.py:11
_________________ ERROR at setup of test_create_job_too_large _________________
file C:\CallAnnotate\tests\test_app_endpoints.py, line 26
  def test_create_job_too_large(client, monkeypatch, tmp_path):
E       fixture 'client' not found
>       available fixtures: _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, event_loop_policy, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, mock_dependencies, monkeypatch, no_cover, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, setup_test_environment, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory
>       use 'pytest --fixtures [testpath]' for help on them.

C:\CallAnnotate\tests\test_app_endpoints.py:26
_____________ ERROR at setup of test_create_job_unsupported_type ______________
file C:\CallAnnotate\tests\test_app_endpoints.py, line 35
  def test_create_job_unsupported_type(client, tmp_path):
E       fixture 'client' not found
>       available fixtures: _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, event_loop_policy, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, mock_dependencies, monkeypatch, no_cover, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, setup_test_environment, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory
>       use 'pytest --fixtures [testpath]' for help on them.

C:\CallAnnotate\tests\test_app_endpoints.py:35
_________________ ERROR at setup of test_delete_job_endpoint __________________
file C:\CallAnnotate\tests\test_app_endpoints.py, line 42
  def test_delete_job_endpoint(client, tmp_path):
E       fixture 'client' not found
>       available fixtures: _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, event_loop_policy, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, mock_dependencies, monkeypatch, no_cover, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, setup_test_environment, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory
>       use 'pytest --fixtures [testpath]' for help on them.

C:\CallAnnotate\tests\test_app_endpoints.py:42
=========================== short test summary info ===========================
ERROR tests/test_app_endpoints.py::test_health_and_info
ERROR tests/test_app_endpoints.py::test_create_job_too_large
ERROR tests/test_app_endpoints.py::test_create_job_unsupported_type
ERROR tests/test_app_endpoints.py::test_delete_job_endpoint
========================= 5 passed, 4 errors in 0.84s =========================
