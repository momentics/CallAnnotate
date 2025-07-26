============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-8.4.1, pluggy-1.5.0
rootdir: C:\CallAnnotate
configfile: pytest.ini
plugins: anyio-4.9.0, asyncio-1.1.0, cov-6.2.1
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 5 items

tests\test_app_endpoints.py FFFFF                                        [100%]

================================== FAILURES ===================================
____________________________ test_health_and_info _____________________________

client = <starlette.testclient.TestClient object at 0x000001B5F3CF8620>

    def test_health_and_info(client):
        # Health
        res = client.get("/health")
        assert res.status_code == status.HTTP_200_OK
        body = res.json()
        assert body["status"] == "healthy"
>       assert "queue_length" in body
E       AssertionError: assert 'queue_length' in {'active_tasks': 0, 'queue_size': 0, 'status': 'healthy', 'version': '1.0.0'}

tests\test_app_endpoints.py:17: AssertionError
---------------------------- Captured stderr call -----------------------------
2025-07-27 01:43:18,541 - httpx - INFO - HTTP Request: GET http://testserver/health "HTTP/1.1 200 OK"
------------------------------ Captured log call ------------------------------
INFO     httpx:_client.py:1025 HTTP Request: GET http://testserver/health "HTTP/1.1 200 OK"
_____________________ test_create_and_get_job_and_result ______________________

client = <starlette.testclient.TestClient object at 0x000001B5F3CDCE90>
tmp_path = WindowsPath('C:/Users/momentics/AppData/Local/Temp/pytest-of-momentics/pytest-21/test_create_and_get_job_and_re0')

    def test_create_and_get_job_and_result(client, tmp_path):
        # Создаём тестовый аудиофайл
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00\x01\x02")
    
        # POST /jobs
        with open(audio, "rb") as f:
            res = client.post(
                "/jobs",
                files={"file": ("test.wav", f, "audio/wav")},
            )
>       assert res.status_code == status.HTTP_201_CREATED
E       assert 404 == 201
E        +  where 404 = <Response [404 Not Found]>.status_code
E        +  and   201 = status.HTTP_201_CREATED

tests\test_app_endpoints.py:37: AssertionError
---------------------------- Captured stderr call -----------------------------
2025-07-27 01:43:18,784 - httpx - INFO - HTTP Request: POST http://testserver/jobs "HTTP/1.1 404 Not Found"
------------------------------ Captured log call ------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/jobs "HTTP/1.1 404 Not Found"
__________________________ test_create_job_too_large __________________________

client = <starlette.testclient.TestClient object at 0x000001B5F3DF7050>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x000001B5F3E2FFB0>
tmp_path = WindowsPath('C:/Users/momentics/AppData/Local/Temp/pytest-of-momentics/pytest-21/test_create_job_too_large0')

    def test_create_job_too_large(client, monkeypatch, tmp_path):
        # Monkey-patch MAX_FILE_SIZE малый
        monkeypatch.setenv("MAX_FILE_SIZE", "2")
        large = tmp_path / "big.mp3"
        large.write_bytes(b"\x00\x01\x02")
        with open(large, "rb") as f:
            res = client.post("/jobs", files={"file": ("big.mp3", f, "audio/mp3")})
>       assert res.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
E       assert 404 == 413
E        +  where 404 = <Response [404 Not Found]>.status_code
E        +  and   413 = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

tests\test_app_endpoints.py:67: AssertionError
---------------------------- Captured stderr call -----------------------------
2025-07-27 01:43:18,811 - httpx - INFO - HTTP Request: POST http://testserver/jobs "HTTP/1.1 404 Not Found"
------------------------------ Captured log call ------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/jobs "HTTP/1.1 404 Not Found"
______________________ test_create_job_unsupported_type _______________________

client = <starlette.testclient.TestClient object at 0x000001B5F3E2FC80>
tmp_path = WindowsPath('C:/Users/momentics/AppData/Local/Temp/pytest-of-momentics/pytest-21/test_create_job_unsupported_ty0')

    def test_create_job_unsupported_type(client, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text("not audio")
        with open(bad, "rb") as f:
            res = client.post("/jobs", files={"file": ("bad.txt", f, "text/plain")})
>       assert res.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
E       assert 404 == 415
E        +  where 404 = <Response [404 Not Found]>.status_code
E        +  and   415 = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

tests\test_app_endpoints.py:74: AssertionError
---------------------------- Captured stderr call -----------------------------
2025-07-27 01:43:18,831 - httpx - INFO - HTTP Request: POST http://testserver/jobs "HTTP/1.1 404 Not Found"
------------------------------ Captured log call ------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/jobs "HTTP/1.1 404 Not Found"
__________________________ test_delete_job_endpoint ___________________________

client = <starlette.testclient.TestClient object at 0x000001B5F3E2E570>
tmp_path = WindowsPath('C:/Users/momentics/AppData/Local/Temp/pytest-of-momentics/pytest-21/test_delete_job_endpoint0')

    def test_delete_job_endpoint(client, tmp_path):
        # Создаём и ждем jobb
        audio = tmp_path / "d.wav"
        audio.write_bytes(b"\x00")
        with open(audio, "rb") as f:
            res = client.post("/jobs", files={"file": ("d.wav", f, "audio/wav")})
>       job_id = res.json()["job_id"]
                 ^^^^^^^^^^^^^^^^^^^^
E       KeyError: 'job_id'

tests\test_app_endpoints.py:82: KeyError
---------------------------- Captured stderr call -----------------------------
2025-07-27 01:43:18,861 - httpx - INFO - HTTP Request: POST http://testserver/jobs "HTTP/1.1 404 Not Found"
------------------------------ Captured log call ------------------------------
INFO     httpx:_client.py:1025 HTTP Request: POST http://testserver/jobs "HTTP/1.1 404 Not Found"
============================== warnings summary ===============================
src\app\app.py:134
  C:\CallAnnotate\tests\..\src\app\app.py:134: DeprecationWarning: 
          on_event is deprecated, use lifespan event handlers instead.
  
          Read more about it in the
          [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
          
    @app.on_event("startup")

..\Users\momentics\AppData\Local\Programs\Python\Python312\Lib\site-packages\fastapi\applications.py:4495
..\Users\momentics\AppData\Local\Programs\Python\Python312\Lib\site-packages\fastapi\applications.py:4495
  C:\Users\momentics\AppData\Local\Programs\Python\Python312\Lib\site-packages\fastapi\applications.py:4495: DeprecationWarning: 
          on_event is deprecated, use lifespan event handlers instead.
  
          Read more about it in the
          [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
          
    return self.router.on_event(event_type)

src\app\app.py:141
  C:\CallAnnotate\tests\..\src\app\app.py:141: DeprecationWarning: 
          on_event is deprecated, use lifespan event handlers instead.
  
          Read more about it in the
          [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
          
    @app.on_event("shutdown")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/test_app_endpoints.py::test_health_and_info - AssertionError: as...
FAILED tests/test_app_endpoints.py::test_create_and_get_job_and_result - asse...
FAILED tests/test_app_endpoints.py::test_create_job_too_large - assert 404 ==...
FAILED tests/test_app_endpoints.py::test_create_job_unsupported_type - assert...
FAILED tests/test_app_endpoints.py::test_delete_job_endpoint - KeyError: 'job...
======================== 5 failed, 4 warnings in 0.39s ========================
