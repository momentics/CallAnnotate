# tests/test_volume_structure.py

from app.utils import ensure_volume_structure

def test_volume_structure_creation(tmp_path):
    """Тест создания структуры volume"""
    volume_path = tmp_path / "test_volume"
    
    # Создаем структуру
    ensure_volume_structure(str(volume_path))
    
    # Проверяем что все необходимые папки созданы
    expected_dirs = [
        "incoming",
        "processing", 
        "completed",
        "failed",
        "logs/system",
        "logs/tasks",
        "models/embeddings",
        "temp"
    ]
    
    for dir_name in expected_dirs:
        dir_path = volume_path / dir_name
        assert dir_path.exists(), f"Директория {dir_name} не создана"
        assert dir_path.is_dir(), f"{dir_name} не является директорией"

def test_volume_structure_idempotent(tmp_path):
    """Тест что повторное создание структуры безопасно"""
    volume_path = tmp_path / "test_volume"
    
    # Создаем структуру дважды
    ensure_volume_structure(str(volume_path))
    ensure_volume_structure(str(volume_path))
    
    # Проверяем что все работает
    assert (volume_path / "incoming").exists()
    assert (volume_path / "processing").exists()
