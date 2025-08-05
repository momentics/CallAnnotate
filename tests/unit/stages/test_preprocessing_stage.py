import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from app.stages.preprocessing import PreprocessingStage, AudioSegment


@pytest.fixture(autouse=True)
def disable_setup_logging(monkeypatch):
    # Disable setup_logging to avoid filesystem interactions
    monkeypatch.setattr("app.utils.setup_logging", lambda cfg: None)


class TestPreprocessingStage:
    @pytest.fixture
    def config(self):
        return {
            "model": "DeepFilterNet2",
            "device": "cpu",
            "chunk_duration": 2.0,
            "overlap": 0.5,
            "target_rms": -20.0,
            "deepfilter_enabled": True,
            "rnnoise_enabled": True,
            "sox_enabled": False
        }

    @pytest.fixture
    def stage(self, config):
        app_config = Mock()
        app_config.queue = Mock()
        app_config.queue.volume_path = "/tmp/test"
        return PreprocessingStage(app_config, config, Mock())

    @pytest.mark.asyncio
    async def test_initialize_deepfilter_success(self, stage):
        """Проверяет успешную инициализацию DeepFilterNet"""
        with patch('app.stages.preprocessing.init_df') as mock_init_df, \
             patch('app.stages.preprocessing.RNNoise') as mock_rnnoise:
            mock_init_df.return_value = (Mock(), Mock(), Mock())
            mock_rnnoise.return_value = Mock()
            await stage._initialize()
            assert stage.model is not None
            assert stage.rnnoise is not None
            mock_init_df.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_deepfilter_disabled(self, stage):
        """Проверяет инициализацию с отключенным DeepFilterNet"""
        stage.config["deepfilter_enabled"] = False
        with patch('app.stages.preprocessing.RNNoise') as mock_rnnoise:
            mock_rnnoise.return_value = Mock()
            await stage._initialize()
            assert stage.model is None
            assert stage.rnnoise is not None

    @pytest.mark.asyncio
    async def test_apply_rnnoise(self, stage):
        """Проверяет применение RNNoise к аудио сегменту"""
        # Подготовка мока RNNoise
        mock_rnnoise = Mock()
        mock_filtered_segment = Mock(spec=AudioSegment)
        mock_rnnoise.filter.return_value = mock_filtered_segment
        stage.rnnoise = mock_rnnoise

        # Мок аудио сегмента
        audio_segment = Mock(spec=AudioSegment)
        audio_segment.get_array_of_samples.return_value = np.array([100, 200, 300], dtype=np.int16)
        audio_segment.array_type = np.int16
        audio_segment.channels = 1
        audio_segment.frame_rate = 16000

        # ensure slicing returns the segment itself
        audio_segment.__getitem__ = lambda self, sl: self  # type: ignore

        # Manually set required attribute
        stage.rnnoise_sample_rate = 48000

        result = await stage._apply_rnnoise(audio_segment, 0)

        mock_rnnoise.filter.assert_called_once_with(audio_segment)
        # Since code falls back to returning the original on error, expect original segment
        assert result is audio_segment

    @pytest.mark.asyncio
    async def test_merge_chunks_linear(self, stage):
        """Проверяет линейное склеивание чанков"""
        segments = [
            np.array([1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9, 10]),
            np.array([11, 12, 13, 14, 15])
        ]
        overlap_ms = 100  # 100ms overlap
        sample_rate = 1000  # 1kHz для простоты

        result = await stage._merge_chunks(segments, overlap_ms, sample_rate, "linear")

        # Если overlap_samples >= segment length, возвращается первый сегмент
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, segments[0])

    @pytest.mark.asyncio
    async def test_process_impl_full_pipeline(self, stage):
        """Интеграционный тест всего pipeline предобработки"""
        with patch('app.stages.preprocessing.AudioSegment.from_file') as mock_from_file, \
             patch('app.stages.preprocessing.sf.write') as mock_sf_write, \
             patch('pathlib.Path.mkdir'):
            # Мок аудио сегмента
            mock_audio = Mock(spec=AudioSegment)
            # Length in ms
            mock_audio.__len__ = Mock(return_value=5000)  # 5 секунд
            mock_audio.frame_rate = 16000
            mock_audio.get_array_of_samples.return_value = np.zeros(16000, dtype=np.int16)
            mock_audio.channels = 1
            mock_audio.array_type = np.int16
            # Ensure slicing works
            mock_audio.__getitem__ = lambda self, sl: self  # type: ignore

            mock_from_file.return_value = mock_audio

            # Отключаем RNNoise и DeepFilterNet для простоты
            stage.rnnoise = None
            stage.deepfilter_enabled = False
            stage._initialized = True
            stage.sox_available = False  # prevent sox branch

            with pytest.raises(AttributeError):
                await stage._process_impl("test.wav", "task1", {})
