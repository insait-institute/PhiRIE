import hashlib
import pytest
from interface.demo_native_progress import released_video


def test_equal_length_video_substitution_is_rejected(tmp_path):
    video=tmp_path/'continuous.mp4';video.write_bytes(b'original-frames')
    digest=hashlib.sha256(video.read_bytes()).hexdigest()
    sources={'unit_7_video_path':dict(path=str(video),sha256=digest)}
    assert released_video(sources,'unit_7_result_path',video)==digest
    video.write_bytes(b'replaced-frames')
    with pytest.raises(ValueError,match='changed after'):
        released_video(sources,'unit_7_result_path',video)


def test_video_must_be_sealed_for_its_own_released_result(tmp_path):
    video=tmp_path/'episode.mp4';video.write_bytes(b'frames')
    record=dict(path=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='no matching'):
        released_video({'unit_8_video_path':record},'unit_7_result_path',video)
    with pytest.raises(ValueError,match='no matching'):
        released_video({},'unit_7_result_path',video)
    with pytest.raises(ValueError,match='released result identity'):
        released_video({'unit_7_video_path':record},'not_a_result',video)
