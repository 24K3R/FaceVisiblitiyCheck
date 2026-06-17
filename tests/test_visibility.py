import pytest

from face_visibility import Face, extract_visible_faces, is_front_facing


def square(face_id, z):
    return Face(
        face_id,
        [
            [-1.0, -1.0, z],
            [1.0, -1.0, z],
            [1.0, 1.0, z],
            [-1.0, 1.0, z],
        ],
    )


def test_front_facing_uses_observer_to_model_direction():
    assert is_front_facing(square("near", 0.0).vertices, [0.0, 0.0, -1.0])
    assert not is_front_facing(square("near", 0.0).vertices, [0.0, 0.0, 1.0])


def test_extract_visible_faces_removes_occluded_back_face():
    near = square("near", 0.0)
    far = square("far", 1.0)

    result = extract_visible_faces([far, near], [0.0, 0.0, 1.0], resolution=64, two_sided=True)

    assert result.visible_ids == ["near"]


def test_extract_visible_faces_keeps_partially_visible_face():
    near = Face("near", [[-1.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    far = square("far", 1.0)

    result = extract_visible_faces([far, near], [0.0, 0.0, 1.0], resolution=64, two_sided=True)

    assert result.visible_ids == ["far", "near"]


def test_rejects_zero_view_direction():
    with pytest.raises(ValueError, match="non-zero"):
        extract_visible_faces([square("face", 0.0)], [0.0, 0.0, 0.0])
