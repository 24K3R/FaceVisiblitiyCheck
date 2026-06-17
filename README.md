# FaceVisiblitiyCheck

Check whether discretized explicit-surface faces are visible from a given view direction.

## Algorithm

For surfaces that are already explicitly represented and discretized into face elements, the implemented visible-face extraction uses an orthographic view:

1. **Back-face culling**: compute each element normal with Newell's method. If the view direction points from the observer toward the model, a one-sided face is potentially visible only when `dot(normal, view_direction) < 0`. Use `two_sided=True` when the surface orientation is unknown or both sides should be visible.
2. **Projection**: build a camera basis perpendicular to the view direction and project all candidate vertices to the view plane.
3. **Depth test**: rasterize candidate polygons into a z-buffer. Depth is `dot(point, view_direction)`, so smaller values are closer to the observer.
4. **Visible extraction**: record only faces that own at least `coverage_threshold` pixels in the depth buffer.

This approach supports triangles, quads, and small convex polygons produced by an existing meshing/discretization method. Increase `resolution` for small elements or more accurate silhouette extraction.

## Usage

```python
from face_visibility import Face, extract_visible_faces

faces = [
    Face("near", [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]),
    Face("far", [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]),
]

# Direction points from observer to model.  Here the observer is on negative Z,
# looking toward positive Z, so the z=0 face hides the z=1 face.
result = extract_visible_faces(faces, [0, 0, 1], resolution=512, two_sided=True)
print(result.visible_ids)  # ["near"]
```

`Face.payload` can store your original element metadata, such as surface type, parametric cell indices, or application-specific IDs.
