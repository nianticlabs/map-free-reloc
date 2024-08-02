import trimesh
import numpy as np
import logging
from PIL import Image
from PIL import ImageOps

# Setup logging levels.
logging.basicConfig(level=logging.WARNING)

THICKNESS = 0.01  # controls how thick the frustum's 'bars' are

origin_frustum_verts = np.array([
    (0., 0., 0.),
    (0.375, -0.5, -0.5),
    (0.375, 0.5, -0.5),
    (-0.375, 0.5, -0.5),
    (-0.375, -0.5, -0.5),
])

frustum_edges = np.array([
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 2),
]) - 1


def get_image_box(
        image_path,
        frustum_pose,
        aspect_ratio=4.0 / 3.0,
        cam_marker_size=1.0,
        flip=False
):
    """ Gets a textured mesh of an image. """

    pil_image = Image.open(image_path)
    pil_image = ImageOps.flip(pil_image)  # flip top/bottom to align with scene space

    width = 0.75
    height = width * aspect_ratio
    width *= cam_marker_size
    height *= cam_marker_size

    if flip:
        pil_image = ImageOps.mirror(pil_image)  # flips left/right
        width = -width

    vertices = np.zeros((4, 3))
    vertices[0, :] = [width / 2, height / 2, -cam_marker_size / 2]
    vertices[1, :] = [width / 2, -height / 2, -cam_marker_size / 2]
    vertices[2, :] = [-width / 2, -height / 2, -cam_marker_size / 2]
    vertices[3, :] = [-width / 2, height / 2, -cam_marker_size / 2]

    faces = np.zeros((2, 3))
    faces[0, :] = [0, 1, 2]
    faces[1, :] = [2, 3, 0]

    uvs = np.zeros((4, 2))

    uvs[0, :] = [1.0, 0]
    uvs[1, :] = [1.0, 1.0]
    uvs[2, :] = [0, 1.0]
    uvs[3, :] = [0, 0]

    face_normals = np.zeros((2, 3))
    face_normals[0, :] = [0.0, 0.0, 1.0]
    face_normals[1, :] = [0.0, 0.0, 1.0]

    material = trimesh.visual.texture.SimpleMaterial(
        image=pil_image,
        ambient=(1.0, 1.0, 1.0, 1.0),
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    texture = trimesh.visual.TextureVisuals(
        uv=uvs,
        image=pil_image,
        material=material,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        visual=texture,
        validate=True,
        process=False
    )

    def transform_trimesh(mesh, transform):
        """ Applies a transform to a trimesh. """
        np_vertices = np.array(mesh.vertices)
        np_vertices = (transform @ np.concatenate([np_vertices, np.ones((np_vertices.shape[0], 1))], 1).T).T
        np_vertices = np_vertices / np_vertices[:, 3][:, None]
        mesh.vertices[:, 0] = np_vertices[:, 0]
        mesh.vertices[:, 1] = np_vertices[:, 1]
        mesh.vertices[:, 2] = np_vertices[:, 2]

        return mesh

    return transform_trimesh(mesh, frustum_pose)


def normalise_vector(vect):
    length = np.sqrt((vect ** 2).sum())
    return vect / length


def cuboid_from_line(line_start, line_end, color=(255, 0, 255)):
    """Approximates a line with a long cuboid
    color is a 3-element RGB tuple, with each element a uint8 value
    """
    # create two vectors which are both (a) perpendicular to the direction of the line and
    # (b) perpendicular to each other.
    direction = normalise_vector(line_end - line_start)
    random_dir = normalise_vector(np.random.rand(3))
    perpendicular_x = normalise_vector(np.cross(direction, random_dir))
    perpendicular_y = normalise_vector(np.cross(direction, perpendicular_x))

    vertices = []
    for node in (line_start, line_end):
        for x_offset in (-1, 1):
            for y_offset in (-1, 1):
                vert = node + THICKNESS * (perpendicular_y * y_offset + perpendicular_x * x_offset)
                vertices.append(vert)

    faces = [
        (4, 5, 1, 0),
        (5, 7, 3, 1),
        (7, 6, 2, 3),
        (6, 4, 0, 2),
        (0, 1, 3, 2),  # end of tube
        (6, 7, 5, 4),  # other end of tube
    ]

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    for c in (0, 1, 2):
        mesh.visual.vertex_colors[:, c] = color[c]

    return mesh


def get_position_marker(marker_pose, marker_color, marker_extent=0.03):
    """
    Generates a cube to signify a singular camera position.

    @param marker_pose: 4x4 camera pose, OpenGL convention
    @param marker_color: RGB color of the marker
    @param marker_extent: size of the marker, marker is a cube of this side length
    """
    current_pos_marker = trimesh.primitives.Box(
        extents=(marker_extent, marker_extent, marker_extent),
        transform=marker_pose)
    for c in (0, 1, 2):
        current_pos_marker.visual.vertex_colors[:, c] = marker_color[c]

    return current_pos_marker


def generate_grid(frame_idx, cmap):
    """
    Generates a grid of lines that fade in over time.

    @param frame_idx: Controls the fade-in of the grid.
    @param cmap: Color map for the grid.
    @return: trimesh object of the grid.
    """

    y_offset = -2.5
    z_offset = -2
    line_count = 100
    width = line_count // 2

    grid_edges_1 = [np.array([-width, y_offset, i + z_offset, width, y_offset, i + z_offset]) for i in
                    range(1, line_count // 2 + 1)]
    grid_edges_2 = [np.array([-width, y_offset, i + z_offset, width, y_offset, i + z_offset]) for i in
                    range(-line_count // 2, 0)]
    grid_edges_2.reverse()
    grid_edges = [val for pair in zip(grid_edges_1, grid_edges_2) for val in pair]
    grid_edges = [np.array([-width, y_offset, z_offset, width, y_offset, z_offset])] + grid_edges

    cuboids = []
    for edge_idx, edge in enumerate(grid_edges):

        opacity = max(0, min(1, (frame_idx - edge_idx) / 10) * 255)
        opacity = max(0, min(opacity, 245 - edge_idx * 5))
        if opacity == 0:
            continue

        color = cmap[int(opacity)] * 255

        line_cuboid = cuboid_from_line(line_start=edge[:3],
                                       line_end=edge[3:],
                                       color=color)
        cuboids.append(line_cuboid)

    grid_edges_1 = [np.array([i, y_offset, -width + z_offset, i, y_offset, width + z_offset]) for i in
                    range(1, line_count // 2 + 1)]
    grid_edges_2 = [np.array([i, y_offset, -width + z_offset, i, y_offset, width + z_offset]) for i in
                    range(-line_count // 2, 0)]
    grid_edges_2.reverse()
    grid_edges = [val for pair in zip(grid_edges_1, grid_edges_2) for val in pair]
    grid_edges = [np.array([0, y_offset, -width + z_offset, 0, y_offset, width + z_offset])] + grid_edges

    for edge_idx, edge in enumerate(grid_edges):

        opacity = max(0, min(1, (frame_idx - edge_idx) / 10) * 255)
        opacity = max(0, min(opacity, 245 - edge_idx * 5))
        if opacity == 0:
            continue

        color = cmap[int(opacity)] * 255

        line_cuboid = cuboid_from_line(line_start=edge[:3],
                                       line_end=edge[3:],
                                       color=color)
        cuboids.append(line_cuboid)

    if len(cuboids) == 0:
        return None
    else:
        return trimesh.util.concatenate(cuboids)


def generate_frustum_at_position(rotation, translation, color, size, border_only=False):
    """Generates a frustum mesh at a specified (rotation, translation), with given color and size
    : rotation is a 3x3 numpy array
    : translation is a 3-long numpy vector
    : color is a 3-long numpy vector or tuple or list; each element is a uint8 RGB value
    : size is a float
    : border_only is a boolean that controls whether to only draw the border of the image
    """

    transformed_frustum_verts = \
        size * rotation.dot(origin_frustum_verts.T).T + translation[None, :]

    cuboids = []
    for edge in frustum_edges:
        line_cuboid = cuboid_from_line(line_start=transformed_frustum_verts[edge[0]],
                                       line_end=transformed_frustum_verts[edge[1]],
                                       color=color)
        cuboids.append(line_cuboid)

    if border_only:
        cuboids = cuboids[4:]

    return trimesh.util.concatenate(cuboids)