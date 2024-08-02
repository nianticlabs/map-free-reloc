import numpy as np
from scipy.linalg import svd

class LazyCamera:
    """Smooth and slightly delayed scene camera.

    Implements a rolling average of last few camera positions.
    Also zooms out to display the whole scene.
    """

    # buffer holding last m camera positions
    m_camera_buffer = None

    m_camera_buffer_size = None
    m_backwards_offset = None

    def __init__(self,
                 camera_buffer_size=20,
                 backwards_offset=4):
        """Constructor.

        Parameters:
            camera_buffer_size: Number of last few cameras to consider
            backwards_offset: Move observing camera backwards from current view, in meters
        """

        self.m_camera_buffer = []
        self.m_camera_buffer_size = camera_buffer_size
        self.m_backwards_offset = backwards_offset

    @staticmethod
    def _orthonormalize_rotation(T):
        """Takes a 4x4 matrix and orthonormalizes the upper left 3x3 using SVD

        Returns:
            T with orthonormalized upper 3x3
        """

        R = T[:3, :3]

        # see https://arxiv.org/pdf/2006.14616.pdf Eq.2
        U, S, Vt = svd(R)
        Z = np.eye(3)
        Z[-1, -1] = np.sign(np.linalg.det(U @ Vt))
        R = U @ Z @ Vt

        T[:3, :3] = R

        return T

    def update_camera(self, view):
        """Update lazy camera with new view.

        Parameters:
            view: New camera view, 4x4 matrix
        """

        observing_camera = view.copy()

        # push observing camera back in z-direction in camera space
        z_vec = np.zeros((3,))
        z_vec[2] = 1
        offset_vector = view[:3, :3] @ z_vec
        observing_camera[:3, 3] += offset_vector * self.m_backwards_offset

        # use moving avage of last X cameras (so that observing camera is smooth and follows with slight delay)
        self.m_camera_buffer.append(observing_camera)

        if len(self.m_camera_buffer) > self.m_camera_buffer_size:
            self.m_camera_buffer = self.m_camera_buffer[1:]

    def get_current_view(self):
        """Get current lazy camera view for rendering.

        Returns:
            4x4 matrix
        """

        if self.m_camera_buffer_size == 1:
            return self.m_camera_buffer[0]

        # naive average of camera pose matrices
        smooth_camera_pose = np.zeros((4, 4))
        for camera_pose in self.m_camera_buffer:
            smooth_camera_pose += camera_pose
        smooth_camera_pose /= len(self.m_camera_buffer)

        return self._orthonormalize_rotation(smooth_camera_pose)