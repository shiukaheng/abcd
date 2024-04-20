
import time
from typing import Generic, List, Literal, NamedTuple, Tuple, TypeVar, Union
import numpy as np
import torch
import viser
from gs.core.GaussianModel import GaussianModel
import threading
from gs.core.View import KnownView
from gs.geometry.bounding_box import BoundingBox
from gs.helpers.image import torch_to_numpy
import threading
from gs.helpers.transforms import rotmat_to_qvec
from gs.trainers.grid.GridGaussianModel import GridGaussianCell
from gs.compositing.gaussian_rendering_fix import fix_default_blended
from gs.visualization.helpers import build_camera

global shared_viser
shared_viser = {
    "viser": None,
    "viewer": None
}

T = TypeVar('T')

def hsv_to_rgb(h, s, v):
    # Ensure h, s, v are within the expected range [0, 1]
    h = h % 1.0  # h values are cyclic [0, 1)
    i = (h * 6.0).int()
    f = (h * 6.0) % 1.0
    
    w = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    condition_zero = (i == 0)
    condition_one = (i == 1)
    condition_two = (i == 2)
    condition_three = (i == 3)
    condition_four = (i == 4)
    condition_five = (i == 5)
    
    r = torch.where(condition_zero, v, torch.where(condition_one, q, torch.where(condition_two, w, torch.where(condition_three, w, torch.where(condition_four, t, v)))))
    g = torch.where(condition_zero, t, torch.where(condition_one, v, torch.where(condition_two, v, torch.where(condition_three, q, torch.where(condition_four, w, w)))))
    b = torch.where(condition_zero, w, torch.where(condition_one, w, torch.where(condition_two, t, torch.where(condition_three, v, torch.where(condition_four, v, q)))))
    
    return torch.stack((r, g, b), dim=0)  # Stack to make the output tensor as (3, h, w)

def false_color_depth(depth: torch.Tensor, alpha: torch.Tensor, range: Union[Literal["auto"], Tuple[float, float]]="auto") -> torch.Tensor:
    """
    Map a mono-channel depth image to a spectrum color false-color image
    """
    # Use range to normalize depth
    if range == "auto":
        min_depth = depth.min().item()
        max_depth = depth.max().item()
        if min_depth == max_depth:
            min_depth = 0
            max_depth = 1
    else:
        min_depth, max_depth = range
    depth = (depth - min_depth) / (max_depth - min_depth)
    # Interpret depth as hue, alpha as value.
    hue = depth
    saturation = torch.ones_like(hue)
    value = (1-depth) * alpha
    rgb = hsv_to_rgb(hue, saturation, value).squeeze(1)
    return rgb

class GroupSceneNodeHandle(NamedTuple):
    scene_node_handles: List[viser.SceneNodeHandle]

    @property
    def visible(self):
        return self.scene_node_handles[0].visible
    
    @visible.setter
    def visible(self, value):
        for handle in self.scene_node_handles:
            handle.visible = value

    def remove(self):
        for handle in self.scene_node_handles:
            handle.remove()

class Viewer(Generic[T]):
    def __init__(self, width=1920, frame_rate=15, reuse_viser=True, auto_start=True,):

        # Initialize the viewer
        self.model = None
        self.render_once_thread = None # Thread for rendering of each frame. Helps ease load on the main thread when rendering is slow
        self.render_once_last_time = 0 # Time of the last render pass. For skipping render passes if they are too close to each other
        self.render_channel = {} # Which channel to display in the viewer

        # Options for reusing viser. Useful to keep the same viser server (browser window) open, while switching models
        if reuse_viser:
            global shared_viser
            if shared_viser["viser"] is None: # If no viser is present, create one
                shared_viser["viser"] = viser.ViserServer()
                shared_viser["viewer"] = self
            else:
                shared_viser["viewer"].stop(stop_viser=False) # If a viser is present, stop the viewer rendering loop only so we don't have two loops running
            self.viser = shared_viser["viser"]
        else:
            self.viser = viser.ViserServer()
        self.running = False # Flag to stop the rendering loop
        self.frame_rate = frame_rate # Target frame rate
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True) # Thread for the rendering loop
        self.width = width # Width of the rendered images

        if auto_start:
            self.start(threaded=True)

        render_channel_switcher = self.viser.add_gui_dropdown("Render channel", ("rgb", "depth", "alpha"))
        def update_render_channel(gui_event: viser.GuiEvent):
            self.render_channel[gui_event.client_id] = render_channel_switcher.value
        render_channel_switcher.on_update(update_render_channel)

    def set_model(self, model: GaussianModel):
        """
        Set the model to visualize
        """
        self.model = model

    def _send_renders(self, renders):
        """
        Send the renders to the clients
        """
        for cid, client in self.viser.get_clients().items():
            if cid in renders:
                # Lets not use depth for now. It looks bad.
                # A single channel depth representation is not informative for volume rendering. It only works when it approximates a 3D surface.
                # Perhaps there are better representations that Splatfacto is using?
                client.set_background_image(renders[cid]["display"])

    @torch.no_grad()
    def render_once(self):
        """
        Do a single render pass. Sends the renders on another thread.
        """
        # If no model, skip this render pass
        if self.model is None:
            return
        # If render_once_thread is running, skip this render pass
        minimum_time = 1.0 / self.frame_rate
        # If the last render was less than minimum_time ago, skip this render pass
        if time.time() - self.render_once_last_time < minimum_time:
            return
        if self.render_once_thread is not None and self.render_once_thread.is_alive():
            return
        clients = self.viser.get_clients()
        output = {}
        for cid, client in clients.items():
            camera = build_camera(client.camera, width=self.width).to(self.model.positions.device)
            render, depth, alpha = self.model.forward(camera)
            render = render.detach().cpu()
            # We need to normalize depth and convert to 3 channels for visualization
            depth = fix_default_blended(depth, alpha)
            depth_raw = depth.detach().cpu()
            depth = false_color_depth(depth, alpha).detach().cpu()
            # We need to convert alpha to 3 channels for visualization
            alpha = alpha.detach().cpu().repeat(1, 3, 1, 1)
            channels = { "rgb": render, "depth": depth, "alpha": alpha }

            # Get channel to display
            if cid not in self.render_channel:
                self.render_channel[cid] = "rgb"
            else:
                self.render_channel[cid] = self.render_channel[cid]
            output[cid] = {
                "display": torch_to_numpy(channels[self.render_channel[cid]]),
                "depth": torch_to_numpy(depth_raw)
            } # Send the render to the client
            del camera
        self.render_once_thread = threading.Thread(target=self._send_renders, args=(output,))
        self.render_once_thread.start()
        self.render_once_last_time = time.time()

    def render_loop(self):
        """
        Run a render loop at the specified frame rate
        """
        with torch.no_grad():
            target_time = 1.0 / self.frame_rate
            while self.running:
                start_time = time.time()
                self.render_once()
                elapsed_time = time.time() - start_time
                sleep_time = target_time - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def start(self, threaded=False):
        """
        Start a rendering loop.
        """
        self.running = True
        stop_button = self.viser.add_gui_button("Exit viewer")
        @stop_button.on_click
        def on_stop_button_click(client: viser.ClientHandle):
            self.stop()
        if threaded:
            if not self.render_thread.is_alive():
                self.render_thread.start()
        else:
            self.render_loop()

    def stop(self, stop_viser=True):
        """
        Stop the rendering loop and close the Viser server
        """
        self.running = False
        self.render_thread.join()
        if stop_viser:
            self.viser.stop()

    def add_camera(self, camera: KnownView[T], camera_scale=0.3, color=(0, 0, 1), show_image=True):
        """
        Add camera to the viewer
        """
        name = f"/cameras/{camera.id}/frustum"
        return self.viser.add_camera_frustum(
            name,
            camera.fov_y,
            camera.aspect_ratio,
            camera_scale,
            color,
            torch_to_numpy(camera.image) if show_image else None,
            "jpeg",
            jpeg_quality=None,
            wxyz=rotmat_to_qvec(camera.R),
            position=camera.center,
        )
    
    def add_cell_bounary(self, cell: GridGaussianCell, color=(255, 255, 255), line_width=2):
        """
        Add bounding box to the viewer
        """
        name = f"/bounding_boxes/{cell.index.to_string_id()}"
        # Unfortuately Viser does not support creating 3D wireframe boxes. We can instead use 6 1x1 grid planes to represent the bounding box
        curve_args = []
        for i, segment in enumerate(cell.bounding_box.get_edges()):
            curve_args.append({
                "name": f"{name}/{i}",
                "positions": segment,
                "control_points": segment,
                "line_width": line_width,
                "color": color,
                "segments": 1,
            })

        return GroupSceneNodeHandle([
            self.viser.add_spline_cubic_bezier(**args) for args in curve_args
        ])
        
        








