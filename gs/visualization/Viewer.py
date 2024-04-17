
import time
import torch
import viser
from gs.core.GaussianModel import GaussianModel
import threading
from gs.helpers.image import torch_to_numpy
import threading
from gs.visualization.helpers import build_camera

global shared_viser
shared_viser = {
    "viser": None,
    "viewer": None
}

class Viewer():
    def __init__(self, model: GaussianModel, width=1920, frame_rate=15, reuse_viser=True, auto_start=True):

        # Initialize the viewer
        self.model = model
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
            self.start()

        render_channel_switcher = self.viser.add_gui_dropdown("Render channel", ("rgb", "depth", "alpha"))
        def update_render_channel(gui_event: viser.GuiEvent):
            self.render_channel[gui_event.client_id] = render_channel_switcher.value
        render_channel_switcher.on_update(update_render_channel)

    def _send_renders(self, renders):
        """
        Send the renders to the clients
        """
        for cid, client in self.viser.get_clients().items():
            if cid in renders:
                client.set_background_image(renders[cid])

    def render_once(self):
        """
        Do a single render pass. Sends the renders on another thread.
        """
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
            render: torch.Tensor = render.detach().cpu()
            # We need to normalize depth and convert to 3 channels for visualization
            depth: torch.Tensor = depth.detach().cpu()
            depth = 1 - (depth / 5.).clamp(0, 1)
            depth = depth.repeat(1, 3, 1, 1)
            # We need to convert alpha to 3 channels for visualization
            alpha: torch.Tensor = alpha.detach().cpu().repeat(1, 3, 1, 1)
            channels = { "rgb": render, "depth": depth, "alpha": alpha }

            # Get channel to display
            if cid not in self.render_channel:
                self.render_channel[cid] = "rgb"
            else:
                self.render_channel[cid] = self.render_channel[cid]
            output[cid] = torch_to_numpy(channels[self.render_channel[cid]]) # Send the render to the client
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