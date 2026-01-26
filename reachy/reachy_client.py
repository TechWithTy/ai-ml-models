
"""
Reachy Mini Client Wrapper.
Wraps the 'reachy_mini' SDK for the AI CLI.
"""
import sys
import numpy as np
import logging

# Setup imports
try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
    REACHY_AVAILABLE = True
except ImportError:
    REACHY_AVAILABLE = False
    
logger = logging.getLogger("ReachyClient")

class ReachyMiniClient:
    def __init__(self, media_backend="default"):
        self.available = REACHY_AVAILABLE
        self.media_backend = media_backend
        if not self.available:
            logger.warning("ReachyMini SDK not installed. Running in MOCK mode.")

    def move(self, head_z=0, antennas_deg=None, body_yaw_deg=0, duration=2.0):
        """
        Executes a move command.
        head_z: Height in mm (default 0)
        antennas_deg: List of 2 angles in degrees (e.g. [45, 45])
        body_yaw_deg: Body rotation in degrees (default 0)
        """
        if not self.available:
            return {"status": "mock", "message": f"Moved to z={head_z}, ants={antennas_deg}, yaw={body_yaw_deg}"}
            
        try:
            with ReachyMini(media_backend=self.media_backend) as mini:
                args = {
                    "duration": duration,
                    "method": "minjerk"
                }
                
                if head_z is not None:
                     args["head"] = create_head_pose(z=head_z, mm=True)
                
                if antennas_deg:
                     args["antennas"] = np.deg2rad(antennas_deg)
                
                if body_yaw_deg is not None:
                     args["body_yaw"] = np.deg2rad(body_yaw_deg)
                     
                mini.goto_target(**args)
                
            return {"status": "success", "message": "Movement execution complete"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def see(self, save_path="reachy_view.jpg"):
        """Captures a frame from Reachy's camera."""
        if not self.available:
             return {"status": "mock", "message": f"Saved image to {save_path}"}
             
        try:
            import cv2
            with ReachyMini(media_backend=self.media_backend) as mini:
                frame = mini.media.get_frame()
                # Frame is RGB? SDK says (h, w, 3). Usually CV2 expects BGR for saving.
                # Assuming RGB from SDK, convert for CV2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, frame_bgr)
            return {"status": "success", "file": save_path}
        except Exception as e:
             return {"status": "error", "message": str(e)}
             
    def speak_audio(self, audio_data: np.ndarray, samplerate=16000):
        """Plays raw audio data."""
        if not self.available:
            return {"status": "mock", "message": f"Played {len(audio_data)} samples"}
            
        try:
            import time
            with ReachyMini(media_backend=self.media_backend) as mini:
                mini.media.start_playing()
                mini.media.push_audio_sample(audio_data)
                # Wait for playback
                duration = len(audio_data) / mini.media.get_output_audio_samplerate()
                time.sleep(duration)
                mini.media.stop_playing()
            return {"status": "success", "message": "Audio verified"}
        except Exception as e:
             return {"status": "error", "message": str(e)}
