from streamlit_webrtc import ClientSettings

#CLASSES = [ 'pistol' ]
CLASSES = [ 'mask', 'nomask', 'person', 'pistol']
#DICT_MODELS = {'mask': 'MASK', 'no_mask': 'MASK', 'person': 'MAIN', 'pistol': 'PISTOL'}
#CLASSES_IDS = {'mask': 0, 'no_mask': 1, 'person': 0, 'pistol': 0}

WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
